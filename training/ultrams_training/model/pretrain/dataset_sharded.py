"""
分片流式数据集模块
==================
用于大规模分布式训练的IterableDataset实现

核心特性：
- 按rank分配shard，天然负载均衡
- 流式读取，内存恒定
- 支持实时指纹计算

Author: Ultraexplorer Team
Date: 2025-12-13
"""

import os
import json
import warnings
import logging
import ctypes
import numpy as np
import torch
import pyarrow.feather as feather
from functools import partial
from typing import List, Dict, Optional, Iterator
from torch.utils.data import IterableDataset, DataLoader

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None
_POSIX_FADV_DONTNEED = 4


def _drop_file_cache(path: str):
    """Tell OS to evict page cache for a file via posix_fadvise(DONTNEED)."""
    if _libc is None:
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        _libc.posix_fadvise(fd, 0, size, _POSIX_FADV_DONTNEED)
    except Exception:
        pass
    finally:
        if fd is not None:
            os.close(fd)


# RDKit日志过滤
for logger_name in ["rdkit", "rdkit.Chem", "rdkit.Chem.AllChem"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except:
    pass


def generate_fingerprint_batch(
    smiles_list: List[str], fingerprint_size: int = 2048
) -> np.ndarray:
    """
    批量生成分子指纹

    Args:
        smiles_list: SMILES列表
        fingerprint_size: 指纹维度

    Returns:
        指纹数组 [batch_size, fingerprint_size]
    """
    fingerprints = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=fingerprint_size
                )
                fp_arr = np.zeros(fingerprint_size, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_arr)
                fingerprints.append(fp_arr)
            else:
                fingerprints.append(np.zeros(fingerprint_size, dtype=np.float32))
        except:
            fingerprints.append(np.zeros(fingerprint_size, dtype=np.float32))

    return np.array(fingerprints, dtype=np.float32)


class ShardedIterableDataset(IterableDataset):
    """
    分片流式数据集

    特性：
    - 按rank分配shard文件
    - 顺序读取，用完释放
    - 支持多worker（每个worker处理不同的shard）
    """

    def __init__(
        self,
        shard_dir: str,
        rank: int = 0,
        world_size: int = 1,
        fingerprint_size: int = 2048,
        shuffle_shards: bool = False,
        seed: int = 42,
        augmentation_enabled: bool = False,
        augmentation_shard_suffix: str = "_aug",
        augmentation_shard_dir: str = None,
    ):
        """
        Args:
            shard_dir: shard文件目录
            rank: 当前进程的rank
            world_size: 总进程数
            fingerprint_size: 指纹维度
            shuffle_shards: 是否每个epoch打乱shard顺序
            seed: 随机种子
            augmentation_enabled: 是否加载增强数据
            augmentation_shard_suffix: 增强shard的后缀
            augmentation_shard_dir: 增强shard的目录（None则使用shard_dir）
        """
        super().__init__()

        self.shard_dir = shard_dir
        self.rank = rank
        self.world_size = world_size
        self.fingerprint_size = fingerprint_size
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0
        self.augmentation_enabled = augmentation_enabled
        self.augmentation_shard_suffix = augmentation_shard_suffix
        self.augmentation_shard_dir = augmentation_shard_dir or shard_dir

        # 加载shard索引
        index_path = os.path.join(shard_dir, "shard_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"找不到shard索引文件: {index_path}")

        with open(index_path, "r") as f:
            self.index_data = json.load(f)

        self.num_shards = self.index_data["num_shards"]
        self.total_samples = self.index_data["total_samples"]

        # 构建shard列表：[(path, num_samples), ...]
        # 原始shard
        self.all_shards = []
        for sinfo in self.index_data["shards"]:
            path = os.path.join(shard_dir, sinfo["filename"])
            self.all_shards.append((path, sinfo["num_samples"]))

        # 增强shard（直接加入列表，和原始shard一样处理）
        if augmentation_enabled:
            aug_index_path = os.path.join(
                self.augmentation_shard_dir,
                f"augmentation_index{augmentation_shard_suffix}.json",
            )
            if os.path.exists(aug_index_path):
                with open(aug_index_path, "r") as f:
                    aug_index = json.load(f)

                aug_count = 0
                for sinfo in aug_index.get("shards", []):
                    path = os.path.join(self.augmentation_shard_dir, sinfo["output"])
                    if os.path.exists(path):
                        self.all_shards.append((path, sinfo["augmented_samples"]))
                        aug_count += 1

                if rank == 0:
                    print(
                        f"  ✓ 增强shard: {aug_count} 个, 总shard: {len(self.all_shards)}"
                    )
            else:
                if rank == 0:
                    print(f"  ⚠️ 增强索引不存在: {aug_index_path}")

        # 分配shard给当前rank
        total_shards = len(self.all_shards)
        raw_shard_ids = list(range(rank, total_shards, world_size))

        # 取最小值确保负载均衡
        min_shards_per_rank = total_shards // world_size
        self.my_shard_ids = raw_shard_ids[:min_shards_per_rank]
        self.dropped_shards = len(raw_shard_ids) - len(self.my_shard_ids)

        # 计算本rank的样本数
        self.my_samples = sum(self.all_shards[sid][1] for sid in self.my_shard_ids)
        self.used_samples = min_shards_per_rank * world_size

    def set_epoch(self, epoch: int):
        """设置当前epoch（用于shard顺序变化）"""
        self.epoch = epoch

    def _get_shard_order(self) -> List[int]:
        """获取当前epoch的shard顺序"""
        shard_ids = self.my_shard_ids.copy()

        if self.shuffle_shards:
            # 使用epoch作为额外种子，确保每个epoch顺序不同
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(shard_ids)

        return shard_ids

    def _load_shard(self, shard_id: int) -> List[Dict]:
        """加载单个shard（从all_shards列表）"""
        shard_path, _ = self.all_shards[shard_id]
        table = feather.read_table(shard_path)
        return self._parse_arrow_table(table)

    def _parse_arrow_table(self, table) -> List[Dict]:
        """解析Arrow表为数据列表"""
        shard_data = []

        mz_col = table["mz"].to_pylist()
        intensity_col = table["intensity"].to_pylist()
        smiles_col = table["smiles"].to_pylist()
        precursor_mz_col = table["precursor_mz"].to_pylist()
        adduct_col = table["adduct"].to_pylist()
        nce_level_col = table["nce_level"].to_pylist()

        for i in range(len(table)):
            mz = mz_col[i]
            intensity = intensity_col[i]
            if len(mz) > 0:
                spectrum = np.column_stack([mz, intensity]).astype(np.float32)
            else:
                spectrum = np.array([], dtype=np.float32).reshape(0, 2)

            nce = nce_level_col[i]
            shard_data.append(
                {
                    "spectrum": spectrum,
                    "smiles": smiles_col[i],
                    "precursor_mz": precursor_mz_col[i],
                    "adduct": adduct_col[i] if adduct_col[i] else None,
                    "nce_level": nce if nce >= 0 else None,
                }
            )

        return shard_data

    def __iter__(self) -> Iterator[Dict]:
        """迭代数据"""
        worker_info = torch.utils.data.get_worker_info()

        # 获取shard顺序
        shard_ids = self._get_shard_order()

        # 如果有多个worker，进一步分配shard
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # 每个worker处理不同的shard
            shard_ids = shard_ids[worker_id::num_workers]
            # 如果worker没有分到shard，直接返回空
            if len(shard_ids) == 0:
                return

        # 遍历分配的shard
        for shard_id in shard_ids:
            shard_data = self._load_shard(shard_id)
            for item in shard_data:
                yield item
            del shard_data

    def __len__(self) -> int:
        """返回本rank的样本数（近似值，用于进度条）"""
        return self.my_samples


def collate_fn_sharded(batch: List[Dict], fingerprint_size: int = 2048) -> Dict:
    """
    分片数据集的collate函数

    批量计算指纹，避免逐条计算的开销
    """
    # 提取字段
    spectra = [item["spectrum"] for item in batch]
    precursor_mzs = torch.tensor(
        [item["precursor_mz"] for item in batch], dtype=torch.float32
    )
    smiles_list = [item["smiles"] for item in batch]
    adducts = [item.get("adduct", None) for item in batch]
    nce_levels = [item.get("nce_level", None) for item in batch]

    # 批量计算指纹
    fingerprints_np = generate_fingerprint_batch(smiles_list, fingerprint_size)
    fingerprints = torch.from_numpy(fingerprints_np)

    return {
        "spectrum": spectra,
        "precursor_mz": precursor_mzs,
        "smiles": smiles_list,
        "fingerprint": fingerprints,
        "adduct": adducts,
        "nce_level": nce_levels,
    }


def create_sharded_dataloader(
    shard_dir: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    fingerprint_size: int = 2048,
    pin_memory: bool = True,
    shuffle_shards: bool = False,
    seed: int = 42,
    augmentation_enabled: bool = False,
    augmentation_shard_suffix: str = "_aug",
    augmentation_shard_dir: str = None,
) -> DataLoader:
    """
    创建分片数据加载器

    Args:
        shard_dir: shard目录
        batch_size: 批次大小
        rank: 当前rank
        world_size: 总进程数
        num_workers: DataLoader worker数
        fingerprint_size: 指纹维度
        pin_memory: 是否pin memory
        shuffle_shards: 是否每epoch打乱shard顺序
        seed: 随机种子
        augmentation_enabled: 是否加载增强数据
        augmentation_shard_suffix: 增强shard的后缀
        augmentation_shard_dir: 增强shard目录（None则使用shard_dir）

    Returns:
        DataLoader实例
    """
    dataset = ShardedIterableDataset(
        shard_dir=shard_dir,
        rank=rank,
        world_size=world_size,
        fingerprint_size=fingerprint_size,
        shuffle_shards=shuffle_shards,
        seed=seed,
        augmentation_enabled=augmentation_enabled,
        augmentation_shard_suffix=augmentation_shard_suffix,
        augmentation_shard_dir=augmentation_shard_dir,
    )

    # 使用partial创建可pickle的collate_fn
    collate_fn = partial(collate_fn_sharded, fingerprint_size=fingerprint_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset不支持shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )

    return dataloader


def get_shard_info(shard_dir: str) -> Dict:
    """获取shard信息"""
    index_path = os.path.join(shard_dir, "shard_index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到shard索引文件: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)

    return index_data


# ============================================================================
# Parquet 格式支持（Ae1 数据）
# ============================================================================
import pyarrow.parquet as pq


POLARITY_TO_ADDUCT = {1: "[M+H]+", 0: "[M-H]-"}


class ParquetShardDataset(IterableDataset):
    """
    读取 Ae1 parquet 分片数据。

    列:
    - spectrum_mz, spectrum_intensity: list<double>，含零填充
    - polarity: int (1=positive, 0=negative)
    - precursor_mz: double
    - collision_energy: double
    """

    def __init__(
        self,
        shard_dir: str,
        rank: int = 0,
        world_size: int = 1,
        max_peaks: int = 100,
        shuffle_shards: bool = True,
        seed: int = 42,
        clean_shard_list: str = None,
        max_precursor_mz: float = None,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.rank = rank
        self.world_size = world_size
        self.max_peaks = max_peaks
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.max_precursor_mz = max_precursor_mz
        import multiprocessing as _mp

        self._shared_epoch = _mp.Value("i", 0)

        if clean_shard_list and os.path.isfile(clean_shard_list):
            with open(clean_shard_list) as f:
                all_files = sorted([line.strip() for line in f if line.strip()])
            if rank == 0:
                print(
                    f"[ParquetShardDataset] Using clean shard list: {len(all_files)} shards"
                )
        else:
            all_files = sorted(
                [f for f in os.listdir(shard_dir) if f.endswith(".parquet")]
            )
        self.all_shards = [os.path.join(shard_dir, f) for f in all_files]
        self.num_shards = len(self.all_shards)

        per_rank = self.num_shards // world_size
        self.my_shard_ids = list(range(rank, per_rank * world_size, world_size))[
            :per_rank
        ]

        if rank == 0:
            print(
                f"[ParquetShardDataset] {self.num_shards} shards, {per_rank} per rank"
            )

    @property
    def epoch(self):
        return self._shared_epoch.value

    def set_epoch(self, epoch: int):
        self._shared_epoch.value = epoch

    def __iter__(self) -> Iterator[Dict]:
        shard_ids = self.my_shard_ids.copy()
        epoch = self.epoch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if self.shuffle_shards:
            rng = np.random.RandomState(self.seed + epoch * 100 + self.rank)
            rng.shuffle(shard_ids)

        if worker_info is not None:
            shard_ids = shard_ids[worker_info.id :: worker_info.num_workers]

        row_rng = np.random.RandomState(
            self.seed + epoch * 1000 + self.rank * 10 + worker_id
        )

        for sid in shard_ids:
            path = self.all_shards[sid]
            try:
                table = pq.read_table(path)
            except Exception as e:
                raise RuntimeError(f"failed to read required Parquet shard: {path}") from e
            cols = table.column_names
            n = table.num_rows
            mz_cname = "spectrum_mz" if "spectrum_mz" in cols else "mz"
            int_cname = (
                "spectrum_intensity" if "spectrum_intensity" in cols else "intensity"
            )

            mz_arrow = table.column(mz_cname).combine_chunks()
            int_arrow = table.column(int_cname).combine_chunks()
            peak_len = int(np.diff(mz_arrow.offsets.to_numpy()[:2])[0])
            mz_2d = mz_arrow.values.to_numpy().astype(np.float32).reshape(n, peak_len)
            int_2d = int_arrow.values.to_numpy().astype(np.float32).reshape(n, peak_len)
            pmz_arr = table.column("precursor_mz").to_numpy().astype(np.float32)

            has_pol = "polarity" in cols
            pol_arr = table.column("polarity").to_numpy() if has_pol else None
            has_add = "adduct" in cols
            add_list = table.column("adduct").to_pylist() if has_add else None
            has_smi = "smiles" in cols
            smi_list = table.column("smiles").to_pylist() if has_smi else None
            has_ce = "collision_energy" in cols
            ce_arr = (
                table.column("collision_energy").to_numpy().astype(np.float32)
                if has_ce
                else None
            )
            has_rt = "RT" in cols
            rt_arr = (
                table.column("RT").to_numpy().astype(np.float32) if has_rt else None
            )

            del table, mz_arrow, int_arrow

            if self.max_precursor_mz is not None:
                pmz_keep = pmz_arr <= self.max_precursor_mz
                valid_rows = np.where(pmz_keep)[0]
                row_order = row_rng.permutation(len(valid_rows))
                row_order = valid_rows[row_order]
            else:
                row_order = row_rng.permutation(n)

            for i in row_order:
                mz_arr = mz_2d[i]
                int_arr = int_2d[i]

                mask = mz_arr > 0
                mz_arr = mz_arr[mask]
                int_arr = int_arr[mask]

                if len(mz_arr) < 3:
                    continue

                max_int = int_arr.max()
                if max_int > 0:
                    int_arr = int_arr / max_int
                int_arr = np.clip(int_arr, 0, 1)

                if len(mz_arr) > self.max_peaks:
                    top_idx = np.argsort(int_arr)[-self.max_peaks :]
                    top_idx = np.sort(top_idx)
                    mz_arr = mz_arr[top_idx]
                    int_arr = int_arr[top_idx]

                prec_mz = float(pmz_arr[i])
                mz_arr = np.concatenate([[prec_mz], mz_arr])
                int_arr = np.concatenate([[1.1], int_arr])

                target_len = self.max_peaks + 1
                if len(mz_arr) < target_len:
                    pad_len = target_len - len(mz_arr)
                    mz_arr = np.concatenate(
                        [mz_arr, np.zeros(pad_len, dtype=np.float32)]
                    )
                    int_arr = np.concatenate(
                        [int_arr, np.zeros(pad_len, dtype=np.float32)]
                    )

                spectrum = np.stack([mz_arr, int_arr], axis=-1)

                adduct = add_list[i] if has_add else None
                polarity = pol_arr[i] if pol_arr is not None else None
                if adduct is None and polarity is not None:
                    try:
                        adduct = POLARITY_TO_ADDUCT.get(int(polarity), None)
                    except (ValueError, TypeError):
                        pass

                rt_val = float(rt_arr[i]) if rt_arr is not None else 0.0
                if np.isnan(rt_val):
                    rt_val = 0.0

                yield {
                    "spectrum": spectrum,
                    "precursor_mz": prec_mz,
                    "adduct": adduct,
                    "polarity": polarity,
                    "smiles": smi_list[i] if has_smi else None,
                    "nce_level": None,
                    "collision_energy": float(ce_arr[i])
                    if ce_arr is not None
                    else None,
                    "rt": rt_val,
                }

            del mz_2d, int_2d, pmz_arr
            if add_list is not None:
                del add_list
            if smi_list is not None:
                del smi_list
            if pol_arr is not None:
                del pol_arr
            if ce_arr is not None:
                del ce_arr
            if rt_arr is not None:
                del rt_arr
            _drop_file_cache(path)


def collate_fn_parquet(batch: List[Dict]) -> Dict:
    """Parquet 数据的 collate 函数"""
    spectra = [item["spectrum"] for item in batch]
    precursor_mzs = torch.tensor(
        [item["precursor_mz"] for item in batch], dtype=torch.float32
    )
    adducts = [item.get("adduct", None) for item in batch]
    nce_levels = [item.get("nce_level", None) for item in batch]
    smiles_list = [item.get("smiles", None) for item in batch]
    polarities = [item.get("polarity", None) for item in batch]
    rt_vals = [item.get("rt", 0.0) for item in batch]

    return {
        "spectrum": spectra,
        "precursor_mz": precursor_mzs,
        "adduct": adducts,
        "nce_level": nce_levels,
        "smiles": smiles_list,
        "polarity": polarities,
        "rt": rt_vals,
    }


def create_parquet_dataloader(
    shard_dir: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    max_peaks: int = 100,
    pin_memory: bool = True,
    shuffle_shards: bool = True,
    seed: int = 42,
    clean_shard_list: str = None,
    max_precursor_mz: float = None,
) -> DataLoader:
    dataset = ParquetShardDataset(
        shard_dir=shard_dir,
        rank=rank,
        world_size=world_size,
        max_peaks=max_peaks,
        shuffle_shards=shuffle_shards,
        seed=seed,
        clean_shard_list=clean_shard_list,
        max_precursor_mz=max_precursor_mz,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_parquet,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )


def get_parquet_shard_info(shard_dir: str) -> Dict:
    files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".parquet")])
    total_rows = 0
    for f in files:
        total_rows += pq.read_metadata(os.path.join(shard_dir, f)).num_rows
    avg_per_shard = total_rows // max(len(files), 1)
    return {
        "num_shards": len(files),
        "total_samples": total_rows,
        "avg_per_shard": avg_per_shard,
    }


# ============================================================================
# MLM-optimized collate: CPU preprocessing done here by DataLoader workers
# ============================================================================


def _prep_spectrum_np(spec_np: np.ndarray, max_peaks: int) -> np.ndarray:
    """Filter, normalize, top-k, sort. Returns (n_peaks, 2) float32."""
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    mx = inten.max()
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0, 1)
    if len(mz) > max_peaks:
        idx = np.argsort(inten)[-max_peaks:]
        idx = np.sort(idx)
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


def _apply_mask_np(spec: np.ndarray, mask_ratio: float):
    """Random masking. Returns (masked, orig, mask_flags) all numpy."""
    n = len(spec)
    if n == 0:
        empty2 = np.zeros((0, 2), dtype=np.float32)
        return empty2, empty2, np.zeros(0, dtype=np.bool_)
    nm = max(1, int(n * mask_ratio))
    idx = np.random.choice(n, size=nm, replace=False)
    mask = np.zeros(n, dtype=np.bool_)
    mask[idx] = True
    masked = spec.copy()
    masked[idx] = 0.0
    return masked, spec, mask


def collate_fn_mlm(batch: List[Dict], max_peaks: int, mask_ratio: float) -> Dict:
    """
    MLM collate: does all CPU preprocessing (prep + mask + pad) so that
    forward() receives ready-to-use padded tensors — no per-sample loops.
    """
    B = len(batch)
    masked_spectra = np.zeros((B, max_peaks, 2), dtype=np.float32)
    orig_spectra = np.zeros((B, max_peaks, 2), dtype=np.float32)
    mask_flags = np.zeros((B, max_peaks), dtype=np.bool_)
    attn_mask = np.zeros((B, max_peaks), dtype=np.int64)
    precursor_mzs = np.empty(B, dtype=np.float32)
    rt_vals = np.zeros(B, dtype=np.float32)
    pol_vals = np.full(B, -1, dtype=np.int64)
    for i, item in enumerate(batch):
        precursor_mzs[i] = item["precursor_mz"]
        rt_vals[i] = item.get("rt", 0.0)
        p = item.get("polarity", None)
        if p is not None and p in (0, 1):
            pol_vals[i] = int(p)
        spec = item["spectrum"]
        if spec.ndim == 2 and spec.shape[0] > 1:
            spec = spec[1:]  # strip prepended precursor (pos 0, int=1.1)
        prepped = _prep_spectrum_np(spec, max_peaks)
        n = len(prepped)
        if n == 0:
            continue
        masked, orig, mf = _apply_mask_np(prepped, mask_ratio)
        masked_spectra[i, :n] = masked
        orig_spectra[i, :n] = orig
        mask_flags[i, :n] = mf
        attn_mask[i, :n] = 1

    return {
        "masked_spectra": torch.from_numpy(masked_spectra),
        "orig_spectra": torch.from_numpy(orig_spectra),
        "mask_flags": torch.from_numpy(mask_flags),
        "attn_mask": torch.from_numpy(attn_mask),
        "precursor_mz": torch.from_numpy(precursor_mzs),
        "rt": torch.from_numpy(rt_vals),
        "polarity": torch.from_numpy(pol_vals),
    }


def _augment_for_vicreg(spec: np.ndarray, n_peaks: int):
    """Peak dropout + intensity noise for VICReg view2.
    spec: (max_peaks, 2) padded array, n_peaks: number of valid peaks."""
    if n_peaks < 3:
        return spec.copy(), n_peaks
    aug = spec.copy()
    n_drop = max(1, int(n_peaks * 0.1))
    drop_idx = np.random.choice(n_peaks, n_drop, replace=False)
    aug[drop_idx] = 0.0
    valid_mask = aug[:n_peaks, 0] > 0
    valid = aug[:n_peaks][valid_mask]
    n_valid = int(len(valid))
    if n_valid < 3:
        return spec.copy(), n_peaks
    noise = np.random.normal(0, 0.03, size=n_valid).astype(np.float32)
    valid[:, 1] = np.clip(valid[:, 1] + noise, 0, 1)
    out = np.zeros_like(spec)
    out[:n_valid] = valid
    return out, n_valid


def collate_fn_mlm_vicreg(batch: List[Dict], max_peaks: int, mask_ratio: float) -> Dict:
    """MLM collate + VICReg view2 (augmented orig)."""
    B = len(batch)
    masked_spectra = np.zeros((B, max_peaks, 2), dtype=np.float32)
    orig_spectra = np.zeros((B, max_peaks, 2), dtype=np.float32)
    mask_flags = np.zeros((B, max_peaks), dtype=np.bool_)
    attn_mask = np.zeros((B, max_peaks), dtype=np.int64)
    precursor_mzs = np.empty(B, dtype=np.float32)
    rt_vals = np.zeros(B, dtype=np.float32)
    view2 = np.zeros((B, max_peaks, 2), dtype=np.float32)
    view2_attn = np.zeros((B, max_peaks), dtype=np.int64)

    for i, item in enumerate(batch):
        precursor_mzs[i] = item["precursor_mz"]
        rt_vals[i] = item.get("rt", 0.0)
        spec = item["spectrum"]
        if spec.ndim == 2 and spec.shape[0] > 1:
            spec = spec[1:]
        prepped = _prep_spectrum_np(spec, max_peaks)
        n = len(prepped)
        if n == 0:
            continue
        masked, orig, mf = _apply_mask_np(prepped, mask_ratio)
        masked_spectra[i, :n] = masked
        orig_spectra[i, :n] = orig
        mask_flags[i, :n] = mf
        attn_mask[i, :n] = 1
        aug, n_aug = _augment_for_vicreg(orig_spectra[i], n)
        view2[i] = aug
        view2_attn[i, :n_aug] = 1

    return {
        "masked_spectra": torch.from_numpy(masked_spectra),
        "orig_spectra": torch.from_numpy(orig_spectra),
        "mask_flags": torch.from_numpy(mask_flags),
        "attn_mask": torch.from_numpy(attn_mask),
        "precursor_mz": torch.from_numpy(precursor_mzs),
        "rt": torch.from_numpy(rt_vals),
        "view2": torch.from_numpy(view2),
        "view2_attn": torch.from_numpy(view2_attn),
    }


def create_parquet_dataloader_mlm_vicreg(
    shard_dir: str,
    batch_size: int,
    max_peaks: int,
    mask_ratio: float,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_shards: bool = True,
    seed: int = 42,
    clean_shard_list: str = None,
    max_precursor_mz: float = None,
) -> DataLoader:
    """DataLoader with MLM + VICReg collate (CPU prep + mask + pad + augment)."""
    from functools import partial

    dataset = ParquetShardDataset(
        shard_dir=shard_dir,
        rank=rank,
        world_size=world_size,
        max_peaks=max_peaks,
        shuffle_shards=shuffle_shards,
        seed=seed,
        clean_shard_list=clean_shard_list,
        max_precursor_mz=max_precursor_mz,
    )
    cfn = partial(collate_fn_mlm_vicreg, max_peaks=max_peaks, mask_ratio=mask_ratio)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cfn,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )


def create_parquet_dataloader_mlm(
    shard_dir: str,
    batch_size: int,
    max_peaks: int,
    mask_ratio: float,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_shards: bool = True,
    seed: int = 42,
    clean_shard_list: str = None,
    max_precursor_mz: float = None,
    drop_last: bool = True,
) -> DataLoader:
    """DataLoader with MLM-optimized collate (CPU prep + mask + pad)."""
    from functools import partial

    dataset = ParquetShardDataset(
        shard_dir=shard_dir,
        rank=rank,
        world_size=world_size,
        max_peaks=max_peaks,
        shuffle_shards=shuffle_shards,
        seed=seed,
        clean_shard_list=clean_shard_list,
        max_precursor_mz=max_precursor_mz,
    )
    cfn = partial(collate_fn_mlm, max_peaks=max_peaks, mask_ratio=mask_ratio)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=pin_memory,
        collate_fn=cfn,
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )
