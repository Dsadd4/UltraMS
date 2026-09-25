"""Convert the public SpecBridge MSnLib MGF and candidate pickle for benchmarking."""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional

# 数据集配置
DATASETS = {'msnlib': {'output_name': 'MSnLib'}}


def parse_mgf_file(mgf_path: str) -> List[Dict]:
    """解析 MGF 文件，返回数据列表"""
    print(f"\n解析 MGF 文件: {mgf_path}")

    try:
        from pyteomics import mgf
    except ImportError:
        raise ImportError("请安装 pyteomics: pip install pyteomics")

    data_list = []
    fold_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}

    with mgf.MGF(mgf_path) as reader:
        for spec in tqdm(reader, desc="读取谱图"):
            params = spec.get('params', {})

            # 提取质谱峰
            mzs = spec.get('m/z array', np.array([]))
            intensities = spec.get('intensity array', np.array([]))

            if len(mzs) == 0:
                continue

            # 提取元数据
            smiles = params.get('SMILES') or params.get('smiles', '')
            if not smiles:
                continue

            # 前体离子质量
            pepmass = params.get('PEPMASS') or params.get('pepmass')
            if pepmass is None:
                precursor_mz = 0.0
            elif isinstance(pepmass, (list, tuple)):
                precursor_mz = float(pepmass[0])
            else:
                precursor_mz = float(pepmass)

            # 加合物
            adduct = params.get('ADDUCT') or params.get('adduct', '')
            if adduct:
                adduct = str(adduct).strip()

            # 碰撞能量 - 直接保留原值
            collision_energy = params.get('COLLISION_ENERGY') or params.get('collision_energy')
            if collision_energy is not None:
                # 处理列表形式的碰撞能量（取第一个有效值）
                if isinstance(collision_energy, (list, tuple)):
                    collision_energy = collision_energy[0] if collision_energy else None
                if isinstance(collision_energy, str):
                    collision_energy = collision_energy.strip('[]').split(',')[0].strip()
                try:
                    collision_energy = float(collision_energy) if collision_energy and collision_energy != 'nan' else None
                except (ValueError, TypeError):
                    collision_energy = None

            # 数据划分
            fold = params.get('FOLD') or params.get('fold', 'unknown')
            fold = str(fold).strip().lower()
            if fold == 'valid' or fold == 'validation':
                fold = 'val'

            fold_counts[fold] = fold_counts.get(fold, 0) + 1

            data_item = {
                'mzs': mzs.astype(np.float32),
                'intensities': intensities.astype(np.float32),
                'smiles': smiles,
                'precursor_mz': precursor_mz,
                'adduct': adduct,
                'collision_energy': collision_energy,
                'fold': fold,
            }
            data_list.append(data_item)

    print(f"✓ 解析完成: {len(data_list)} 条谱图")
    print(f"  Fold 分布: {fold_counts}")

    return data_list


def convert_to_arrow_shards(
    data_list: List[Dict],
    output_dir: str,
    num_gpus: int = 6,
    shard_size: Optional[int] = None
):
    """将数据列表转换为 Arrow shards（用于训练）"""
    import pyarrow as pa
    import pyarrow.feather as feather
    os.makedirs(output_dir, exist_ok=True)

    total_samples = len(data_list)

    # 计算 shard 大小
    if shard_size is None:
        target_shards_per_gpu = 11
        target_num_shards = num_gpus * target_shards_per_gpu
        shard_size = total_samples // target_num_shards
        num_shards = (total_samples // shard_size // num_gpus) * num_gpus
        shard_size = total_samples // num_shards
    else:
        num_shards = total_samples // shard_size
        num_shards = (num_shards // num_gpus) * num_gpus
        shard_size = total_samples // num_shards

    print(f"\n生成 Arrow Shards:")
    print(f"  - 总样本数: {total_samples:,}")
    print(f"  - Shard数量: {num_shards}")
    print(f"  - 每个Shard: {shard_size:,} 样本")
    print(f"  - 输出目录: {output_dir}")

    # 打乱数据
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    shuffled_data = [data_list[i] for i in indices]

    shard_info = []

    for shard_idx in tqdm(range(num_shards), desc="保存Shards"):
        start_idx = shard_idx * shard_size
        end_idx = start_idx + shard_size
        shard_data = shuffled_data[start_idx:end_idx]

        # 构建 Arrow Table
        mz_list = [item['mzs'].tolist() for item in shard_data]
        intensity_list = [item['intensities'].tolist() for item in shard_data]
        smiles_list = [item['smiles'] for item in shard_data]
        precursor_mz_list = [item['precursor_mz'] for item in shard_data]
        adduct_list = [item['adduct'] if item['adduct'] else '' for item in shard_data]
        nce_level_list = [int(item['collision_energy']) if item['collision_energy'] is not None else -1
                         for item in shard_data]

        table = pa.table({
            'mz': pa.array(mz_list, type=pa.list_(pa.float32())),
            'intensity': pa.array(intensity_list, type=pa.list_(pa.float32())),
            'smiles': pa.array(smiles_list, type=pa.string()),
            'precursor_mz': pa.array(precursor_mz_list, type=pa.float32()),
            'adduct': pa.array(adduct_list, type=pa.string()),
            'nce_level': pa.array(nce_level_list, type=pa.int16())
        })

        shard_filename = f"shard_{shard_idx:03d}.arrow"
        shard_path = os.path.join(output_dir, shard_filename)
        feather.write_feather(table, shard_path, compression='lz4')

        file_size = os.path.getsize(shard_path) / 1024**3
        shard_info.append({
            'filename': shard_filename,
            'num_samples': len(shard_data),
            'size_gb': file_size
        })

    # 保存索引文件
    index_path = os.path.join(output_dir, "shard_index.json")
    index_data = {
        'format': 'arrow',
        'num_shards': num_shards,
        'total_samples': num_shards * shard_size,
        'original_samples': total_samples,
        'shard_size': shard_size,
        'num_gpus': num_gpus,
        'shards_per_gpu': num_shards // num_gpus,
        'shards': shard_info
    }
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)

    total_size = sum(s['size_gb'] for s in shard_info)
    print(f"✓ Shards 生成完成!")
    print(f"  - 共 {num_shards} 个shard")
    print(f"  - 总大小: {total_size:.2f} GB")


def convert_to_csv(data_list: List[Dict], output_path: str):
    """将数据列表转换为 CSV（用于测试）"""
    print(f"\n生成 CSV 文件: {output_path}")

    rows = []
    for item in tqdm(data_list, desc="转换"):
        mzs_str = ','.join(map(str, item['mzs'].tolist()))
        intensities_str = ','.join(map(str, item['intensities'].tolist()))

        rows.append({
            'mzs': mzs_str,
            'intensities': intensities_str,
            'smiles': item['smiles'],
            'precursor_mz': item['precursor_mz'],
            'adduct': item['adduct'] if item['adduct'] else '',
            'collision_energy': item['collision_energy'] if item['collision_energy'] is not None else '',
            'fold': item['fold'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"✓ CSV 保存完成: {len(df)} 条记录")


def convert_candidates(pkl_path: str, json_path: str):
    """将 pkl 候选集转换为 JSON"""
    print(f"\n转换候选集: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        candidates = pickle.load(f)

    with open(json_path, 'w') as f:
        json.dump(candidates, f)

    print(f"✓ 候选集转换完成: {len(candidates)} 个分子")
    print(f"  输出: {json_path}")


def process_dataset(dataset_name: str, output_base_dir: str, num_gpus: int = 6,
                    *, mgf_path: str, candidates_path: str, make_shards: bool = False,
                    spectra_only: bool = False):
    """处理单个数据集"""
    if dataset_name not in DATASETS:
        print(f"❌ 未知数据集: {dataset_name}")
        return

    config = dict(DATASETS[dataset_name])
    config['mgf_path'] = mgf_path
    config['candidates_path'] = candidates_path
    output_name = config['output_name']

    print(f"\n{'='*60}")
    print(f"处理数据集: {output_name}")
    print(f"{'='*60}")

    # 检查文件存在
    if not os.path.exists(config['mgf_path']):
        raise FileNotFoundError(config['mgf_path'])
    if not spectra_only and not os.path.exists(config['candidates_path']):
        raise FileNotFoundError(config['candidates_path'])

    # 创建输出目录
    dataset_dir = os.path.join(output_base_dir, output_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. 解析 MGF
    data_list = parse_mgf_file(config['mgf_path'])

    # 2. 按 fold 划分
    train_data = [d for d in data_list if d['fold'] == 'train']
    test_data = [d for d in data_list if d['fold'] == 'test']
    val_data = [d for d in data_list if d['fold'] == 'val']
    observed = (len(data_list), len(train_data), len(val_data), len(test_data))
    expected = (560084, 446667, 55980, 57437)
    if observed != expected:
        raise ValueError(f'MSnLib spectrum and fold counts differ: {observed} != {expected}')

    print(f"\n数据划分:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Val: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")

    # 3. 训练集 → Arrow shards
    if train_data and make_shards:
        shards_dir = os.path.join(dataset_dir, 'shards')
        convert_to_arrow_shards(train_data, shards_dir, num_gpus=num_gpus)

    # 4. 测试集 → CSV
    if test_data:
        test_csv_path = os.path.join(dataset_dir, f'{output_name}_test.csv')
        convert_to_csv(test_data, test_csv_path)

    # 5. 验证集 → CSV（可选）
    if val_data:
        val_csv_path = os.path.join(dataset_dir, f'{output_name}_val.csv')
        convert_to_csv(val_data, val_csv_path)

    # 6. 完整数据集 CSV（包含所有 fold，用于兼容现有脚本）
    full_csv_path = os.path.join(dataset_dir, f'{output_name}.csv')
    convert_to_csv(data_list, full_csv_path)

    # 7. 候选集 pkl → JSON
    if not spectra_only:
        candidates_json = os.path.join(dataset_dir, f'{output_name}_candidates.json')
        convert_candidates(config['candidates_path'], candidates_json)

    print(f"\n{'='*60}")
    print(f"✅ {output_name} 处理完成!")
    print(f"{'='*60}")
    print(f"输出目录: {dataset_dir}")
    print(f"  - 训练 shards: {dataset_dir}/shards/")
    print(f"  - 测试 CSV: {dataset_dir}/{output_name}_test.csv")
    if not spectra_only:
        print(f"  - 候选集 JSON: {dataset_dir}/{output_name}_candidates.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=str, required=True, help='Benchmark workspace root')
    parser.add_argument('--num_gpus', type=int, default=6,
                        help='GPU 数量（用于计算 shard 数）')
    parser.add_argument('--make-shards', action='store_true',
                        help='Also create optional Arrow training shards')
    parser.add_argument('--spectra-only', action='store_true',
                        help='Prepare the spectra table without the candidate pickle')

    args = parser.parse_args()

    source_dir = os.path.join(args.root, 'datasets', 'MSnLib')
    process_dataset('msnlib', os.path.join(args.root, 'datasets'), args.num_gpus,
                    mgf_path=os.path.join(source_dir, 'SpecBridge_MSnLib_dataset.mgf'),
                    candidates_path=os.path.join(source_dir, 'SpecBridge_MSnLib_candidates.pkl'),
                    make_shards=args.make_shards, spectra_only=args.spectra_only)

    print("\n🎉 全部转换完成!")


if __name__ == '__main__':
    main()
