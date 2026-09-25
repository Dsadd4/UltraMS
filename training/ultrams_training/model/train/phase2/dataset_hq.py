"""
High-Quality Mixed Dataset for Phase 2 Training
=================================================
Loads MassSpecGym, MSnLib (with RT), Spectraverse train sets.
Supports:
  - Standard MLM masking (same as v9)
  - Same-molecule pair sampling for VICReg
  - RT targets (MSnLib only)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import defaultdict

MAX_PEAKS = 150
MASK_RATIO = 0.15

DATASETS = {
    "masspecgym": {
        "csv": "datasets/hq_rt_v1/MassSpecGym.csv",
        "has_rt": False,
    },
    "msnlib": {
        "csv": "datasets/hq_rt_v1/MSnLib_with_rt.csv",
        "has_rt": True,
    },
    "spectraverse": {
        "csv": "datasets/hq_rt_v1/Spectraverse.csv",
        "has_rt": False,
    },
}


def _parse_spectrum(mzs_str, ints_str, max_peaks=MAX_PEAKS):
    mzs = np.array([float(x) for x in str(mzs_str).split(",")], dtype=np.float32)
    ints = np.array([float(x) for x in str(ints_str).split(",")], dtype=np.float32)
    valid = mzs > 0
    mzs, ints = mzs[valid], ints[valid]
    if len(mzs) < 3:
        return None
    mx = ints.max()
    if mx > 0:
        ints = ints / mx
    ints = np.clip(ints, 0, 1)
    if len(mzs) > max_peaks:
        idx = np.argsort(ints)[-max_peaks:]
        idx = np.sort(idx)
        mzs, ints = mzs[idx], ints[idx]
    order = np.argsort(mzs)
    return np.stack([mzs[order], ints[order]], axis=-1).astype(np.float32)


class HQMixedDataset(Dataset):
    """Combined high-quality dataset with same-molecule indexing."""

    def __init__(self, dataset_names=None, fold="train", max_peaks=MAX_PEAKS):
        super().__init__()
        if dataset_names is None:
            dataset_names = list(DATASETS.keys())
        self.max_peaks = max_peaks
        self.samples = []
        self.smiles_to_indices = defaultdict(list)

        for ds_name in dataset_names:
            cfg = DATASETS[ds_name]
            df = pd.read_csv(cfg["csv"])
            if "fold" in df.columns:
                df = df[df["fold"] == fold]
            has_rt = cfg["has_rt"] and "rt" in df.columns

            print(f"[{ds_name}] Loading {len(df)} {fold} samples (has_rt={has_rt})")

            for _, row in df.iterrows():
                spec = _parse_spectrum(row["mzs"], row["intensities"], max_peaks)
                if spec is None:
                    continue
                sm = str(row.get("smiles", ""))
                if not sm or sm == "nan":
                    continue

                raw_rt = (
                    float(row["rt"])
                    if has_rt
                    and pd.notna(row.get("rt", None))
                    and float(row.get("rt", 0)) > 0
                    else 0.0
                )
                rt_val = raw_rt / 600.0 if raw_rt > 0 else 0.0
                idx = len(self.samples)
                self.samples.append(
                    {
                        "spectrum": spec,
                        "precursor_mz": float(row["precursor_mz"]),
                        "smiles": sm,
                        "rt": rt_val,
                        "source": ds_name,
                    }
                )
                self.smiles_to_indices[sm].append(idx)

        self.multi_spec_smiles = [
            sm for sm, idxs in self.smiles_to_indices.items() if len(idxs) >= 2
        ]
        n_multi = sum(
            len(idxs)
            for sm in self.multi_spec_smiles
            for idxs in [self.smiles_to_indices[sm]]
        )
        n_rt = sum(1 for s in self.samples if s["rt"] > 0)
        print(
            f"Total: {len(self.samples)} samples, {len(self.smiles_to_indices)} unique SMILES, "
            f"{len(self.multi_spec_smiles)} molecules with >=2 spectra ({n_multi} spectra), "
            f"{n_rt} with RT"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_pair_index(self, idx):
        """Get a different spectrum index of the same molecule, or -1 if singleton."""
        sm = self.samples[idx]["smiles"]
        indices = self.smiles_to_indices[sm]
        if len(indices) < 2:
            return -1
        candidates = [i for i in indices if i != idx]
        return candidates[np.random.randint(len(candidates))]


def _pad_spectrum(spec, max_peaks):
    out = np.zeros((max_peaks, 2), dtype=np.float32)
    n = min(len(spec), max_peaks)
    out[:n] = spec[:n]
    return out, n


def _apply_mask(spec, n_peaks, mask_ratio=MASK_RATIO):
    n_mask = max(1, int(n_peaks * mask_ratio))
    mask_idx = np.random.choice(n_peaks, n_mask, replace=False)
    masked = spec.copy()
    masked[mask_idx] = 0.0
    flags = np.zeros(spec.shape[0], dtype=np.int64)
    flags[mask_idx] = 1
    return masked, flags


def _augment_spectrum(spec, n_peaks):
    """Light augmentation for VICReg when no same-molecule pair exists."""
    aug = spec.copy()
    if n_peaks < 3:
        return aug
    n_drop = max(1, int(n_peaks * 0.1))
    drop_idx = np.random.choice(n_peaks, n_drop, replace=False)
    aug[drop_idx] = 0.0
    keep = aug[:, 0] > 0
    valid = aug[keep]
    if len(valid) < 3:
        return spec.copy()
    out = np.zeros_like(aug)
    out[: len(valid)] = valid
    noise = np.random.normal(0, 0.02, size=len(valid))
    out[: len(valid), 1] = np.clip(out[: len(valid), 1] + noise, 0, 1)
    return out


def collate_fn_phase2(batch):
    """Collate with MLM masking + VICReg pairs + RT."""
    dataset = batch[0].get("_dataset_ref")
    B = len(batch)
    P = MAX_PEAKS

    masked_spectra = np.zeros((B, P, 2), dtype=np.float32)
    orig_spectra = np.zeros((B, P, 2), dtype=np.float32)
    mask_flags = np.zeros((B, P), dtype=np.int64)
    attn_mask = np.zeros((B, P), dtype=np.int64)
    precursor_mz = np.zeros(B, dtype=np.float32)
    rt = np.zeros(B, dtype=np.float32)

    view1 = np.zeros((B, P, 2), dtype=np.float32)
    view2 = np.zeros((B, P, 2), dtype=np.float32)
    view1_attn = np.zeros((B, P), dtype=np.int64)
    view2_attn = np.zeros((B, P), dtype=np.int64)
    has_pair = np.zeros(B, dtype=np.int64)

    for i, item in enumerate(batch):
        spec = item["spectrum"]
        padded, n = _pad_spectrum(spec, P)
        masked, flags = _apply_mask(padded, n)

        orig_spectra[i] = padded
        masked_spectra[i] = masked
        mask_flags[i] = flags
        attn_mask[i, :n] = 1
        precursor_mz[i] = item["precursor_mz"]
        rt[i] = item.get("rt", 0.0)

        v1_pad, v1_n = _pad_spectrum(spec, P)
        view1[i] = v1_pad
        view1_attn[i, :v1_n] = 1

        pair_idx = item.get("_pair_idx", -1)
        if pair_idx >= 0:
            pair_item = item["_pair_data"]
            v2_pad, v2_n = _pad_spectrum(pair_item["spectrum"], P)
            view2[i] = v2_pad
            view2_attn[i, :v2_n] = 1
            has_pair[i] = 1
        else:
            aug = _augment_spectrum(padded, n)
            v2_pad, v2_n = _pad_spectrum(aug, P)
            n2 = int((v2_pad[:, 0] > 0).sum())
            view2[i] = v2_pad
            view2_attn[i, :n2] = 1
            has_pair[i] = 1

    return {
        "masked_spectra": torch.from_numpy(masked_spectra),
        "orig_spectra": torch.from_numpy(orig_spectra),
        "mask_flags": torch.from_numpy(mask_flags),
        "attn_mask": torch.from_numpy(attn_mask),
        "precursor_mz": torch.from_numpy(precursor_mz),
        "rt": torch.from_numpy(rt),
        "view1": torch.from_numpy(view1),
        "view2": torch.from_numpy(view2),
        "view1_attn": torch.from_numpy(view1_attn),
        "view2_attn": torch.from_numpy(view2_attn),
        "has_pair": torch.from_numpy(has_pair),
    }


class PairSamplerWrapper(Dataset):
    """Wraps HQMixedDataset to inject pair info for collate_fn."""

    def __init__(self, dataset: HQMixedDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        pair_idx = self.dataset.get_pair_index(idx)
        if pair_idx >= 0:
            item["_pair_idx"] = pair_idx
            item["_pair_data"] = self.dataset[pair_idx]
        else:
            item["_pair_idx"] = -1
        return item


def create_hq_dataloader(
    batch_size=512,
    num_workers=4,
    max_peaks=MAX_PEAKS,
    rank=0,
    world_size=1,
    dataset_names=None,
):
    ds = HQMixedDataset(dataset_names=dataset_names, fold="train", max_peaks=max_peaks)
    wrapped = PairSamplerWrapper(ds)

    sampler = (
        DistributedSampler(wrapped, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn_phase2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, ds


if __name__ == "__main__":
    print("Testing HQMixedDataset...")
    loader, ds = create_hq_dataloader(batch_size=8, num_workers=0)
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
    print(f"  RT values: {batch['rt']}")
    print(f"  has_pair: {batch['has_pair']}")
    print("OK")
