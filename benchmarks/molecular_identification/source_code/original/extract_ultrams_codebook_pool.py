#!/usr/bin/env python3
"""Extract a transformer-free UltraMS peak-codebook spectrum representation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline_common import atomic_write_json, parse_peak_strings, sha256_file


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN = ROOT / "train"


def import_retrieval_module():
    path = HERE / "evaluate_molecule_identification.py"
    spec = importlib.util.spec_from_file_location("fig3_codebook_retrieval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_row(mzs_text: object, intensities_text: object, max_peaks: int) -> np.ndarray:
    mzs, intensities = parse_peak_strings(mzs_text, intensities_text)
    keep = (mzs > 0) & (intensities >= 0)
    mzs, intensities = mzs[keep], intensities[keep]
    if len(mzs) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    maximum = float(intensities.max(initial=0.0))
    if maximum > 0:
        intensities = np.clip(intensities / maximum, 0.0, 1.0)
    if len(mzs) > max_peaks:
        chosen = np.argsort(intensities, kind="stable")[-max_peaks:]
        mzs, intensities = mzs[chosen], intensities[chosen]
    order = np.argsort(mzs, kind="stable")
    return np.stack((mzs[order], intensities[order]), axis=-1).astype(np.float32)


@torch.no_grad()
def extract_batch(model, rows: list, device: torch.device) -> np.ndarray:
    batch = len(rows)
    max_peaks = int(model.max_peaks)
    padded = np.zeros((batch, max_peaks, 2), dtype=np.float32)
    mask = np.zeros((batch, max_peaks), dtype=bool)
    for idx, row in enumerate(rows):
        spectrum = prepare_row(row.mzs, row.intensities, max_peaks)
        count = min(max_peaks, len(spectrum))
        if count:
            padded[idx, :count] = spectrum[:count]
            mask[idx, :count] = True
    peaks = torch.from_numpy(padded).to(device)
    valid = torch.from_numpy(mask).to(device)
    # peak_encoder is the learned multi-resolution m/z and intensity codebook
    # followed by its fixed 1024-d shared-space projection.  The transformer,
    # CLS token, positional embeddings and all Phase-2 encoder layers are not run.
    encoded = model.peak_encoder(peaks)
    weights = peaks[:, :, 1].clamp_min(0.0) * valid.to(peaks.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
    return pooled.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    done_path = output / "DONE.json"
    if done_path.exists() and not args.force:
        print(done_path.read_text())
        return

    csv_path = Path(args.csv)
    n_rows = sum(1 for _ in csv_path.open()) - 1
    if n_rows != 560_084:
        raise RuntimeError(f"unexpected MSnLib row count: {n_rows}")
    device = torch.device(args.device)
    retrieval = import_retrieval_module()
    model, dimension = retrieval.load_model("rt_only_d11", device)
    if dimension != 1024:
        raise RuntimeError(dimension)
    checkpoint_path = TRAIN / "output" / "phase2_rt_only" / "stage_d_epoch_11.pt"
    cache_path = output / "ultrams_codebook_pool_float32.npy"
    cache = np.lib.format.open_memmap(cache_path, "w+", dtype=np.float32, shape=(n_rows, 1024))

    started = time.time()
    offset = 0
    fold_counts: dict[str, int] = {}
    for frame in pd.read_csv(
        csv_path,
        usecols=["mzs", "intensities", "fold"],
        chunksize=args.chunk_size,
    ):
        for fold, count in frame.fold.value_counts().items():
            fold_counts[str(fold)] = fold_counts.get(str(fold), 0) + int(count)
        row_list = list(frame.itertuples(index=False))
        for start in range(0, len(row_list), args.batch_size):
            block = row_list[start : start + args.batch_size]
            cache[offset : offset + len(block)] = extract_batch(model, block, device)
            offset += len(block)
        if offset == len(frame) or offset % 20_000 < len(frame) or offset == n_rows:
            print(f"rows={offset:,}/{n_rows:,}", flush=True)
    if offset != n_rows:
        raise RuntimeError((offset, n_rows))
    cache.flush()

    probe = np.asarray(cache[: min(20_000, n_rows)], dtype=np.float32)
    if not np.isfinite(probe).all():
        raise RuntimeError("non-finite codebook embeddings")
    summary = {
        "status": "complete",
        "method": "UltraMS transformer-free peak-codebook intensity-weighted pooling",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256_file(csv_path),
        "n_rows": n_rows,
        "shape": [n_rows, 1024],
        "dtype": "float32",
        "fold_counts": fold_counts,
        "excluded_components": [
            "CLS token",
            "precursor token",
            "positional embedding",
            "transformer encoder",
            "Phase-2 projection head",
        ],
        "pooling": "maximum-normalized fragment-intensity-weighted mean over at most 150 peaks",
        "cache_sha256": sha256_file(cache_path),
        "runtime_seconds": time.time() - started,
        "batch_size": args.batch_size,
    }
    atomic_write_json(done_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
