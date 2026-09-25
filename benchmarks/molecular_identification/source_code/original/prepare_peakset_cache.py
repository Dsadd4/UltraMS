#!/usr/bin/env python3
"""Prepare a full-precision fixed-length raw-peak cache for MassSpecGym DeepSets."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_common import atomic_write_json, parse_peak_strings, sha256_file


def tokenize_row(mzs_text: object, intensities_text: object, precursor_mz: float) -> tuple[np.ndarray, int]:
    mzs, intensities = parse_peak_strings(mzs_text, intensities_text)
    keep = (mzs >= 10.0) & (mzs <= 1005.0) & (intensities >= 0)
    mzs, intensities = mzs[keep], intensities[keep]
    before = int(len(mzs))
    if len(mzs) > 60:
        # Retain the 60 strongest peaks, then restore m/z order. Sum pooling is
        # order-invariant, but this also reproduces the official tokenizer.
        chosen = np.argpartition(intensities, -60)[-60:]
        chosen = chosen[np.argsort(mzs[chosen], kind="stable")]
        mzs, intensities = mzs[chosen], intensities[chosen]
    maximum = float(intensities.max(initial=0.0))
    if maximum > 0:
        intensities = intensities / maximum
    out = np.zeros((61, 2), dtype=np.float32)
    out[0] = (float(precursor_mz), 1.1)
    count = min(60, len(mzs))
    if count:
        out[1 : count + 1, 0] = mzs[:count]
        out[1 : count + 1, 1] = intensities[:count]
    return out, before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ffn-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    done_path = output / "DONE.json"
    if done_path.exists() and not args.force:
        print(done_path.read_text())
        return

    started = time.time()
    frame = pd.read_csv(args.csv, usecols=["mzs", "intensities", "precursor_mz"])
    folds = np.load(Path(args.ffn_cache_dir) / "fold_codes_uint8.npy", mmap_mode="r")
    if len(frame) != len(folds) or len(frame) != 560_084:
        raise RuntimeError(f"unexpected row count frame={len(frame)} folds={len(folds)}")
    if not np.isfinite(frame.precursor_mz.to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite precursor m/z in MSnLib")

    # Do not store m/z in float16. Its spacing is 0.0625 Da at m/z 100 and
    # 0.5 Da at m/z 1000, which destroys the fixed high-frequency Fourier
    # representation used by the official DeepSets baseline.
    cache_path = output / "peaksets_float32.npy"
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(len(frame), 61, 2))
    original_counts = np.zeros(len(frame), dtype=np.int32)
    retained_counts = np.zeros(len(frame), dtype=np.int16)
    for idx, row in enumerate(frame.itertuples(index=False)):
        values, before = tokenize_row(row.mzs, row.intensities, row.precursor_mz)
        cache[idx] = values
        original_counts[idx] = before
        retained_counts[idx] = int(np.count_nonzero(values[1:, 1] > 0))
        if idx == 0 or (idx + 1) % 50_000 == 0 or idx + 1 == len(frame):
            print(f"rows={idx + 1:,}/{len(frame):,}", flush=True)
    cache.flush()
    np.save(output / "original_peak_counts_int32.npy", original_counts)
    np.save(output / "retained_peak_counts_int16.npy", retained_counts)

    summary = {
        "status": "complete",
        "protocol": {
            "mz_from": 10.0,
            "mz_to": 1005.0,
            "n_max_peaks": 60,
            "intensity_normalization": "divide by maximum after m/z filtering and top-60 retention",
            "precursor_row": ["precursor_mz", 1.1],
            "padding": "zeros to 61 rows",
            "dtype": "float32",
            "precision_reason": "preserve exact m/z resolution for fixed Fourier features",
        },
        "n_rows": len(frame),
        "shape": [len(frame), 61, 2],
        "fold_counts": {str(code): int(np.sum(folds == code)) for code in (0, 1, 2)},
        "original_peak_count": {
            "minimum": int(original_counts.min()),
            "median": float(np.median(original_counts)),
            "mean": float(original_counts.mean()),
            "maximum": int(original_counts.max()),
        },
        "retained_peak_count": {
            "minimum": int(retained_counts.min()),
            "median": float(np.median(retained_counts)),
            "mean": float(retained_counts.mean()),
            "maximum": int(retained_counts.max()),
        },
        "n_rows_truncated_to_60": int(np.sum(original_counts > 60)),
        "cache_sha256": sha256_file(cache_path),
        "source_csv_sha256": sha256_file(args.csv),
        "ffn_prepare_summary_sha256": sha256_file(Path(args.ffn_cache_dir) / "prepare_summary.json"),
        "runtime_seconds": time.time() - started,
    }
    atomic_write_json(done_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
