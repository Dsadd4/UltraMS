#!/usr/bin/env python3
"""Prepare ChemBERTa-key-filtered MSnLib candidate lists for candidate ranking."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline_common import atomic_write_json, read_csv_rows, sha256_file, sha256_lines


def frozen_jsonl_counts(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sample_idx, counts, smiles = [], [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sample_idx.append(int(row["sample_idx"]))
            counts.append(int(row["n_candidates"]))
            smiles.append(str(row["smiles"]))
    return np.asarray(sample_idx, dtype=np.int64), np.asarray(counts, dtype=np.int64), smiles


def freeze(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_json = out / "MSnLib_candidates_effective_chemberta.json"
    if output_json.exists() and not args.force:
        print(f"already exists: {output_json}")
        return
    started = time.time()
    print(f"loading ChemBERTa cache key set: {args.chemberta_cache}", flush=True)
    molecule_cache = torch.load(args.chemberta_cache, map_location="cpu", weights_only=True)
    cache_keys = set(molecule_cache)
    n_cache_keys = len(cache_keys)
    del molecule_cache
    gc.collect()
    print(f"ChemBERTa keys={n_cache_keys:,}", flush=True)

    with open(args.candidates) as handle:
        raw = json.load(handle)
    keys_by_fold = {"val": set(), "test": set()}
    for row in read_csv_rows(args.csv):
        fold = row.get("fold")
        if fold in keys_by_fold:
            keys_by_fold[fold].add(str(row["smiles"]).strip())
    all_keys = sorted(set().union(*keys_by_fold.values()))
    effective: dict[str, list[str]] = {}
    raw_counts, effective_counts = [], []
    gt_retained = 0
    missing_keys = []
    for number, key in enumerate(all_keys, start=1):
        if key not in raw:
            missing_keys.append(key)
            continue
        candidate_list = list(raw[key])
        if key not in candidate_list:
            candidate_list.append(key)
        filtered = [candidate for candidate in candidate_list if candidate in cache_keys]
        effective[key] = filtered
        raw_counts.append(len(candidate_list))
        effective_counts.append(len(filtered))
        gt_retained += int(key in filtered)
        if number % 1000 == 0 or number == len(all_keys):
            print(f"effective candidate keys={number:,}/{len(all_keys):,}", flush=True)
    if missing_keys:
        raise RuntimeError(f"{len(missing_keys)} validation/test keys absent from candidate JSON")
    atomic_write_json(output_json, effective)

    ultra_idx, ultra_counts, ultra_smiles = frozen_jsonl_counts(Path(args.ultrams_jsonl))
    dreams_idx, dreams_counts, dreams_smiles = frozen_jsonl_counts(Path(args.dreams_jsonl))
    if not np.array_equal(ultra_idx, dreams_idx) or ultra_smiles != dreams_smiles:
        raise RuntimeError("UltraMS and DreaMS frozen JSONL query identity/order differ")
    if not np.array_equal(ultra_counts, dreams_counts):
        raise RuntimeError("UltraMS and DreaMS frozen JSONL candidate counts differ")
    replay_counts = np.asarray([len(effective[smi]) for smi in ultra_smiles], dtype=np.int64)
    if not np.array_equal(replay_counts, ultra_counts):
        mismatch = np.where(replay_counts != ultra_counts)[0]
        raise RuntimeError(
            f"effective candidate replay mismatches frozen JSONL on {len(mismatch)} rows; "
            f"first row={int(mismatch[0]) if len(mismatch) else None}"
        )

    raw_counts_arr = np.asarray(raw_counts, dtype=np.int64)
    effective_counts_arr = np.asarray(effective_counts, dtype=np.int64)
    audit = {
        "status": "complete",
        "csv": str(Path(args.csv).resolve()),
        "csv_sha256": sha256_file(args.csv),
        "raw_candidates": str(Path(args.candidates).resolve()),
        "raw_candidates_sha256": sha256_file(args.candidates),
        "chemberta_cache": str(Path(args.chemberta_cache).resolve()),
        "chemberta_cache_sha256": sha256_file(args.chemberta_cache),
        "n_chemberta_cache_keys": n_cache_keys,
        "fold_query_keys": {fold: len(values) for fold, values in keys_by_fold.items()},
        "n_effective_candidate_keys": len(effective),
        "ordered_candidate_key_sha256": sha256_lines(all_keys),
        "n_ground_truth_retained": gt_retained,
        "ground_truth_retained_fraction": gt_retained / len(all_keys),
        "raw_count": {
            "total": int(raw_counts_arr.sum()),
            "mean": float(raw_counts_arr.mean()),
            "median": float(np.median(raw_counts_arr)),
            "minimum": int(raw_counts_arr.min()),
            "maximum": int(raw_counts_arr.max()),
        },
        "effective_count": {
            "total": int(effective_counts_arr.sum()),
            "mean": float(effective_counts_arr.mean()),
            "median": float(np.median(effective_counts_arr)),
            "minimum": int(effective_counts_arr.min()),
            "maximum": int(effective_counts_arr.max()),
        },
        "removed_occurrences": int((raw_counts_arr - effective_counts_arr).sum()),
        "frozen_test_jsonl_regression": {
            "n_rows": len(ultra_counts),
            "sample_idx_sha256": sha256_lines(map(str, ultra_idx.tolist())),
            "candidate_count_sha256": sha256_lines(map(str, ultra_counts.tolist())),
            "candidate_count_mean": float(ultra_counts.mean()),
            "candidate_count_median": float(np.median(ultra_counts)),
            "candidate_count_minimum": int(ultra_counts.min()),
            "candidate_count_maximum": int(ultra_counts.max()),
            "ultrams_jsonl_sha256": sha256_file(args.ultrams_jsonl),
            "dreams_jsonl_sha256": sha256_file(args.dreams_jsonl),
            "replay_exact": True,
        },
        "artifact": {
            "path": str(output_json),
            "sha256": sha256_file(output_json),
            "bytes": output_json.stat().st_size,
        },
        "runtime_seconds": time.time() - started,
    }
    atomic_write_json(out / "effective_candidate_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--chemberta-cache", required=True)
    parser.add_argument("--ultrams-jsonl", required=True)
    parser.add_argument("--dreams-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    freeze(args)


if __name__ == "__main__":
    main()
