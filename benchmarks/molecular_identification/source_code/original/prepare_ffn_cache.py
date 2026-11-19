#!/usr/bin/env python3
"""Build resumable MSnLib spectrum bins and 4096-bit target fingerprints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from baseline_common import (
    FOLD_TO_CODE,
    atomic_write_json,
    fingerprint_ffn_bins,
    parse_peak_strings,
    read_csv_rows,
    sha256_file,
)


EXPECTED_ROWS = 560_084


def fingerprint_smiles(smiles: str, n_bits: int = 4096, radius: int = 2) -> tuple[np.ndarray, bool]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits // 8, dtype=np.uint8), False
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    bitvect = generator.GetFingerprint(mol)
    unpacked = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, unpacked)
    return np.packbits(unpacked), True


def prepare(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "prepare_state.json"
    bins_path = out / "ffn_bins_float16.npy"
    folds_path = out / "fold_codes_uint8.npy"
    smiles_idx_path = out / "row_smiles_index_int32.npy"
    unique_path = out / "unique_target_smiles.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "status": "building_bins",
            "next_row": 0,
            "unique_smiles": [],
            "n_peaks_total": 0,
            "n_peaks_ge_1005": 0,
            "intensity_total": 0.0,
            "intensity_ge_1005": 0.0,
            "invalid_fold_rows": 0,
            "empty_spectrum_rows": 0,
        }

    if bins_path.exists():
        bins = open_memmap(bins_path, mode="r+")
        folds = open_memmap(folds_path, mode="r+")
        row_smiles_index = open_memmap(smiles_idx_path, mode="r+")
        if bins.shape != (args.expected_rows, args.max_mz):
            raise RuntimeError(f"unexpected bins shape {bins.shape}")
    else:
        bins = open_memmap(
            bins_path,
            mode="w+",
            dtype=np.float16,
            shape=(args.expected_rows, args.max_mz),
        )
        folds = open_memmap(folds_path, mode="w+", dtype=np.uint8, shape=(args.expected_rows,))
        row_smiles_index = open_memmap(
            smiles_idx_path, mode="w+", dtype=np.int32, shape=(args.expected_rows,)
        )

    unique_smiles = list(state.get("unique_smiles", []))
    smiles_to_idx = {value: idx for idx, value in enumerate(unique_smiles)}
    start_row = int(state.get("next_row", 0))
    started = time.time()
    seen_rows = 0
    if state.get("status") == "building_bins":
        for row_idx, row in enumerate(read_csv_rows(args.csv)):
            seen_rows = row_idx + 1
            if row_idx < start_row:
                continue
            if row_idx >= args.expected_rows:
                raise RuntimeError(f"CSV has more than expected {args.expected_rows} rows")
            mzs, intensities = parse_peak_strings(row["mzs"], row["intensities"])
            bins[row_idx] = fingerprint_ffn_bins(
                mzs, intensities, max_mz=args.max_mz, bin_width=1.0
            ).astype(np.float16)
            fold = row.get("fold", "")
            if fold not in FOLD_TO_CODE:
                state["invalid_fold_rows"] += 1
                folds[row_idx] = 255
            else:
                folds[row_idx] = FOLD_TO_CODE[fold]
            smi = str(row.get("smiles", "")).strip()
            if smi not in smiles_to_idx:
                smiles_to_idx[smi] = len(unique_smiles)
                unique_smiles.append(smi)
            row_smiles_index[row_idx] = smiles_to_idx[smi]
            state["n_peaks_total"] += int(mzs.size)
            high = mzs >= float(args.max_mz)
            state["n_peaks_ge_1005"] += int(high.sum())
            state["intensity_total"] += float(intensities.sum(dtype=np.float64))
            state["intensity_ge_1005"] += float(intensities[high].sum(dtype=np.float64))
            state["empty_spectrum_rows"] += int(not np.any(bins[row_idx]))

            if (row_idx + 1) % args.flush_every == 0:
                bins.flush()
                folds.flush()
                row_smiles_index.flush()
                state["next_row"] = row_idx + 1
                state["unique_smiles"] = unique_smiles
                state["elapsed_seconds"] = time.time() - started
                atomic_write_json(state_path, state)
                print(
                    f"rows={row_idx + 1:,}/{args.expected_rows:,} "
                    f"unique_smiles={len(unique_smiles):,}",
                    flush=True,
                )

        if seen_rows != args.expected_rows:
            raise RuntimeError(f"CSV row count {seen_rows} != expected {args.expected_rows}")
        bins.flush()
        folds.flush()
        row_smiles_index.flush()
        state["next_row"] = args.expected_rows
        state["unique_smiles"] = unique_smiles
        state["status"] = "building_target_fingerprints"
        atomic_write_json(state_path, state)
        atomic_write_json(unique_path, unique_smiles)

    unique_smiles = json.loads(unique_path.read_text()) if unique_path.exists() else unique_smiles
    fp_path = out / "unique_target_morgan_r2_4096_packed.npy"
    valid_path = out / "unique_target_morgan_valid.npy"
    if fp_path.exists():
        fingerprints = open_memmap(fp_path, mode="r+")
        valid = open_memmap(valid_path, mode="r+")
    else:
        fingerprints = open_memmap(
            fp_path, mode="w+", dtype=np.uint8, shape=(len(unique_smiles), args.fp_bits // 8)
        )
        valid = open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(len(unique_smiles),))

    if state.get("status") == "building_target_fingerprints":
        fp_start = int(state.get("next_fp", 0))
        for idx in range(fp_start, len(unique_smiles)):
            fingerprints[idx], valid[idx] = fingerprint_smiles(
                unique_smiles[idx], n_bits=args.fp_bits, radius=args.fp_radius
            )
            if (idx + 1) % args.fp_flush_every == 0:
                fingerprints.flush()
                valid.flush()
                state["next_fp"] = idx + 1
                atomic_write_json(state_path, state)
                print(f"fingerprints={idx + 1:,}/{len(unique_smiles):,}", flush=True)
        fingerprints.flush()
        valid.flush()
        state["next_fp"] = len(unique_smiles)
        state["status"] = "complete"
        atomic_write_json(state_path, state)

    fold_counts = {name: int(np.sum(folds == code)) for name, code in FOLD_TO_CODE.items()}
    intensity_total = float(state["intensity_total"])
    summary = {
        "status": state["status"],
        "csv": str(Path(args.csv).resolve()),
        "csv_sha256": sha256_file(args.csv),
        "n_rows": int(args.expected_rows),
        "fold_counts": fold_counts,
        "n_unique_target_smiles": len(unique_smiles),
        "invalid_target_smiles": int((~np.asarray(valid)).sum()),
        "input": {
            "max_mz_exclusive": int(args.max_mz),
            "bin_width": 1.0,
            "aggregation": "sum",
            "normalization": "per-spectrum maximum",
            "peak_cap": None,
            "include_precursor": False,
            "include_adduct": False,
        },
        "target": {
            "type": "Morgan bit fingerprint",
            "radius": int(args.fp_radius),
            "bits": int(args.fp_bits),
        },
        "high_mz_audit": {
            "n_peaks_total": int(state["n_peaks_total"]),
            "n_peaks_ge_1005": int(state["n_peaks_ge_1005"]),
            "peak_fraction_ge_1005": state["n_peaks_ge_1005"] / max(1, state["n_peaks_total"]),
            "intensity_total": intensity_total,
            "intensity_ge_1005": float(state["intensity_ge_1005"]),
            "intensity_fraction_ge_1005": float(state["intensity_ge_1005"]) / max(1e-30, intensity_total),
        },
        "empty_spectrum_rows": int(state["empty_spectrum_rows"]),
        "invalid_fold_rows": int(state["invalid_fold_rows"]),
        "artifacts": {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    for path in (bins_path, folds_path, smiles_idx_path, unique_path, fp_path, valid_path):
        summary["artifacts"][path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    atomic_write_json(out / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--max-mz", type=int, default=1005)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=4096)
    parser.add_argument("--flush-every", type=int, default=10_000)
    parser.add_argument("--fp-flush-every", type=int, default=1_000)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
