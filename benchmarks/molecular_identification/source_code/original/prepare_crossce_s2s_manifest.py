#!/usr/bin/env python3
"""Freeze the leakage-safe Figure 3c cross-CE transfer evaluation.

The eligible molecules are restricted to the compound-disjoint Figure 3h test
partition.  A test molecule is retained only when it has at least one strict-
adduct spectrum at low collision energy (<=20) and one at high collision energy
(>=40).  Exactly one spectrum from each side is selected by a stable source-row
hash, yielding an aligned one-query/one-library-spectrum-per-molecule task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


MODE_CONFIG = {
    "pos": {"ion_mode": "[M+H]+", "selection_seed": 202608241},
    "neg": {"ion_mode": "[M-H]-", "selection_seed": 202608242},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_i64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def stable_choice(rows: list[int], source_rows: np.ndarray, seed: int) -> int:
    def key(row: int) -> bytes:
        return hashlib.blake2b(
            f"{seed}\0{int(source_rows[row])}".encode("ascii"), digest_size=16
        ).digest()

    return min(rows, key=key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig3h-manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ce = pd.to_numeric(
        pd.read_csv(args.csv, usecols=["collision_energy"])["collision_energy"],
        errors="coerce",
    ).to_numpy()
    summaries: dict[str, dict] = {}
    h5_path = args.fig3h_manifest / "msnlib_fig3h_spectra.h5"
    with h5py.File(h5_path, "r") as h5:
        for mode, config in MODE_CONFIG.items():
            group = h5[mode]
            labels = np.asarray(group["smiles_index"], dtype=np.int64)
            source_rows = np.asarray(group["source_row"], dtype=np.int64)
            smiles = json.loads((args.fig3h_manifest / f"smiles_{mode}.json").read_text())
            row_ce = ce[source_rows]
            fig3h = np.load(args.fig3h_manifest / f"protocol_{mode}.npz")
            test_labels = set(labels[np.asarray(fig3h["test_query_rows"], dtype=np.int64)].tolist())

            low: dict[int, list[int]] = {}
            high: dict[int, list[int]] = {}
            for row, (label, value) in enumerate(zip(labels, row_ce)):
                label = int(label)
                if label not in test_labels or not np.isfinite(value):
                    continue
                if value <= 20:
                    low.setdefault(label, []).append(row)
                if value >= 40:
                    high.setdefault(label, []).append(row)

            eligible = sorted(set(low) & set(high), key=lambda label: smiles[label])
            query_rows = np.asarray(
                [stable_choice(low[label], source_rows, int(config["selection_seed"])) for label in eligible],
                dtype=np.int64,
            )
            library_rows = np.asarray(
                [stable_choice(high[label], source_rows, int(config["selection_seed"]) + 1) for label in eligible],
                dtype=np.int64,
            )
            eligible_labels = np.asarray(eligible, dtype=np.int64)
            if not np.array_equal(labels[query_rows], eligible_labels):
                raise RuntimeError(f"{mode}: query labels are not aligned")
            if not np.array_equal(labels[library_rows], eligible_labels):
                raise RuntimeError(f"{mode}: library labels are not aligned")
            if len(np.intersect1d(query_rows, library_rows)):
                raise RuntimeError(f"{mode}: query/library self-overlap")

            out_path = args.output_dir / f"protocol_{mode}.npz"
            np.savez_compressed(
                out_path,
                test_query_rows=query_rows,
                test_library_rows=library_rows,
                query_rows=query_rows,
                library_rows=library_rows,
                query_source_rows=source_rows[query_rows],
                library_source_rows=source_rows[library_rows],
                evaluation_labels=eligible_labels,
            )
            summary = {
                "mode": mode,
                "ion_mode": config["ion_mode"],
                "source_split": "Figure 3h strict-adduct compound-disjoint test split",
                "n_fig3h_test_compounds": len(test_labels),
                "n_crossce_eligible_compounds": len(eligible),
                "low_collision_energy_max": 20.0,
                "high_collision_energy_min": 40.0,
                "selection_seed": int(config["selection_seed"]),
                "query_rows_sha256_i64le": sha256_i64(query_rows),
                "library_rows_sha256_i64le": sha256_i64(library_rows),
                "query_source_rows_sha256_i64le": sha256_i64(source_rows[query_rows]),
                "library_source_rows_sha256_i64le": sha256_i64(source_rows[library_rows]),
                "protocol_sha256": sha256_file(out_path),
            }
            (args.output_dir / f"summary_{mode}.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            summaries[mode] = summary
            print(json.dumps(summary), flush=True)

    manifest = {
        "status": "complete",
        "experiment": "Figure 3c cross-collision-energy spectrum-to-spectrum transfer",
        "training_supervision": (
            "Figure 3h encoders are frozen; simple controls were trained only with "
            "same-molecule spectrum pairs under symmetric InfoNCE; no ChemBERTa or fingerprint target"
        ),
        "evaluation": (
            "strict-adduct Figure 3h test compounds with one <=20 CE query and one >=40 CE "
            "library spectrum per molecule; exact cosine ranking across the held-out library"
        ),
        "source_h5": str(h5_path),
        "source_h5_sha256": sha256_file(h5_path),
        "source_csv": str(args.csv),
        "source_csv_sha256": sha256_file(args.csv),
        "modes": summaries,
    }
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
