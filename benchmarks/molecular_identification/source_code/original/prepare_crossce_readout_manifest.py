#!/usr/bin/env python3
"""Freeze a test-only cross-CE pool with auditable H5/source rows.

The four simple readouts are trained only on the 70% cross-CE training
compounds.  Spectrum choices are replayed from the historical 30% pool and then
restricted to the true 15% test compounds, so existing UltraMS/DreaMS records
and all new baselines can be compared on identical spectra without validation
or the legacy one-compound rounding leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from baseline_common import atomic_write_json, sha256_file


MODES = {
    "pos": {"adduct": "[M+H]+", "split_seed": 42, "pool_offset": 99},
    "neg": {"adduct": "[M-H]-", "split_seed": 43, "pool_offset": 199},
}


def rows_sha256(rows: np.ndarray) -> str:
    payload = "".join(f"{int(row)}\n" for row in rows).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.manifest_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ce = pd.to_numeric(pd.read_csv(args.csv, usecols=["collision_energy"])["collision_energy"], errors="coerce").to_numpy()
    summaries = {}
    with h5py.File(source / "msnlib_fig3h_spectra.h5", "r") as h5:
        for mode, config in MODES.items():
            group = h5[mode]
            labels = np.asarray(group["smiles_index"], dtype=np.int64)
            source_rows = np.asarray(group["source_row"], dtype=np.int64)
            smiles = json.loads((source / f"smiles_{mode}.json").read_text())
            row_ce = ce[source_rows]
            low_by_label: dict[int, list[int]] = {}
            high_by_label: dict[int, list[int]] = {}
            for row, (label, value) in enumerate(zip(labels, row_ce)):
                if not np.isfinite(value):
                    continue
                if value <= 20:
                    low_by_label.setdefault(int(label), []).append(row)
                elif value >= 40:
                    high_by_label.setdefault(int(label), []).append(row)
            valid_labels = sorted(set(low_by_label) & set(high_by_label), key=lambda label: smiles[label])
            shuffled = [smiles[label] for label in valid_labels]
            rng = random.Random(int(config["split_seed"]))
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = int(n * 0.15)
            n_val = int(n * 0.15)
            test_smiles = shuffled[:n_test]
            val_smiles = shuffled[n_test:n_test + n_val]
            train_smiles = shuffled[n_test + n_val:]
            label_of = {value: index for index, value in enumerate(smiles)}
            train_labels = np.asarray([label_of[value] for value in train_smiles], dtype=np.int64)
            train_rows = np.flatnonzero(np.isin(labels, train_labels)).astype(np.int64)

            # Reproduce the historical spectrum choices first, then retain only
            # the true test identities.  Building directly on test would advance
            # Python's RNG in a different order and select different spectra.
            legacy_nontraining = set(shuffled[:int(n * 0.30)])
            legacy_smiles = sorted(legacy_nontraining)
            pool_rng = random.Random(42 + int(config["pool_offset"]))
            replay = {}
            for value in legacy_smiles:
                label = label_of[value]
                replay[value] = (
                    pool_rng.choice(low_by_label[label]),
                    pool_rng.choice(high_by_label[label]),
                )
            evaluation_smiles = sorted(test_smiles)
            query_rows = [replay[value][0] for value in evaluation_smiles]
            library_rows = [replay[value][1] for value in evaluation_smiles]
            query_rows = np.asarray(query_rows, dtype=np.int64)
            library_rows = np.asarray(library_rows, dtype=np.int64)
            query_source_rows = source_rows[query_rows]
            library_source_rows = source_rows[library_rows]
            np.savez_compressed(
                output / f"protocol_{mode}.npz",
                train_rows=train_rows,
                train_labels=train_labels,
                val_labels=np.asarray([label_of[value] for value in val_smiles], dtype=np.int64),
                test_labels=np.asarray([label_of[value] for value in test_smiles], dtype=np.int64),
                evaluation_labels=np.asarray([label_of[value] for value in evaluation_smiles], dtype=np.int64),
                query_rows=query_rows,
                library_rows=library_rows,
                query_source_rows=query_source_rows,
                library_source_rows=library_source_rows,
            )
            summary = {
                "mode": mode,
                "adduct": config["adduct"],
                "split_seed": config["split_seed"],
                "pool_seed": 42 + int(config["pool_offset"]),
                "n_valid_compounds": n,
                "n_train_compounds": len(train_smiles),
                "n_val_compounds": len(val_smiles),
                "n_test_compounds": len(test_smiles),
                "n_evaluation_compounds": len(evaluation_smiles),
                "n_train_spectra": len(train_rows),
                "query_source_rows_sha256": rows_sha256(query_source_rows),
                "library_source_rows_sha256": rows_sha256(library_source_rows),
                "protocol_sha256": sha256_file(output / f"protocol_{mode}.npz"),
            }
            atomic_write_json(output / f"summary_{mode}.json", summary)
            summaries[mode] = summary
            print(json.dumps(summary), flush=True)
    atomic_write_json(output / "manifest_summary.json", {
        "status": "complete",
        "experiment": "cross-CE ChemBERTa-readout transfer baselines",
        "source_h5": str(source / "msnlib_fig3h_spectra.h5"),
        "source_h5_sha256": sha256_file(source / "msnlib_fig3h_spectra.h5"),
        "source_csv": str(args.csv),
        "source_csv_sha256": sha256_file(Path(args.csv)),
        "evaluation_scope": "true 15% test compounds only; historical per-sample spectrum choices replayed before filtering",
        "checkpoint_rule": "fixed final epoch; evaluation pool never used for selection",
        "modes": summaries,
    })


if __name__ == "__main__":
    main()
