#!/usr/bin/env python3
"""Select the reported collision-energy test queries from backbone search results."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--crossce-manifest-dir", type=Path, required=True)
    parser.add_argument("--backbone-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}
    with h5py.File(args.manifest_dir / "msnlib_fig3h_spectra.h5", "r") as spectra:
        for mode in ("pos", "neg"):
            protocol = np.load(args.crossce_manifest_dir / f"protocol_{mode}.npz")
            query_rows = protocol["query_rows"]
            source_rows = protocol["query_source_rows"]
            manifest_source_rows = np.asarray(spectra[mode]["source_row"], dtype=np.int64)
            labels = np.asarray(spectra[mode]["smiles_index"], dtype=np.int64)
            if not np.array_equal(manifest_source_rows[query_rows], source_rows):
                raise ValueError(f"{mode}: query identities differ from the spectrum manifest")
            smiles = json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text())
            selected_smiles = [smiles[index] for index in labels[query_rows]]
            original = json.loads(
                (args.backbone_results_dir / f"per_sample_chem_{mode}.json").read_text()
            )
            by_smiles = {row["smiles"]: row for row in original}
            if len(by_smiles) != len(original):
                raise ValueError(f"{mode}: duplicate backbone query identities")
            for method, rank_key in (("UltraMS", "rank_ultra_ft"), ("DreaMS", "rank_dream_ft")):
                hits = []
                for source_row, molecule in zip(source_rows, selected_smiles):
                    record = by_smiles.get(molecule)
                    if record is None:
                        raise ValueError(f"{mode}: backbone result missing {molecule}")
                    hit = int(record[rank_key] == 1)
                    hits.append(hit)
                    rows.append((method, mode, int(source_row), hit))
                summary[f"{method}_{mode}"] = {
                    "n_queries": len(hits),
                    "top1_correct": sum(hits),
                    "top1_accuracy_percent": 100 * sum(hits) / len(hits),
                }

    result = args.output_dir / "collision_energy_backbone_test.csv.gz"
    temporary = result.with_name(result.name + ".part")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "mode", "query_source_row", "top1_correct"))
        writer.writerows(rows)
    os.replace(temporary, result)
    summary_path = args.output_dir / "collision_energy_backbone_test.json"
    temporary = summary_path.with_name(summary_path.name + ".part")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
