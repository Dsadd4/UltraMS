"""Combine the six methods' low-to-high collision-energy search results."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


BASELINES = {
    "DeepSets": "deepsets",
    "Fourier": "fourier_projection",
    "Codebook": "ultrams_codebook",
    "Linear": "linear",
}
CARBOXYLIC_ACID = Chem.MolFromSmarts("[C;X3](=[O;X1])[O;H1,-1]")
HALOGENS = {9, 17, 35, 53}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--backbone-results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_directory
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "mode", "query_source_row", "top1_correct",
            "stereochemically_complex", "multi_ring_aromatics",
            "carboxylic_acids", "halogenated_compounds",
        ])
        writer.writeheader()
        for mode in ("pos", "neg"):
            identity_path = source / "crossce_readout_fixed_v1" / "search" / "linear" / mode / "seed_0" / "per_sample.json"
            identity = json.loads(identity_path.read_text())
            legacy_path = args.backbone_results_dir / f"per_sample_chem_{mode}.json"
            legacy = {row["smiles"]: row for row in json.loads(legacy_path.read_text())}
            selected = [legacy[row["smiles"]] for row in identity]
            for method in ("UltraMS", "DreaMS", *BASELINES):
                if method == "UltraMS":
                    hits = [row["rank_ultra_ft"] == 1 for row in selected]
                elif method == "DreaMS":
                    hits = [row["rank_dream_ft"] == 1 for row in selected]
                else:
                    path = source / "crossce_readout_fixed_v1" / "search" / BASELINES[method] / mode / "seed_0" / "retrieval.npz"
                    with np.load(path, allow_pickle=False) as data:
                        hits = data["top1_correct"].tolist()
                if len(hits) != len(identity):
                    raise ValueError(f"{method}/{mode}: query count differs")
                for source_row, row, hit in zip(identity, selected, hits):
                    mol = Chem.MolFromSmiles(row["smiles"])
                    if mol is None:
                        raise ValueError(row["smiles"])
                    writer.writerow({
                        "method": method,
                        "mode": mode,
                        "query_source_row": source_row["query_source_row"],
                        "top1_correct": int(hit),
                        "stereochemically_complex": int(bool(row["props"]["hi_stereo"])),
                        "multi_ring_aromatics": int(row["props"]["n_arom_rings"] >= 3),
                        "carboxylic_acids": int(mol.HasSubstructMatch(CARBOXYLIC_ACID)),
                        "halogenated_compounds": int(any(atom.GetAtomicNum() in HALOGENS for atom in mol.GetAtoms())),
                    })


if __name__ == "__main__":
    main()
