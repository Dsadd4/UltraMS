"""Freeze large SpectraVerse negative-ion structure-family benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.cluster import KMeans

from prepare_spectraverse_isomer_benchmark import (
    excluded_connectivities,
    parse_vector,
    sha256,
    top_peaks,
)


def molecule_labels(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return (
        Chem.MolToSmiles(mol, isomericSmiles=True),
        rdMolDescriptors.CalcMolFormula(mol),
        inchi.MolToInchiKey(mol).split("-")[0],
        inchi.MolToInchiKey(mol),
    )


def valid_peak_count(row) -> int:
    mz, intensity = parse_vector(row["mzs"]), parse_vector(row["intensities"])
    if len(mz) != len(intensity):
        return 0
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    return int(valid.sum())


def fingerprint_matrix(smiles: list[str], n_bits: int) -> np.ndarray:
    generator = GetMorganGenerator(radius=2, fpSize=n_bits)
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, value in enumerate(smiles):
        fp = generator.GetFingerprint(Chem.MolFromSmiles(value))
        matrix[i] = np.asarray(fp, dtype=np.uint8)
    return matrix


def freeze_fold(source: pd.DataFrame, fold: str, out_dir: Path, n_peaks: int,
                n_clusters: int, fp_bits: int) -> dict:
    frame = source[source.fold == fold].copy()
    frame = frame.sort_values(
        ["connectivity", "valid_peak_count", "source_row"],
        ascending=[True, False, True],
    ).drop_duplicates("connectivity", keep="first").reset_index(drop=True)
    fingerprints = fingerprint_matrix(frame.canonical_smiles.tolist(), fp_bits)
    labels = KMeans(n_clusters=n_clusters, n_init=50, random_state=42).fit_predict(fingerprints)
    frame["structure_cluster"] = labels
    frame["benchmark_id"] = [f"{fold}_{i:06d}" for i in range(len(frame))]
    frame["energy_role"] = "representative"

    mzs = np.zeros((len(frame), n_peaks), np.float32)
    intensities = np.zeros((len(frame), n_peaks), np.float32)
    lengths = np.zeros(len(frame), np.int16)
    for i, row in frame.iterrows():
        mz, intensity = top_peaks(row.mzs, row.intensities, n_peaks)
        lengths[i] = len(mz)
        mzs[i, :len(mz)], intensities[i, :len(mz)] = mz, intensity
    metadata = frame[[
        "benchmark_id", "source_row", "fold", "adduct", "energy_role", "precursor_mz",
        "canonical_smiles", "formula", "connectivity", "inchikey", "structure_cluster",
    ]].copy()
    metadata.insert(1, "spectrum_id", metadata.benchmark_id)
    metadata.to_csv(out_dir / f"{fold}_metadata.csv", index=False)
    np.savez_compressed(out_dir / f"{fold}_top{n_peaks}_peaks.npz", mzs=mzs, intensities=intensities, lengths=lengths)
    np.save(out_dir / f"{fold}_morgan_r2_{fp_bits}.npy", fingerprints)
    counts = metadata.structure_cluster.value_counts().sort_index()
    return {
        "fold": fold, "n_spectra": len(frame), "n_connectivities": frame.connectivity.nunique(),
        "n_structure_clusters": n_clusters,
        "cluster_sizes": {str(int(k)): int(v) for k, v in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mona", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adduct", default="[M-H]-")
    parser.add_argument("--n-peaks", type=int, default=150)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--fp-bits", type=int, default=2048)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.csv)
    source["source_row"] = np.arange(len(source))
    source = source[source.adduct == args.adduct].copy()
    source["valid_peak_count"] = source.apply(valid_peak_count, axis=1)
    invalid = int((source.valid_peak_count < 3).sum())
    source = source[source.valid_peak_count >= 3].copy()
    labels = [molecule_labels(value) for value in source.smiles]
    valid = np.array([value is not None for value in labels])
    source = source.loc[valid].copy()
    source[["canonical_smiles", "formula", "connectivity", "inchikey"]] = pd.DataFrame(
        [value for value in labels if value is not None], index=source.index
    )
    excluded = excluded_connectivities(args.mona)
    source = source[~source.connectivity.isin(excluded)].copy()
    fold_counts = source.groupby("connectivity").fold.nunique()
    crossing = set(fold_counts.index[fold_counts > 1])
    source = source[~source.connectivity.isin(crossing)].copy()
    reports = [
        freeze_fold(source, fold, args.out_dir, args.n_peaks, args.n_clusters, args.fp_bits)
        for fold in ("val", "test")
    ]
    manifest = {
        "source": str(args.csv), "source_sha256": sha256(args.csv), "adduct": args.adduct,
        "sampling": "one spectrum per connectivity; maximum valid peak count, then source-row tie-break",
        "structure_labels": f"KMeans(n={args.n_clusters}, seed=42, n_init=50) on Morgan radius-2 {args.fp_bits}-bit fingerprints",
        "excluded_mona_connectivities": len(excluded),
        "cross_fold_connectivities_removed": len(crossing),
        "invalid_spectra_removed": invalid,
        "reports": reports,
    }
    (args.out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
