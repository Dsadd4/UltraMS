"""Freeze a supervised-model SpectraVerse negative-ion isomer benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors


def patch_legacy_msml() -> None:
    if "msml.utils.spectra" in sys.modules:
        return
    msml, utils, spectra = types.ModuleType("msml"), types.ModuleType("msml.utils"), types.ModuleType("msml.utils.spectra")
    class MSnSpectrum:
        pass
    MSnSpectrum.__module__ = "msml.utils.spectra"
    spectra.MSnSpectrum, utils.spectra, msml.utils = MSnSpectrum, spectra, utils
    sys.modules.update({"msml": msml, "msml.utils": utils, "msml.utils.spectra": spectra})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vector(value) -> np.ndarray:
    return np.fromstring(value, sep=",", dtype=np.float32) if isinstance(value, str) else np.asarray(value, dtype=np.float32)


def top_peaks(mzs, intensities, n_peaks: int):
    mz, intensity = parse_vector(mzs), parse_vector(intensities)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) < 3:
        raise ValueError("fewer than three valid peaks")
    if len(mz) > n_peaks:
        keep = np.argpartition(intensity, -n_peaks)[-n_peaks:]
        mz, intensity = mz[keep], intensity[keep]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    intensity = intensity / max(float(intensity.max()), 1e-12)
    return mz.astype(np.float32), intensity.astype(np.float32)


def has_valid_spectrum(row) -> bool:
    try:
        top_peaks(row["mzs"], row["intensities"], 150)
        return True
    except ValueError:
        return False


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


def excluded_connectivities(path: Path) -> set[str]:
    patch_legacy_msml()
    return set(pd.read_pickle(path)["inchi14"].dropna().astype(str))


def freeze_split(source: pd.DataFrame, fold: str, out_dir: Path, n_peaks: int) -> dict:
    frame = source[source["fold"] == fold].copy()
    counts = frame.groupby("connectivity").size()
    frame = frame[frame["connectivity"].isin(counts.index[counts >= 2])].copy()
    formula_sizes = frame[["formula", "connectivity"]].drop_duplicates().groupby("formula").size()
    frame = frame[frame["formula"].isin(formula_sizes.index[formula_sizes >= 2])].copy()
    frame = frame.sort_values(["formula", "connectivity", "source_row"]).reset_index(drop=True)
    frame["within_structure_index"] = frame.groupby("connectivity").cumcount()
    frame["energy_role"] = np.where(frame["within_structure_index"] % 2 == 0, "query", "reference")
    role = frame.groupby(["connectivity", "energy_role"]).size().unstack(fill_value=0)
    eligible = role.index[(role.get("query", 0) > 0) & (role.get("reference", 0) > 0)]
    frame = frame[frame["connectivity"].isin(eligible)].copy()
    formula_sizes = frame[["formula", "connectivity"]].drop_duplicates().groupby("formula").size()
    frame = frame[frame["formula"].isin(formula_sizes.index[formula_sizes >= 2])].reset_index(drop=True)
    frame["benchmark_id"] = [f"{fold}_{i:06d}" for i in range(len(frame))]

    mzs = np.zeros((len(frame), n_peaks), np.float32)
    intensities = np.zeros((len(frame), n_peaks), np.float32)
    lengths = np.zeros(len(frame), np.int16)
    for i, row in frame.iterrows():
        mz, intensity = top_peaks(row["mzs"], row["intensities"], n_peaks)
        lengths[i] = len(mz)
        mzs[i, :len(mz)], intensities[i, :len(mz)] = mz, intensity
    metadata = frame[[
        "benchmark_id", "source_row", "fold", "adduct", "energy_role", "precursor_mz",
        "canonical_smiles", "formula", "connectivity", "inchikey",
    ]].copy()
    metadata.insert(1, "spectrum_id", metadata["benchmark_id"])
    metadata.to_csv(out_dir / f"{fold}_metadata.csv", index=False)
    np.savez_compressed(out_dir / f"{fold}_top{n_peaks}_peaks.npz", mzs=mzs, intensities=intensities, lengths=lengths)
    return {
        "fold": fold, "n_spectra": len(frame),
        "n_queries": int((frame.energy_role == "query").sum()),
        "n_references": int((frame.energy_role == "reference").sum()),
        "n_connectivities": int(frame.connectivity.nunique()),
        "n_formula_groups": int(frame.formula.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--mona", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adduct", default="[M-H]-")
    parser.add_argument("--n-peaks", type=int, default=150)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.csv)
    source["source_row"] = np.arange(len(source))
    source = source[source.adduct == args.adduct].copy()
    n_before_peak_gate = len(source)
    peak_valid = source.apply(has_valid_spectrum, axis=1)
    source = source[peak_valid].copy()
    n_invalid_spectra = n_before_peak_gate - len(source)
    labels = [molecule_labels(value) for value in source.smiles]
    valid = np.array([value is not None for value in labels])
    source = source.loc[valid].copy()
    labels = [value for value in labels if value is not None]
    source[["canonical_smiles", "formula", "connectivity", "inchikey"]] = pd.DataFrame(labels, index=source.index)
    excluded = excluded_connectivities(args.mona)
    source = source[~source.connectivity.isin(excluded)].copy()
    fold_crossing = source.groupby("connectivity").fold.nunique()
    crossing = set(fold_crossing.index[fold_crossing > 1])
    source = source[~source.connectivity.isin(crossing)].copy()
    reports = [freeze_split(source, fold, args.out_dir, args.n_peaks) for fold in ("val", "test")]
    manifest = {
        "source": str(args.csv), "source_sha256": sha256(args.csv),
        "adduct": args.adduct, "structure_identity": "InChIKey connectivity block",
        "formula": "RDKit molecular formula", "split": "native SpectraVerse val/test",
        "query_reference_rule": "stable source-row order; alternating rows within each connectivity",
        "excluded_mona_connectivities": len(excluded),
        "cross_fold_connectivities_removed": len(crossing),
        "invalid_spectra_removed_before_grouping": n_invalid_spectra,
        "reports": reports,
    }
    (args.out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
