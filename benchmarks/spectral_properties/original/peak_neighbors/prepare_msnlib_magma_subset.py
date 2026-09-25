#!/usr/bin/env python
"""Prepare a compact MSnLib subset for MAGMa peak annotation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_MAGMA_ADDUCTS = {
    "[M+Cl]-", "[M+Cl]1-", "[M+H-2H2O]+", "[M+H-2H2O]1+",
    "[M+H-H2O]+", "[M+H-H2O]1+", "[M+H3N+H]+", "[M+H3N+H]1+",
    "[M+H]+", "[M+H]1+", "[M+K]+", "[M+K]1+", "[M+NH3+H]+",
    "[M+NH3+H]1+", "[M+NH4]+", "[M+NH4]1+", "[M+Na]+",
    "[M+Na]1+", "[M-H-CO2]-", "[M-H-CO2]1-", "[M-H-H2O]-",
    "[M-H-H2O]1-", "[M-H2O+H]+", "[M-H2O+H]1+",
    "[M-H2O-H]-", "[M-H2O-H]1-", "[M-H4O2+H]+",
    "[M-H4O2+H]1+", "[M-H]-", "[M-H]1-", "[M]+", "[M]1+",
}


def project_root() -> Path:
    return Path(os.environ.get('ULTRAMS_EXPERIMENT_ROOT', Path(__file__).resolve().parents[4]))


def split_floats(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(x) for x in text.split(",") if x], dtype=np.float32)


def molecule_label(smiles: str) -> str:
    try:
        from rdkit import Chem
    except Exception:
        return "Other"
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return "Other"
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    counts = {e: elems.count(e) for e in set(elems)}
    if any(counts.get(x, 0) for x in ("F", "Cl", "Br", "I")):
        return "Halogenated"
    if counts.get("P", 0) or counts.get("S", 0):
        return "S/P-containing"
    if counts.get("N", 0):
        return "N-containing"
    if counts.get("O", 0) >= 3:
        return "O-rich"
    if set(counts).issubset({"C", "H"}):
        return "Hydrocarbon"
    return "Other"


def peak_tuple_string(mzs: np.ndarray, intensities: np.ndarray) -> str:
    if len(mzs) != len(intensities):
        n = min(len(mzs), len(intensities))
        mzs, intensities = mzs[:n], intensities[:n]
    if len(intensities) and float(np.nanmax(intensities)) > 0:
        intensities = intensities / float(np.nanmax(intensities))
    pairs = [(f"{float(m):.6f}", f"{float(i):.6f}") for m, i in zip(mzs, intensities)]
    return "[" + ", ".join(f"('{m}', '{i}')" for m, i in pairs) + "]"


def pick_rows(df: pd.DataFrame, n_spectra: int, max_per_smiles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["molecule_label", "adduct", "_rand"])

    selected = []
    per_smiles = {}
    groups = list(df.groupby(["molecule_label", "adduct"], dropna=False))
    if not groups:
        return df.head(0)
    rng.shuffle(groups)

    pointers = {key: 0 for key, _ in groups}
    group_rows = {key: g.reset_index(drop=True) for key, g in groups}
    while len(selected) < n_spectra:
        progressed = False
        for key, _ in groups:
            g = group_rows[key]
            ptr = pointers[key]
            while ptr < len(g):
                row = g.iloc[ptr]
                ptr += 1
                smi = row["smiles"]
                if per_smiles.get(smi, 0) >= max_per_smiles:
                    continue
                selected.append(row)
                per_smiles[smi] = per_smiles.get(smi, 0) + 1
                progressed = True
                break
            pointers[key] = ptr
            if len(selected) >= n_spectra:
                break
        if not progressed:
            break

    out = pd.DataFrame(selected).drop(columns=["_rand"], errors="ignore")
    return out.sort_values("spec_id").reset_index(drop=True)


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--n-spectra", type=int, default=360)
    ap.add_argument("--min-peaks", type=int, default=18)
    ap.add_argument("--max-peaks", type=int, default=130)
    ap.add_argument("--max-per-smiles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--msnlib-csv", type=Path, default=None,
                    help="Full MSnLib table with its existing fold column; alternatively use MSnLib_val/test.csv")
    ap.add_argument("--output", type=Path, default=root / "main_figure/fig2/data/task5_peak_embedding_umap/msnlib_magma_subset.csv")
    args = ap.parse_args()

    frames = []
    full = pd.read_csv(args.msnlib_csv) if args.msnlib_csv is not None else None
    for split in args.splits:
        if full is not None:
            df = full[full["fold"] == split].copy()
        else:
            path = root / f"datasets/MSnLib/MSnLib_{split}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
        df = df.reset_index(drop=False).rename(columns={"index": "source_row"})
        df["source_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for i, row in df.iterrows():
        mzs = split_floats(row["mzs"])
        ints = split_floats(row["intensities"])
        if len(mzs) < args.min_peaks or len(mzs) > args.max_peaks:
            continue
        smiles = str(row.get("smiles", ""))
        adduct = str(row.get("adduct", row.get("prec_type", "[M+H]+")))
        if not smiles or smiles == "nan" or not adduct or adduct == "nan":
            continue
        if adduct not in SUPPORTED_MAGMA_ADDUCTS:
            continue
        spec_id = f"MSnLib_{row['source_split']}_{int(row['source_row']):07d}"
        rows.append({
            "spec_id": spec_id,
            "source_split": row["source_split"],
            "source_row": int(row["source_row"]),
            "smiles": smiles,
            "prec_type": adduct,
            "adduct": adduct,
            "precursor_mz": float(row["precursor_mz"]),
            "collision_energy": row.get("collision_energy", ""),
            "fold": row.get("fold", row["source_split"]),
            "n_peaks": int(len(mzs)),
            "molecule_label": molecule_label(smiles),
            "mzs": ",".join(f"{float(x):.6f}" for x in mzs),
            "intensities": ",".join(f"{float(x):.6f}" for x in ints),
            "peaks": peak_tuple_string(mzs, ints),
        })

    candidates = pd.DataFrame(rows)
    subset = pick_rows(candidates, args.n_spectra, args.max_per_smiles, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)

    summary = {
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(subset)),
        "splits": args.splits,
        "n_spectra_requested": args.n_spectra,
        "min_peaks": args.min_peaks,
        "max_peaks": args.max_peaks,
        "label_counts": subset["molecule_label"].value_counts().to_dict(),
        "adduct_counts": subset["adduct"].value_counts().to_dict(),
    }
    with open(args.output.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
