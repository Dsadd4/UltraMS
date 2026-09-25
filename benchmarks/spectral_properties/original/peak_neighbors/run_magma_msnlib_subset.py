#!/usr/bin/env python
"""Run MAGMa annotation for the prepared MSnLib subset.

It writes the per-peak MAGMa annotations used by the peak-neighbor benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


def fragment_atom_indices(fe, frag_id: int) -> list[int]:
    entry = fe.frag_to_entry.get(int(frag_id))
    if entry is None:
        return []
    frag_bits = int(entry["frag"])
    return [i for i in range(fe.natoms) if frag_bits & (1 << i)]


def fragment_smiles(fe, atom_indices: list[int]) -> str | None:
    if not atom_indices:
        return None
    try:
        return Chem.MolFragmentToSmiles(
            fe.mol,
            atomsToUse=atom_indices,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def split_floats(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(x) for x in text.split(",") if x], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ultrago-root", type=Path,
                    default=Path(os.environ.get(
                        "ULTRAGO_DIR", Path(__file__).resolve().parents[2] / "magma_support"
                    )))
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-canonicalize", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.ultrago_root))
    from fragments.magma.op.Magma4MassSpecGYm import run_magma

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    samples = []
    t0 = time.time()
    for i, row in df.iterrows():
        mzs = split_floats(row["mzs"])
        intensities = split_floats(row["intensities"])
        adduct = str(row.get("adduct", row.get("prec_type", "[M+H]+")))
        spec_id = str(row.get("spec_id", f"row_{i}"))
        item = {
            "spec_id": spec_id,
            "source_split": str(row.get("source_split", "")),
            "source_row": int(row.get("source_row", i)),
            "smiles": str(row["smiles"]),
            "adduct": adduct,
            "precursor_mz": float(row["precursor_mz"]),
            "molecule_label": str(row.get("molecule_label", "Other")),
            "n_peaks": int(len(mzs)),
        }
        try:
            fe, result = run_magma(
                smiles=item["smiles"],
                peaks=mzs,
                peak_intensities=intensities,
                ppm_threshold=args.ppm,
                adduct=adduct,
                verbose=False,
                profile=False,
                skip_canonicalization=args.no_canonicalize,
            )
            annotations = []
            for peak_id, ann in result.reset_index(drop=True).iterrows():
                matched = pd.notna(ann.get("matched_mass"))
                frag_atoms: list[int] = []
                frag_smi = None
                if matched and pd.notna(ann.get("frag_id")):
                    frag_atoms = fragment_atom_indices(fe, int(ann.get("frag_id")))
                    frag_smi = fragment_smiles(fe, frag_atoms)
                annotations.append({
                    "peak_id": int(peak_id),
                    "mz_observed": float(ann["observed_mz"]),
                    "intensity": None if pd.isna(ann.get("intensity")) else float(ann.get("intensity")),
                    "assigned": bool(matched),
                    "ppm_diff": None if pd.isna(ann.get("ppm_diff")) else float(ann.get("ppm_diff")),
                    "frag_mass": None if not matched else float(ann["matched_mass"]),
                    "frag_h_shift": None if pd.isna(ann.get("h_shift")) else int(ann.get("h_shift")),
                    "frag_formula": None if pd.isna(ann.get("formula")) else str(ann.get("formula")),
                    "frag_id": None if pd.isna(ann.get("frag_id")) else int(ann.get("frag_id")),
                    "frag_atom_indices": frag_atoms,
                    "frag_smiles": frag_smi,
                    "score": None if pd.isna(ann.get("score")) else float(ann.get("score")),
                    "n_candidates": int(ann.get("n_matches", 0) or 0),
                })
            n_matched = sum(x["assigned"] for x in annotations)
            item.update({
                "success": True,
                "error": None,
                "n_matched": int(n_matched),
                "match_rate": float(n_matched / max(1, len(annotations))),
                "annotations": annotations,
            })
        except Exception as exc:
            item.update({
                "success": False,
                "error": repr(exc),
                "n_matched": 0,
                "match_rate": 0.0,
                "annotations": [],
            })
        samples.append(item)
        if not args.quiet and (len(samples) == 1 or len(samples) % 25 == 0):
            ok = sum(x["success"] for x in samples)
            mean_rate = np.mean([x["match_rate"] for x in samples if x["success"]]) if ok else 0.0
            print(f"{len(samples)}/{len(df)} annotated, success={ok}, mean_match={mean_rate:.3f}", flush=True)

    payload = {
        "stage": "msnlib_magma_peak_annotation",
        "input": str(args.input),
        "ppm": args.ppm,
        "n_samples": len(samples),
        "n_success": int(sum(x["success"] for x in samples)),
        "elapsed_sec": round(time.time() - t0, 3),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({k: payload[k] for k in ["n_samples", "n_success", "elapsed_sec"]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
