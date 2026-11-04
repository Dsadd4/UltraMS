#!/usr/bin/env python3
"""Freeze candidate-blind top topology selection and reveal pairwise compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--registered-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.registered_protocol.read_text(encoding="utf-8"))
    if "global_top_family_sum" not in protocol.get("sensitivity_rules", []):
        raise RuntimeError("Top-family rule was not registered before execution")

    families = pd.read_csv(args.families)
    required_family = {
        "peak_id", "query_mz", "frag_connectivity_key", "representative_frag_smiles",
        "max_cosine", "median_cosine", "n_reference_rows", "n_reference_spectra",
        "n_reference_parent_connectivities", "best_parent_diverse_rank",
        "minimum_delta_ppm",
    }
    if not required_family.issubset(families.columns):
        raise ValueError(f"Missing family columns: {sorted(required_family - set(families))}")

    # Candidate-blind selection: no candidate identity or compatibility table is
    # read until this table has been selected and written.
    blind_top = (
        families.sort_values(
            ["peak_id", "max_cosine", "frag_connectivity_key"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("peak_id", as_index=False)
        .first()
        .sort_values("peak_id", kind="mergesort")
        .reset_index(drop=True)
    )
    blind_path = args.output_dir / "blind_top_topology_per_peak.csv"
    blind_top.to_csv(blind_path, index=False)

    candidates = pd.read_csv(args.candidates)
    if candidates.connectivity_key.nunique() != 60:
        raise RuntimeError("Expected 60 connectivity-distinct candidates")
    target = candidates.loc[candidates.is_true.eq(1), "connectivity_key"]
    fixed = candidates.loc[candidates.is_fixed_decoy.eq(1), "connectivity_key"]
    if len(target) != 1 or len(fixed) != 1:
        raise RuntimeError("Expected exactly one target and one fixed decoy")
    target_key, fixed_key = str(target.iloc[0]), str(fixed.iloc[0])

    compatibility = pd.read_csv(args.compatibility)
    if compatibility.matched.dtype == object:
        compatibility["matched"] = compatibility.matched.astype(str).str.lower().isin(
            ["true", "1"]
        )
    else:
        compatibility["matched"] = compatibility.matched.astype(bool)
    pair = compatibility.loc[
        compatibility.connectivity_key.isin([target_key, fixed_key]),
        ["connectivity_key", "peak_id", "frag_connectivity_key", "matched"],
    ]
    pivot = pair.pivot(
        index=["peak_id", "frag_connectivity_key"],
        columns="connectivity_key",
        values="matched",
    ).reset_index()
    revealed = blind_top.merge(
        pivot,
        on=["peak_id", "frag_connectivity_key"],
        how="left",
        validate="one_to_one",
    ).rename(columns={target_key: "target_compatible", fixed_key: "pleurostimin_compatible"})
    revealed[["target_compatible", "pleurostimin_compatible"]] = revealed[
        ["target_compatible", "pleurostimin_compatible"]
    ].fillna(False).astype(bool)

    def relation(row: pd.Series) -> str:
        if row.target_compatible and not row.pleurostimin_compatible:
            return "target_only"
        if row.pleurostimin_compatible and not row.target_compatible:
            return "pleurostimin_only"
        if row.target_compatible and row.pleurostimin_compatible:
            return "shared"
        return "neither"

    revealed["pairwise_relation"] = revealed.apply(relation, axis=1)
    revealed_path = args.output_dir / "revealed_pairwise_compatibility.csv"
    revealed.to_csv(revealed_path, index=False)

    selected_316 = revealed.loc[revealed.peak_id.eq(51)]
    if len(selected_316) != 1:
        raise RuntimeError("Expected one frozen top family for peak 51")
    peak316 = selected_316.iloc[0]
    relation_counts = revealed.pairwise_relation.value_counts().to_dict()
    informative = revealed.loc[
        revealed.target_compatible | revealed.pleurostimin_compatible
    ]
    discriminative = revealed.loc[
        revealed.target_compatible.ne(revealed.pleurostimin_compatible)
    ]

    summary = {
        "experiment": "Figure 3c candidate-blind top-topology selection",
        "selection_rule": (
            "For each query peak, select the source-valid fragment topology with "
            "the highest contextual peak-embedding cosine; ties are resolved by "
            "fragment connectivity key. Candidate identities and compatibility "
            "are revealed only after this table is frozen."
        ),
        "rule_registered_before_execution_as": "global_top_family_sum",
        "n_query_peaks_with_source_valid_topology": int(len(blind_top)),
        "pairwise_relation_counts": {key: int(value) for key, value in relation_counts.items()},
        "n_pairwise_informative_topologies": int(len(informative)),
        "n_pairwise_discriminative_topologies": int(len(discriminative)),
        "all_discriminative_topologies_favor_target": bool(
            len(discriminative) > 0
            and discriminative.target_compatible.all()
            and (~discriminative.pleurostimin_compatible).all()
        ),
        "peak_316": {
            "peak_id": int(peak316.peak_id),
            "query_mz": float(peak316.query_mz),
            "selected_frag_connectivity_key": str(peak316.frag_connectivity_key),
            "max_cosine": float(peak316.max_cosine),
            "median_cosine": float(peak316.median_cosine),
            "n_reference_spectra": int(peak316.n_reference_spectra),
            "n_reference_parent_connectivities": int(
                peak316.n_reference_parent_connectivities
            ),
            "minimum_delta_ppm": float(peak316.minimum_delta_ppm),
            "target_compatible": bool(peak316.target_compatible),
            "pleurostimin_compatible": bool(peak316.pleurostimin_compatible),
            "pairwise_relation": str(peak316.pairwise_relation),
        },
        "blindness_audit": {
            "blind_table_written_before_candidate_file_read": True,
            "candidate_identity_used_to_select_topology": False,
            "candidate_compatibility_used_to_select_topology": False,
        },
        "input_sha256": {
            "families": sha256(args.families),
            "compatibility": sha256(args.compatibility),
            "candidates": sha256(args.candidates),
            "registered_protocol": sha256(args.registered_protocol),
        },
        "output_sha256": {
            "blind_top_topology_per_peak": sha256(blind_path),
            "revealed_pairwise_compatibility": sha256(revealed_path),
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
