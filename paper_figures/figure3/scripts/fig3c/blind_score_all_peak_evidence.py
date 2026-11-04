#!/usr/bin/env python3
"""Blindly score all 60 Figure 3c candidates from all query-peak evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from blind_score_peak_evidence import as_bool, connectivity_key, rank_interval, sha256


PRIMARY_RULE = "peak_best_cosine_sum"
SENSITIVITY_RULES = (
    "peak_coverage_count",
    "peak_best_cosine_intensity_sum",
    "peak_best_parent_weighted_sum",
    "global_top_family_sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--query-metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = pd.read_csv(args.neighbors, low_memory=False)
    query = pd.read_csv(args.query_metadata)[
        ["peak_id", "mz", "intensity", "encoded_intensity"]
    ].rename(columns={
        "mz": "frozen_query_mz",
        "intensity": "query_intensity",
        "encoded_intensity": "query_encoded_intensity",
    })
    if len(query) != int(protocol["query_peak_count"]) or query.peak_id.nunique() != len(query):
        raise RuntimeError("Query peak manifest does not reproduce")

    source_valid = (
        ~as_bool(rows.magma_failed)
        & as_bool(rows.magma_peak_index_verified)
        & as_bool(rows.magma_assigned)
        & as_bool(rows.nontrivial_fragment)
        & as_bool(rows.nonintact_fragment)
        & as_bool(rows.source_parent_substructure_compatible)
        & rows.peak_embedding_cosine.ge(float(protocol["minimum_cosine"]))
    )
    valid = rows.loc[source_valid].copy()
    if valid.empty:
        raise RuntimeError("No source-intrinsically valid annotations")
    valid = valid.merge(query, on="peak_id", how="left", validate="many_to_one")
    if (
        valid.query_encoded_intensity.isna().any()
        or not np.allclose(
            valid.query_mz, valid.frozen_query_mz, rtol=0.0, atol=2e-5
        )
    ):
        raise RuntimeError("Query metadata did not merge")

    fragment_mols: dict[str, Chem.Mol] = {}
    for row in valid.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.frag_smiles))
        if mol is None or connectivity_key(mol) != str(row.frag_connectivity_key):
            raise RuntimeError(f"Fragment failed: {row.frag_connectivity_key}")
        fragment_mols.setdefault(str(row.frag_connectivity_key), mol)

    families = (
        valid.groupby(["peak_id", "query_mz", "frag_connectivity_key"], as_index=False)
        .agg(
            representative_frag_smiles=("frag_smiles", "first"),
            max_cosine=("peak_embedding_cosine", "max"),
            median_cosine=("peak_embedding_cosine", "median"),
            n_reference_rows=("reference_store_index", "size"),
            n_reference_spectra=("reference_store_index", "nunique"),
            n_reference_parent_connectivities=("reference_parent_key", "nunique"),
            best_parent_diverse_rank=("parent_diverse_rank", "min"),
            minimum_delta_ppm=("delta_ppm", "min"),
            query_intensity=("query_intensity", "first"),
            query_encoded_intensity=("query_encoded_intensity", "first"),
        )
        .sort_values(
            ["peak_id", "max_cosine", "frag_connectivity_key"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    families["global_family_rank_within_peak"] = (
        families.groupby("peak_id").cumcount() + 1
    ).astype(int)

    candidates_full = pd.read_csv(args.candidates)
    required = {"connectivity_key", "candidate_smiles", "is_true", "is_fixed_decoy"}
    if not required.issubset(candidates_full.columns):
        raise ValueError(f"Candidate file lacks {sorted(required - set(candidates_full))}")
    if len(candidates_full) != 60 or candidates_full.connectivity_key.nunique() != 60:
        raise RuntimeError("Expected 60 connectivity-distinct candidates")
    candidates = candidates_full[["connectivity_key", "candidate_smiles"]].copy()
    candidate_mols: list[Chem.Mol] = []
    for row in candidates.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.candidate_smiles))
        if (
            mol is None
            or connectivity_key(mol) != str(row.connectivity_key)
            or rdMolDescriptors.CalcMolFormula(mol) != protocol["candidate_formula"]
        ):
            raise RuntimeError(f"Candidate failed: {row.connectivity_key}")
        candidate_mols.append(mol)

    compatibility_rows: list[dict] = []
    score_rows: list[dict] = []
    candidate_peak_rows: list[dict] = []
    for candidate, candidate_mol in zip(candidates.itertuples(index=False), candidate_mols):
        local_records: list[dict] = []
        for family in families.itertuples(index=False):
            frag = fragment_mols[str(family.frag_connectivity_key)]
            matched = bool(candidate_mol.HasSubstructMatch(frag, useChirality=False))
            record = {
                "connectivity_key": candidate.connectivity_key,
                "peak_id": int(family.peak_id),
                "query_mz": float(family.query_mz),
                "frag_connectivity_key": family.frag_connectivity_key,
                "matched": matched,
                "max_cosine": float(family.max_cosine),
                "n_reference_parent_connectivities": int(
                    family.n_reference_parent_connectivities
                ),
                "query_encoded_intensity": float(family.query_encoded_intensity),
                "global_family_rank_within_peak": int(
                    family.global_family_rank_within_peak
                ),
            }
            compatibility_rows.append(record)
            if matched:
                local_records.append(record)

        matched = pd.DataFrame(local_records)
        if matched.empty:
            per_peak = pd.DataFrame(columns=["peak_id"])
        else:
            matched["parent_weighted"] = matched.max_cosine * matched[
                "n_reference_parent_connectivities"
            ].map(lambda value: math.log2(1.0 + int(value)))
            per_peak = (
                matched.sort_values(
                    ["peak_id", "max_cosine", "frag_connectivity_key"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
                .groupby("peak_id", as_index=False)
                .agg(
                    best_cosine=("max_cosine", "max"),
                    query_encoded_intensity=("query_encoded_intensity", "first"),
                    best_parent_weighted=("parent_weighted", "max"),
                )
            )
        per_peak_full = query[["peak_id", "query_encoded_intensity"]].merge(
            per_peak[["peak_id", "best_cosine", "best_parent_weighted"]],
            on="peak_id", how="left", validate="one_to_one",
        )
        per_peak_full[["best_cosine", "best_parent_weighted"]] = per_peak_full[
            ["best_cosine", "best_parent_weighted"]
        ].fillna(0.0)
        for contribution in per_peak_full.itertuples(index=False):
            candidate_peak_rows.append({
                "connectivity_key": candidate.connectivity_key,
                "peak_id": int(contribution.peak_id),
                "best_cosine": float(contribution.best_cosine),
                "best_parent_weighted": float(contribution.best_parent_weighted),
                "query_encoded_intensity": float(contribution.query_encoded_intensity),
            })
        primary = float(per_peak_full.best_cosine.sum())
        coverage = int(per_peak_full.best_cosine.gt(0).sum())
        intensity_sum = float(
            (per_peak_full.best_cosine * per_peak_full.query_encoded_intensity).sum()
        )
        parent_weighted_sum = float(per_peak_full.best_parent_weighted.sum())
        top_family = matched.loc[
            matched.global_family_rank_within_peak.eq(1)
        ] if len(matched) else matched
        global_top_sum = float(top_family.max_cosine.sum()) if len(top_family) else 0.0
        score_rows.append(
            {
                "connectivity_key": candidate.connectivity_key,
                "candidate_smiles": candidate.candidate_smiles,
                PRIMARY_RULE: primary,
                "peak_coverage_count": coverage,
                "peak_best_cosine_intensity_sum": intensity_sum,
                "peak_best_parent_weighted_sum": parent_weighted_sum,
                "global_top_family_sum": global_top_sum,
            }
        )

    blind_scores = pd.DataFrame(score_rows)
    rules = (PRIMARY_RULE,) + SENSITIVITY_RULES
    for rule in rules:
        rank_min, rank_max, tie_size = rank_interval(blind_scores[rule])
        blind_scores[f"{rule}_rank_min"] = rank_min
        blind_scores[f"{rule}_rank_max"] = rank_max
        blind_scores[f"{rule}_tie_size"] = tie_size
    blind_scores = blind_scores.sort_values(
        [f"{PRIMARY_RULE}_rank_min", PRIMARY_RULE, "connectivity_key"],
        ascending=[True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    blind_path = args.output_dir / "blind_candidate_scores.csv"
    blind_scores.to_csv(blind_path, index=False)
    pd.DataFrame(compatibility_rows).to_csv(
        args.output_dir / "blind_candidate_family_compatibility.csv", index=False
    )
    candidate_peak = pd.DataFrame(candidate_peak_rows)
    candidate_peak.to_csv(
        args.output_dir / "blind_candidate_peak_contributions.csv", index=False
    )
    valid.to_csv(args.output_dir / "blind_source_valid_neighbor_annotations.csv", index=False)
    families.to_csv(args.output_dir / "blind_fragment_topology_families.csv", index=False)

    labels = candidates_full[
        ["connectivity_key", "candidate_index", "is_true", "is_fixed_decoy"]
    ]
    revealed = blind_scores.merge(labels, on="connectivity_key", how="left", validate="one_to_one")
    revealed.to_csv(args.output_dir / "revealed_candidate_scores.csv", index=False)

    peak51 = candidate_peak.loc[candidate_peak.peak_id.eq(51)].copy()
    without51 = (
        candidate_peak.loc[~candidate_peak.peak_id.eq(51)]
        .groupby("connectivity_key", as_index=False)
        .best_cosine.sum()
        .rename(columns={"best_cosine": "primary_score_without_peak51"})
    )
    rank_min, rank_max, tie_size = rank_interval(without51.primary_score_without_peak51)
    without51["rank_min"] = rank_min
    without51["rank_max"] = rank_max
    without51["tie_size"] = tie_size
    without51 = without51.merge(labels, on="connectivity_key", how="left", validate="one_to_one")
    without51.to_csv(args.output_dir / "revealed_primary_scores_without_peak51.csv", index=False)

    results: dict[str, dict] = {}
    for rule in rules:
        target = revealed.loc[revealed.is_true.eq(1)].iloc[0]
        decoy = revealed.loc[revealed.is_fixed_decoy.eq(1)].iloc[0]
        top = revealed.loc[revealed[f"{rule}_rank_min"].eq(1)]
        results[rule] = {
            "target_score": float(target[rule]),
            "target_rank_interval": [
                int(target[f"{rule}_rank_min"]), int(target[f"{rule}_rank_max"])
            ],
            "target_tie_size": int(target[f"{rule}_tie_size"]),
            "fixed_decoy_score": float(decoy[rule]),
            "fixed_decoy_rank_interval": [
                int(decoy[f"{rule}_rank_min"]), int(decoy[f"{rule}_rank_max"])
            ],
            "fixed_decoy_tie_size": int(decoy[f"{rule}_tie_size"]),
            "target_minus_fixed_decoy": float(target[rule] - decoy[rule]),
            "top_candidate_keys": sorted(top.connectivity_key.astype(str).tolist()),
        }

    primary = results[PRIMARY_RULE]
    sensitivity_above_decoy = sum(
        results[rule]["target_minus_fixed_decoy"] > 0 for rule in SENSITIVITY_RULES
    )
    target_without51 = without51.loc[without51.is_true.eq(1)].iloc[0]
    decoy_without51 = without51.loc[without51.is_fixed_decoy.eq(1)].iloc[0]
    target_peak51 = peak51.merge(labels, on="connectivity_key", how="left").loc[
        lambda frame: frame.is_true.eq(1)
    ].iloc[0]
    decoy_peak51 = peak51.merge(labels, on="connectivity_key", how="left").loc[
        lambda frame: frame.is_fixed_decoy.eq(1)
    ].iloc[0]
    full_margin = float(primary["target_minus_fixed_decoy"])
    peak51_margin = float(target_peak51.best_cosine - decoy_peak51.best_cosine)
    summary = {
        "experiment": "Figure 3c all-peak candidate-blind evidence scoring",
        "query_peak_count": len(query),
        "candidate_formula": protocol["candidate_formula"],
        "n_candidates": len(blind_scores),
        "n_selected_neighbors_before_blind_source_gate": len(rows),
        "n_blind_source_valid_neighbor_annotations": len(valid),
        "n_blind_fragment_topology_families": len(families),
        "n_query_peaks_with_blind_valid_families": int(families.peak_id.nunique()),
        "primary_rule": PRIMARY_RULE,
        "sensitivity_rules": list(SENSITIVITY_RULES),
        "results_by_rule": results,
        "success_criteria": protocol["success_criteria"],
        "criteria_outcome": {
            "primary_target_above_fixed_decoy": bool(
                primary["target_minus_fixed_decoy"] > 0
            ),
            "primary_unique_global_top1": bool(
                primary["target_rank_interval"] == [1, 1]
            ),
            "sensitivity_rules_target_above_fixed_decoy": int(sensitivity_above_decoy),
            "sensitivity_rule_count": len(SENSITIVITY_RULES),
        },
        "peak51_ablation": {
            "target_contribution": float(target_peak51.best_cosine),
            "fixed_decoy_contribution": float(decoy_peak51.best_cosine),
            "pairwise_margin_contribution": peak51_margin,
            "fraction_of_full_primary_pairwise_margin": (
                None if full_margin == 0 else float(peak51_margin / full_margin)
            ),
            "target_score_without_peak51": float(
                target_without51.primary_score_without_peak51
            ),
            "target_rank_interval_without_peak51": [
                int(target_without51.rank_min), int(target_without51.rank_max)
            ],
            "fixed_decoy_score_without_peak51": float(
                decoy_without51.primary_score_without_peak51
            ),
            "fixed_decoy_rank_interval_without_peak51": [
                int(decoy_without51.rank_min), int(decoy_without51.rank_max)
            ],
            "target_minus_fixed_decoy_without_peak51": float(
                target_without51.primary_score_without_peak51
                - decoy_without51.primary_score_without_peak51
            ),
        },
        "blindness_audit": {
            "source_validity_gate_uses_candidate_structure": False,
            "all_60_candidates_scored_by_identical_rule": True,
            "target_and_fixed_decoy_labels_merged_after_blind_score_write": True,
            "forbidden_filter_fields": protocol["forbidden_during_evidence_retention"],
        },
        "input_sha256": {
            "neighbors": sha256(args.neighbors),
            "candidates": sha256(args.candidates),
            "query_metadata": sha256(args.query_metadata),
            "protocol": sha256(args.protocol),
        },
        "output_sha256": {"blind_candidate_scores": sha256(blind_path)},
        "boundary": (
            "This is a descriptive blind rescore of one frozen case using "
            "computational MAGMa hypotheses, not independent structural confirmation."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
