#!/usr/bin/env python3
"""Blindly score all Figure 3c candidates from one retrieved peak's annotations.

The retrieval and evidence-retention stages are candidate-label blind.  Candidate
structures are used only after source-intrinsic MAGMa validation, and every one
of the 60 candidates is scored by the same substructure-compatibility rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


PRIMARY_RULE = "family_max_cosine_sum"
SENSITIVITY_RULES = (
    "family_count",
    "family_parent_weighted_cosine_sum",
    "row_cosine_sum",
    "best_family_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def connectivity_key(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol).split("-")[0]


def rank_interval(scores: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact-tie-aware minimum rank, maximum rank and tie size."""
    values = scores.to_numpy(dtype=np.float64)
    rank_min = np.empty(len(values), dtype=np.int64)
    rank_max = np.empty(len(values), dtype=np.int64)
    tie_size = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        greater = int(np.sum(values > value))
        equal = np.isclose(values, value, rtol=1e-12, atol=1e-12)
        rank_min[index] = greater + 1
        tie_size[index] = int(equal.sum())
        rank_max[index] = greater + tie_size[index]
    return rank_min, rank_max, tie_size


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    peak_id = int(protocol["query_peak_id"])
    query_mz = float(protocol["query_mz"])

    rows = pd.read_csv(args.neighbors, low_memory=False)
    rows = rows.loc[rows.peak_id.eq(peak_id)].copy()
    if rows.empty or not np.allclose(rows.query_mz, query_mz, rtol=0.0, atol=1e-9):
        raise RuntimeError("Frozen query peak identity does not reproduce")

    # The blind evidence gate deliberately excludes every field computed from
    # Figure 3c candidate structures, including pairwise_relation and the
    # target/fixed-isomer compatibility flags.
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
        raise RuntimeError("No source-intrinsically valid fragment annotations")
    if valid.frag_smiles.isna().any() or valid.frag_connectivity_key.isna().any():
        raise RuntimeError("Blind-valid annotations contain missing fragment structures")

    fragment_mols: dict[str, Chem.Mol] = {}
    for row in valid.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.frag_smiles))
        if mol is None:
            raise RuntimeError(f"Invalid fragment SMILES: {row.frag_smiles}")
        reproduced = connectivity_key(mol)
        if reproduced != str(row.frag_connectivity_key):
            raise RuntimeError(
                f"Fragment connectivity key mismatch: {reproduced} != "
                f"{row.frag_connectivity_key}"
            )
        fragment_mols.setdefault(reproduced, mol)

    families = (
        valid.groupby("frag_connectivity_key", as_index=False)
        .agg(
            representative_frag_smiles=("frag_smiles", "first"),
            max_cosine=("peak_embedding_cosine", "max"),
            median_cosine=("peak_embedding_cosine", "median"),
            n_reference_rows=("reference_store_index", "size"),
            n_reference_spectra=("reference_store_index", "nunique"),
            n_reference_parent_connectivities=("reference_parent_key", "nunique"),
            best_parent_diverse_rank=("parent_diverse_rank", "min"),
            minimum_delta_ppm=("delta_ppm", "min"),
        )
        .sort_values(["max_cosine", "frag_connectivity_key"], ascending=[False, True])
        .reset_index(drop=True)
    )

    candidates_full = pd.read_csv(args.candidates)
    required = {"connectivity_key", "candidate_smiles", "is_true", "is_fixed_decoy"}
    if not required.issubset(candidates_full.columns):
        raise ValueError(f"Candidate file lacks {sorted(required - set(candidates_full))}")
    if len(candidates_full) != 60 or candidates_full.connectivity_key.nunique() != 60:
        raise RuntimeError("Expected exactly 60 connectivity-distinct candidates")

    # Score from a deliberately unlabeled view.  Labels are merged only after
    # the blind score table has been fully computed and written to disk.
    candidates = candidates_full[["connectivity_key", "candidate_smiles"]].copy()
    candidate_mols: list[Chem.Mol] = []
    for row in candidates.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.candidate_smiles))
        if mol is None or connectivity_key(mol) != str(row.connectivity_key):
            raise RuntimeError(f"Candidate structure failed: {row.connectivity_key}")
        if rdMolDescriptors.CalcMolFormula(mol) != protocol["candidate_formula"]:
            raise RuntimeError(f"Candidate formula failed: {row.connectivity_key}")
        candidate_mols.append(mol)

    family_records: list[dict] = []
    row_records: list[dict] = []
    for candidate, mol in zip(candidates.itertuples(index=False), candidate_mols):
        matched_families: list[pd.Series] = []
        for _, family in families.iterrows():
            frag = fragment_mols[str(family.frag_connectivity_key)]
            matched = bool(mol.HasSubstructMatch(frag, useChirality=False))
            family_records.append(
                {
                    "connectivity_key": candidate.connectivity_key,
                    "frag_connectivity_key": family.frag_connectivity_key,
                    "matched": matched,
                    "max_cosine": float(family.max_cosine),
                    "n_reference_parent_connectivities": int(
                        family.n_reference_parent_connectivities
                    ),
                }
            )
            if matched:
                matched_families.append(family)

        matched_rows = []
        for evidence in valid.itertuples(index=False):
            frag = fragment_mols[str(evidence.frag_connectivity_key)]
            if mol.HasSubstructMatch(frag, useChirality=False):
                matched_rows.append(evidence)

        family_max_cosine_sum = float(sum(float(x.max_cosine) for x in matched_families))
        family_count = int(len(matched_families))
        family_parent_weighted = float(sum(
            float(x.max_cosine)
            * math.log2(1.0 + int(x.n_reference_parent_connectivities))
            for x in matched_families
        ))
        row_cosine_sum = float(sum(float(x.peak_embedding_cosine) for x in matched_rows))
        best_family_cosine = float(
            max((float(x.max_cosine) for x in matched_families), default=0.0)
        )
        row_records.append(
            {
                "connectivity_key": candidate.connectivity_key,
                "candidate_smiles": candidate.candidate_smiles,
                PRIMARY_RULE: family_max_cosine_sum,
                "family_count": family_count,
                "family_parent_weighted_cosine_sum": family_parent_weighted,
                "row_cosine_sum": row_cosine_sum,
                "best_family_cosine": best_family_cosine,
            }
        )

    blind_scores = pd.DataFrame(row_records)
    rules = (PRIMARY_RULE,) + SENSITIVITY_RULES
    for rule in rules:
        rank_min, rank_max, tie_size = rank_interval(blind_scores[rule])
        blind_scores[f"{rule}_rank_min"] = rank_min
        blind_scores[f"{rule}_rank_max"] = rank_max
        blind_scores[f"{rule}_tie_size"] = tie_size
    blind_scores = blind_scores.sort_values(
        [f"{PRIMARY_RULE}_rank_min", PRIMARY_RULE, "connectivity_key"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    blind_path = args.output_dir / "blind_candidate_scores.csv"
    blind_scores.to_csv(blind_path, index=False)

    compatibility = pd.DataFrame(family_records)
    compatibility.to_csv(args.output_dir / "blind_candidate_family_compatibility.csv", index=False)
    valid.to_csv(args.output_dir / "blind_source_valid_neighbor_annotations.csv", index=False)
    families.to_csv(args.output_dir / "blind_fragment_topology_families.csv", index=False)

    labels = candidates_full[
        ["connectivity_key", "candidate_index", "is_true", "is_fixed_decoy"]
    ].copy()
    revealed = blind_scores.merge(labels, on="connectivity_key", how="left", validate="one_to_one")
    revealed.to_csv(args.output_dir / "revealed_candidate_scores.csv", index=False)

    per_rule: dict[str, dict] = {}
    for rule in rules:
        target = revealed.loc[revealed.is_true.eq(1)].iloc[0]
        decoy = revealed.loc[revealed.is_fixed_decoy.eq(1)].iloc[0]
        top = revealed.loc[revealed[f"{rule}_rank_min"].eq(1)]
        per_rule[rule] = {
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

    primary = per_rule[PRIMARY_RULE]
    criteria = protocol["success_criteria"]
    primary_pairwise_success = primary["target_minus_fixed_decoy"] > 0
    primary_global_top1 = primary["target_rank_interval"] == [1, 1]
    sensitivity_target_above_decoy = sum(
        per_rule[rule]["target_minus_fixed_decoy"] > 0 for rule in SENSITIVITY_RULES
    )
    summary = {
        "experiment": "Figure 3c m/z 316 candidate-blind peak-evidence scoring",
        "query_peak_id": peak_id,
        "query_mz": query_mz,
        "candidate_formula": protocol["candidate_formula"],
        "n_candidates": len(blind_scores),
        "n_selected_neighbors_before_blind_source_gate": len(rows),
        "n_blind_source_valid_neighbor_annotations": len(valid),
        "n_blind_fragment_topology_families": len(families),
        "primary_rule": PRIMARY_RULE,
        "sensitivity_rules": list(SENSITIVITY_RULES),
        "results_by_rule": per_rule,
        "success_criteria": criteria,
        "criteria_outcome": {
            "primary_target_above_fixed_decoy": bool(primary_pairwise_success),
            "primary_unique_global_top1": bool(primary_global_top1),
            "sensitivity_rules_target_above_fixed_decoy": int(
                sensitivity_target_above_decoy
            ),
            "sensitivity_rule_count": len(SENSITIVITY_RULES),
            "pairwise_claim_pass": bool(
                primary_pairwise_success
                and sensitivity_target_above_decoy >= int(
                    criteria["minimum_sensitivity_rules_target_above_fixed_decoy"]
                )
            ),
            "global_blind_identification_claim_pass": bool(primary_global_top1),
        },
        "blindness_audit": {
            "retrieval_score_uses_candidate_structure": False,
            "source_validity_gate_uses_candidate_structure": False,
            "candidate_compatibility_applied_after_source_validity": True,
            "all_60_candidates_scored_by_identical_rule": True,
            "target_and_fixed_decoy_labels_merged_after_blind_score_write": True,
            "forbidden_filter_fields": [
                "target_substructure_compatible",
                "fixed_decoy_substructure_compatible",
                "pairwise_relation",
                "compatible_candidates_of_60",
                "compatible_candidate_keys",
                "strict_retained_substructure",
            ],
        },
        "input_sha256": {
            "neighbors": sha256(args.neighbors),
            "candidates": sha256(args.candidates),
            "protocol": sha256(args.protocol),
        },
        "output_sha256": {
            "blind_candidate_scores": sha256(blind_path),
        },
        "boundary": (
            "This single-peak descriptive result uses computational MAGMa fragment "
            "hypotheses. It is not an authentic-standard, MS3 or NMR confirmation."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
