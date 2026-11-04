#!/usr/bin/env python3
"""Rebuild the quantitative audit supporting the Fig. 3e supplement 1 story."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
DESCRIPTOR_FILE = SOURCE / "four_peaks_break_molecular_tie_nature_candidate_descriptor_distribution_source.csv"
MAGMA_FILE = SOURCE / "same_formula_magma_annotations.json"
OUTPUT = SOURCE / "chemical_significance_audit.json"

PROPERTIES = [
    "tpsa_a2",
    "clogp",
    "phenolic_oh_groups",
    "carbonyl_groups",
    "rotatable_bonds",
    "fraction_csp3",
]


def scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def property_match_mask(df: pd.DataFrame, target: pd.Series) -> np.ndarray:
    """Compare displayed values with tolerance for CSV floating-point round trips."""
    mask = np.ones(len(df), dtype=bool)
    for prop in PROPERTIES:
        mask &= np.isclose(df[prop].to_numpy(), float(target[prop]), rtol=0.0, atol=1e-6)
    return mask


def top20_summary(df: pd.DataFrame, model: str, target: pd.Series) -> dict:
    subset = df.nsmallest(20, f"{model}_rank")
    non_target = subset.loc[~subset["is_true"].astype(bool)].copy()
    return {
        "n_non_target": int(len(non_target)),
        "morgan_tanimoto_mean": float(non_target["morgan_tanimoto_to_target"].mean()),
        "morgan_tanimoto_median": float(non_target["morgan_tanimoto_to_target"].median()),
        "morgan_tanimoto_rank_weighted_mean": float(
            np.average(
                non_target["morgan_tanimoto_to_target"],
                weights=1.0 / non_target[f"{model}_rank"],
            )
        ),
        "morgan_tanimoto_ge_0_7": int((non_target["morgan_tanimoto_to_target"] >= 0.7).sum()),
        "all_six_properties_equal_target": int(property_match_mask(non_target, target).sum()),
        "mean_absolute_property_difference": {
            prop: float((non_target[prop] - target[prop]).abs().mean()) for prop in PROPERTIES
        },
    }


def main() -> None:
    df = pd.read_csv(DESCRIPTOR_FILE)
    target = df.loc[df["is_true"].astype(bool)].iloc[0]
    fixed = df.loc[df["is_fixed_decoy"].astype(bool)].iloc[0]
    non_target = df.loc[~df["is_true"].astype(bool)]

    property_match = property_match_mask(df, target)
    property_matched = df.loc[property_match].copy()

    model_summary = {}
    for model in ("ultrams", "dreams"):
        score = f"{model}_score"
        rank = f"{model}_rank"
        rho, pvalue = spearmanr(df[score], df["morgan_tanimoto_to_target"])
        target_score = float(target[score])
        model_summary[model] = {
            "target_rank_all_60": int(target[rank]),
            "target_rank_among_descriptor_matched": int(
                property_matched[score].rank(method="min", ascending=False).loc[target.name]
            ),
            "target_score": target_score,
            "target_z_above_non_target_mean": float(
                (target_score - non_target[score].mean()) / non_target[score].std(ddof=1)
            ),
            "target_lead_over_best_non_target": float(target_score - non_target[score].max()),
            "target_margin_over_fixed_decoy": float(target_score - fixed[score]),
            "score_vs_morgan_spearman_rho": float(rho),
            "score_vs_morgan_spearman_p": float(pvalue),
            "top20_non_target": top20_summary(df, model, target),
        }

    magma = json.loads(MAGMA_FILE.read_text())
    samples = {sample["spec_id"]: sample for sample in magma["samples"]}
    true_sample = samples["sample_545225_true"]
    decoy_sample = samples["sample_545225_dreams_top1_decoy"]
    true_ann = {int(a["peak_id"]): a for a in true_sample["annotations"]}
    decoy_ann = {int(a["peak_id"]): a for a in decoy_sample["annotations"]}
    shared_assigned = set(k for k, v in true_ann.items() if v["assigned"]) & set(
        k for k, v in decoy_ann.items() if v["assigned"]
    )
    same_formula = [k for k in shared_assigned if true_ann[k]["frag_formula"] == decoy_ann[k]["frag_formula"]]
    same_smiles = [k for k in same_formula if true_ann[k]["frag_smiles"] == decoy_ann[k]["frag_smiles"]]

    ultra_top = set(df.nsmallest(20, "ultrams_rank").loc[lambda x: ~x["is_true"].astype(bool), "connectivity_key"])
    dreams_top = set(df.nsmallest(20, "dreams_rank").loc[lambda x: ~x["is_true"].astype(bool), "connectivity_key"])

    audit = {
        "case": {
            "target_name": "8-methoxyquercetin 7-O-beta-D-glucopyranoside",
            "target_pubchem_cid": 75528773,
            "target_inchikey": "XTWQFJODFQFTIT-UAEGTEGVSA-N",
            "fixed_decoy_name": "Pleurostimin 7-glucoside",
            "fixed_decoy_pubchem_cid": 163016934,
            "formula": "C22H22O13",
            "precursor_mz": 493.09877,
            "n_measured_peaks": 54,
            "n_benchmark_entries": 62,
            "n_connectivity_candidates": int(len(df)),
        },
        "descriptor_tie": {
            "properties": PROPERTIES,
            "n_candidates_matching_target_on_all_six_including_target": int(property_match.sum()),
            "n_non_target_candidates_matching_target_on_all_six": int(property_match.sum() - 1),
            "fixed_decoy_matches_target_on_all_six": bool(
                np.allclose(fixed[PROPERTIES].astype(float), target[PROPERTIES].astype(float), rtol=0.0, atol=1e-6)
            ),
        },
        "models": model_summary,
        "top20_non_target_overlap": {
            "intersection": int(len(ultra_top & dreams_top)),
            "union": int(len(ultra_top | dreams_top)),
            "jaccard": float(len(ultra_top & dreams_top) / len(ultra_top | dreams_top)),
        },
        "fixed_decoy": {
            "ultrams_rank": int(fixed["ultrams_rank"]),
            "dreams_rank": int(fixed["dreams_rank"]),
            "morgan_tanimoto_to_target": float(fixed["morgan_tanimoto_to_target"]),
        },
        "magma": {
            "true_assigned_peaks": int(true_sample["n_matched"]),
            "true_assignment_rate": float(true_sample["match_rate"]),
            "fixed_decoy_assigned_peaks": int(decoy_sample["n_matched"]),
            "fixed_decoy_assignment_rate": float(decoy_sample["match_rate"]),
            "peaks_assigned_for_both": int(len(shared_assigned)),
            "shared_fragment_formula": int(len(same_formula)),
            "shared_fragment_formula_and_smiles": int(len(same_smiles)),
            "same_formula_different_candidate_connectivity": int(len(same_formula) - len(same_smiles)),
            "interpretation": "MAGMa assignments are candidate interpretations, not confirmed pathways or model attribution.",
        },
        "claim_boundary": (
            "The audit supports connectivity-level discrimination within a same-formula, descriptor-matched set. "
            "It does not support a claim that UltraMS retrieves globally more similar non-targets or that the four displayed peaks caused the ranking."
        ),
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
