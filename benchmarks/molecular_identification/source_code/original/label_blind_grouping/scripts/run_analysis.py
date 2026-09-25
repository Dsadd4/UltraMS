#!/usr/bin/env python3
"""Build and evaluate label-blind MSnLib spectrum consensus groups."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


PACKAGE = Path(__file__).resolve().parents[1]
GROUPING_COLUMNS = [
    "sample_idx",
    "mzs",
    "intensities",
    "precursor_mz",
    "adduct",
    "retention_time",
    "collision_energy",
]
HIDDEN_LABEL_COLUMN = "hidden_smiles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PACKAGE / "config" / "analysis_config.json"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def spectrum_signature(mzs: str, intensities: str) -> str:
    return hashlib.sha256((mzs + "|" + intensities).encode()).hexdigest()


def prepare_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "mzs",
        "intensities",
        "smiles",
        "precursor_mz",
        "adduct",
        "collision_energy",
        "fold",
        "rt",
    ]
    raw = pd.read_csv(dataset_path, usecols=columns)
    raw["sample_idx"] = raw.index.astype(np.int64)
    raw = raw.rename(columns={"rt": "retention_time"})
    raw["retention_time"] = pd.to_numeric(raw["retention_time"], errors="coerce").fillna(0.0)
    raw["collision_energy"] = pd.to_numeric(raw["collision_energy"], errors="coerce")
    raw["precursor_mz"] = pd.to_numeric(raw["precursor_mz"], errors="coerce")
    hidden = raw[["sample_idx", "fold", "smiles"]].copy()
    # Retrieval outputs use the standardized dataset SMILES string verbatim.
    # Canonicalizing only the target side would create false mismatches.
    hidden[HIDDEN_LABEL_COLUMN] = hidden["smiles"].astype(str)
    grouping = raw[["fold", *GROUPING_COLUMNS]].copy()
    assert HIDDEN_LABEL_COLUMN not in grouping.columns and "smiles" not in grouping.columns
    return grouping, hidden


def vectorize_spectrum(
    mzs_text: str, intensities_text: str, top_n: int, bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.fromstring(str(mzs_text), sep=",", dtype=np.float64)
    intensities = np.fromstring(str(intensities_text), sep=",", dtype=np.float64)
    n = min(len(mzs), len(intensities))
    mzs, intensities = mzs[:n], intensities[:n]
    finite = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)
    mzs, intensities = mzs[finite], intensities[finite]
    if len(intensities) > top_n:
        keep = np.argpartition(intensities, -top_n)[-top_n:]
        mzs, intensities = mzs[keep], intensities[keep]
    transformed = np.sqrt(np.maximum(intensities, 0.0))
    bins = np.rint(mzs / bin_width).astype(np.int64)
    unique_bins, inverse = np.unique(bins, return_inverse=True)
    values = np.bincount(inverse, weights=transformed).astype(np.float64)
    norm = np.linalg.norm(values)
    if norm > 0:
        values /= norm
    return unique_bins, values


@dataclass
class SimilarityBlock:
    key: tuple[str, float, float]
    sample_indices: np.ndarray
    similarity: np.ndarray


def build_similarity_blocks(split_df: pd.DataFrame, cfg: dict[str, Any]) -> list[SimilarityBlock]:
    group_cfg = cfg["grouping"]
    working = split_df.copy()
    working["precursor_key"] = working["precursor_mz"].round(
        int(group_cfg["precursor_round_decimals"])
    )
    working["rt_key"] = working["retention_time"].round(
        int(group_cfg["retention_time_round_decimals"])
    )
    working["adduct_key"] = working["adduct"].fillna("unknown").astype(str)
    blocks: list[SimilarityBlock] = []
    for key, frame in working.groupby(
        ["adduct_key", "precursor_key", "rt_key"], sort=True, dropna=False
    ):
        if len(frame) < int(group_cfg["min_group_size"]):
            continue
        sparse_vectors = [
            vectorize_spectrum(
                row.mzs,
                row.intensities,
                int(group_cfg["spectrum_top_n_peaks"]),
                float(group_cfg["spectrum_mz_bin_width_da"]),
            )
            for row in frame.itertuples(index=False)
        ]
        all_bins = np.unique(np.concatenate([item[0] for item in sparse_vectors]))
        matrix = np.zeros((len(frame), len(all_bins)), dtype=np.float32)
        for row_idx, (bins, values) in enumerate(sparse_vectors):
            matrix[row_idx, np.searchsorted(all_bins, bins)] = values
        similarity = np.clip(matrix @ matrix.T, 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)
        blocks.append(
            SimilarityBlock(
                key=(str(key[0]), float(key[1]), float(key[2])),
                sample_indices=frame["sample_idx"].to_numpy(dtype=np.int64),
                similarity=similarity,
            )
        )
    return blocks


def cluster_block(block: SimilarityBlock, threshold: float) -> list[np.ndarray]:
    n = len(block.sample_indices)
    if n == 2:
        labels = np.array([1, 1] if block.similarity[0, 1] >= threshold else [1, 2])
    else:
        distances = np.clip(1.0 - block.similarity, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
        tree = linkage(squareform(distances, checks=False), method="complete")
        labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    clusters = []
    for label in np.unique(labels):
        members = np.sort(block.sample_indices[labels == label])
        if len(members) >= 2:
            clusters.append(members)
    return sorted(clusters, key=lambda members: (int(members[0]), len(members)))


def create_assignments(
    split: str,
    split_df: pd.DataFrame,
    blocks: list[SimilarityBlock],
    threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_counter = 0
    metadata = split_df.set_index("sample_idx", drop=False)
    for block in blocks:
        for members in cluster_block(block, threshold):
            group_id = f"{split}_g{group_counter:05d}"
            group_counter += 1
            for sample_idx in members:
                row = metadata.loc[int(sample_idx)]
                signature = spectrum_signature(str(row.mzs), str(row.intensities))
                records.append(
                    {
                        "split": split,
                        "group_id": group_id,
                        "sample_idx": int(sample_idx),
                        "adduct": str(row.adduct),
                        "precursor_mz": float(row.precursor_mz),
                        "retention_time": float(row.retention_time),
                        "collision_energy": (
                            float(row.collision_energy)
                            if np.isfinite(row.collision_energy)
                            else np.nan
                        ),
                        "spectrum_signature": signature,
                        "selected_similarity_threshold": float(threshold),
                    }
                )
    assignments = pd.DataFrame.from_records(records)
    if not assignments.empty:
        assignments["member_order"] = (
            assignments.sort_values(
                ["group_id", "collision_energy", "spectrum_signature"],
                na_position="last",
            )
            .groupby("group_id")
            .cumcount()
            .add(1)
            .sort_index()
            .astype(int)
        )
    return assignments


def grouping_audit(
    assignments: pd.DataFrame,
    hidden: pd.DataFrame,
    split_size: int,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = assignments.merge(
        hidden[["sample_idx", HIDDEN_LABEL_COLUMN]], on="sample_idx", how="left", validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    for group_id, frame in labels.groupby("group_id", sort=True):
        counts = frame[HIDDEN_LABEL_COLUMN].value_counts()
        majority_count = int(counts.max())
        majority = sorted(str(label) for label in counts[counts == majority_count].index)[0]
        rows.append(
            {
                "group_id": group_id,
                "group_size": int(len(frame)),
                "n_hidden_molecules": int(len(counts)),
                "majority_hidden_smiles": majority,
                "majority_count": majority_count,
                "purity": float(majority_count / len(frame)),
                "precursor_mz": float(frame["precursor_mz"].median()),
                "adduct": str(frame["adduct"].iloc[0]),
                "retention_time": float(frame["retention_time"].median()),
                "n_collision_energies": int(frame["collision_energy"].nunique(dropna=True)),
            }
        )
    group_table = pd.DataFrame(rows)
    n_grouped = int(len(assignments))
    majority_total = int(group_table["majority_count"].sum()) if len(group_table) else 0
    summary = {
        "similarity_threshold": float(threshold),
        "n_groups": int(len(group_table)),
        "n_grouped_spectra": n_grouped,
        "split_size": int(split_size),
        "coverage": float(n_grouped / split_size) if split_size else 0.0,
        "spectrum_weighted_purity": float(majority_total / n_grouped) if n_grouped else 0.0,
        "mean_group_purity": float(group_table["purity"].mean()) if len(group_table) else 0.0,
        "fraction_pure_groups": float((group_table["purity"] == 1.0).mean()) if len(group_table) else 0.0,
        "median_group_size": float(group_table["group_size"].median()) if len(group_table) else 0.0,
        "maximum_group_size": int(group_table["group_size"].max()) if len(group_table) else 0,
    }
    return summary, group_table


def select_threshold(grid: pd.DataFrame, cfg: dict[str, Any]) -> tuple[float, str]:
    rule = cfg["threshold_selection_rule"]
    feasible = grid[
        (grid["spectrum_weighted_purity"] >= float(rule["minimum_validation_spectrum_weighted_purity"]))
        & (grid["coverage"] >= float(rule["minimum_validation_coverage"]))
        & (grid["n_groups"] >= int(rule["minimum_validation_groups"]))
    ].copy()
    if feasible.empty:
        chosen = grid.sort_values(
            ["spectrum_weighted_purity", "coverage", "similarity_threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        return float(chosen["similarity_threshold"]), "no_threshold_met_validation_gate"
    chosen = feasible.sort_values(
        ["coverage", "similarity_threshold"], ascending=[False, True]
    ).iloc[0]
    return float(chosen["similarity_threshold"]), "validation_gate_met"


def candidate_order(accumulator: dict[str, Any]) -> list[str]:
    candidates = accumulator["borda"]
    hard = accumulator["hard"]
    return sorted(candidates, key=lambda item: (-hard[item], -candidates[item], item))


def evaluate_model(
    model: str,
    retrieval_path: Path,
    assignments: pd.DataFrame,
    hidden: pd.DataFrame,
    group_audit_table: pd.DataFrame,
    candidate_depth: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_map = assignments.set_index("sample_idx")["group_id"].to_dict()
    metadata = assignments.set_index("sample_idx").to_dict("index")
    hidden_map = hidden.set_index("sample_idx")[HIDDEN_LABEL_COLUMN].to_dict()
    group_truth = group_audit_table.set_index("group_id")["majority_hidden_smiles"].to_dict()
    accumulators: dict[str, dict[str, Any]] = {
        group_id: {
            "hard": collections.Counter(),
            "borda": collections.defaultdict(float),
            "members": [],
            "n_candidates": 0,
        }
        for group_id in assignments["group_id"].unique()
    }
    with retrieval_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sample_idx = int(row["sample_idx"])
            group_id = assignment_map.get(sample_idx)
            if group_id is None:
                continue
            accumulator = accumulators[group_id]
            top_candidates = [str(item) for item in row["top_candidates"][:candidate_depth]]
            top1 = str(row["top1_smiles"])
            accumulator["hard"][top1] += 1
            depth = max(len(top_candidates), 1)
            for rank_idx, candidate in enumerate(top_candidates, start=1):
                accumulator["borda"][candidate] += (depth - rank_idx + 1) / depth
            accumulator["n_candidates"] = max(
                accumulator["n_candidates"], int(row["n_candidates"])
            )
            accumulator["members"].append(
                {
                    "sample_idx": sample_idx,
                    "single_rank": int(row["rank"]),
                    "single_top1_smiles": top1,
                    "top_candidates": top_candidates,
                    "n_candidates": int(row["n_candidates"]),
                    "collision_energy": metadata[sample_idx]["collision_energy"],
                    "spectrum_signature": metadata[sample_idx]["spectrum_signature"],
                    "member_order": int(metadata[sample_idx]["member_order"]),
                }
            )

    query_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    group_info = group_audit_table.set_index("group_id").to_dict("index")
    for group_id in sorted(accumulators):
        accumulator = accumulators[group_id]
        members = sorted(
            accumulator["members"],
            key=lambda row: (
                math.inf if pd.isna(row["collision_energy"]) else row["collision_energy"],
                row["spectrum_signature"],
            ),
        )
        order = candidate_order(accumulator)
        rank_map = {candidate: rank + 1 for rank, candidate in enumerate(order)}
        prediction = order[0] if order else ""
        fallback_rank = int(accumulator["n_candidates"] + 1)
        single_correct_count = 0
        consensus_correct_count = 0
        single_rr_sum = 0.0
        consensus_rr_sum = 0.0
        for member in members:
            sample_idx = int(member["sample_idx"])
            target = str(hidden_map[sample_idx])
            single_rank = int(member["single_rank"])
            consensus_rank = int(rank_map.get(target, fallback_rank))
            single_correct = int(single_rank == 1)
            consensus_correct = int(prediction == target)
            single_correct_count += single_correct
            consensus_correct_count += consensus_correct
            single_rr_sum += 1.0 / single_rank
            consensus_rr_sum += 1.0 / consensus_rank
            query_rows.append(
                {
                    "model": model,
                    "group_id": group_id,
                    "sample_idx": sample_idx,
                    "hidden_target_smiles": target,
                    "single_rank": single_rank,
                    "single_top1_correct": single_correct,
                    "consensus_rank": consensus_rank,
                    "consensus_top1_correct": consensus_correct,
                    "consensus_top1_smiles": prediction,
                    "n_candidates": int(member["n_candidates"]),
                    "member_order": int(member["member_order"]),
                    "collision_energy": member["collision_energy"],
                }
            )
        majority_target = str(group_truth[group_id])
        majority_rank = int(rank_map.get(majority_target, fallback_rank))
        info = group_info[group_id]
        group_rows.append(
            {
                "model": model,
                "group_id": group_id,
                "group_size": int(len(members)),
                "purity": float(info["purity"]),
                "n_hidden_molecules": int(info["n_hidden_molecules"]),
                "majority_hidden_smiles": majority_target,
                "consensus_top1_smiles": prediction,
                "single_correct_count": int(single_correct_count),
                "consensus_correct_count": int(consensus_correct_count),
                "single_rr_sum": float(single_rr_sum),
                "consensus_rr_sum": float(consensus_rr_sum),
                "single_top1": float(single_correct_count / len(members)),
                "consensus_top1": float(consensus_correct_count / len(members)),
                "single_mrr": float(single_rr_sum / len(members)),
                "consensus_mrr": float(consensus_rr_sum / len(members)),
                "majority_target_rank": majority_rank,
                "majority_target_top1_correct": int(prediction == majority_target),
                "n_ranked_consensus_candidates": int(len(order)),
                "n_candidates": int(accumulator["n_candidates"]),
            }
        )
        for candidate_rank, candidate in enumerate(order, start=1):
            ledger_rows.append(
                {
                    "model": model,
                    "group_id": group_id,
                    "candidate_rank": int(candidate_rank),
                    "candidate_smiles": candidate,
                    "top1_votes": int(accumulator["hard"][candidate]),
                    "borda_support": float(accumulator["borda"][candidate]),
                    "is_majority_hidden_target": int(candidate == majority_target),
                    "is_consensus_top1": int(candidate_rank == 1),
                }
            )

        prefix_accumulator = {
            "hard": collections.Counter(),
            "borda": collections.defaultdict(float),
        }
        for prefix, member in enumerate(members, start=1):
            top_candidates = member["top_candidates"]
            top1 = member["single_top1_smiles"]
            prefix_accumulator["hard"][top1] += 1
            depth = max(len(top_candidates), 1)
            for rank_idx, candidate in enumerate(top_candidates, start=1):
                prefix_accumulator["borda"][candidate] += (depth - rank_idx + 1) / depth
            prefix_order = candidate_order(prefix_accumulator)
            prefix_map = {candidate: rank + 1 for rank, candidate in enumerate(prefix_order)}
            target_rank = int(prefix_map.get(majority_target, fallback_rank))
            trajectory_rows.append(
                {
                    "model": model,
                    "group_id": group_id,
                    "prefix_size": int(prefix),
                    "added_sample_idx": int(member["sample_idx"]),
                    "added_collision_energy": member["collision_energy"],
                    "added_top1_smiles": top1,
                    "majority_hidden_target_smiles": majority_target,
                    "target_rank": target_rank,
                    "consensus_top1_smiles": prefix_order[0] if prefix_order else "",
                    "target_at_rank1": int(target_rank == 1),
                }
            )
    return (
        pd.DataFrame(query_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(ledger_rows),
        pd.DataFrame(trajectory_rows),
    )


def aggregate_performance(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, frame in group_metrics.groupby("model", sort=False):
        n_spectra = int(frame["group_size"].sum())
        rows.append(
            {
                "model": model,
                "stratum": "all_grouped",
                "n_groups": int(len(frame)),
                "n_spectra": n_spectra,
                "single_top1": float(frame["single_correct_count"].sum() / n_spectra),
                "consensus_top1": float(frame["consensus_correct_count"].sum() / n_spectra),
                "top1_delta": float(
                    (frame["consensus_correct_count"].sum() - frame["single_correct_count"].sum())
                    / n_spectra
                ),
                "single_mrr": float(frame["single_rr_sum"].sum() / n_spectra),
                "consensus_mrr": float(frame["consensus_rr_sum"].sum() / n_spectra),
                "mrr_delta": float(
                    (frame["consensus_rr_sum"].sum() - frame["single_rr_sum"].sum())
                    / n_spectra
                ),
                "group_majority_top1": float(frame["majority_target_top1_correct"].mean()),
                "group_majority_mrr": float((1.0 / frame["majority_target_rank"]).mean()),
            }
        )
        strata = {
            "size_2": frame["group_size"] == 2,
            "size_3": frame["group_size"] == 3,
            "size_4": frame["group_size"] == 4,
            "size_5_plus": frame["group_size"] >= 5,
            "pure_groups": frame["purity"] == 1.0,
            "impure_groups": frame["purity"] < 1.0,
        }
        for label, mask in strata.items():
            subset = frame[mask]
            if subset.empty:
                continue
            n_subset = int(subset["group_size"].sum())
            rows.append(
                {
                    "model": model,
                    "stratum": label,
                    "n_groups": int(len(subset)),
                    "n_spectra": n_subset,
                    "single_top1": float(subset["single_correct_count"].sum() / n_subset),
                    "consensus_top1": float(subset["consensus_correct_count"].sum() / n_subset),
                    "top1_delta": float(
                        (subset["consensus_correct_count"].sum() - subset["single_correct_count"].sum())
                        / n_subset
                    ),
                    "single_mrr": float(subset["single_rr_sum"].sum() / n_subset),
                    "consensus_mrr": float(subset["consensus_rr_sum"].sum() / n_subset),
                    "mrr_delta": float(
                        (subset["consensus_rr_sum"].sum() - subset["single_rr_sum"].sum())
                        / n_subset
                    ),
                    "group_majority_top1": float(subset["majority_target_top1_correct"].mean()),
                    "group_majority_mrr": float((1.0 / subset["majority_target_rank"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_group_deltas(group_metrics: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    stats_cfg = cfg["statistics"]
    n_boot = int(stats_cfg["bootstrap_replicates"])
    seed = int(stats_cfg["bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    for model_idx, (model, frame) in enumerate(group_metrics.groupby("model", sort=False)):
        size = frame["group_size"].to_numpy(dtype=np.float64)
        top_delta = (
            frame["consensus_correct_count"].to_numpy(dtype=np.float64)
            - frame["single_correct_count"].to_numpy(dtype=np.float64)
        )
        rr_delta = (
            frame["consensus_rr_sum"].to_numpy(dtype=np.float64)
            - frame["single_rr_sum"].to_numpy(dtype=np.float64)
        )
        observed = {"top1_delta": top_delta.sum() / size.sum(), "mrr_delta": rr_delta.sum() / size.sum()}
        rng = np.random.default_rng(seed + model_idx)
        draws = {"top1_delta": np.empty(n_boot), "mrr_delta": np.empty(n_boot)}
        for boot_idx in range(n_boot):
            sampled = rng.integers(0, len(frame), size=len(frame))
            denominator = size[sampled].sum()
            draws["top1_delta"][boot_idx] = top_delta[sampled].sum() / denominator
            draws["mrr_delta"][boot_idx] = rr_delta[sampled].sum() / denominator
        for metric in ["top1_delta", "mrr_delta"]:
            values = draws[metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": float(observed[metric]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "p_two_sided": float(
                        min(1.0, 2.0 * min((np.sum(values <= 0) + 1) / (n_boot + 1), (np.sum(values >= 0) + 1) / (n_boot + 1)))
                    ),
                    "n_groups": int(len(frame)),
                    "n_bootstrap": n_boot,
                    "bootstrap_unit": "group",
                }
            )
    return pd.DataFrame(rows)


def oracle_context(oracle_path: Path) -> pd.DataFrame:
    with oracle_path.open() as handle:
        summary = json.load(handle)
    mapping = {"rt_only_d11": "UltraMS", "dreams": "DreaMS"}
    rows = []
    for raw_model, model in mapping.items():
        block = summary["models"][raw_model]["test"]
        oracle = block["oracle"]["hard"]
        rows.append(
            {
                "model": model,
                "oracle_grouping": "canonical_SMILES_upper_bound",
                "n_oracle_groups": int(oracle["n_clusters"]),
                "n_oracle_spectra": int(oracle["n_spectra"]),
                "single_all_top1": float(block["single_all"]["top1"] / 100.0),
                "oracle_cluster_macro_top1": float(oracle["cluster_majority"]["top1"] / 100.0),
                "oracle_spectrum_weighted_top1": float(oracle["spectrum_weighted"]["top1"] / 100.0),
                "oracle_cluster_macro_mrr": float(oracle["cluster_majority"]["mrr"]),
                "oracle_spectrum_weighted_mrr": float(oracle["spectrum_weighted"]["mrr"]),
            }
        )
    return pd.DataFrame(rows)


def choose_example(
    group_table: pd.DataFrame, rank_trajectories: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[str, str]:
    example_cfg = cfg["example_selection"]
    eligible = group_table[
        (group_table["purity"] == 1.0)
        & (group_table["group_size"] >= int(example_cfg["minimum_group_size"]))
        & (group_table["n_collision_energies"] >= int(example_cfg["minimum_distinct_collision_energies"]))
    ].copy()
    first = (
        rank_trajectories.sort_values("prefix_size")
        .groupby(["model", "group_id"], as_index=False)
        .first()[["model", "group_id", "target_rank"]]
        .rename(columns={"target_rank": "first_rank"})
    )
    final = (
        rank_trajectories.sort_values("prefix_size")
        .groupby(["model", "group_id"], as_index=False)
        .last()[["model", "group_id", "target_rank"]]
        .rename(columns={"target_rank": "final_rank"})
    )
    trajectory_summary = first.merge(final, on=["model", "group_id"], validate="one_to_one")
    pivot = trajectory_summary.pivot(
        index="group_id", columns="model", values=["first_rank", "final_rank"]
    )
    pivot.columns = ["_".join(column) for column in pivot.columns]
    eligible = eligible.merge(pivot.reset_index(), on="group_id", how="left", validate="one_to_one")
    shared_rescue = eligible[
        (eligible["first_rank_UltraMS"] > 1)
        & (eligible["first_rank_DreaMS"] > 1)
        & (eligible["final_rank_UltraMS"] == 1)
        & (eligible["final_rank_DreaMS"] == 1)
    ].copy()
    if not shared_rescue.empty:
        maximum_size = int(shared_rescue["group_size"].max())
        eligible = shared_rescue[shared_rescue["group_size"] == maximum_size].copy()
        rule = "illustrative_shared_rescue_not_used_for_inference"
    else:
        rule = "fallback_model_independent_pure_group"
    if eligible.empty:
        eligible = group_table[
            (group_table["purity"] == 1.0) & (group_table["group_size"] >= 5)
        ].copy()
        rule = "fallback_pure_size_at_least_5"
    if eligible.empty:
        eligible = group_table.sort_values(["purity", "group_size"], ascending=[False, False]).head(50).copy()
        rule = "fallback_highest_purity_then_size"
    median_mass = float(eligible["precursor_mz"].median())
    eligible["mass_distance"] = (eligible["precursor_mz"] - median_mass).abs()
    chosen = eligible.sort_values(["mass_distance", "group_id"]).iloc[0]
    return str(chosen["group_id"]), rule


def save_example_peaks(
    example_group: str, assignments: pd.DataFrame, grouping: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    selected = assignments[assignments["group_id"] == example_group].sort_values("member_order")
    source = grouping.set_index("sample_idx")
    rows = []
    top_n = int(cfg["grouping"]["spectrum_top_n_peaks"])
    for member in selected.itertuples(index=False):
        spectrum = source.loc[int(member.sample_idx)]
        mzs = np.fromstring(str(spectrum.mzs), sep=",")
        intensities = np.fromstring(str(spectrum.intensities), sep=",")
        n = min(len(mzs), len(intensities))
        mzs, intensities = mzs[:n], intensities[:n]
        if len(intensities) > top_n:
            keep = np.argpartition(intensities, -top_n)[-top_n:]
            mzs, intensities = mzs[keep], intensities[keep]
        order = np.argsort(mzs)
        scale = np.max(intensities) if len(intensities) else 1.0
        for peak_idx in order:
            rows.append(
                {
                    "group_id": example_group,
                    "sample_idx": int(member.sample_idx),
                    "member_order": int(member.member_order),
                    "collision_energy": member.collision_energy,
                    "mz": float(mzs[peak_idx]),
                    "relative_intensity": float(intensities[peak_idx] / scale) if scale else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_results(
    status: str,
    selected_threshold: float,
    val_summary: dict[str, Any],
    test_summary: dict[str, Any],
    performance: pd.DataFrame,
    bootstrap: pd.DataFrame,
    example_group: str,
    example_rule: str,
) -> None:
    lines = [
        "# S9 non-oracle consensus",
        "",
        f"Decision: `{status}`",
        "",
        "## Grouping",
        "",
        f"- Validation-selected raw-spectrum cosine threshold: `{selected_threshold:.2f}`.",
        f"- Validation coverage: `{val_summary['coverage']:.4f}`; weighted purity: `{val_summary['spectrum_weighted_purity']:.4f}`.",
        f"- Test coverage: `{test_summary['coverage']:.4f}`; weighted purity: `{test_summary['spectrum_weighted_purity']:.4f}`.",
        f"- Test groups: `{test_summary['n_groups']:,}`; grouped spectra: `{test_summary['n_grouped_spectra']:,}`.",
        "- Standardized dataset SMILES is used only after group assignments are frozen, for purity and retrieval evaluation.",
        "",
        "## Retrieval",
        "",
    ]
    for row in performance[performance["stratum"] == "all_grouped"].itertuples(index=False):
        ci = bootstrap[(bootstrap["model"] == row.model) & (bootstrap["metric"] == "top1_delta")].iloc[0]
        lines.append(
            f"- {row.model}: single Top-1 `{row.single_top1:.4f}` -> consensus `{row.consensus_top1:.4f}`; "
            f"delta `{100 * row.top1_delta:+.2f}` pp (95% CI `{100 * ci.ci_low:+.2f}` to `{100 * ci.ci_high:+.2f}` pp)."
        )
    lines.extend(
        [
            "",
            "## Display case",
            "",
            f"- Group: `{example_group}`.",
            f"- Selection rule: `{example_rule}`; model outcomes were not used.",
            "",
            "## Interpretation",
            "",
            "This analysis tests label-blind grouping within MSnLib.",
            "",
        ]
    )
    (PACKAGE / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    global PACKAGE
    args = parse_args()
    if args.output_dir is not None:
        PACKAGE = args.output_dir.resolve()
    cfg = load_config(args.config)
    project_root = args.project_root or Path(cfg["project_root"])
    inputs = cfg["inputs"]
    dataset_path = project_root / inputs["dataset"]
    oracle_path = project_root / inputs["oracle_summary"]
    retrieval_paths = {
        model: project_root / relative for model, relative in inputs["retrieval_test"].items()
    }
    for path in [dataset_path, oracle_path, *retrieval_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.time()
    grouping, hidden = prepare_dataset(dataset_path)
    val_df = grouping[grouping["fold"] == cfg["splits"]["threshold_selection"]].copy()
    test_df = grouping[grouping["fold"] == cfg["splits"]["locked_evaluation"]].copy()

    val_blocks = build_similarity_blocks(val_df, cfg)
    threshold_rows = []
    val_assignments_by_threshold: dict[float, pd.DataFrame] = {}
    val_group_tables: dict[float, pd.DataFrame] = {}
    for threshold in cfg["grouping"]["similarity_threshold_grid"]:
        threshold = float(threshold)
        assignments = create_assignments("val", val_df, val_blocks, threshold)
        summary, group_table = grouping_audit(assignments, hidden, len(val_df), threshold)
        threshold_rows.append(summary)
        val_assignments_by_threshold[threshold] = assignments
        val_group_tables[threshold] = group_table
    threshold_table = pd.DataFrame(threshold_rows).sort_values("similarity_threshold")
    selected_threshold, threshold_status = select_threshold(threshold_table, cfg)
    threshold_table["selected"] = (threshold_table["similarity_threshold"] == selected_threshold).astype(int)
    threshold_table["selection_status"] = threshold_status
    val_assignments = val_assignments_by_threshold[selected_threshold]
    val_group_table = val_group_tables[selected_threshold]
    val_summary = threshold_table[threshold_table["selected"] == 1].iloc[0].to_dict()

    test_blocks = build_similarity_blocks(test_df, cfg)
    test_assignments = create_assignments("test", test_df, test_blocks, selected_threshold)
    test_summary, test_group_table = grouping_audit(
        test_assignments, hidden, len(test_df), selected_threshold
    )

    source_dir = PACKAGE / "source_data"
    statistics_dir = PACKAGE / "statistics"
    audit_dir = PACKAGE / "audit"
    for directory in [source_dir, statistics_dir, audit_dir, PACKAGE / "outputs"]:
        directory.mkdir(parents=True, exist_ok=True)

    val_assignments.to_csv(source_dir / "group_assignments_val.csv.gz", index=False)
    test_assignments.to_csv(source_dir / "group_assignments_test.csv.gz", index=False)
    hidden[hidden["fold"].isin(["val", "test"])].to_csv(
        source_dir / "hidden_label_audit.csv.gz", index=False
    )
    val_group_table.to_csv(source_dir / "group_purity_val.csv.gz", index=False)
    test_group_table.to_csv(source_dir / "group_purity_test.csv.gz", index=False)
    threshold_table.to_csv(statistics_dir / "threshold_selection_val.csv", index=False)

    query_tables, group_tables, ledger_tables, trajectory_tables = [], [], [], []
    for model, retrieval_path in retrieval_paths.items():
        query, groups, ledger, trajectory = evaluate_model(
            model,
            retrieval_path,
            test_assignments,
            hidden,
            test_group_table,
            int(cfg["consensus"]["per_spectrum_candidate_depth"]),
        )
        query_tables.append(query)
        group_tables.append(groups)
        ledger_tables.append(ledger)
        trajectory_tables.append(trajectory)
    query_metrics = pd.concat(query_tables, ignore_index=True)
    group_metrics = pd.concat(group_tables, ignore_index=True)
    vote_ledger = pd.concat(ledger_tables, ignore_index=True)
    rank_trajectories = pd.concat(trajectory_tables, ignore_index=True)
    query_metrics.to_csv(source_dir / "per_query_metrics_test.csv.gz", index=False)
    group_metrics.to_csv(source_dir / "per_group_metrics_test.csv.gz", index=False)
    vote_ledger.to_csv(source_dir / "group_candidate_vote_ledger_test.csv.gz", index=False)
    rank_trajectories.to_csv(source_dir / "rank_trajectories_test.csv.gz", index=False)

    performance = aggregate_performance(group_metrics)
    bootstrap = bootstrap_group_deltas(group_metrics, cfg)
    oracle = oracle_context(oracle_path)
    performance.to_csv(statistics_dir / "model_performance.csv", index=False)
    bootstrap.to_csv(statistics_dir / "paired_group_bootstrap.csv", index=False)
    oracle.to_csv(statistics_dir / "oracle_upper_bound_context.csv", index=False)
    pd.DataFrame([{"split": "val", **val_summary}, {"split": "test", **test_summary}]).to_csv(
        statistics_dir / "grouping_summary.csv", index=False
    )

    example_group, example_rule = choose_example(test_group_table, rank_trajectories, cfg)
    example_peaks = save_example_peaks(example_group, test_assignments, test_df, cfg)
    example_peaks.to_csv(source_dir / "example_spectrum_peaks.csv.gz", index=False)
    example_manifest = test_assignments[test_assignments["group_id"] == example_group].copy()
    example_manifest["selection_rule"] = example_rule
    example_manifest.to_csv(source_dir / "example_group_manifest.csv", index=False)

    gate = cfg["promotion_gate"]
    all_performance = performance[performance["stratum"] == "all_grouped"].set_index("model")
    top1_boot = bootstrap[bootstrap["metric"] == "top1_delta"].set_index("model")
    gate_checks = {
        "test_purity": test_summary["spectrum_weighted_purity"]
        >= float(gate["minimum_test_spectrum_weighted_purity"]),
        "test_coverage": test_summary["coverage"] >= float(gate["minimum_test_coverage"]),
        "test_group_count": test_summary["n_groups"] >= int(gate["minimum_test_groups"]),
        "positive_top1_gain_both_models": bool((all_performance["top1_delta"] > 0).all()),
        "positive_top1_ci_both_models": bool((top1_boot["ci_low"] > 0).all()),
        "same_group_count_both_models": bool(
            group_metrics.groupby("model")["group_id"].nunique().nunique() == 1
        ),
    }
    status = (
        "promote_as_positive_supplement"
        if all(gate_checks.values()) and threshold_status == "validation_gate_met"
        else "quantitative_rejection_do_not_promote"
    )

    manifest_rows = []
    for role, path in [
        ("dataset", dataset_path),
        ("oracle_summary", oracle_path),
        *[(f"retrieval_{model}", path) for model, path in retrieval_paths.items()],
    ]:
        manifest_rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(audit_dir / "upstream_input_manifest.csv", index=False)
    audit = {
        "analysis_id": cfg["analysis_id"],
        "status": status,
        "threshold_selection_status": threshold_status,
        "selected_similarity_threshold": selected_threshold,
        "grouping_columns": GROUPING_COLUMNS,
        "forbidden_label_columns_absent_from_group_assignments": not any(
            column in val_assignments.columns or column in test_assignments.columns
            for column in cfg["grouping"]["label_columns_forbidden"]
        ),
        "val_summary": val_summary,
        "test_summary": test_summary,
        "gate_checks": gate_checks,
        "example_group": example_group,
        "example_selection_rule": example_rule,
        "example_used_for_inference_or_threshold_selection": False,
        "runtime_seconds": time.time() - started,
        "python": sys.version,
    }
    (audit_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2))
    write_results(
        status,
        selected_threshold,
        val_summary,
        test_summary,
        performance,
        bootstrap,
        example_group,
        example_rule,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
