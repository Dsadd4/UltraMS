#!/usr/bin/env python3
"""Evaluate new retrieval baselines on the frozen Figure 3b groups.

This script never rebuilds groups.  It replays the published top-50
plurality+Borda consensus, first regresses the two legacy methods against the
frozen evidence package, then evaluates all five seeds of each learned
baseline.  Model-seed variation and group-cluster uncertainty are kept
separate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = {
    "UltraMS": 20260722,
    "DreaMS": 20260723,
    "Fingerprint FFN": 20260724,
    "Spec2Vec": 20260725,
    "Linear": 20260726,
    "Binned FFN": 20260727,
    "DeepSets": 20260728,
    "DeepSets + Fourier": 20260729,
    "Linear FP": 20260730,
    "Fourier projection": 20260731,
    "UltraMS codebook": 20260801,
}
PRIMARY_MODEL_SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def parse_input(text: str) -> tuple[str, int | None, Path]:
    parts = text.split("|", 2)
    if len(parts) != 3:
        raise ValueError("--input must be METHOD|SEED_OR_NONE|JSONL")
    method, seed_text, path = parts
    return method, None if seed_text.lower() == "none" else int(seed_text), Path(path)


def candidate_order(hard: collections.Counter[str], borda: dict[str, float]) -> list[str]:
    return sorted(borda, key=lambda candidate: (-hard[candidate], -borda[candidate], candidate))


def evaluate_one(
    method: str,
    seed: int | None,
    retrieval: Path,
    assignments: pd.DataFrame,
    hidden: pd.DataFrame,
    purity: pd.DataFrame,
    candidate_depth: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_map = assignments.set_index("sample_idx")["group_id"].to_dict()
    metadata = assignments.set_index("sample_idx").to_dict("index")
    hidden_map = hidden.set_index("sample_idx")["hidden_smiles"].astype(str).to_dict()
    truth = purity.set_index("group_id")["majority_hidden_smiles"].astype(str).to_dict()
    accumulators = {
        group: {
            "hard": collections.Counter(),
            "borda": collections.defaultdict(float),
            "members": [],
            "n_candidates": 0,
        }
        for group in assignments["group_id"].unique()
    }
    seen: set[int] = set()
    with retrieval.open() as handle:
        for line in handle:
            row = json.loads(line)
            sample_idx = int(row["sample_idx"])
            group = assignment_map.get(sample_idx)
            if group is None:
                continue
            if sample_idx in seen:
                raise AssertionError(f"duplicate grouped query {sample_idx} in {retrieval}")
            seen.add(sample_idx)
            candidates = [str(value) for value in row["top_candidates"][:candidate_depth]]
            top1 = str(row["top1_smiles"])
            acc = accumulators[group]
            acc["hard"][top1] += 1
            depth = max(len(candidates), 1)
            for rank_index, candidate in enumerate(candidates, start=1):
                acc["borda"][candidate] += (depth - rank_index + 1) / depth
            acc["n_candidates"] = max(acc["n_candidates"], int(row["n_candidates"]))
            acc["members"].append(
                {
                    "sample_idx": sample_idx,
                    "single_rank": int(row["rank"]),
                    "single_top1_smiles": top1,
                    "n_candidates": int(row["n_candidates"]),
                }
            )
    expected = set(int(value) for value in assignments["sample_idx"])
    if seen != expected:
        raise AssertionError(f"grouped query coverage mismatch: missing={len(expected-seen)} extra={len(seen-expected)}")

    purity_info = purity.set_index("group_id").to_dict("index")
    query_rows, group_rows = [], []
    for group in sorted(accumulators):
        acc = accumulators[group]
        order = candidate_order(acc["hard"], acc["borda"])
        rank_map = {candidate: rank + 1 for rank, candidate in enumerate(order)}
        prediction = order[0] if order else ""
        fallback_rank = int(acc["n_candidates"] + 1)
        single_correct = consensus_correct = 0
        single_rr = consensus_rr = 0.0
        for member in acc["members"]:
            sample_idx = member["sample_idx"]
            target = hidden_map[sample_idx]
            single_rank = member["single_rank"]
            consensus_rank = int(rank_map.get(target, fallback_rank))
            single_correct += int(single_rank == 1)
            consensus_correct += int(prediction == target)
            single_rr += 1.0 / single_rank
            consensus_rr += 1.0 / consensus_rank
            query_rows.append(
                {
                    "method": method,
                    "seed": np.nan if seed is None else seed,
                    "group_id": group,
                    "sample_idx": sample_idx,
                    "hidden_target_smiles": target,
                    "single_rank": single_rank,
                    "single_top1_correct": int(single_rank == 1),
                    "consensus_rank": consensus_rank,
                    "consensus_top1_correct": int(prediction == target),
                    "consensus_top1_smiles": prediction,
                    "n_candidates": member["n_candidates"],
                }
            )
        majority = truth[group]
        info = purity_info[group]
        group_rows.append(
            {
                "method": method,
                "seed": np.nan if seed is None else seed,
                "group_id": group,
                "group_size": len(acc["members"]),
                "purity": float(info["purity"]),
                "n_hidden_molecules": int(info["n_hidden_molecules"]),
                "majority_hidden_smiles": majority,
                "consensus_top1_smiles": prediction,
                "single_correct_count": single_correct,
                "consensus_correct_count": consensus_correct,
                "single_rr_sum": single_rr,
                "consensus_rr_sum": consensus_rr,
                "single_top1": single_correct / len(acc["members"]),
                "consensus_top1": consensus_correct / len(acc["members"]),
                "single_mrr": single_rr / len(acc["members"]),
                "consensus_mrr": consensus_rr / len(acc["members"]),
                "majority_target_rank": int(rank_map.get(majority, fallback_rank)),
                "majority_target_top1_correct": int(prediction == majority),
                "n_ranked_consensus_candidates": len(order),
                "n_candidates": int(acc["n_candidates"]),
            }
        )
    return pd.DataFrame(query_rows), pd.DataFrame(group_rows)


def performance(frame: pd.DataFrame) -> dict[str, float | int]:
    n = int(frame["group_size"].sum())
    return {
        "n_groups": len(frame),
        "n_spectra": n,
        "single_top1": float(frame["single_correct_count"].sum() / n),
        "consensus_top1": float(frame["consensus_correct_count"].sum() / n),
        "top1_delta": float((frame["consensus_correct_count"].sum() - frame["single_correct_count"].sum()) / n),
        "single_mrr": float(frame["single_rr_sum"].sum() / n),
        "consensus_mrr": float(frame["consensus_rr_sum"].sum() / n),
        "mrr_delta": float((frame["consensus_rr_sum"].sum() - frame["single_rr_sum"].sum()) / n),
        "group_majority_top1": float(frame["majority_target_top1_correct"].mean()),
        "group_majority_mrr": float(np.mean(1.0 / frame["majority_target_rank"])),
    }


def bootstrap(frame: pd.DataFrame, *, seed: int, n_boot: int) -> list[dict[str, float | int | str]]:
    size = frame["group_size"].to_numpy(dtype=np.float64)
    values = {
        "top1_delta": (frame["consensus_correct_count"] - frame["single_correct_count"]).to_numpy(dtype=np.float64),
        "mrr_delta": (frame["consensus_rr_sum"] - frame["single_rr_sum"]).to_numpy(dtype=np.float64),
    }
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(n_boot) for metric in values}
    for index in range(n_boot):
        sampled = rng.integers(0, len(frame), size=len(frame))
        denominator = size[sampled].sum()
        for metric, numerators in values.items():
            draws[metric][index] = numerators[sampled].sum() / denominator
    rows = []
    for metric, distribution in draws.items():
        observed = values[metric].sum() / size.sum()
        rows.append(
            {
                "metric": metric,
                "estimate": observed,
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "p_two_sided": float(min(1.0, 2.0 * min((np.sum(distribution <= 0) + 1) / (n_boot + 1), (np.sum(distribution >= 0) + 1) / (n_boot + 1)))),
                "n_groups": len(frame),
                "n_bootstrap": n_boot,
                "bootstrap_seed": seed,
                "bootstrap_unit": "group",
            }
        )
    return rows


def assert_legacy_regression(package: Path, groups: pd.DataFrame, performance_rows: pd.DataFrame, bootstrap_rows: pd.DataFrame) -> dict[str, object]:
    frozen_groups = pd.read_csv(package / "source_data/per_group_metrics_test.csv.gz")
    frozen_perf = pd.read_csv(package / "statistics/model_performance.csv")
    frozen_boot = pd.read_csv(package / "statistics/paired_group_bootstrap.csv")
    audit: dict[str, object] = {}
    for method in ("UltraMS", "DreaMS"):
        actual = groups[(groups.method == method) & groups.seed.isna()].copy()
        expected = frozen_groups[frozen_groups.model == method].copy()
        actual = actual.sort_values("group_id").reset_index(drop=True)
        expected = expected.sort_values("group_id").reset_index(drop=True)
        columns = [
            "group_id", "group_size", "purity", "n_hidden_molecules", "majority_hidden_smiles",
            "consensus_top1_smiles", "single_correct_count", "consensus_correct_count", "single_rr_sum",
            "consensus_rr_sum", "single_top1", "consensus_top1", "single_mrr", "consensus_mrr",
            "majority_target_rank", "majority_target_top1_correct", "n_ranked_consensus_candidates", "n_candidates",
        ]
        if len(actual) != len(expected):
            raise AssertionError((method, len(actual), len(expected)))
        for column in columns:
            left, right = actual[column], expected[column]
            if pd.api.types.is_numeric_dtype(right):
                if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), atol=1e-15, rtol=1e-14, equal_nan=True):
                    raise AssertionError(f"legacy group regression failed: {method} {column}")
            elif not np.array_equal(left.astype(str), right.astype(str)):
                raise AssertionError(f"legacy group regression failed: {method} {column}")
        actual_perf = performance_rows[
            (performance_rows.method == method)
            & (performance_rows.seed.isna() | performance_rows.seed.astype(str).eq(""))
        ].iloc[0]
        expected_perf = frozen_perf[(frozen_perf.model == method) & (frozen_perf.stratum == "all_grouped")].iloc[0]
        for column in ["n_groups", "n_spectra", "single_top1", "consensus_top1", "top1_delta", "single_mrr", "consensus_mrr", "mrr_delta", "group_majority_top1", "group_majority_mrr"]:
            if not np.isclose(float(actual_perf[column]), float(expected_perf[column]), atol=1e-15, rtol=1e-14):
                raise AssertionError(f"legacy performance regression failed: {method} {column}")
        for metric in ("top1_delta", "mrr_delta"):
            actual_boot = bootstrap_rows[(bootstrap_rows.method == method) & (bootstrap_rows.metric == metric)].iloc[0]
            expected_boot = frozen_boot[(frozen_boot.model == method) & (frozen_boot.metric == metric)].iloc[0]
            for column in ("estimate", "ci_low", "ci_high", "p_two_sided"):
                if not np.isclose(float(actual_boot[column]), float(expected_boot[column]), atol=1e-15, rtol=1e-14):
                    raise AssertionError(f"legacy bootstrap regression failed: {method} {metric} {column}")
        audit[method] = {"per_group_rows": len(actual), "regression": "exact_or_1e-14_float_tolerance"}
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-package", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--allow-incomplete-methods",
        action="store_true",
        help="Infrastructure/regression test only; formal runs must omit this flag.",
    )
    parser.add_argument(
        "--formal-primary-six",
        action="store_true",
        help="Require the six displayed methods, using one frozen/seed-0 checkpoint each.",
    )
    parser.add_argument(
        "--formal-chemberta-six",
        action="store_true",
        help="Require UltraMS, DreaMS, and the four matched ChemBERTa-readout baselines.",
    )
    parser.add_argument(
        "--formal-final-six",
        action="store_true",
        help="Require the final user-selected six methods.",
    )
    args = parser.parse_args()

    formal_modes = int(args.formal_primary_six) + int(args.formal_chemberta_six) + int(args.formal_final_six)
    if formal_modes > 1 or (formal_modes and args.allow_incomplete_methods):
        raise ValueError("formal modes and --allow-incomplete-methods are mutually exclusive")

    package = args.frozen_package
    assignments = pd.read_csv(package / "source_data/group_assignments_test.csv.gz")
    hidden = pd.read_csv(package / "source_data/hidden_label_audit.csv.gz")
    purity = pd.read_csv(package / "source_data/group_purity_test.csv.gz")
    if len(purity) != 15541 or len(assignments) != 52070:
        raise AssertionError((len(purity), len(assignments)))

    query_frames, group_frames, provenance = [], [], []
    for method, seed, path in [parse_input(value) for value in args.input]:
        if method not in BOOTSTRAP_SEED:
            raise ValueError(f"no registered bootstrap seed for {method}")
        if not path.exists():
            raise FileNotFoundError(path)
        query, groups = evaluate_one(method, seed, path, assignments, hidden, purity, args.candidate_depth)
        query_frames.append(query)
        group_frames.append(groups)
        provenance.append({"method": method, "seed": "" if seed is None else seed, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    query_metrics = pd.concat(query_frames, ignore_index=True)
    group_metrics = pd.concat(group_frames, ignore_index=True)

    if args.formal_final_six:
        methods = set(group_metrics.method)
        required = {
            "UltraMS", "DreaMS", "Linear", "DeepSets",
            "Fourier projection", "UltraMS codebook",
        }
        if methods != required:
            raise ValueError(f"unexpected formal final six-method set: {sorted(methods)}")
        for frozen in ("UltraMS", "DreaMS"):
            frame = group_metrics[group_metrics.method == frozen]
            if len(frame) != len(purity) or frame.seed.notna().any():
                raise ValueError(f"{frozen} requires one frozen result for every group")
        for learned in required.difference({"UltraMS", "DreaMS"}):
            seeds = sorted(int(value) for value in group_metrics.loc[group_metrics.method == learned, "seed"].dropna().unique())
            if seeds != [0]:
                raise ValueError(f"{learned} requires exactly primary seed 0; got {seeds}")
    elif args.formal_chemberta_six:
        methods = set(group_metrics.method)
        required = {
            "UltraMS", "DreaMS", "Linear", "Binned FFN",
            "DeepSets", "DeepSets + Fourier",
        }
        if methods != required:
            raise ValueError(f"unexpected formal ChemBERTa six-method set: {sorted(methods)}")
        for frozen in ("UltraMS", "DreaMS"):
            frame = group_metrics[group_metrics.method == frozen]
            if len(frame) != len(purity) or frame.seed.notna().any():
                raise ValueError(f"{frozen} requires one frozen result for every group")
        for learned in required.difference({"UltraMS", "DreaMS"}):
            seeds = sorted(int(value) for value in group_metrics.loc[group_metrics.method == learned, "seed"].dropna().unique())
            if seeds != [0]:
                raise ValueError(f"{learned} requires exactly primary seed 0; got {seeds}")
    elif args.formal_primary_six:
        methods = set(group_metrics.method)
        required = {
            "UltraMS", "DreaMS", "Fingerprint FFN", "Linear FP",
            "DeepSets", "DeepSets + Fourier",
        }
        if methods != required:
            raise ValueError(f"unexpected formal six-method set: {sorted(methods)}")
        for frozen in ("UltraMS", "DreaMS"):
            frame = group_metrics[group_metrics.method == frozen]
            if len(frame) != len(purity) or frame.seed.notna().any():
                raise ValueError(f"{frozen} requires one frozen result for every group")
        for learned in required.difference({"UltraMS", "DreaMS"}):
            seeds = sorted(int(value) for value in group_metrics.loc[group_metrics.method == learned, "seed"].dropna().unique())
            if seeds != [0]:
                raise ValueError(f"{learned} requires exactly primary seed 0; got {seeds}")
    elif not args.allow_incomplete_methods:
        methods = set(group_metrics.method)
        required = {"UltraMS", "DreaMS", "Fingerprint FFN", "Spec2Vec"}
        if methods != required:
            raise ValueError(f"unexpected formal method set: {sorted(methods)}")
        for frozen in ("UltraMS", "DreaMS"):
            frame = group_metrics[group_metrics.method == frozen]
            if len(frame) != len(purity) or frame.seed.notna().any():
                raise ValueError(f"{frozen} requires one frozen result for every group")
        for learned in ("Fingerprint FFN", "Spec2Vec"):
            seeds = sorted(int(value) for value in group_metrics.loc[group_metrics.method == learned, "seed"].dropna().unique())
            if seeds != [0, 1, 2, 3, 4]:
                raise ValueError(f"{learned} requires model seeds 0-4; got {seeds}")

    performance_rows = []
    for (method, seed), frame in group_metrics.groupby(["method", "seed"], sort=False, dropna=False):
        performance_rows.append({"method": method, "seed": "" if pd.isna(seed) else int(seed), **performance(frame)})
    performance_table = pd.DataFrame(performance_rows)

    method_group_frames: dict[str, pd.DataFrame] = {}
    summary_rows, bootstrap_rows = [], []
    for method, frame in group_metrics.groupby("method", sort=False):
        if frame["seed"].notna().any():
            primary = frame[frame.seed == PRIMARY_MODEL_SEED].copy()
            primary_seed: int | str = PRIMARY_MODEL_SEED
        else:
            primary = frame.copy()
            primary_seed = ""
        if len(primary) != len(purity):
            raise ValueError(f"{method} primary group rows {len(primary)} != {len(purity)}")
        method_group_frames[method] = primary
        method_seed_perf = performance_table[performance_table.method == method]
        row: dict[str, object] = {
            "method": method,
            "primary_seed": primary_seed,
            "n_model_seeds": len(method_seed_perf),
            "n_groups": len(primary),
            "n_spectra": int(primary.group_size.sum()),
        }
        primary_performance = performance(primary)
        for metric in ["single_top1", "consensus_top1", "top1_delta", "single_mrr", "consensus_mrr", "mrr_delta", "group_majority_top1", "group_majority_mrr"]:
            seed_values = method_seed_perf[metric].to_numpy(dtype=float)
            row[f"{metric}_primary"] = float(primary_performance[metric])
            row[f"{metric}_mean_across_seeds"] = float(seed_values.mean())
            row[f"{metric}_sd_across_seeds"] = float(seed_values.std(ddof=1)) if len(seed_values) > 1 else 0.0
        summary_rows.append(row)
        for boot_row in bootstrap(primary, seed=BOOTSTRAP_SEED[method], n_boot=args.bootstrap_replicates):
            bootstrap_rows.append(
                {
                    "method": method,
                    "primary_seed": primary_seed,
                    "comparison_policy": "single frozen checkpoint; learned baseline seed 0",
                    **boot_row,
                }
            )
    summary_table = pd.DataFrame(summary_rows)
    bootstrap_table = pd.DataFrame(bootstrap_rows)

    legacy_audit = assert_legacy_regression(package, group_metrics, performance_table, bootstrap_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_metrics.to_csv(args.output_dir / "per_query_metrics_all_runs.csv.gz", index=False)
    group_metrics.to_csv(args.output_dir / "per_group_metrics_all_runs.csv.gz", index=False)
    performance_table.to_csv(args.output_dir / "per_seed_performance.csv", index=False)
    summary_table.to_csv(args.output_dir / "method_summary.csv", index=False)
    bootstrap_table.to_csv(args.output_dir / "paired_group_bootstrap.csv", index=False)
    pd.DataFrame(provenance).to_csv(args.output_dir / "input_manifest.csv", index=False)
    audit = {
        "status": "complete",
        "group_source": "frozen; no regrouping",
        "n_groups": len(purity),
        "n_grouped_spectra": len(assignments),
        "candidate_depth": args.candidate_depth,
        "formal_method_completeness_enforced": (
            args.formal_primary_six or args.formal_chemberta_six or args.formal_final_six or not args.allow_incomplete_methods
        ),
        "formal_primary_six": args.formal_primary_six,
        "formal_chemberta_six": args.formal_chemberta_six,
        "formal_final_six": args.formal_final_six,
        "primary_model_seed": PRIMARY_MODEL_SEED,
        "main_result_policy": "single checkpoint per method; learned baselines use seed 0; no seed ensemble",
        "stability_model_seeds": (
            [] if args.formal_primary_six or args.formal_chemberta_six else [1, 2, 3, 4]
        ),
        "bootstrap_seed_by_method": BOOTSTRAP_SEED,
        "legacy_regression": legacy_audit,
        "frozen_inputs": {
            relative: sha256_file(package / relative)
            for relative in [
                "source_data/group_assignments_test.csv.gz",
                "source_data/group_purity_test.csv.gz",
                "source_data/hidden_label_audit.csv.gz",
                "source_data/per_group_metrics_test.csv.gz",
                "statistics/model_performance.csv",
                "statistics/paired_group_bootstrap.csv",
            ]
        },
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
