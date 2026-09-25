#!/usr/bin/env python3
"""Evaluate exact same-molecule Top-50 consensus for Figure 3a.

Unlike the historical Top-1-only replay, this evaluator constructs a real
candidate ordering from each member spectrum's frozen Top-50 list. Top-1
plurality is the primary vote and normalized Borda support breaks vote ties.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_SINGLE = "single_per_spectrum"
CONDITION_CLUSTER = "oracle_top50_consensus_cluster_majority"
CONDITION_WEIGHTED = "oracle_top50_consensus_spectrum_weighted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def parse_input(text: str) -> tuple[str, int | None, Path]:
    method, seed_text, path_text = text.split("|", 2)
    return method, None if seed_text.lower() == "none" else int(seed_text), Path(path_text)


def rank_metrics(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(ranks) == 0 or np.any(ranks < 1):
        raise ValueError("rank vector must be nonempty and positive")
    return {
        "n": len(ranks),
        "hit1": float(np.mean(ranks <= 1)),
        "hit5": float(np.mean(ranks <= 5)),
        "hit10": float(np.mean(ranks <= 10)),
        "hit20": float(np.mean(ranks <= 20)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.percentile(ranks, 90)),
        "p95_rank": float(np.percentile(ranks, 95)),
        "p99_rank": float(np.percentile(ranks, 99)),
    }


def consensus_order(members: list[dict[str, object]], depth: int) -> list[str]:
    votes: collections.Counter[str] = collections.Counter()
    support: collections.defaultdict[str, float] = collections.defaultdict(float)
    for member in members:
        row = [str(value) for value in member["top_candidates"][:depth]]
        if not row:
            continue
        votes[row[0]] += 1
        denominator = len(row)
        for rank0, candidate in enumerate(row):
            support[candidate] += (denominator - rank0) / denominator
    return sorted(support, key=lambda candidate: (-votes[candidate], -support[candidate], candidate))


def evaluate(path: Path, method: str, seed: int | None, depth: int) -> tuple[list[dict], pd.DataFrame]:
    groups: dict[str, list[dict[str, object]]] = {}
    single_ranks: list[int] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            target = str(row["smiles"])
            top_candidates = list(row["top_candidates"])
            if not top_candidates:
                raise ValueError(f"empty top-candidate list in {path}")
            groups.setdefault(target, []).append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "rank": int(row["rank"]),
                    "top_candidates": top_candidates,
                }
            )
            single_ranks.append(int(row["rank"]))
    if len(single_ranks) != 57_437:
        raise AssertionError((path, len(single_ranks)))

    group_rows: list[dict[str, object]] = []
    for cluster_id, (target, members) in enumerate(
        item for item in groups.items() if len(item[1]) >= 2
    ):
        order = consensus_order(members, depth)
        target_present = target in order
        # Every member stores its exact Top-50 list.  If the target is absent
        # from their union, it is known to be outside the displayed Top-50
        # consensus support and must not receive the historical
        # ``len(observed_candidates) + 1`` pseudo-rank.  That old convention
        # made absent targets look like Top-20 hits whenever a small group had
        # fewer than 20 distinct voted candidates.  Rank 51 is a conservative
        # right-censoring value sufficient for the pre-registered Hit@1/5/20
        # endpoints; larger observed unions retain their natural lower bound.
        rank = (
            order.index(target) + 1
            if target_present
            else max(depth + 1, len(order) + 1)
        )
        group_rows.append(
            {
                "method": method,
                "seed": np.nan if seed is None else seed,
                "cluster_id": cluster_id,
                "canonical_smiles": target,
                "group_size": len(members),
                "consensus_rank": rank,
                "consensus_top1_smiles": order[0],
                "target_present_in_top50_union": int(target_present),
                "rank_right_censored": int(not target_present),
                "n_unique_top50_candidates": len(order),
                "member_sample_indices": ",".join(str(member["sample_idx"]) for member in members),
            }
        )
    groups_frame = pd.DataFrame(group_rows)
    if len(groups_frame) != 5_002 or int(groups_frame.group_size.sum()) != 57_436:
        raise AssertionError((path, len(groups_frame), int(groups_frame.group_size.sum())))
    impossible = groups_frame.loc[
        groups_frame["rank_right_censored"].eq(1)
        & groups_frame["consensus_rank"].le(depth)
    ]
    if len(impossible):
        raise AssertionError(
            f"{method}: {len(impossible)} absent targets were assigned rank <= {depth}"
        )
    cluster_ranks = groups_frame.consensus_rank.to_numpy(dtype=np.int64)
    weighted_ranks = np.repeat(cluster_ranks, groups_frame.group_size.to_numpy(dtype=np.int64))
    seed_value = np.nan if seed is None else seed
    metrics = [
        {"method": method, "seed": seed_value, "condition": CONDITION_SINGLE, **rank_metrics(np.asarray(single_ranks))},
        {"method": method, "seed": seed_value, "condition": CONDITION_CLUSTER, **rank_metrics(cluster_ranks)},
        {"method": method, "seed": seed_value, "condition": CONDITION_WEIGHTED, **rank_metrics(weighted_ranks)},
    ]
    return metrics, groups_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=50)
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
    if args.depth != 50:
        raise ValueError("the formal protocol fixes consensus depth at 50")

    metric_rows: list[dict] = []
    group_frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for method, seed, path in [parse_input(value) for value in args.input]:
        if not path.exists():
            raise FileNotFoundError(path)
        metrics, groups = evaluate(path, method, seed, args.depth)
        metric_rows.extend(metrics)
        group_frames.append(groups)
        manifest.append(
            {
                "method": method,
                "seed": np.nan if seed is None else seed,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    group_table = pd.concat(group_frames, ignore_index=True)
    required = (
        {
            "UltraMS", "DreaMS", "Linear", "DeepSets",
            "Fourier projection", "UltraMS codebook",
        }
        if args.formal_final_six
        else
        {
            "UltraMS", "DreaMS", "Linear", "Binned FFN",
            "DeepSets", "DeepSets + Fourier",
        }
        if args.formal_chemberta_six
        else {
            "UltraMS", "DreaMS", "Fingerprint FFN", "Linear FP",
            "DeepSets", "DeepSets + Fourier",
        }
    )
    if set(metrics.method) != required:
        raise ValueError(f"unexpected method set: {sorted(metrics.method.unique())}")
    for method in required:
        frame = metrics[metrics.method == method]
        if len(frame) != 3:
            raise ValueError(f"{method} has {len(frame)} metric rows")
        seeds = frame.seed.dropna().unique().tolist()
        if method in {"UltraMS", "DreaMS"}:
            if len(seeds) != 0:
                raise ValueError(f"{method} must use its frozen checkpoint without a model seed")
        elif seeds != [0]:
            raise ValueError(f"{method} must use exactly primary seed 0; got {seeds}")

    summaries: list[dict[str, object]] = []
    for (method, condition), frame in metrics.groupby(["method", "condition"], sort=False):
        if len(frame) != 1:
            raise ValueError(f"{method}/{condition} is not a single frozen checkpoint")
        source = frame.iloc[0]
        row: dict[str, object] = {
            "method": method,
            "condition": condition,
            "primary_seed": "" if pd.isna(source.seed) else int(source.seed),
            "n_model_seeds": 1,
            "n": int(source.n),
        }
        for metric in (
            "hit1", "hit5", "hit10", "hit20", "mrr", "mean_rank",
            "median_rank", "p90_rank", "p95_rank", "p99_rank",
        ):
            row[f"{metric}_primary"] = float(source[metric])
            row[f"{metric}_mean"] = float(source[metric])
            row[f"{metric}_sd_across_seeds"] = 0.0
        summaries.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "method_summary.csv", index=False)
    group_table.to_csv(args.output_dir / "per_group_top50_consensus.csv.gz", index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir / "input_manifest.csv", index=False)
    audit = (
        group_table.groupby(["method", "seed"], dropna=False)
        .agg(
            n_groups=("cluster_id", "size"),
            target_present_fraction=("target_present_in_top50_union", "mean"),
            n_rank_right_censored=("rank_right_censored", "sum"),
            median_union_size=("n_unique_top50_candidates", "median"),
        )
        .reset_index()
    )
    audit.to_csv(args.output_dir / "consensus_audit.csv", index=False)
    done = {
        "status": "complete",
        "grouping": "oracle exact canonical SMILES; groups of size >=2",
        "ranking": "Top-1 plurality, normalized-Borda support over each member's frozen Top-50 list, lexical final tie break",
        "depth": args.depth,
        "primary_model_seed": 0,
        "main_result_policy": "one frozen checkpoint per method; learned baselines use seed 0",
        "formal_chemberta_six": args.formal_chemberta_six,
        "formal_final_six": args.formal_final_six,
        "n_oracle_groups": 5_002,
        "n_oracle_spectra": 57_436,
        "methods": sorted(required),
        "artifacts": {
            path.name: sha256_file(path)
            for path in args.output_dir.iterdir()
            if path.is_file()
        },
    }
    (args.output_dir / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done, indent=2), flush=True)


if __name__ == "__main__":
    main()
