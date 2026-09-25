#!/usr/bin/env python
"""Build repeat-level summaries from attn_subspectrum_eval_v1.full_cache.pkl."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate_element_peaks import (  # noqa: E402
    METHOD_ORDER,
    TARGET_ATOMS,
    aggregate_metrics,
    evaluate_one,
    parse_top_n,
    project_root,
    resolve_path,
)


HIT_METHODS = ["UltraMS", "DreaMS", "Linear SVM", "Random forest"]
HIT_ATOMS = ["S", "F", "Cl", "Br"]


def hit_recall_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> tuple[float, float]:
    count = min(len(scores), len(truth))
    if count <= 0:
        return float("nan"), float("nan")
    values = np.nan_to_num(np.asarray(scores[:count], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.asarray(truth[:count], dtype=bool)
    positives = int(labels.sum())
    if positives <= 0:
        return float("nan"), float("nan")
    found = int(labels[np.argsort(-values)[: min(int(k), count)]].sum())
    return float(found > 0), float(found / positives)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_hit_recall(payload: dict, out_dir: Path, k_values: list[int]) -> None:
    records = {int(rec["test_index"]): rec for rec in payload["records"]}
    repeat_rows: list[dict] = []
    for repeat_id, indices in enumerate(payload["repeat_indices"], 1):
        for atom in HIT_ATOMS:
            for method in HIT_METHODS:
                for k in k_values:
                    values = {"hit_rate": [], "recall": []}
                    for test_index in indices:
                        rec = records[int(test_index)]
                        scores = rec["method_scores"].get(method, {}).get(atom)
                        if scores is None:
                            continue
                        mask_key = "dreams_magma_mask_by_atom" if method == "DreaMS" else "magma_mask_by_atom"
                        hit_rate, recall = hit_recall_at_k(scores, rec[mask_key][atom], k)
                        if np.isfinite(hit_rate):
                            values["hit_rate"].append(hit_rate)
                            values["recall"].append(recall)
                    for metric, items in values.items():
                        repeat_rows.append({
                            "repeat_id": repeat_id,
                            "atom": atom,
                            "method": method,
                            "metric": metric,
                            "k": int(k),
                            "n": len(items),
                            "value": float(np.mean(items)) if items else float("nan"),
                        })

    summary_rows: list[dict] = []
    for atom in HIT_ATOMS:
        for method in HIT_METHODS:
            for metric in ("hit_rate", "recall"):
                for k in k_values:
                    selected = [row for row in repeat_rows if row["atom"] == atom and row["method"] == method
                                and row["metric"] == metric and row["k"] == k]
                    values = [row["value"] for row in selected if np.isfinite(row["value"])]
                    sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                    summary_rows.append({
                        "atom": atom,
                        "method": method,
                        "metric": metric,
                        "k": int(k),
                        "n": sum(row["n"] for row in selected),
                        "repeat_n": len(values),
                        "mean": float(np.mean(values)) if values else float("nan"),
                        "sem": sem,
                    })
    write_rows(out_dir / "attn_subspectrum_hit_recall_curve_v1.repeat_curves.csv", repeat_rows)
    write_rows(out_dir / "attn_subspectrum_hit_recall_curve_v1.summary_curves.csv", summary_rows)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="main_figure/fig2/output/attn_subspectrum_eval_v1")
    parser.add_argument("--top-n", default=None)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=150)
    parser.add_argument("--k-step", type=int, default=1)
    args = parser.parse_args()
    if args.k_min < 1 or args.k_max < args.k_min or args.k_step < 1:
        parser.error("k values must satisfy 1 <= k-min <= k-max and k-step >= 1")

    out_dir = resolve_path(root, args.out_dir)
    full_cache = out_dir / "attn_subspectrum_eval_v1.full_cache.pkl"
    with full_cache.open("rb") as f:
        payload = pickle.load(f)

    config = payload.get("config", {})
    top_n_values = parse_top_n(args.top_n or config.get("top_n", "5,10,20"))
    repeat_indices = payload["repeat_indices"]
    records = payload["records"]
    rec_by_test = {int(rec["test_index"]): rec for rec in records}

    metric_rows = []
    for repeat_id, indices in enumerate(repeat_indices, 1):
        for rank_in_repeat, test_index in enumerate(indices, 1):
            rec = rec_by_test[int(test_index)]
            spectrum = np.asarray(rec["spectrum"], dtype=float)
            for atom in TARGET_ATOMS:
                for method in METHOD_ORDER:
                    scores = rec["method_scores"].get(method, {}).get(atom)
                    if scores is None:
                        continue
                    if method == "DreaMS":
                        truth = np.asarray(rec["dreams_magma_mask_by_atom"][atom], dtype=bool)
                        ref_intensity = np.asarray(rec["dreams_peaks"], dtype=float)[:, 1]
                    else:
                        truth = np.asarray(rec["magma_mask_by_atom"][atom], dtype=bool)
                        ref_intensity = spectrum[:, 1]
                    for metrics in evaluate_one(np.asarray(scores, dtype=float), truth, ref_intensity, top_n_values):
                        metric_rows.append(
                            {
                                "repeat_id": repeat_id,
                                "rank_in_repeat": rank_in_repeat,
                                "sample_id": rec["sample_id"],
                                "test_index": int(test_index),
                                "atom": atom,
                                "method": method,
                                "molecule_atom_count": int(rec["target_atom_counts"][atom]),
                                **metrics,
                            }
                        )

    repeat_metrics = pd.DataFrame(metric_rows)
    repeat_metrics_path = out_dir / "attn_subspectrum_eval_v1.repeat_metrics.csv"
    repeat_metrics.to_csv(repeat_metrics_path, index=False)

    valid = repeat_metrics[repeat_metrics["n_element_peaks"] > 0].copy()
    repeat_means = (
        valid.groupby(["repeat_id", "method", "atom", "top_n"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            cosine=("cosine", "mean"),
            precision_at_n=("precision_at_n", "mean"),
            element_intensity_capture_at_n=("element_intensity_capture_at_n", "mean"),
        )
        .reset_index()
    )
    repeat_means_path = out_dir / "attn_subspectrum_eval_v1.repeat_means.csv"
    repeat_means.to_csv(repeat_means_path, index=False)

    summary = (
        repeat_means.groupby(["method", "atom", "top_n"], dropna=False)
        .agg(
            n=("n", "sum"),
            repeat_n=("repeat_id", "count"),
            cosine_mean=("cosine", "mean"),
            cosine_sem=("cosine", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(np.isfinite(x).sum(), 1)))),
            precision_mean=("precision_at_n", "mean"),
            precision_sem=(
                "precision_at_n",
                lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(np.isfinite(x).sum(), 1))),
            ),
            capture_mean=("element_intensity_capture_at_n", "mean"),
            capture_sem=(
                "element_intensity_capture_at_n",
                lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(max(np.isfinite(x).sum(), 1))),
            ),
        )
        .reset_index()
    )
    repeat_summary_path = out_dir / "attn_subspectrum_eval_v1.repeat_summary_metrics.csv"
    summary.to_csv(repeat_summary_path, index=False)

    manifest = {
        "stage": "element_peak_localization_repeat_summary",
        "source_full_cache": str(full_cache),
        "repeat_metrics_csv": str(repeat_metrics_path),
        "repeat_means_csv": str(repeat_means_path),
        "repeat_summary_metrics_csv": str(repeat_summary_path),
        "top_n_values": top_n_values,
        "repeat_count": len(repeat_indices),
        "samples_per_repeat": [len(x) for x in repeat_indices],
    }
    (out_dir / "attn_subspectrum_eval_v1.repeat_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    summarize_hit_recall(payload, out_dir, list(range(args.k_min, args.k_max + 1, args.k_step)))
    print(f"Saved repeat-level metrics under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
