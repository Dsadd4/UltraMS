#!/usr/bin/env python3
"""Plot element hit-rate@k and recall@k curves from saved subspectrum cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ATOM_ORDER = ["S", "F", "Cl", "Br"]
METHOD_ORDER = ["UltraMS", "DreaMS", "Linear SVM", "Random forest"]
METHOD_COLORS = {
    "UltraMS": "#e94b3c",
    "DreaMS": "#4db9d1",
    "Linear SVM": "#7c5aa6",
    "Random forest": "#3f8c4a",
}
ATOM_COLORS = {"S": "#f28e63", "Cl": "#4fa83d", "F": "#7b4ab2", "Br": "#c67a2a"}
METRIC_LABELS = {
    "hit_rate": "Element hit rate at k (%)",
    "recall": "Element peak recall at k (%)",
}
METRIC_BASENAMES = {
    "hit_rate": "attn_subspectrum_accuracy_curve_v1",
    "recall": "attn_subspectrum_recall_curve_v1",
}
ERROR_K_VALUES = [5, 10, 20, 50, 100, 150]
ROW4_FIGSIZE = (10.8, 2.08)
ROW4_LEGEND_ANCHOR = (0.5, 1.025)
ROW4_ADJUST = {"top": 0.82, "bottom": 0.24}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def topk_metrics(scores: np.ndarray, truth: np.ndarray, k: int) -> dict[str, float]:
    n = min(len(scores), len(truth))
    if n <= 0:
        return {"hit_rate": float("nan"), "recall": float("nan")}
    scores = np.nan_to_num(np.asarray(scores[:n], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    truth = np.asarray(truth[:n], dtype=bool)
    n_true = int(truth.sum())
    if n_true <= 0:
        return {"hit_rate": float("nan"), "recall": float("nan")}
    k_eff = min(int(k), n)
    order = np.argsort(-scores)[:k_eff]
    hits = int(truth[order].sum())
    return {
        "hit_rate": float(hits > 0),
        "recall": float(hits / n_true),
    }


def method_truth(rec: dict, method: str, atom: str) -> np.ndarray:
    if method == "DreaMS":
        return np.asarray(rec["dreams_magma_mask_by_atom"][atom], dtype=bool)
    return np.asarray(rec["magma_mask_by_atom"][atom], dtype=bool)


def compute_curves(payload: dict, k_values: list[int]) -> tuple[list[dict], list[dict]]:
    records = payload["records"]
    repeat_indices = payload["repeat_indices"]
    rec_by_test = {int(rec["test_index"]): rec for rec in records}
    repeat_rows: list[dict] = []
    summary_rows: list[dict] = []

    for repeat_id, indices in enumerate(repeat_indices, 1):
        for atom in ATOM_ORDER:
            for method in METHOD_ORDER:
                for k in k_values:
                    metric_values = {"hit_rate": [], "recall": []}
                    for test_index in indices:
                        rec = rec_by_test[int(test_index)]
                        scores = rec["method_scores"].get(method, {}).get(atom)
                        if scores is None:
                            continue
                        truth = method_truth(rec, method, atom)
                        values = topk_metrics(np.asarray(scores, dtype=float), truth, k)
                        for metric, value in values.items():
                            if np.isfinite(value):
                                metric_values[metric].append(value)
                    for metric, values in metric_values.items():
                        repeat_rows.append(
                            {
                                "repeat_id": repeat_id,
                                "atom": atom,
                                "method": method,
                                "metric": metric,
                                "k": int(k),
                                "n": int(len(values)),
                                "value": float(np.mean(values)) if values else float("nan"),
                            }
                        )

    for atom in ATOM_ORDER:
        for method in METHOD_ORDER:
            for metric in ("hit_rate", "recall"):
                for k in k_values:
                    rows = [
                        row
                        for row in repeat_rows
                        if row["atom"] == atom
                        and row["method"] == method
                        and row["metric"] == metric
                        and row["k"] == int(k)
                    ]
                    values = [row["value"] for row in rows if np.isfinite(row["value"])]
                    summary_rows.append(
                        {
                            "atom": atom,
                            "method": method,
                            "metric": metric,
                            "k": int(k),
                            "n": int(sum(row["n"] for row in rows)),
                            "repeat_n": int(len(values)),
                            "mean": float(np.mean(values)) if values else float("nan"),
                            "sem": sem(values),
                        }
                    )
    return repeat_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_for(summary_rows: list[dict], atom: str, method: str, metric: str) -> list[dict]:
    return sorted(
        [row for row in summary_rows if row["atom"] == atom and row["method"] == method and row["metric"] == metric],
        key=lambda row: int(row["k"]),
    )


def style_axis(
    ax,
    k_values: list[int],
    atom: str,
    metric: str,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    compact_ylabel: bool = False,
) -> None:
    ax.set_xscale("log")
    xticks = [k for k in [5, 10, 20, 50, 100, 150] if k_values[0] <= k <= k_values[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(k_values[0], k_values[-1])
    if metric == "hit_rate":
        ax.set_ylim(55, 101)
        ax.set_yticks([60, 70, 80, 90, 100])
        ax.text(
            -0.035,
            0.0,
            "//",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#333333",
        )
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    if show_xlabel:
        ax.set_xlabel("Top-k peaks")
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ylabel = {"hit_rate": "Hit rate at k (%)", "recall": "Peak recall at k (%)"}[metric] if compact_ylabel else METRIC_LABELS[metric]
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    ax.set_title(atom, color=ATOM_COLORS.get(atom, "#333333"), fontweight="bold", pad=2 if not show_xlabel else 4)


def add_error_bars(ax, x: np.ndarray, y: np.ndarray, e: np.ndarray, color: str, k_values: list[int]) -> None:
    error_x = [k for k in ERROR_K_VALUES if k_values[0] <= k <= k_values[-1]]
    if not error_x:
        return
    for k in error_x:
        idx = int(np.argmin(np.abs(x - k)))
        if not np.isfinite(y[idx]) or not np.isfinite(e[idx]):
            continue
        ax.errorbar(
            x[idx],
            y[idx],
            yerr=e[idx],
            color=color,
            fmt="none",
            elinewidth=0.75,
            capsize=1.7,
            capthick=0.75,
            alpha=0.88,
            zorder=5,
        )


def plot_atom_curve(summary_rows: list[dict], atom: str, metric: str, k_values: list[int], out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.55, 2.35))
    for method in METHOD_ORDER:
        rows = rows_for(summary_rows, atom, method, metric)
        if not rows:
            continue
        x = np.asarray([row["k"] for row in rows], dtype=float)
        y = np.asarray([row["mean"] for row in rows], dtype=float) * 100.0
        e = np.asarray([row["sem"] for row in rows], dtype=float) * 100.0
        color = METHOD_COLORS.get(method, "#666666")
        ax.plot(x, y, drawstyle="steps-post", color=color, linewidth=1.45, label=method)
        ax.fill_between(x, y - e, y + e, step="post", color=color, alpha=0.12, linewidth=0)
        add_error_bars(ax, x, y, e, color, k_values)
    style_axis(ax, k_values, atom, metric)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)


def plot_combined(summary_rows: list[dict], metric: str, k_values: list[int], out_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.05), gridspec_kw={"wspace": 0.30, "hspace": 0.38})
    legend_handles = {}
    for ax, atom in zip(axes.ravel(), ATOM_ORDER):
        for method in METHOD_ORDER:
            rows = rows_for(summary_rows, atom, method, metric)
            if not rows:
                continue
            x = np.asarray([row["k"] for row in rows], dtype=float)
            y = np.asarray([row["mean"] for row in rows], dtype=float) * 100.0
            e = np.asarray([row["sem"] for row in rows], dtype=float) * 100.0
            color = METHOD_COLORS.get(method, "#666666")
            line, = ax.plot(x, y, drawstyle="steps-post", color=color, linewidth=1.35, label=method)
            legend_handles.setdefault(method, line)
            ax.fill_between(x, y - e, y + e, step="post", color=color, alpha=0.11, linewidth=0)
            add_error_bars(ax, x, y, e, color, k_values)
        style_axis(ax, k_values, atom, metric)
    handles = [legend_handles[m] for m in METHOD_ORDER if m in legend_handles]
    labels = [m for m in METHOD_ORDER if m in legend_handles]
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(top=0.90)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)


def plot_row4(summary_rows: list[dict], metric: str, k_values: list[int], out_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=ROW4_FIGSIZE, gridspec_kw={"wspace": 0.34})
    legend_handles = {}
    for ax, atom in zip(axes, ATOM_ORDER):
        for method in METHOD_ORDER:
            rows = rows_for(summary_rows, atom, method, metric)
            if not rows:
                continue
            x = np.asarray([row["k"] for row in rows], dtype=float)
            y = np.asarray([row["mean"] for row in rows], dtype=float) * 100.0
            e = np.asarray([row["sem"] for row in rows], dtype=float) * 100.0
            color = METHOD_COLORS.get(method, "#666666")
            line, = ax.plot(x, y, drawstyle="steps-post", color=color, linewidth=1.35, label=method)
            legend_handles.setdefault(method, line)
            ax.fill_between(x, y - e, y + e, step="post", color=color, alpha=0.11, linewidth=0)
            add_error_bars(ax, x, y, e, color, k_values)
        style_axis(ax, k_values, atom, metric)
    handles = [legend_handles[m] for m in METHOD_ORDER if m in legend_handles]
    labels = [m for m in METHOD_ORDER if m in legend_handles]
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=ROW4_LEGEND_ANCHOR,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(**ROW4_ADJUST)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="main_figure/fig2/output/attn_subspectrum_eval_v1")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=150)
    parser.add_argument("--k-step", type=int, default=1)
    args = parser.parse_args()

    out_dir = resolve_path(root, args.out_dir)
    cache_path = resolve_path(root, args.cache) if args.cache else out_dir / "attn_subspectrum_eval_v1.full_cache.pkl"
    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    repeat_rows, summary_rows = compute_curves(payload, k_values)
    write_csv(out_dir / "attn_subspectrum_hit_recall_curve_v1.repeat_curves.csv", repeat_rows)
    write_csv(out_dir / "attn_subspectrum_hit_recall_curve_v1.summary_curves.csv", summary_rows)

    for metric, basename in METRIC_BASENAMES.items():
        for atom in ATOM_ORDER:
            plot_atom_curve(summary_rows, atom, metric, k_values, out_dir / f"{basename}_{atom}")
        plot_combined(summary_rows, metric, k_values, out_dir / f"{basename}_2x2")
        plot_row4(summary_rows, metric, k_values, out_dir / f"{basename}_row4")

    plot_data = {
        "stage": "fig2_attn_subspectrum_hit_recall_curve_v1",
        "source_full_cache": str(cache_path),
        "atoms": ATOM_ORDER,
        "methods": METHOD_ORDER,
        "k_values": k_values,
        "repeat_count": len(payload["repeat_indices"]),
        "samples_per_repeat": [len(x) for x in payload["repeat_indices"]],
        "metric_definitions": {
            "hit_rate": "Top-k accuracy: 1 if at least one MAGMa-supported element peak is in top-k, else 0.",
            "recall": "Fraction of MAGMa-supported element peaks recovered by the top-k scored peaks.",
        },
        "plot_parameters": {
            "row4_figsize_inches": list(ROW4_FIGSIZE),
            "row4_legend_anchor": list(ROW4_LEGEND_ANCHOR),
            "row4_subplots_adjust": ROW4_ADJUST,
            "error_bar_k_values": ERROR_K_VALUES,
            "hit_rate_ylim_percent": [55, 101],
            "recall_ylim_percent": [0, 100],
        },
        "outputs": {
            "repeat_curves_csv": str(out_dir / "attn_subspectrum_hit_recall_curve_v1.repeat_curves.csv"),
            "summary_curves_csv": str(out_dir / "attn_subspectrum_hit_recall_curve_v1.summary_curves.csv"),
        },
    }
    (out_dir / "attn_subspectrum_hit_recall_curve_v1.plot_data.json").write_text(
        json.dumps(plot_data, indent=2),
        encoding="utf-8",
    )
    print(f"Saved hit-rate/recall@k curves under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
