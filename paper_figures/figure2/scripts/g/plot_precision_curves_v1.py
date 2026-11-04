#!/usr/bin/env python
"""Plot element peak precision@k curves from saved subspectrum full cache."""

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


def precision_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> float:
    n = min(len(scores), len(truth))
    if n <= 0:
        return float("nan")
    scores = np.nan_to_num(np.asarray(scores[:n], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    truth = np.asarray(truth[:n], dtype=bool)
    if int(truth.sum()) <= 0:
        return float("nan")
    k_eff = min(int(k), n)
    order = np.argsort(-scores)[:k_eff]
    return float(truth[order].sum() / max(k_eff, 1))


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
                    values = []
                    for test_index in indices:
                        rec = rec_by_test[int(test_index)]
                        scores = rec["method_scores"].get(method, {}).get(atom)
                        if scores is None:
                            continue
                        truth = method_truth(rec, method, atom)
                        value = precision_at_k(np.asarray(scores, dtype=float), truth, k)
                        if np.isfinite(value):
                            values.append(value)
                    repeat_rows.append(
                        {
                            "repeat_id": repeat_id,
                            "atom": atom,
                            "method": method,
                            "k": int(k),
                            "n": int(len(values)),
                            "precision": float(np.mean(values)) if values else float("nan"),
                        }
                    )

    for atom in ATOM_ORDER:
        for method in METHOD_ORDER:
            for k in k_values:
                values = [
                    row["precision"]
                    for row in repeat_rows
                    if row["atom"] == atom and row["method"] == method and row["k"] == int(k)
                    and np.isfinite(row["precision"])
                ]
                n = sum(
                    row["n"]
                    for row in repeat_rows
                    if row["atom"] == atom and row["method"] == method and row["k"] == int(k)
                )
                summary_rows.append(
                    {
                        "atom": atom,
                        "method": method,
                        "k": int(k),
                        "n": int(n),
                        "repeat_n": int(len(values)),
                        "precision_mean": float(np.mean(values)) if values else float("nan"),
                        "precision_sem": sem(values),
                    }
                )
    return repeat_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_for(summary_rows: list[dict], atom: str, method: str) -> list[dict]:
    return sorted(
        [row for row in summary_rows if row["atom"] == atom and row["method"] == method],
        key=lambda row: int(row["k"]),
    )


def style_precision_axis(ax, k_values: list[int], atom: str) -> None:
    ax.set_xscale("log")
    xticks = [k for k in [5, 10, 20, 50, 100, 150] if k_values[0] <= k <= k_values[-1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlim(k_values[0], k_values[-1])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.set_xlabel("Top-k peaks")
    ax.set_ylabel("Element peak precision at k (%)")
    ax.set_title(atom, color=ATOM_COLORS.get(atom, "#333333"), fontweight="bold", pad=4)


def plot_atom_curve(summary_rows: list[dict], atom: str, k_values: list[int], out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.55, 2.35))
    for method in METHOD_ORDER:
        rows = rows_for(summary_rows, atom, method)
        if not rows:
            continue
        x = np.asarray([row["k"] for row in rows], dtype=float)
        y = np.asarray([row["precision_mean"] for row in rows], dtype=float) * 100.0
        e = np.asarray([row["precision_sem"] for row in rows], dtype=float) * 100.0
        color = METHOD_COLORS.get(method, "#666666")
        ax.plot(x, y, drawstyle="steps-post", color=color, linewidth=1.45, label=method)
        ax.fill_between(x, y - e, y + e, step="post", color=color, alpha=0.12, linewidth=0)
    style_precision_axis(ax, k_values, atom)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)


def plot_combined(summary_rows: list[dict], k_values: list[int], out_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.05), gridspec_kw={"wspace": 0.30, "hspace": 0.38})
    legend_handles = {}
    for ax, atom in zip(axes.ravel(), ATOM_ORDER):
        for method in METHOD_ORDER:
            rows = rows_for(summary_rows, atom, method)
            if not rows:
                continue
            x = np.asarray([row["k"] for row in rows], dtype=float)
            y = np.asarray([row["precision_mean"] for row in rows], dtype=float) * 100.0
            e = np.asarray([row["precision_sem"] for row in rows], dtype=float) * 100.0
            color = METHOD_COLORS.get(method, "#666666")
            line, = ax.plot(x, y, drawstyle="steps-post", color=color, linewidth=1.35, label=method)
            legend_handles.setdefault(method, line)
            ax.fill_between(x, y - e, y + e, step="post", color=color, alpha=0.11, linewidth=0)
        style_precision_axis(ax, k_values, atom)
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
    write_csv(out_dir / "attn_subspectrum_precision_curve_v1.repeat_curves.csv", repeat_rows)
    write_csv(out_dir / "attn_subspectrum_precision_curve_v1.summary_curves.csv", summary_rows)
    plot_data = {
        "stage": "fig2_attn_subspectrum_precision_curve_v1",
        "source_full_cache": str(cache_path),
        "atoms": ATOM_ORDER,
        "methods": METHOD_ORDER,
        "k_values": k_values,
        "repeat_count": len(payload["repeat_indices"]),
        "samples_per_repeat": [len(x) for x in payload["repeat_indices"]],
        "note": "Precision@k is computed per repeat and averaged over five repeat means. Samples with no MAGMa element peak for the queried element are excluded for that element.",
        "outputs": {
            "repeat_curves_csv": str(out_dir / "attn_subspectrum_precision_curve_v1.repeat_curves.csv"),
            "summary_curves_csv": str(out_dir / "attn_subspectrum_precision_curve_v1.summary_curves.csv"),
        },
    }
    (out_dir / "attn_subspectrum_precision_curve_v1.plot_data.json").write_text(
        json.dumps(plot_data, indent=2),
        encoding="utf-8",
    )

    for atom in ATOM_ORDER:
        plot_atom_curve(summary_rows, atom, k_values, out_dir / f"attn_subspectrum_precision_curve_v1_{atom}")
    plot_combined(summary_rows, k_values, out_dir / "attn_subspectrum_precision_curve_v1_2x2")
    print(f"Saved precision@k curves under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
