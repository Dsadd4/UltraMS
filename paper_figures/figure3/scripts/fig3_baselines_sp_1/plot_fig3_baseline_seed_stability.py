#!/usr/bin/env python3
"""Render supplementary five-seed stability evidence for Figure 3 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLORS = {"Fingerprint FFN": "#666666", "Spec2Vec": "#5AA69A"}
INK = "#25282B"
MID = "#737B82"
GRID = "#D9DEE2"
MM = 1 / 25.4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_formal(directory: Path, marker: str) -> None:
    path = directory / marker
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or payload.get("formal_method_completeness_enforced") is not True:
        raise RuntimeError(f"formal completion marker required: {path}")
    if payload.get("primary_model_seed") != 0:
        raise RuntimeError(f"primary seed-0 policy required: {path}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def draw_seed_group(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    seeds: np.ndarray,
    color: str,
) -> None:
    order = np.argsort(seeds)
    values = np.asarray(values, dtype=float)[order]
    seeds = np.asarray(seeds, dtype=int)[order]
    if not np.array_equal(seeds, np.arange(5)):
        raise ValueError(f"expected model seeds 0--4, got {seeds.tolist()}")
    offsets = np.linspace(-0.07, 0.07, 5)
    ax.scatter(
        x + offsets[1:], values[1:], s=7.2, marker="o", color=color,
        alpha=0.42, linewidth=0, zorder=2,
    )
    ax.scatter(
        [x + offsets[0]], [values[0]], s=15, marker="o", color=color,
        edgecolor="white", linewidth=0.55, zorder=4,
    )
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    ax.errorbar(
        x + 0.115, mean, yerr=sd, fmt="_", color=color, markersize=5,
        markeredgewidth=1.0, elinewidth=0.75, capsize=1.4, zorder=3,
    )


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    categories: list[str],
    category_column: str,
    value_column: str,
    title: str,
    labels: list[str] | None = None,
) -> None:
    displayed_values: list[float] = []
    tick_labels: list[str] = []
    for index, category in enumerate(categories):
        subset = frame[frame[category_column] == category].sort_values("seed")
        color = COLORS.get(category, COLORS["Spec2Vec"])
        absolute = 100 * subset[value_column].to_numpy(dtype=float)
        mean = float(absolute.mean())
        deviations = absolute - mean
        displayed_values.extend(deviations.tolist())
        base_label = (labels or categories)[index]
        tick_labels.append(f"{base_label}\n{mean:.2f}% mean")
        draw_seed_group(
            ax,
            float(index),
            deviations,
            subset.seed.to_numpy(dtype=int),
            color,
        )
    ax.set_xticks(np.arange(len(categories)), tick_labels)
    ax.set_xlim(-0.28, len(categories) - 0.72)
    # Centering each independently scaled endpoint on its five-seed mean keeps
    # small FFN variation visible beside the much higher Spec2Vec endpoints.
    # Absolute means remain explicit in the tick labels and source tables.
    limit = max(0.10, 1.22 * max(abs(value) for value in displayed_values))
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color=MID, linestyle=":", linewidth=0.65, zorder=1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_title(title, pad=3, fontweight="normal")
    clean_axis(ax)


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, object] = {}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(output.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-dir", required=True, type=Path)
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--fig3h-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require_formal(args.retrieval_dir, "DONE.json")
    require_formal(args.group_dir, "AUDIT.json")
    require_formal(args.fig3h_dir, "DONE.json")

    retrieval = pd.read_csv(args.retrieval_dir / "per_seed_metrics.csv")
    retrieval = retrieval[retrieval.method.isin(COLORS) & retrieval.seed.notna()].copy()
    retrieval.seed = retrieval.seed.astype(int)
    grouped = pd.read_csv(args.group_dir / "per_seed_performance.csv")
    grouped = grouped[grouped.method.isin(COLORS) & grouped.seed.notna()].copy()
    grouped.seed = grouped.seed.astype(int)
    spectral = pd.read_csv(args.fig3h_dir / "low_fdr_mean_recall_per_seed.csv")
    spectral = spectral[
        (spectral.method == "Spec2Vec")
        & (spectral.split == "test")
        & spectral.seed.notna()
    ].copy()
    spectral.seed = spectral.seed.astype(int)

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(115 * MM, 50 * MM))
    draw_panel(
        axes[0], retrieval, ["Fingerprint FFN", "Spec2Vec"], "method", "top1",
        "Single-spectrum Top-1", labels=["FFN", "Spec2Vec"],
    )
    draw_panel(
        axes[1], grouped, ["Fingerprint FFN", "Spec2Vec"], "method", "consensus_top1",
        "Grouped-spectrum Top-1", labels=["FFN", "Spec2Vec"],
    )
    draw_panel(
        axes[2], spectral, ["pos", "neg"], "mode", "mean_identification_recall",
        "Low-FDR mean recall", labels=["[M+H]+", "[M-H]-"],
    )
    axes[0].set_ylabel("Deviation from five-seed mean (pp)")

    primary = plt.Line2D([], [], marker="o", linestyle="none", color=INK,
                         markeredgecolor="white", markeredgewidth=0.55,
                         markersize=3.2, label="Primary seed 0")
    repeats = plt.Line2D([], [], marker="o", linestyle="none", color=INK,
                         alpha=0.42, markersize=2.3, label="Seeds 1--4")
    mean_sd = plt.Line2D([], [], marker="_", linestyle="none", color=INK,
                         markersize=5, markeredgewidth=1.0, label="Mean ± s.d.")
    fig.legend(handles=[primary, repeats, mean_sd], loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=3, handletextpad=0.25,
               columnspacing=0.65)
    fig.subplots_adjust(left=0.105, right=0.955, bottom=0.30, top=0.90, wspace=0.42)

    output = args.output_dir / "fig3_baseline_seed_stability"
    save(fig, output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval.to_csv(args.output_dir / "seed_stability_retrieval_source.csv", index=False)
    grouped.to_csv(args.output_dir / "seed_stability_group_source.csv", index=False)
    spectral.to_csv(args.output_dir / "seed_stability_spectral_source.csv", index=False)
    sources = [
        args.retrieval_dir / "per_seed_metrics.csv",
        args.group_dir / "per_seed_performance.csv",
        args.fig3h_dir / "low_fdr_mean_recall_per_seed.csv",
    ]
    audit = {
        "status": "complete",
        "primary_model_seed": 0,
        "stability_model_seeds": [1, 2, 3, 4],
        "point_definition": "each point is one independently trained model seed",
        "summary_definition": "mean plus or minus sample standard deviation across five seeds",
        "sources": [
            {
                "source_id": f"{path.parent.name}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sources
        ],
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.glob("fig3_baseline_seed_stability.*"))
        },
    }
    (args.output_dir / "PLOT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
