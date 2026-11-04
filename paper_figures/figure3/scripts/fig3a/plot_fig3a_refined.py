#!/usr/bin/env python3
"""Draw Figure 3a from the frozen MSnLib retrieval summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PANEL_DIR = SCRIPT_DIR.parent
SOURCE_DIR = PANEL_DIR / "source_data"
OUT_DIR = PANEL_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_ULTRA = "#3965BB"
C_DREAM = "#E9A0AA"
C_LINEAR = "#A7A7A7"
C_DEEPSETS = "#C48A4A"
C_FOURIER = "#8767AD"
C_CODEBOOK = "#5AA69A"
C_GRID = "#E2E5E9"
C_AXIS = "#33373D"

METRICS = [("Hit@1", "hit1"), ("Hit@5", "hit5"), ("Hit@20", "hit20")]
MODELS = [
    "DeepSets", "Fourier projection", "UltraMS codebook", "Linear",
    "DreaMS", "UltraMS",
]
COLORS = {
    "DreaMS": C_DREAM,
    "Linear": C_LINEAR,
    "DeepSets": C_DEEPSETS,
    "Fourier projection": C_FOURIER,
    "UltraMS codebook": C_CODEBOOK,
    "UltraMS": C_ULTRA,
}
DISPLAY = {
    "Fourier projection": "Fourier",
    "UltraMS codebook": "Codebook",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 6.2,
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
        "savefig.dpi": 450,
    })


def load_rows(kind: str) -> pd.DataFrame:
    formal = SOURCE_DIR / "formal_final_six_v1"
    if kind == "single":
        path = formal / "single_method_summary.csv"
        rows = pd.read_csv(path).rename(
            columns={
                "method": "model",
                "n_queries": "n",
                "top1_primary": "hit1",
                "top5_primary": "hit5",
                "top20_primary": "hit20",
            }
        )
    elif kind == "oracle":
        path = formal / "oracle_method_summary.csv"
        rows = pd.read_csv(path)
        rows = rows.loc[
            rows["condition"].eq("oracle_top50_consensus_cluster_majority")
        ].rename(
            columns={
                "method": "model",
                "hit1_primary": "hit1",
                "hit5_primary": "hit5",
                "hit20_primary": "hit20",
            }
        )
    else:
        raise ValueError(kind)
    rows = rows.set_index("model")
    missing = set(MODELS).difference(rows.index)
    if missing:
        raise ValueError(f"Missing model rows in {path}: {sorted(missing)}")
    return rows


def draw_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    title: str,
    ylim: tuple[float, float],
    *,
    show_legend: bool = False,
) -> None:
    x = np.arange(len(METRICS), dtype=float)
    width = 0.13
    offsets = width * (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2)
    for metric_index, (_, key) in enumerate(METRICS):
        ranked_models = sorted(
            MODELS, key=lambda model: float(rows.loc[model, key]),
        )
        for model, offset in zip(ranked_models, offsets):
            ax.bar(
                x[metric_index] + offset,
                float(rows.loc[model, key]) * 100,
                width=width,
                color=COLORS[model], edgecolor="white", linewidth=0.45,
                zorder=3,
            )

    ax.set_xticks(x, [label for label, _ in METRICS])
    plt.setp(ax.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.48, 2.48)
    ax.set_title(title, fontweight="normal", pad=3, linespacing=1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax.spines["left"].set_color(C_AXIS)
    # Keep the categorical baseline visually quiet: the bar positions and
    # labels already define the x axis, so a dark horizontal spine adds no
    # information and competes with the shortest bars.
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", pad=1)
    # The two panels use one shared legend, added at figure level in main().


def export_plot_data(single: pd.DataFrame, consensus: pd.DataFrame) -> None:
    records = []
    for condition, rows, unit in (
        ("Each spectrum alone", single, "query spectrum"),
        ("Combine votes from related spectra", consensus, "ground-truth molecule group"),
    ):
        for model in MODELS:
            for label, key in METRICS:
                records.append({
                    "evaluation": condition,
                    "analysis_unit": unit,
                    "model": model,
                    "metric": label,
                    "hit_rate_percent": float(rows.loc[model, key]) * 100,
                    "n": int(rows.loc[model, "n"]),
                })
    pd.DataFrame(records).to_csv(OUT_DIR / "fig3a_plot_data.csv", index=False)


def main() -> None:
    set_style()
    single = load_rows("single")
    consensus = load_rows("oracle")
    export_plot_data(single, consensus)

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(112 * mm, 62 * mm))
    draw_panel(axes[0], single, "Each spectrum\nranked separately", (0, 52))
    draw_panel(
        axes[1], consensus, "Known same-molecule spectra\nvoted together", (0, 70),
        show_legend=False,
    )
    axes[1].set_yticks(np.arange(0, 71, 10))
    axes[0].set_ylabel("Hit rate (%)")
    # Matplotlib fills multi-row legends by columns; this input order produces
    # two readable rows ranked from the weakest to the strongest methods.
    legend_models = [
        "DeepSets", "Fourier projection", "UltraMS codebook",
        "Linear", "DreaMS", "UltraMS",
    ]
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=COLORS[model], edgecolor="none",
            label=DISPLAY.get(model, model),
        )
        for model in legend_models
    ]
    fig.legend(
        # Keep the shared two-row key immediately below the rotated category
        # labels, matching the compact author reference without touching them.
        handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.130),
        ncol=3, handlelength=1.0, handletextpad=0.35, columnspacing=0.9,
        labelspacing=0.25, borderaxespad=0, fontsize=5.2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.34, top=0.89, wspace=0.32)

    for ext in ("png", "pdf", "svg"):
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(OUT_DIR / f"fig3a.{ext}", **kwargs)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'fig3a.svg'}")


if __name__ == "__main__":
    main()
