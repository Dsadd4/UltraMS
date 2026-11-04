#!/usr/bin/env python3
"""Plot Top-k retrieval accuracy as a Nature-style horizontal-bar matrix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
OUT_DIR = PACKAGE_DIR / "output"
SOURCE_DATA = DATA_DIR / "benchmark_top1_source_data.csv"
PLOT_DATA = PACKAGE_DIR / "fig3a_sp1_plot_data.csv"
CONTRACT = PACKAGE_DIR / "fig3a_sp1_chart_contract.json"
OUT_STEM = PACKAGE_DIR / "fig3a_sp1"


PALETTE = {
    "ink": "#272B33",
    "grid": "#E5E9EC",
    "axis": "#454B52",
    "ultra": "#3965BB",
    "dreams": "#E9A0AA",
    "mist": "#B0A0D2",
    "jestr": "#5AA69A",
    "esp": "#F4D49E",
    "deepsets": "#8FB9A8",
    "ffn": "#DAA06D",
    "random": "#D9D9D9",
}

FAMILY_COLORS = {
    "UltraMS": PALETTE["ultra"],
    "DreaMS": PALETTE["dreams"],
    "MIST": PALETTE["mist"],
    "JESTR": PALETTE["jestr"],
    "ESP": PALETTE["esp"],
    "DeepSets": PALETTE["deepsets"],
    "FFN": PALETTE["ffn"],
    "Random": PALETTE["random"],
}

DISPLAY_LABELS = {
    "UltraMS fair fusion": "UltraMS",
    "DreaMS fair fusion": "DreaMS",
    "DeepSets + Fourier features": "DeepSets+Fourier",
}

PANEL_DEFS = [
    {
        "panel": "NPLIB1",
        "title": "NPLIB1",
        "variants": {
            "NPLIB1/CANOPUS public natural product retrieval",
            "NPLIB1 ms2peaks_min50 three-fold source fusion",
        },
    },
    {
        "panel": "MassSpecGym mass",
        "title": "MassSpecGym: mass candidates",
        "variants": {
            "Official molecule retrieval leaderboard",
            "Mass/all fair two-source fingerprint retrieval",
        },
    },
    {
        "panel": "MassSpecGym formula",
        "title": "MassSpecGym: formula candidates",
        "variants": {
            "Official molecule retrieval bonus leaderboard",
            "Formula/all fair two-source fingerprint retrieval",
        },
    },
]

METRIC_DEFS = [
    {"metric": "Top-1", "column": "top1_percent", "xlim": 36.5, "tick_step": 8},
    {"metric": "Top-5", "column": "top5_percent", "xlim": 65.0, "tick_step": 20},
    {"metric": "Top-20", "column": "top20_percent", "xlim": 90.0, "tick_step": 20},
]


def clean_method_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label), str(label))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.7,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.7,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def build_plot_data(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for panel_def in PANEL_DEFS:
        sub = source[
            source["include_in_primary_plot"].astype(bool)
            & source["benchmark_variant"].isin(panel_def["variants"])
        ].copy()
        if sub.empty:
            raise ValueError(f"No rows for panel {panel_def['panel']}")

        sub = sub.sort_values(
            ["top1_percent", "model"], ascending=[False, True]
        ).reset_index(drop=True)
        sub["rank_for_plot"] = np.arange(len(sub))

        for _, row in sub.iterrows():
            for metric_def in METRIC_DEFS:
                value = row.get(metric_def["column"])
                rows.append(
                    {
                        "panel": panel_def["panel"],
                        "benchmark_variant": row["benchmark_variant"],
                        "model": row["model"],
                        "display_model": clean_method_label(row["model"]),
                        "model_family": row["model_family"],
                        "result_group": row["result_group"],
                        "metric": metric_def["metric"],
                        "retrieval_percent": float(value) if pd.notna(value) else np.nan,
                        "rank_for_plot": int(row["rank_for_plot"]),
                        "bar_color": FAMILY_COLORS.get(row["model_family"], PALETTE["random"]),
                        "is_ultrams": row["model_family"] == "UltraMS",
                        "is_dreams": row["model_family"] == "DreaMS",
                        "source_label": row["source_label"],
                        "source_url_or_path": row["source_url_or_path"],
                    }
                )
    return pd.DataFrame(rows)


def format_percent_axis(ax: plt.Axes, metric_def: dict) -> None:
    xlim = metric_def["xlim"]
    ticks = np.arange(0, xlim + 0.01, metric_def["tick_step"])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlim(0, xlim)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=PALETTE["ink"], pad=2)
    ax.tick_params(axis="y", colors=PALETTE["ink"], pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])


def plot_metric_panel(
    ax: plt.Axes,
    panel_data: pd.DataFrame,
    metric_def: dict,
    show_method_labels: bool,
) -> None:
    metric_data = panel_data[panel_data["metric"].eq(metric_def["metric"])].copy()
    available = metric_data[metric_data["retrieval_percent"].notna()].copy()

    ax.barh(
        available["rank_for_plot"],
        available["retrieval_percent"],
        height=0.62,
        color=available["bar_color"],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )

    xlim = metric_def["xlim"]
    for _, row in available.iterrows():
        ax.text(
            min(float(row["retrieval_percent"]) + xlim * 0.018, xlim * 0.985),
            row["rank_for_plot"],
            f"{row['retrieval_percent']:.1f}",
            ha="left",
            va="center",
            fontsize=5.9,
            color=PALETTE["ink"],
        )

    methods = (
        panel_data[["rank_for_plot", "display_model"]]
        .drop_duplicates()
        .sort_values("rank_for_plot")
    )
    y = methods["rank_for_plot"].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(methods["display_model"] if show_method_labels else [])
    ax.set_ylim(-0.65, len(methods) - 0.35)
    ax.invert_yaxis()
    format_percent_axis(ax, metric_def)


def write_contract() -> None:
    contract = {
        "analytical_question": "How does cumulative candidate retrieval change from Top-1 to Top-5 and Top-20 across external benchmarks?",
        "takeaway": "UltraMS exceeds the matched DreaMS comparator at every reported k, while reported JESTR Top-5 and Top-20 values are higher on both MassSpecGym tasks.",
        "chart_family": "Comparison & Ranking",
        "variant": "three-by-three horizontal bar matrix",
        "metric": "Top-k retrieval accuracy / Hit rate at k (%)",
        "missing_value_policy": "Missing values, if present, are retained as NA in plot data and shown as an empty row, without interpolation or a zero-valued bar.",
        "data_source": str(SOURCE_DATA.relative_to(PACKAGE_DIR)),
        "derived_plot_data": str(PLOT_DATA.relative_to(PACKAGE_DIR)),
        "panels": [p["panel"] for p in PANEL_DEFS],
        "metrics": [m["metric"] for m in METRIC_DEFS],
        "palette_policy": "Method-family colours are fixed across every row and metric column.",
        "qa_requirements": [
            "three benchmark rows and three Top-k columns are present",
            "method order within each benchmark is fixed by descending Top-1 accuracy",
            "all available values are shown as bars with direct value labels",
            "the MIST full Top-1, Top-5, and Top-20 values match official Supplementary Table 8",
            "no error bars, reference-category legend, line segments, or grey annotation text are rendered",
            "PNG, PDF, and SVG are exported",
        ],
    }
    CONTRACT.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_DATA)
    plot_data = build_plot_data(source)
    plot_data.to_csv(PLOT_DATA, index=False)
    write_contract()

    # Nature double-column matrix: 188 x 145 mm.
    fig, axes = plt.subplots(3, 3, figsize=(7.40, 5.71), dpi=300)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.155, right=0.992, top=0.925, bottom=0.090, wspace=0.17, hspace=0.46)

    for row_index, panel_def in enumerate(PANEL_DEFS):
        panel_data = plot_data[plot_data["panel"].eq(panel_def["panel"])].copy()
        for column_index, metric_def in enumerate(METRIC_DEFS):
            plot_metric_panel(
                axes[row_index, column_index],
                panel_data,
                metric_def,
                show_method_labels=column_index == 0,
            )

    for column_index, metric_def in enumerate(METRIC_DEFS):
        position = axes[0, column_index].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.965,
            metric_def["metric"],
            ha="center",
            va="center",
            fontsize=7.2,
            color=PALETTE["ink"],
        )

    for row_index, panel_def in enumerate(PANEL_DEFS):
        position = axes[row_index, 0].get_position()
        fig.text(
            position.x0,
            position.y1 + 0.014,
            panel_def["title"],
            ha="left",
            va="bottom",
            fontsize=7.1,
            color=PALETTE["ink"],
        )

    fig.text(
        0.575,
        0.025,
        "Retrieval accuracy (%)",
        ha="center",
        va="center",
        fontsize=6.8,
        color=PALETTE["ink"],
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=500 if ext == "png" else None)
    plt.close(fig)
    print(f"saved {OUT_STEM}.png")
    print(f"saved {OUT_STEM}.pdf")
    print(f"saved {OUT_STEM}.svg")
    print(f"saved {PLOT_DATA}")
    print(f"saved {CONTRACT}")


if __name__ == "__main__":
    main()
