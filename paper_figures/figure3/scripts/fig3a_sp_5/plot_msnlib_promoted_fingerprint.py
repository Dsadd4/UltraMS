#!/usr/bin/env python3
"""Plot MSnLib promoted UltraMS/DreaMS molecular-fingerprint retrieval."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
PANEL = HERE.parent
SOURCE = PANEL / "source_data" / "msnlib_promoted_fp_metrics.csv"
OUTPUT = PANEL / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {"DreaMS FP": "#E9A0AA", "UltraMS FP": "#3965BB"}
DISPLAY = {"DreaMS FP": "DreaMS", "UltraMS FP": "UltraMS"}
METRICS = [("Top-1", "top1"), ("Top-5", "top5"), ("Top-20", "top20")]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def main() -> None:
    setup_style()
    data = pd.read_csv(SOURCE)
    required = {"DreaMS FP", "UltraMS FP"}
    if set(data["method"]) != required or len(data) != 2:
        raise ValueError(f"Expected exactly {sorted(required)}; got {data['method'].tolist()}")
    if not data["n_queries"].eq(57_437).all():
        raise ValueError("Formal test denominator must be 57,437")

    x = list(range(len(METRICS)))
    plot_rows = []
    fig, ax = plt.subplots(figsize=(82 / 25.4, 55 / 25.4))
    for method in ("DreaMS FP", "UltraMS FP"):
        row = data.loc[data["method"].eq(method)].iloc[0]
        y = [100 * float(row[key]) for _, key in METRICS]
        ax.plot(
            x, y, color=COLORS[method], linewidth=1.35, marker="o",
            markersize=4.2, markeredgecolor="white", markeredgewidth=0.55,
            label=DISPLAY[method], zorder=3,
        )
        for xpos, ((metric, _), value) in enumerate(zip(METRICS, y)):
            label_y = value + (1.15 if method == "UltraMS FP" else 0.35)
            label_x = xpos + (0.06 if method == "UltraMS FP" else -0.06)
            ax.text(
                label_x, label_y, f"{value:.2f}", ha="center", va="bottom",
                color=COLORS[method], fontsize=6.0,
            )
            plot_rows.append({
                "method": method,
                "display_method": DISPLAY[method],
                "metric": metric,
                "retrieval_percent": value,
                "n_queries": int(row["n_queries"]),
                "validation_selected_global_weight": float(row["source_weight_first"]),
            })

    ax.set_xticks(x, [label for label, _ in METRICS])
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, max(r["retrieval_percent"] for r in plot_rows) + 8)
    ax.set_xlim(-0.18, 2.18)
    ax.set_title("MSnLib molecular-fingerprint retrieval", pad=6)
    ax.grid(axis="y", color="#E2E5E9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, handlelength=1.5, columnspacing=0.9)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.86)

    pd.DataFrame(plot_rows).to_csv(OUTPUT / "fig3a_sp_5_plot_data.csv", index=False)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(OUTPUT / f"fig3a_sp_5.{ext}", **kwargs)
        fig.savefig(PANEL / f"fig3a_sp_5.{ext}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
