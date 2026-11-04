#!/usr/bin/env python3
"""Render the Nature-style S7 source-fusion decision figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


HERE = Path(__file__).resolve()
S7_DIR = HERE.parents[1]
SOURCE_DIR = S7_DIR / "source_data"
OUTPUT_DIR = S7_DIR / "output"

# Shared supplementary-figure palette.
ULTRA = "#3965BB"
DREAMS = "#E9A0AA"
TEAL = "#1B9E8A"
GOLD = "#C9982E"
HARM = "#B64F59"
TEXT = "#292D32"
MID = "#7B858E"
GRID = "#DDE3E7"
PALE = "#F1F3F4"
WHITE = "#FFFFFF"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.unicode_minus": False,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(colors=TEXT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)


def panel_heading(ax: plt.Axes, letter: str, title: str, subtitle: str | None = None) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.96,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.035,
        0.96,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=TEXT,
    )
    if subtitle:
        ax.text(
            0.035,
            0.28,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
            color=MID,
        )


def outcome_labels(source_rank: pd.Series, final_rank: pd.Series) -> np.ndarray:
    return np.select(
        [
            (source_rank > 1) & (final_rank == 1),
            (source_rank == 1) & (final_rank == 1),
            (source_rank == 1) & (final_rank > 1),
        ],
        ["rescued", "preserved", "harmed"],
        default="unresolved",
    )


def plot_rank_pair(
    ax: plt.Axes,
    queries: pd.DataFrame,
    metrics: pd.DataFrame,
    source_column: str,
    source_view: str,
    source_label: str,
    show_y: bool,
) -> None:
    work = queries[[source_column, "final_rank"]].copy()
    work["outcome"] = outcome_labels(work[source_column], work["final_rank"])
    colors = {
        "rescued": TEAL,
        "preserved": ULTRA,
        "harmed": HARM,
        "unresolved": MID,
    }
    alphas = {"rescued": 0.24, "preserved": 0.055, "harmed": 0.30, "unresolved": 0.016}
    widths = {"rescued": 0.52, "preserved": 0.34, "harmed": 0.62, "unresolved": 0.24}
    for lane in ["unresolved", "preserved", "rescued", "harmed"]:
        values = work.loc[work["outcome"] == lane, [source_column, "final_rank"]].to_numpy(dtype=float)
        for source_rank, final_rank in values:
            ax.plot(
                [0, 1],
                [source_rank, final_rank],
                color=colors[lane],
                alpha=alphas[lane],
                lw=widths[lane],
                solid_capstyle="round",
                zorder=1,
            )

    ax.axvline(0, color=GRID, lw=0.7, zorder=0)
    ax.axvline(1, color=GRID, lw=0.7, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(54, 0.83)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(-0.10, 1.10)
    ax.set_xticks([0, 1], [source_label, "Final\nfusion"])
    if show_y:
        ax.set_ylabel("Target-molecule rank")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    clean_axis(ax, "y")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)

    lookup = metrics.set_index("view")
    source_top1 = float(lookup.loc[source_view, "top1_percent"])
    final_top1 = float(lookup.loc["UltraMS final fusion", "top1_percent"])
    ax.text(0, 0.94, f"{source_top1:.1f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=TEXT)
    ax.text(1, 0.94, f"{final_top1:.1f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=TEAL)
def plot_rank_rivers(fig: plt.Figure, slot, queries: pd.DataFrame, metrics: pd.DataFrame) -> None:
    grid = slot.subgridspec(2, 3, height_ratios=[0.58, 1.42], hspace=0.03, wspace=0.16)
    heading_ax = fig.add_subplot(grid[0, :])
    panel_heading(
        heading_ax,
        "a",
        "Independent spectrum views feed one molecular decision",
        "Validation: global + raw in folds 1/3; global + peak in fold 2 (0.5 each).",
    )
    specs = [
        ("global_rank", "UltraMS global", "Global"),
        ("peak_rank", "UltraMS projected peak", "Projected\npeak"),
        ("rawproj_rank", "UltraMS raw projection", "Raw\nprojection"),
    ]
    axes = []
    for index, (column, view, label) in enumerate(specs):
        ax = fig.add_subplot(grid[1, index])
        plot_rank_pair(ax, queries, metrics, column, view, label, index == 0)
        axes.append(ax)

    handles = [
        mpl.lines.Line2D([], [], color=TEAL, lw=1.4, label="rescued"),
        mpl.lines.Line2D([], [], color=ULTRA, lw=1.4, label="preserved"),
        mpl.lines.Line2D([], [], color=HARM, lw=1.4, label="harmed"),
        mpl.lines.Line2D([], [], color=MID, lw=1.4, alpha=0.65, label="unresolved"),
    ]
    heading_ax.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.33),
        handlelength=1.3,
        columnspacing=0.9,
        borderaxespad=0,
    )


def plot_transition_ledger(ax: plt.Axes, stats: pd.DataFrame) -> None:
    row_specs = [
        ("UltraMS", "global_to_final", "Global", ULTRA),
        ("UltraMS", "projected_peak_to_final", "Projected peak", TEAL),
        ("UltraMS", "raw_projection_to_final", "Raw projection", GOLD),
        ("DreaMS", "global_to_final", "DreaMS global", DREAMS),
    ]
    rows = []
    for model, comparison, label, color in row_specs:
        match = stats.loc[(stats["model"] == model) & (stats["comparison"] == comparison)]
        if len(match) != 1:
            raise ValueError(f"Expected one {model} {comparison} row")
        rows.append((match.iloc[0], label, color))

    y_positions = [6, 4, 2, 0]
    for y, (row, label, color) in zip(y_positions, rows):
        harm = -float(row["harm_percent"])
        rescue = float(row["rescue_percent"])
        ax.barh(y, harm, color=HARM, height=0.34, zorder=2)
        ax.barh(y, rescue, color=TEAL, height=0.34, zorder=2)
        ax.errorbar(
            harm,
            y,
            xerr=[[abs(harm + row["harm_ci_high"])], [abs(-row["harm_ci_low"] - harm)]],
            fmt="none",
            ecolor=TEXT,
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )
        ax.errorbar(
            rescue,
            y,
            xerr=[[rescue - row["rescue_ci_low"]], [row["rescue_ci_high"] - rescue]],
            fmt="none",
            ecolor=TEXT,
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )
        ax.text(-3.40, y, f"{int(row['harmed'])}", ha="right", va="center", color=HARM, fontsize=6.4, fontweight="bold")
        rescue_label_x = 5.35 if y == 0 else 6.20
        ax.text(rescue_label_x, y, f"{int(row['rescued'])}", ha="left", va="center", color=TEAL, fontsize=6.4, fontweight="bold")
        preserved_y = y + 0.50 if y == 0 else y - 0.44
        preserved_va = "bottom" if y == 0 else "top"
        ax.text(
            1.55,
            preserved_y,
            f"n={int(row['preserved']):,} preserved  |  net {row['net_top1_gain_pp']:+.2f} pp",
            ha="center",
            va=preserved_va,
            color=color,
            fontsize=5.9,
        )

    ax.axvline(0, color=TEXT, lw=0.75)
    ax.set_yticks(y_positions, ["Global", "Peak", "Raw", "DreaMS"])
    ax.set_xlim(-3.75, 6.85)
    ax.set_ylim(-0.92, 6.72)
    ax.set_xlabel("Fraction of all queries (%)")
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{abs(value):.0f}"))
    ax.text(-1.75, 6.24, "harmed", color=HARM, fontsize=6.2, ha="center", va="bottom")
    ax.text(3.15, 6.24, "rescued", color=TEAL, fontsize=6.2, ha="center", va="bottom")
    clean_axis(ax, "x")
    ax.set_title("b  Fusion changes individual decisions", loc="left", fontsize=8.5, fontweight="bold", color=TEXT, pad=5)


def plot_strata(ax: plt.Axes, strata: pd.DataFrame) -> None:
    level_order = [
        "1-9 candidates",
        "10-49 candidates",
        "50 candidates",
        "50-99 peaks",
        "100-149 peaks",
        "150 peaks",
        "<=25 heavy atoms",
        "26-40 heavy atoms",
        ">40 heavy atoms",
    ]
    positions = {level: len(level_order) - index - 1 for index, level in enumerate(level_order)}
    offsets = {"UltraMS": 0.13, "DreaMS": -0.13}
    markers = {"UltraMS": "o", "DreaMS": "D"}
    colors = {"UltraMS": ULTRA, "DreaMS": DREAMS}
    for _, row in strata.iterrows():
        y = positions[row["stratum"]] + offsets[row["model"]]
        ax.errorbar(
            row["net_top1_gain_pp"],
            y,
            xerr=[[row["net_top1_gain_pp"] - row["net_gain_ci_low"]], [row["net_gain_ci_high"] - row["net_top1_gain_pp"]]],
            fmt=markers[row["model"]],
            color=colors[row["model"]],
            ecolor=colors[row["model"]],
            ms=3.5,
            elinewidth=0.9,
            capsize=1.8,
            markeredgewidth=0,
            zorder=3,
        )
    counts = strata.loc[strata["model"] == "UltraMS"].set_index("stratum")["n_queries"]
    ax.set_yticks(
        [positions[level] for level in level_order],
        [f"{level.replace('candidates', 'cand.')}  ({int(counts[level]):,})" for level in level_order],
    )
    ax.tick_params(axis="y", labelsize=6.1)
    ax.axvline(0, color=TEXT, lw=0.75)
    ax.axhline(5.5, color=GRID, lw=0.7)
    ax.axhline(2.5, color=GRID, lw=0.7)
    ax.set_xlim(-2.5, 8.5)
    ax.set_xlabel("Fusion-attributable top-1 gain (percentage points)")
    handles = [
        mpl.lines.Line2D([], [], marker="o", color=ULTRA, lw=0, label="UltraMS", markersize=4.1),
        mpl.lines.Line2D([], [], marker="D", color=DREAMS, lw=0, label="DreaMS", markersize=3.9),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0,
    )
    clean_axis(ax, "x")
    ax.set_title("c  Gain by stratum", loc="left", fontsize=8.5, fontweight="bold", color=TEXT, pad=5)


def plot_massspecgym(ax: plt.Axes, metrics: pd.DataFrame, task: str, show_y: bool) -> None:
    subset = metrics.loc[metrics["task"] == task].set_index("view")
    names = {
        "UltraMS": ["UltraMS global", "UltraMS projected peak", "UltraMS global + peak fusion"],
        "DreaMS": ["DreaMS global", "DreaMS peak", "DreaMS global + peak fusion"],
    }
    x = np.arange(3)
    for model, color, marker in [("UltraMS", ULTRA, "o"), ("DreaMS", DREAMS, "D")]:
        values = [float(subset.loc[name, "top1_percent"]) for name in names[model]]
        ax.plot(x, values, color=color, lw=1.25, marker=marker, ms=3.8, markeredgewidth=0, zorder=3)
        ax.text(2.08, values[-1], f"{model}  {values[-1]:.2f}%", color=color, va="center", ha="left", fontsize=6.3, fontweight="bold")
    ax.set_xticks(x, ["Global", "Peak", "Fusion"])
    ax.set_xlim(-0.12, 3.00)
    ax.set_ylim(10.4, 17.7)
    ax.set_yticks([11, 13, 15, 17])
    if show_y:
        ax.set_ylabel("Top-1 (%)")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    clean_axis(ax, "y")
    ax.set_title(f"{task.capitalize()} candidates  (n=16,440)", loc="left", fontsize=7.4, fontweight="bold", color=TEXT, pad=3)


def plot_replication(fig: plt.Figure, slot, metrics: pd.DataFrame) -> None:
    grid = slot.subgridspec(2, 2, height_ratios=[0.48, 1.0], hspace=0.02, wspace=0.22)
    heading_ax = fig.add_subplot(grid[0, :])
    panel_heading(
        heading_ax,
        "d",
        "The same two-view gain replicates on MassSpecGym",
    )
    mass_ax = fig.add_subplot(grid[1, 0])
    formula_ax = fig.add_subplot(grid[1, 1], sharey=mass_ax)
    plot_massspecgym(mass_ax, metrics, "mass", True)
    plot_massspecgym(formula_ax, metrics, "formula", False)


def molecule_image(smiles: str, size: tuple[int, int] = (1000, 300)) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Cannot render SMILES: {smiles}")
    image = Draw.MolToImage(mol, size=size, fitImage=True)
    rgba = np.asarray(image.convert("RGBA")).copy()
    white = np.all(rgba[:, :, :3] > 248, axis=2)
    rgba[white, 3] = 0
    return rgba


def plot_case_spectrum(ax: plt.Axes, peaks: pd.DataFrame, color: str, show_y: bool, show_x: bool) -> None:
    ax.vlines(peaks["mz"], 0, peaks["relative_intensity"], color="#CBD2D7", lw=0.42)
    strongest = peaks.nlargest(8, "relative_intensity")
    ax.vlines(strongest["mz"], 0, strongest["relative_intensity"], color=color, lw=0.9)
    xmax = max(float(peaks["precursor_mz"].iloc[0]) * 1.02, float(peaks["mz"].max()) * 1.02)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 1] if show_y else [])
    ax.set_xticks([round(xmax / 2)] if show_x else [])
    if show_y:
        ax.set_ylabel("Rel. intensity", labelpad=2)
    if show_x:
        ax.set_xlabel("m/z", labelpad=1)
    clean_axis(ax)
    ax.spines["left"].set_visible(show_y)
    ax.tick_params(axis="both", pad=1)


def plot_case_rank(ax: plt.Axes, case: pd.Series, color: str, show_y: bool, show_x: bool) -> None:
    rank_columns = ["global_rank", "peak_rank", "rawproj_rank", "final_rank"]
    ranks = np.array([case[column] for column in rank_columns], dtype=float)
    x = np.arange(4)
    ax.plot(x, ranks, color=color, lw=1.25, zorder=2)
    ax.scatter(
        x,
        ranks,
        s=19,
        c=[ULTRA, TEAL, GOLD, color],
        edgecolors=WHITE,
        linewidths=0.5,
        zorder=3,
    )
    for xpos, value in zip(x, ranks):
        if value <= 1:
            ax.text(xpos, value + 0.62, f"#{int(value)}", ha="center", va="top", fontsize=6.1, color=TEXT)
        else:
            ax.text(xpos, value - 0.30, f"#{int(value)}", ha="center", va="bottom", fontsize=6.1, color=TEXT)
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(11.2, 0.45)
    ax.set_yticks([1, 5, 10] if show_y else [])
    ax.set_xticks(x, ["G", "P", "R", "F"] if show_x else [])
    if show_y:
        ax.set_ylabel("Target rank", labelpad=2)
    if show_x:
        ax.set_xlabel("Spectrum view", labelpad=1)
    clean_axis(ax, "y")
    ax.spines["left"].set_visible(show_y)
    ax.tick_params(axis="both", pad=1)


def plot_case_structure(ax: plt.Axes, case: pd.Series, formula: str, color: str) -> None:
    ax.imshow(molecule_image(case["smiles"]), interpolation="lanczos")
    ax.set_axis_off()
    ax.text(0.01, 0.98, formula, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=TEXT, fontweight="bold")
    ax.plot([0.01, 0.18], [0.08, 0.08], transform=ax.transAxes, color=color, lw=1.4, clip_on=False)


def plot_cases(fig: plt.Figure, slot, cases: pd.DataFrame, spectra: pd.DataFrame) -> None:
    grid = slot.subgridspec(
        4,
        3,
        height_ratios=[0.78, 1.0, 1.0, 1.0],
        width_ratios=[0.27, 0.29, 0.44],
        hspace=0.10,
        wspace=0.14,
    )
    heading_ax = fig.add_subplot(grid[0, :])
    panel_heading(
        heading_ax,
        "e",
        "Fixed-rule molecular outcomes",
    )
    heading_ax.text(
        1.0,
        0.96,
        "fixed median rule  |  50 candidates  |  150 retained peaks  |  [M+H]+",
        transform=heading_ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=MID,
    )

    lane_specs = [
        ("rescued", "Rescued", TEAL),
        ("preserved", "Preserved", ULTRA),
        ("harmed", "Harmed", HARM),
    ]
    indexed = cases.set_index("case_lane")
    for row_index, (lane, label, color) in enumerate(lane_specs, start=1):
        case = indexed.loc[lane]
        peaks = spectra.loc[spectra["case_lane"] == lane].sort_values("mz")
        spectrum_ax = fig.add_subplot(grid[row_index, 0])
        rank_ax = fig.add_subplot(grid[row_index, 1])
        structure_ax = fig.add_subplot(grid[row_index, 2])
        show_x = row_index == 3
        show_y = row_index == 2
        plot_case_spectrum(spectrum_ax, peaks, color, show_y, show_x)
        plot_case_rank(rank_ax, case, color, show_y, show_x)
        plot_case_structure(structure_ax, case, str(peaks["formula"].iloc[0]), color)
        if row_index == 1:
            spectrum_ax.set_title("Measured spectrum", loc="center", fontsize=6.5, color=MID, pad=6)
            rank_ax.set_title("Source ranks", loc="center", fontsize=6.5, color=MID, pad=6)
            structure_ax.set_title("Target molecule", loc="center", fontsize=6.5, color=MID, pad=6)
        spectrum_ax.text(
            0.01,
            0.93,
            label,
            transform=spectrum_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=color,
        )
        spectrum_ax.annotate(
            "",
            xy=(1.09, 0.50),
            xytext=(1.01, 0.50),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=GRID, lw=0.8),
            annotation_clip=False,
        )
        rank_ax.annotate(
            "",
            xy=(1.07, 0.50),
            xytext=(1.00, 0.50),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=GRID, lw=0.8),
            annotation_clip=False,
        )
        if row_index < 3:
            for ax in [spectrum_ax, rank_ax, structure_ax]:
                ax.plot([0, 1], [-0.10, -0.10], transform=ax.transAxes, color=PALE, lw=0.7, clip_on=False)


def write_caption(stats: pd.DataFrame) -> None:
    primary = stats.loc[stats["comparison"] == "global_to_final"].set_index("model")
    ultra = primary.loc["UltraMS"]
    dreams = primary.loc["DreaMS"]
    text = (
        "Supplementary Figure S7 | Global and peak views assemble a molecular decision. "
        "NPLIB1 spectrum-to-molecule retrieval used 3,699 held-out ms2peaks_min50 fold-query evaluations "
        "from 3,352 unique spectra and three molecule-disjoint folds. (a) Paired target-rank paths from each "
        "UltraMS source view to the same validation-selected final fusion. The sources are shown independently, "
        "not as sequential computation stages. Folds 1 and 3 used equal global and raw-projection weights; fold 2 "
        "used equal global and projected-peak weights. (b) Exact source-to-final top-1 transitions. UltraMS "
        f"global-to-final rescued {int(ultra['rescued'])}, preserved {int(ultra['preserved'])} and harmed "
        f"{int(ultra['harmed'])} queries (net {ultra['net_top1_gain_pp']:+.2f} percentage points); DreaMS "
        f"global-to-fusion rescued {int(dreams['rescued'])}, preserved {int(dreams['preserved'])} and harmed "
        f"{int(dreams['harmed'])} queries (net {dreams['net_top1_gain_pp']:+.2f} percentage points). Error bars "
        "are 95% target-molecule cluster-bootstrap intervals. (c) Net top-1 change across strata defined without "
        "model outputs. (d) Global-plus-peak replication on 16,440 MassSpecGym test queries per candidate task; "
        "raw projection was not part of this protocol. (e) Deterministic median-transition cases from the shared "
        "hard subset of queries with 50 candidates and 150 retained peaks. Cases illustrate the population-defined "
        "outcome lanes and are not independent statistical evidence."
    )
    (OUTPUT_DIR / "s7_caption_draft.txt").write_text(text)


def main() -> None:
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(SOURCE_DIR / "nplib1_query_decomposition.csv")
    metrics = pd.read_csv(SOURCE_DIR / "nplib1_view_metrics.csv")
    outcomes = pd.read_csv(SOURCE_DIR / "nplib1_outcome_stats.csv")
    strata = pd.read_csv(SOURCE_DIR / "nplib1_stratum_stats.csv")
    cases = pd.read_csv(SOURCE_DIR / "nplib1_case_selection.csv")
    spectra = pd.read_csv(SOURCE_DIR / "nplib1_case_spectra.csv")
    mass_metrics = pd.read_csv(SOURCE_DIR / "massspecgym_view_metrics.csv")

    # 179 mm wide; compact 196 mm height at final print size.
    fig = plt.figure(figsize=(7.05, 7.72), constrained_layout=False)
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.08, 2.06, 1.28, 2.05],
        width_ratios=[0.43, 0.57],
        left=0.095,
        right=0.988,
        bottom=0.040,
        top=0.988,
        hspace=0.30,
        wspace=0.48,
    )

    plot_rank_rivers(fig, outer[0, :], queries, metrics)
    ledger_ax = fig.add_subplot(outer[1, 0])
    strata_ax = fig.add_subplot(outer[1, 1])
    plot_transition_ledger(ledger_ax, outcomes)
    plot_strata(strata_ax, strata)
    plot_replication(fig, outer[2, :], mass_metrics)
    plot_cases(fig, outer[3, :], cases, spectra)

    for extension in ["png", "pdf", "svg"]:
        path = OUTPUT_DIR / f"s7_source_fusion_decision.{extension}"
        fig.savefig(path, dpi=600 if extension == "png" else None, bbox_inches=None, pad_inches=0)
    write_caption(outcomes)
    plt.close(fig)
    print(f"Figure outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
