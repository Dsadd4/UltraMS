#!/usr/bin/env python3
"""Add the formal baselines to the existing Figure 3b visual grammar.

The panel types are intentionally unchanged: absolute Top-1 bars above
paired-bootstrap density ribbons, plus the same signed purity heatmap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import gaussian_kde


SCRIPT = Path(__file__).resolve()
PANEL = SCRIPT.parents[1]
SOURCE = PANEL / "source_data" / "formal_final_six_v1"
OUTPUT = PANEL / "output"
MAIN_OUTPUT = PANEL.parents[1] / "output"

MODELS = [
    "DeepSets", "Fourier projection", "UltraMS codebook", "Linear",
    "DreaMS", "UltraMS",
]
DISPLAY = {
    "Fourier projection": "Fourier",
    "UltraMS codebook": "Codebook",
}
COLORS = {
    "UltraMS": "#3965BB",
    "DreaMS": "#E9A0AA",
    "Linear": "#A7A7A7",
    "DeepSets": "#C48A4A",
    "Fourier projection": "#8767AD",
    "UltraMS codebook": "#5AA69A",
}
INK = "#25282B"
MID = "#737B82"
LIGHT = "#D9DEE2"
MM = 1 / 25.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def primary_groups(groups: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = groups[groups.method == method].copy()
    if frame.seed.notna().any():
        frame = frame[frame.seed == 0].copy()
    if len(frame) != 15_541 or frame.group_id.nunique() != 15_541:
        raise ValueError(f"expected 15,541 frozen groups for {method}; got {len(frame)}")
    return frame


def load_formal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marker = json.loads((SOURCE / "AUDIT.json").read_text())
    if (
        marker.get("status") != "complete"
        or marker.get("primary_model_seed") != 0
    ):
        raise RuntimeError("audited final six-method Figure 3b marker is required")
    summary = pd.read_csv(SOURCE / "method_summary.csv")
    bootstrap = pd.read_csv(SOURCE / "paired_group_bootstrap.csv")
    groups = pd.read_csv(SOURCE / "per_group_metrics_all_runs.csv.gz")
    if not set(MODELS).issubset(set(summary.method)) or not set(MODELS).issubset(set(bootstrap.method)):
        raise ValueError("formal source summary is missing a displayed method")
    return summary, bootstrap, groups


def performance_table(summary: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in MODELS:
        frame = primary_groups(groups, method)
        for stratum, subset in (
            ("all_grouped", frame),
            ("pure_groups", frame[frame.purity == 1.0]),
            ("impure_groups", frame[frame.purity < 1.0]),
        ):
            denominator = float(subset.group_size.sum())
            single = float(subset.single_correct_count.sum() / denominator)
            consensus = float(subset.consensus_correct_count.sum() / denominator)
            records.append(
                {
                    "model": method,
                    "stratum": stratum,
                    "n_groups": int(len(subset)),
                    "n_spectra": int(denominator),
                    "single_top1": single,
                    "consensus_top1": consensus,
                    "top1_delta": consensus - single,
                    "primary_model_seed": 0 if method not in ("UltraMS", "DreaMS") else "frozen",
                }
            )
        expected = summary[summary.method == method].iloc[0]
        observed = records[-3]
        for observed_key, expected_key in (
            ("single_top1", "single_top1_primary"),
            ("consensus_top1", "consensus_top1_primary"),
            ("top1_delta", "top1_delta_primary"),
        ):
            if not np.isclose(observed[observed_key], expected[expected_key], atol=1e-12, rtol=0):
                raise ValueError(f"formal aggregate mismatch for {method}/{observed_key}")
    return pd.DataFrame(records)


def replay_bootstrap(bootstrap: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for method in MODELS:
        frame = primary_groups(groups, method)
        size = frame.group_size.to_numpy(dtype=np.float64)
        delta = (
            frame.consensus_correct_count.to_numpy(dtype=np.float64)
            - frame.single_correct_count.to_numpy(dtype=np.float64)
        )
        row = bootstrap[(bootstrap.method == method) & (bootstrap.metric == "top1_delta")].iloc[0]
        n_bootstrap = int(row.n_bootstrap)
        rng = np.random.default_rng(int(row.bootstrap_seed))
        values = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            sampled = rng.integers(0, len(frame), size=len(frame))
            values[index] = delta[sampled].sum() / size[sampled].sum()
        observed = np.quantile(values, [0.025, 0.975])
        expected = np.asarray([row.ci_low, row.ci_high], dtype=float)
        if not np.allclose(observed, expected, atol=1e-12, rtol=0):
            raise ValueError(f"bootstrap replay mismatch for {method}: {observed} != {expected}")
        records.append(
            pd.DataFrame(
                {
                    "model": method,
                    "bootstrap_index": np.arange(n_bootstrap),
                    "top1_gain_pp": 100 * values,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def draw_bars(ax: plt.Axes, performance: pd.DataFrame) -> None:
    overall = performance[performance.stratum == "all_grouped"].set_index("model")
    width = 0.125
    offsets = width * (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2)
    for metric, center in (("single_top1", 0.0), ("consensus_top1", 1.0)):
        for method, offset in zip(MODELS, offsets):
            value = 100 * float(overall.loc[method, metric])
            position = center + offset
            ax.bar(
                position, value, width=width, color=COLORS[method],
                edgecolor="white", linewidth=0.45, zorder=3,
            )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="none", label=DISPLAY.get(m, m))
        for m in MODELS
    ]
    ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        ncol=3, handlelength=0.75, handletextpad=0.25, columnspacing=0.5,
        borderaxespad=0, fontsize=6.0,
    )
    ax.set_ylabel("Top-1 accuracy (%)", labelpad=2)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 32.2)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xticks([0, 1], ["Single\nspectrum", "Grouped\nspectra"])
    ax.yaxis.grid(True, color=LIGHT, lw=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(axis="y", colors=INK)
    ax.tick_params(axis="x", length=0, pad=2)


def draw_ridges(ax: plt.Axes, bootstrap: pd.DataFrame, draws: pd.DataFrame) -> None:
    low = 100 * float(bootstrap[bootstrap.metric == "top1_delta"].ci_low.min())
    high = 100 * float(bootstrap[bootstrap.metric == "top1_delta"].ci_high.max())
    x_grid = np.linspace(low - 0.18, high + 0.18, 500)
    ridge_y = dict(zip(MODELS, np.arange(len(MODELS) - 1, -1, -1, dtype=float)))
    for method in MODELS:
        values = draws.loc[draws.model == method, "top1_gain_pp"].to_numpy(dtype=float)
        row = bootstrap[(bootstrap.method == method) & (bootstrap.metric == "top1_delta")].iloc[0]
        estimate, ci_low, ci_high = 100 * np.asarray([row.estimate, row.ci_low, row.ci_high])
        density = gaussian_kde(values)(x_grid)
        density = 0.44 * density / density.max()
        base = ridge_y[method]
        color = COLORS[method]
        ax.fill_between(x_grid, base, base + density, color=color, alpha=0.18, linewidth=0)
        central = (x_grid >= ci_low) & (x_grid <= ci_high)
        ax.fill_between(
            x_grid[central], base, base + density[central], color=color, alpha=0.72, linewidth=0,
        )
        ax.plot(x_grid, base + density, color=color, lw=0.7)
        ax.scatter(
            estimate, base + 0.03, marker="D", s=8.5, color=color,
            edgecolor="white", linewidth=0.35, zorder=4,
        )
        ax.text(
            x_grid[-1], base + 0.53, f"{estimate:.2f} [{ci_low:.2f}, {ci_high:.2f}]",
            color=color, ha="right", va="center", fontsize=6.0,
        )
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(-0.08, len(MODELS) - 0.20)
    ticks = np.linspace(np.ceil(x_grid[0] * 2) / 2, np.floor(x_grid[-1] * 2) / 2, 6)
    ax.set_xticks(ticks)
    ax.set_yticks([ridge_y[m] for m in MODELS], [DISPLAY.get(m, m) for m in MODELS])
    for tick, method in zip(ax.get_yticklabels(), MODELS):
        tick.set_color(COLORS[method])
        tick.set_fontweight("normal")
    ax.set_xlabel("Paired Top-1 gain (pp)", labelpad=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", colors=INK)


def effect_matrix(performance: pd.DataFrame) -> np.ndarray:
    matrix = np.empty((2, len(MODELS)), dtype=float)
    for row_index, stratum in enumerate(("pure_groups", "impure_groups")):
        for column_index, method in enumerate(MODELS):
            row = performance[(performance.model == method) & (performance.stratum == stratum)].iloc[0]
            matrix[row_index, column_index] = 100 * float(row.top1_delta)
    return matrix


def draw_heatmap(
    ax: plt.Axes,
    performance: pd.DataFrame,
    fig: plt.Figure,
    *,
    transpose: bool = False,
) -> None:
    raw_matrix = effect_matrix(performance)
    matrix = raw_matrix.T if transpose else raw_matrix
    cmap = LinearSegmentedColormap.from_list("signed_effect", ["#4D4A86", "#EEEFF2", "#D59A32"])
    norm = TwoSlopeNorm(vmin=-15.0, vcenter=0.0, vmax=3.0)
    # Draw the heatmap as vector rectangles so SVG/PDF stay fully editable.
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    if transpose:
        ax.set_xticks([0, 1], ["Same-molecule\ngroups", "Mixed-molecule\ngroups"])
        ax.tick_params(axis="x", top=False, bottom=False, pad=2)
        ax.get_xticklabels()[0].set_ha("right")
        ax.get_xticklabels()[1].set_ha("left")
        ax.set_yticks(np.arange(len(MODELS)), [DISPLAY.get(m, m) for m in MODELS])
        ax.yaxis.tick_right()
        ax.tick_params(
            axis="y", left=False, right=False, labelleft=False, labelright=True, pad=2,
        )
        for tick, method in zip(ax.get_yticklabels(), MODELS):
            tick.set_color(COLORS[method])
            tick.set_fontweight("normal")
            tick.set_fontsize(6.0)
    else:
        ax.set_xticks(np.arange(len(MODELS)), [DISPLAY.get(m, m) for m in MODELS])
        ax.tick_params(axis="x", top=False, bottom=False, pad=2)
        for tick, method in zip(ax.get_xticklabels(), MODELS):
            tick.set_color(COLORS[method])
            tick.set_fontweight("normal")
            tick.set_fontsize(6.0)
        ax.set_yticks([0, 1], ["Same\nmolecule", "Different\nmolecules"])
        ax.tick_params(axis="y", left=False, pad=3)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value < -8.0 or value > 1.7 else INK
            ax.text(
                column, row, f"{value:+.2f}", ha="center", va="center",
                color=text_color, fontsize=6.0, fontweight="normal",
            )
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(
        image, ax=ax, orientation="horizontal",
        fraction=0.080 if transpose else 0.095,
        pad=0.14 if transpose else 0.19,
        aspect=10 if transpose else 18,
        ticks=[-15, 0, 3],
    )
    cbar.set_label("Top-1 change (pp)", labelpad=1.5)
    cbar.solids.set_rasterized(False)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2.0, width=0.5, pad=1.5, labelsize=6.0)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, object] = {}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def main() -> None:
    set_style()
    summary, bootstrap, groups = load_formal()
    performance = performance_table(summary, groups)
    draws = replay_bootstrap(bootstrap, groups)

    fig_left = plt.figure(figsize=(50 * MM, 58 * MM))
    grid = fig_left.add_gridspec(2, 1, height_ratios=(1.45, 1.3), hspace=0.48)
    draw_bars(fig_left.add_subplot(grid[0]), performance)
    draw_ridges(fig_left.add_subplot(grid[1]), bootstrap, draws)
    fig_left.subplots_adjust(left=0.25, right=0.985, bottom=0.14, top=0.88)
    save(fig_left, OUTPUT / "fig3b_1")

    fig_heat, ax_heat = plt.subplots(figsize=(70 * MM, 43 * MM))
    draw_heatmap(ax_heat, performance, fig_heat)
    fig_heat.subplots_adjust(left=0.20, right=0.985, bottom=0.18, top=0.80)
    save(fig_heat, OUTPUT / "fig3b_3")

    # Half of the 205-mm manuscript width is 102.5 mm.  Keep the two evidence
    # blocks side by side and transpose the right heatmap to six rows by two
    # columns, using height instead of extending the figure horizontally.
    combined = plt.figure(figsize=(102.5 * MM, 68 * MM))
    outer = combined.add_gridspec(1, 2, width_ratios=(1.45, 0.62), wspace=0.15)
    left = outer[0].subgridspec(2, 1, height_ratios=(1.45, 1.3), hspace=0.48)
    draw_bars(combined.add_subplot(left[0]), performance)
    draw_ridges(combined.add_subplot(left[1]), bootstrap, draws)
    draw_heatmap(combined.add_subplot(outer[1]), performance, combined, transpose=True)
    combined.subplots_adjust(left=0.13, right=0.87, bottom=0.14, top=0.88)
    save(combined, OUTPUT / "fig3b")
    MAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        (MAIN_OUTPUT / f"fig3b.{extension}").write_bytes(
            (OUTPUT / f"fig3b.{extension}").read_bytes()
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    performance.to_csv(OUTPUT / "fig3b_plot_data.csv", index=False)
    bootstrap.to_csv(OUTPUT / "fig3b_bootstrap_data.csv", index=False)
    draws.to_csv(OUTPUT / "fig3b_bootstrap_draws.csv.gz", index=False)
    audit = {
        "status": "complete",
        "visual_types_preserved": ["grouped Top-1 bars", "paired-bootstrap density ribbons", "signed purity heatmap"],
        "primary_model_seed": 0,
        "methods": MODELS,
        "excluded_from_plot": ["Spec2Vec"],
        "exclusion_scope": "display only; formal result artifacts are retained",
        "n_groups": 15_541,
        "sources": {
            path.name: sha256(path)
            for path in (
                SOURCE / "AUDIT.json",
                SOURCE / "method_summary.csv",
                SOURCE / "paired_group_bootstrap.csv",
                SOURCE / "per_group_metrics_all_runs.csv.gz",
            )
        },
    }
    (OUTPUT / "fig3b_formal_plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
