#!/usr/bin/env python3
"""Render the S9 label-blind consensus figure from packaged source data."""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "source_data"
STATS = PACKAGE / "statistics"
OUTPUT = PACKAGE / "outputs"

BLUE = "#3965BB"
ORANGE = "#E9A0AA"
TEAL = "#1B9E8A"
GOLD = "#C9982E"
INK = "#292D32"
MID = "#7B858E"
LIGHT = "#DDE3E7"
PALE = "#FAFAF9"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def molecule_rgba(smiles: str, size: tuple[int, int] = (360, 210)) -> np.ndarray | None:
    try:
        from PIL import Image
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    image = Draw.MolToImage(mol, size=size, kekulize=True)
    rgb = np.asarray(image.convert("RGB"))
    alpha = np.where(rgb.min(axis=2) > 245, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.045, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, weight="bold", fontsize=9.2, va="bottom", color=INK)


def clean_axis(ax: plt.Axes, grid_y: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    if grid_y:
        ax.grid(axis="y", color=LIGHT, lw=0.55, zorder=0)
    ax.tick_params(colors=INK)


def draw_spectrum(ax: plt.Axes, frame: pd.DataFrame, color: str, linewidth: float = 0.55) -> None:
    ax.vlines(frame["mz"], 0, frame["relative_intensity"], color=color, lw=linewidth)
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, max(100.0, frame["mz"].max() * 1.02))
    ax.axis("off")


def main() -> None:
    set_style()
    audit = json.loads((PACKAGE / "audit" / "analysis_audit.json").read_text())
    if audit["status"] != "promote_as_positive_supplement":
        raise SystemExit("S9 failed the promotion gate; no positive figure was generated.")
    assignments = pd.read_csv(SOURCE / "group_assignments_test.csv.gz")
    group_purity = pd.read_csv(SOURCE / "group_purity_test.csv.gz")
    trajectories = pd.read_csv(SOURCE / "rank_trajectories_test.csv.gz")
    ledger = pd.read_csv(SOURCE / "group_candidate_vote_ledger_test.csv.gz")
    peaks = pd.read_csv(SOURCE / "example_spectrum_peaks.csv.gz")
    example = pd.read_csv(SOURCE / "example_group_manifest.csv")
    performance = pd.read_csv(STATS / "model_performance.csv")
    bootstrap = pd.read_csv(STATS / "paired_group_bootstrap.csv")
    oracle = pd.read_csv(STATS / "oracle_upper_bound_context.csv")
    grouping_summary = pd.read_csv(STATS / "grouping_summary.csv")

    group_id = str(example["group_id"].iloc[0])
    example_traj = trajectories[trajectories["group_id"] == group_id].copy()
    example_ledger = ledger[ledger["group_id"] == group_id].copy()
    example_purity = group_purity[group_purity["group_id"] == group_id].iloc[0]
    target = str(example_purity["majority_hidden_smiles"])
    test_summary = grouping_summary[grouping_summary["split"] == "test"].iloc[0]

    candidate_pool = [target]
    for model in ["UltraMS", "DreaMS"]:
        subset = example_ledger[example_ledger["model"] == model].sort_values("candidate_rank")
        candidate_pool.extend(subset["candidate_smiles"].head(5).astype(str).tolist())
    candidates = []
    for candidate in candidate_pool:
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) == 4:
            break
    candidate_ids = {candidate: chr(ord("A") + idx) for idx, candidate in enumerate(candidates)}

    fig = plt.figure(figsize=(7.15, 5.45), facecolor="white")
    outer = GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.56, 2.05, 1.33], width_ratios=[1.46, 0.88],
        hspace=0.40, wspace=0.27,
        left=0.070, right=0.985, top=0.925, bottom=0.105,
    )
    fig.text(0.070, 0.977, "Label-blind spectral consensus", fontsize=9.2, weight="bold", color=INK, va="top")

    # a. A compact, left-to-right grouping path.
    ax_gate = fig.add_subplot(outer[0, :])
    ax_gate.set_xlim(0, 1)
    ax_gate.set_ylim(0, 1)
    ax_gate.axis("off")
    add_panel_label(ax_gate, "a", x=-0.045, y=0.91)
    ax_gate.text(0.00, 0.91, "Deployable metadata links replicate spectra", weight="bold", color=INK, va="bottom")
    shown_members = sorted(peaks["member_order"].unique())[:4]
    x_positions = np.linspace(0.015, 0.205, len(shown_members))
    for xpos, member_order in zip(x_positions, shown_members):
        inset = ax_gate.inset_axes([xpos, 0.17, 0.047, 0.48])
        member_peaks = peaks[peaks["member_order"] == member_order]
        draw_spectrum(inset, member_peaks, TEAL, linewidth=0.45)
    ax_gate.text(0.115, 0.02, "replicate MS/MS", ha="center", color=MID, fontsize=6.0)
    ax_gate.annotate("", xy=(0.285, 0.42), xytext=(0.235, 0.42), arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": MID})
    gate_labels = [
        (0.34, "adduct"),
        (0.47, "precursor m/z"),
        (0.61, "retention time"),
        (0.76, f"spectral cosine ≥ {audit['selected_similarity_threshold']:.2f}"),
    ]
    for idx, (xpos, label) in enumerate(gate_labels):
        ax_gate.scatter([xpos], [0.42], s=47, facecolor="white", edgecolor=TEAL, linewidth=1.0, zorder=3)
        ax_gate.text(xpos, 0.08, label, ha="center", fontsize=5.9, color=INK)
        if idx < len(gate_labels) - 1:
            ax_gate.plot([xpos + 0.023, gate_labels[idx + 1][0] - 0.023], [0.42, 0.42], color=LIGHT, lw=1.0)
    ax_gate.annotate("", xy=(0.88, 0.42), xytext=(0.80, 0.42), arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": MID})
    ax_gate.scatter([0.91], [0.42], s=104, facecolor=PALE, edgecolor=INK, linewidth=0.75)
    ax_gate.text(0.91, 0.42, f"n={int(example_purity.group_size)}", ha="center", va="center", fontsize=6.1, weight="bold")
    ax_gate.text(0.985, 0.82,
                 f"{int(test_summary.n_grouped_spectra):,} spectra  |  {int(test_summary.n_groups):,} groups  |  "
                 f"{100*test_summary.spectrum_weighted_purity:.1f}% purity",
                 ha="right", color=TEAL, weight="bold", fontsize=6.4)

    # b. Candidate structures and the eight per-spectrum votes.
    ledger_spec = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1, 0], height_ratios=[0.78, 1.0, 1.0], hspace=0.16)
    ax_struct = fig.add_subplot(ledger_spec[0])
    ax_struct.set_xlim(-0.55, max(3.55, len(candidates) - 0.45))
    ax_struct.set_ylim(0, 1)
    ax_struct.axis("off")
    add_panel_label(ax_struct, "b", x=-0.055, y=1.02)
    ax_struct.text(-0.48, 1.02, "Each spectrum casts one molecular vote", weight="bold", color=INK, va="bottom")
    for col_idx, candidate in enumerate(candidates):
        image = molecule_rgba(candidate, size=(330, 175))
        if image is not None:
            inset = ax_struct.inset_axes([0.018 + col_idx * 0.245, 0.17, 0.215, 0.70])
            inset.imshow(image)
            inset.axis("off")
        label = candidate_ids[candidate]
        color = TEAL if candidate == target else MID
        ax_struct.text(col_idx, 0.10, f"{label}{'  target' if candidate == target else ''}", ha="center", color=color,
                       weight="bold" if candidate == target else "normal", fontsize=6.8)

    max_members = int(example_purity.group_size)
    vote_axes = []
    for row_idx, model in enumerate(["UltraMS", "DreaMS"], start=1):
        ax = fig.add_subplot(ledger_spec[row_idx])
        vote_axes.append(ax)
        color = BLUE if model == "UltraMS" else ORANGE
        model_traj = example_traj[example_traj["model"] == model].sort_values("prefix_size")
        ax.set_xlim(-0.65, max(3.48, len(candidates) - 0.52))
        ax.set_ylim(max_members + 0.55, 0.45)
        ax.axvspan(-0.28, 0.28, color="#EAF5F2", zorder=-3)
        for y in range(1, max_members + 1):
            ax.axhline(y, color="#EDF0F2", lw=0.5, zorder=0)
        for col_idx in range(len(candidates)):
            ax.axvline(col_idx, color="#EDF0F2", lw=0.5, zorder=0)
        for row in model_traj.itertuples(index=False):
            if row.added_top1_smiles in candidate_ids:
                col = candidates.index(row.added_top1_smiles)
                ax.scatter(col, row.prefix_size, s=24, marker="o" if model == "UltraMS" else "D",
                           facecolor=color, edgecolor="white", linewidth=0.45, zorder=3)
            else:
                ax.scatter(len(candidates) - 0.10, row.prefix_size, s=12, facecolor="white", edgecolor=LIGHT, linewidth=0.6)
        ce_values = model_traj["added_collision_energy"].tolist()
        ax.set_yticks(range(1, max_members + 1), [f"{value:g}" if np.isfinite(value) else "-" for value in ce_values])
        ax.set_xticks(range(len(candidates)), [candidate_ids[candidate] for candidate in candidates] if row_idx == 2 else [])
        ax.text(0.005, 1.01, model, transform=ax.transAxes, color=color, weight="bold", va="bottom", fontsize=7.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=1.5)
    vote_axes[1].set_xlabel("Candidate receiving the top-1 vote", labelpad=2)
    fig.text(0.021, 0.525, "collision energy (eV)", rotation=90, ha="center", va="center", fontsize=6.6, color=INK)

    # c. The same votes are accumulated into one rank trajectory.
    ax_rank = fig.add_subplot(outer[1, 1])
    add_panel_label(ax_rank, "c", x=-0.12, y=1.04)
    ax_rank.set_title("Votes lift the target to rank 1", loc="left", weight="bold", color=INK, pad=2)
    for model, color, marker in [("UltraMS", BLUE, "o"), ("DreaMS", ORANGE, "D")]:
        frame = example_traj[example_traj["model"] == model].sort_values("prefix_size")
        ax_rank.plot(frame["prefix_size"], frame["target_rank"], color=color, lw=1.35, marker=marker, ms=3.2,
                     mfc=color, mec="white", mew=0.4, zorder=3)
    ax_rank.set_yscale("log")
    max_rank = max(10, int(example_traj["target_rank"].max()))
    ax_rank.set_ylim(max_rank * 1.28, 0.78)
    ax_rank.set_xlim(0.8, max_members + 0.72)
    ax_rank.set_xlabel("Spectra accumulated")
    ax_rank.set_ylabel("Target-molecule rank")
    ax_rank.set_xticks(range(1, max_members + 1))
    clean_axis(ax_rank)
    ax_rank.text(1.05, 4.1, "UltraMS", color=BLUE, weight="bold", fontsize=6.5, va="center")
    ax_rank.text(1.05, 21.0, "DreaMS", color=ORANGE, weight="bold", fontsize=6.5, va="center")
    ax_rank.annotate("rank 1 after 4 spectra", xy=(4, 1), xytext=(4.35, 1.9), color=BLUE, fontsize=6.0,
                     arrowprops={"arrowstyle": "-", "lw": 0.65, "color": BLUE})
    ax_rank.annotate("after 5", xy=(5, 1), xytext=(5.35, 1.38), color=ORANGE, fontsize=6.0,
                     arrowprops={"arrowstyle": "-", "lw": 0.65, "color": ORANGE})

    # d. Population evidence: paired gain and dependence on group size.
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, :], width_ratios=[0.96, 1.08], wspace=0.30)
    ax_gain = fig.add_subplot(bottom[0])
    add_panel_label(ax_gain, "d", x=-0.09, y=1.04)
    ax_gain.set_title("Population-level gain", loc="left", weight="bold", color=INK, pad=5)
    all_perf = performance[performance["stratum"] == "all_grouped"].set_index("model")
    for model, color, marker in [("UltraMS", BLUE, "o"), ("DreaMS", ORANGE, "D")]:
        row = all_perf.loc[model]
        ci = bootstrap[(bootstrap["model"] == model) & (bootstrap["metric"] == "top1_delta")].iloc[0]
        ax_gain.plot([0, 1], 100 * np.array([row.single_top1, row.consensus_top1]), color=color, lw=1.35)
        ax_gain.scatter([0], [100 * row.single_top1], s=29, facecolor="white", edgecolor=color, marker=marker, linewidth=1.0, zorder=3)
        ax_gain.scatter([1], [100 * row.consensus_top1], s=31, facecolor=color, edgecolor="white", marker=marker, linewidth=0.45, zorder=3)
        ax_gain.text(1.06, 100 * row.consensus_top1,
                     f"{model}  {100*row.top1_delta:+.2f} pp\n95% CI {100*ci.ci_low:+.2f} to {100*ci.ci_high:+.2f}",
                     color=color, va="center", fontsize=5.9, linespacing=1.15)
    ax_gain.set_xlim(-0.18, 1.82)
    ax_gain.set_xticks([0, 1], ["single spectrum", "label-blind\nconsensus"])
    ax_gain.set_ylabel("Spectrum-weighted Top-1 (%)")
    ax_gain.set_ylim(18, 32)
    clean_axis(ax_gain)

    ax_size = fig.add_subplot(bottom[1])
    ax_size.set_title("Evidence accumulates in larger groups", loc="left", weight="bold", color=INK, pad=5)
    strata = ["size_2", "size_3", "size_4", "size_5_plus"]
    labels = ["2", "3", "4", "≥5"]
    for model, color, marker in [("UltraMS", BLUE, "o"), ("DreaMS", ORANGE, "D")]:
        frame = performance[(performance["model"] == model) & performance["stratum"].isin(strata)].copy()
        frame["order"] = frame["stratum"].map({value: idx for idx, value in enumerate(strata)})
        frame = frame.sort_values("order")
        ax_size.plot(frame["order"], 100 * frame["top1_delta"], color=color, marker=marker, ms=3.5, lw=1.3,
                     mfc=color, mec="white", mew=0.4)
        end = frame.iloc[-1]
        ax_size.text(3.08, 100 * end.top1_delta, model, color=color, va="center", fontsize=6.1, weight="bold")
    ax_size.axhline(0, color=MID, lw=0.65)
    ax_size.set_xticks(range(len(labels)), labels)
    ax_size.set_xlabel("Spectra per group")
    ax_size.set_ylabel("Consensus Top-1 gain (pp)")
    ax_size.set_ylim(-2.5, 5.5)
    ax_size.set_xlim(-0.15, 3.65)
    clean_axis(ax_size)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT / f"s9_nonoracle_consensus.{extension}",
            dpi=600 if extension == "png" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
