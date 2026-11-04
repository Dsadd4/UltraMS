#!/usr/bin/env python
"""Nature-style layout for the eight-peak molecular-decision story."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from matplotlib.lines import Line2D


INK = "#202428"
MID = "#697178"
PALE = "#D9DEE2"
GRID = "#E7EBEE"
ULTRA = "#3965BB"
DREAMS = "#E9A0AA"
TEAL = "#009E73"
GOLD = "#E69F00"
STEM = "eight_peaks_assemble_a_molecular_decision_nature"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.55,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def molecule_rgba(smiles: str, width: int, height: int, bond_width: float = 1.45) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.025
    options.bondLineWidth = bond_width
    options.minFontSize = 8
    options.maxFontSize = 18
    options.addStereoAnnotation = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return np.asarray(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA"))


def panel_label(ax: plt.Axes, label: str, x: float = -0.055, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom")


def formula_text(formula: str) -> str:
    """Render molecular-formula counts as typographic subscripts."""
    import re

    return "$" + re.sub(r"(\d+)", r"_{\1}", str(formula)) + "$"


def plot_molecule_header(fig: plt.Figure, spec, molecules: pd.DataFrame) -> None:
    measured = molecules[molecules.is_measured.eq(1)].iloc[0]
    isomer = molecules[molecules.is_fixed_isomer.eq(1)].iloc[0]
    header = spec.subgridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.03)

    title_ax = fig.add_subplot(header[0, 0])
    title_ax.axis("off")
    title_ax.text(0.0, 0.82, "Eight peaks assemble a molecular decision", fontsize=10.0,
                  fontweight="bold", color=INK, va="top")
    title_ax.text(0.0, 0.47,
                  "A frozen negative-ion case tested across all 256 peak combinations",
                  fontsize=6.6, color=MID, va="top")
    title_ax.plot([0.0, 0.16], [0.15, 0.15], color=ULTRA, lw=1.6, clip_on=False)
    title_ax.text(0.18, 0.15, "UltraMS", fontsize=6.4, color=ULTRA, va="center", fontweight="bold")
    title_ax.plot([0.39, 0.55], [0.15, 0.15], color=DREAMS, lw=1.6, clip_on=False)
    title_ax.text(0.57, 0.15, "DreaMS", fontsize=6.4, color=DREAMS, va="center", fontweight="bold")
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)

    pair = header[0, 1].subgridspec(1, 3, width_ratios=[1.0, 0.34, 1.0], wspace=0.0)
    for column, row, label, color in [
        (0, measured, "measured", TEAL),
        (2, isomer, "same-formula isomer", DREAMS),
    ]:
        ax = fig.add_subplot(pair[0, column])
        ax.imshow(molecule_rgba(row.smiles, 780, 470, 1.5))
        ax.axis("off")
        ax.text(0.5, -0.03, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
                color=color, fontweight="bold", clip_on=False)
    mid = fig.add_subplot(pair[0, 1])
    mid.axis("off")
    mid.text(0.5, 0.64, formula_text(measured.formula), ha="center", va="center", fontsize=5.8, color=MID)
    mid.annotate("", xy=(0.88, 0.42), xytext=(0.12, 0.42),
                 arrowprops={"arrowstyle": "<->", "lw": 0.55, "color": PALE})


def plot_spectrum(ax: plt.Axes, frame: pd.DataFrame, fragments: pd.DataFrame) -> None:
    background = frame[frame.is_support.eq(0)]
    prior = frame[frame.is_previous_support.eq(1)]
    current = frame[frame.is_new_peak.eq(1)]
    included_ids = set(frame[frame.is_support.eq(1)].peak_id.astype(int))
    absent = fragments[~fragments.peak_id.astype(int).isin(included_ids)]
    ax.vlines(absent.mz, 0, absent.relative_intensity, color=GOLD, lw=0.35, alpha=0.18,
              linestyles="dotted", zorder=0)
    ax.vlines(background.mz, 0, background.relative_intensity, color=PALE, lw=0.42, zorder=1)
    ax.vlines(prior.mz, 0, prior.relative_intensity, color=TEAL, lw=0.85, zorder=2)
    ax.vlines(current.mz, 0, current.relative_intensity, color=GOLD, lw=1.45, zorder=3)
    if len(current):
        ax.scatter(current.mz, current.relative_intensity, s=7, color=GOLD, edgecolor="white",
                   linewidth=0.3, zorder=4)
    ax.set_xlim(50, 340)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)


def add_filmstrip(fig: plt.Figure, spec, steps: pd.DataFrame, spectra: pd.DataFrame,
                  fragments: pd.DataFrame) -> pd.DataFrame:
    wrapper = fig.add_subplot(spec)
    wrapper.axis("off")
    panel_label(wrapper, "a", x=-0.045, y=1.0)
    wrapper.text(0.0, 1.0, "Sequential peak add-back", fontsize=7.6, fontweight="bold", va="bottom")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=GOLD, markersize=3.4, label="new peak"),
        Line2D([], [], marker="o", linestyle="none", color=TEAL, markersize=3.4, label="retained peak"),
        Line2D([], [], marker="o", linestyle="none", color=PALE, markersize=3.4, label="unchanged spectrum"),
    ]
    wrapper.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
                   frameon=False, fontsize=5.7, handletextpad=0.25, columnspacing=0.85,
                   borderaxespad=0.0, labelcolor=MID)

    strip = spec.subgridspec(4, 9, height_ratios=[0.70, 0.23, 0.68, 0.34], hspace=0.02, wspace=0.12)
    source = []
    for step in range(9):
        top = fig.add_subplot(strip[0, step])
        top.axis("off")
        if step == 0:
            top.text(0.5, 0.55, "46 other peaks\nremain", ha="center", va="center", fontsize=5.7, color=MID)
        else:
            frag = fragments[fragments.addback_step.eq(step)].iloc[0]
            top.imshow(molecule_rgba(frag.fragment_smiles, 450, 300, 1.3))

        formula_ax = fig.add_subplot(strip[1, step])
        formula_ax.axis("off")
        if step > 0:
            formula_ax.text(0.5, 0.55, f"{formula_text(frag.fragment_formula)}  {frag.mz:.2f}",
                            ha="center", va="center", fontsize=5.5, color=GOLD,
                            fontweight="bold")

        spectrum_ax = fig.add_subplot(strip[2, step])
        frame = spectra[spectra.addback_step.eq(step)]
        plot_spectrum(spectrum_ax, frame, fragments)
        if step == 0:
            spectrum_ax.text(0.5, 0.66, "8 selected", transform=spectrum_ax.transAxes,
                             ha="center", va="center", fontsize=5.3, color=MID)
            spectrum_ax.text(0.5, 0.40, "peaks absent", transform=spectrum_ax.transAxes,
                             ha="center", va="center", fontsize=5.3, color=MID)

        label_ax = fig.add_subplot(strip[3, step])
        label_ax.axis("off")
        model_rows = steps[steps.addback_step.eq(step)].set_index("model")
        u_rank = int(model_rows.loc["UltraMS", "true_rank"])
        d_rank = int(model_rows.loc["DreaMS", "true_rank"])
        label_ax.text(0.5, 0.92, f"{step}/8", ha="center", va="top", fontsize=6.2, fontweight="bold")
        label_ax.text(0.47, 0.03, f"U #{u_rank}", ha="right", va="bottom", fontsize=5.4,
                      color=ULTRA, fontweight="bold")
        label_ax.text(0.53, 0.03, f"D #{d_rank}", ha="left", va="bottom", fontsize=5.4,
                      color=DREAMS, fontweight="bold")
        source.append({"addback_step": step, "ultrams_rank": u_rank, "dreams_rank": d_rank,
                       "new_peak_id": model_rows.iloc[0].new_peak_id,
                       "new_peak_mz": model_rows.iloc[0].new_peak_mz})
    return pd.DataFrame(source)


def deterministic_jitter(frame: pd.DataFrame) -> np.ndarray:
    if len(frame) <= 1:
        return np.zeros(len(frame))
    order = np.argsort(frame.coalition_mask.to_numpy(dtype=int))
    values = np.linspace(-0.32, 0.32, len(frame))
    jitter = np.empty(len(frame))
    jitter[order] = values
    return jitter


def plot_rank_space(fig: plt.Figure, spec, beads: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    ax = fig.add_subplot(spec)
    panel_label(ax, "b", x=-0.045, y=1.04)
    styles = {"UltraMS": (ULTRA, "o", -0.012), "DreaMS": (DREAMS, "D", 0.012)}
    plotted = []
    for (model, size), frame in beads.groupby(["model", "coalition_size"]):
        color, marker, offset = styles[model]
        frame = frame.sort_values("coalition_mask").copy()
        frame["plot_x"] = float(size) + deterministic_jitter(frame) + offset
        frame["plot_y"] = frame.true_rank.astype(float)
        ax.scatter(frame.plot_x, frame.plot_y, s=5.2, marker=marker, facecolor=color,
                   edgecolor="none", alpha=0.22, zorder=1)
        plotted.extend(frame.to_dict("records"))

    for model in ["UltraMS", "DreaMS"]:
        frame = steps[steps.model.eq(model)].sort_values("addback_step")
        color, marker, _ = styles[model]
        ax.plot(frame.addback_step, frame.true_rank, color=color, lw=1.15, marker=marker,
                ms=3.2, mfc=color, mec="white", mew=0.4, zorder=5)
        for row in frame.itertuples(index=False):
            if model == "DreaMS" or row.addback_step in {0, 8}:
                y = row.true_rank - 3.2 if row.true_rank > 4 else row.true_rank + 2.3
                x = float(row.addback_step)
                align = "center"
                if model == "DreaMS" and row.addback_step == 1:
                    x += 0.14
                    y -= 1.0
                    align = "left"
                ax.text(x, y, f"#{int(row.true_rank)}", color=color,
                        fontsize=5.3, ha=align, va="center", fontweight="bold")

    ax.set_xlim(-0.45, 8.45)
    ax.set_ylim(60, -2)
    ax.set_xticks(range(9), [f"{step}/8" for step in range(9)])
    ax.set_yticks([1, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Selected peaks restored", fontsize=6.4, labelpad=2)
    ax.set_ylabel("Measured-molecule rank", fontsize=6.4, labelpad=3)
    ax.tick_params(axis="both", labelsize=5.8, pad=1.5)
    ax.grid(axis="y", color=GRID, lw=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, 1.04, "All 256 combinations", transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="bottom")
    ax.text(0.67, 1.04, "UltraMS  256/256 at rank 1", transform=ax.transAxes,
            fontsize=6.1, color=ULTRA, fontweight="bold", ha="right", va="bottom")
    ax.text(1.0, 1.04, "DreaMS  51/256 at rank 1", transform=ax.transAxes,
            fontsize=6.1, color=DREAMS, fontweight="bold", ha="right", va="bottom")
    return pd.DataFrame(plotted)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    steps = pd.read_csv(args.data_dir / "addback_decision_path.csv")
    spectra = pd.read_csv(args.data_dir / "addback_spectrum_frames.csv")
    fragments = pd.read_csv(args.data_dir / "addback_fragment_steps.csv")
    beads = pd.read_csv(args.data_dir / "addback_exhaustive_coalitions.csv")
    molecules = pd.read_csv(args.data_dir / "addback_candidate_molecules.csv")
    summary = json.loads((args.data_dir / "addback_story_summary.json").read_text())

    fig = plt.figure(figsize=(7.2, 4.45))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.63, 1.32, 1.38], left=0.065, right=0.985,
                            top=0.975, bottom=0.090, hspace=0.31)
    plot_molecule_header(fig, outer[0], molecules)
    film_source = add_filmstrip(fig, outer[1], steps, spectra, fragments)
    cloud_source = plot_rank_space(fig, outer[2], beads, steps)

    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(args.output_dir / f"{STEM}.{suffix}", dpi=600)
    plt.close(fig)

    film_source.to_csv(args.output_dir / f"{STEM}_filmstrip_source.csv", index=False)
    cloud_source.to_csv(args.output_dir / f"{STEM}_rank_cloud_source.csv", index=False)
    steps.to_csv(args.output_dir / f"{STEM}_decision_source.csv", index=False)
    spectra.to_csv(args.output_dir / f"{STEM}_spectrum_source.csv", index=False)
    fragments.to_csv(args.output_dir / f"{STEM}_fragment_source.csv", index=False)
    molecules.to_csv(args.output_dir / f"{STEM}_molecule_source.csv", index=False)

    audit = {
        "figure": STEM,
        "reference": "Natureserises.pdf",
        "final_size_inches": [7.2, 4.45],
        "minimum_planned_font_pt": 5.3,
        "palette": {"UltraMS": ULTRA, "DreaMS": DREAMS, "retained_peak": TEAL, "new_peak": GOLD},
        "n_exact_coalitions": summary["n_exact_coalitions"],
        "rank1_across_all_256": summary["rank1_across_all_256"],
        "training_or_inference_performed": False,
        "formats_complete": all((args.output_dir / f"{STEM}.{s}").exists() for s in ["png", "pdf", "svg"]),
        "source_tables_complete": True,
    }
    (args.output_dir / f"{STEM}_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
