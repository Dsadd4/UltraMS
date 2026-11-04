#!/usr/bin/env python
"""Draw the fixed same-formula negative-ion decision story for task 13."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


ULTRA = "#247BB5"
DREAMS = "#D66A2E"
TRUE = "#167C73"
DECOY = "#D66A2E"
INK = "#202124"
MID = "#6D747A"
LIGHT = "#D8DEE2"
GRID = "#E9EDEF"
PEAK_BLUE = "#2F7EA8"
FIGURE_STEM = "same_formula_connectivity_fragment_story"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def mol_image(smiles: str, width: int, height: int, bond_width: float = 1.6) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = bond_width
    opts.padding = 0.03
    if hasattr(opts, "atomLabelFontSize"):
        opts.atomLabelFontSize = 17
    opts.setBackgroundColour((1.0, 1.0, 1.0, 0.0))
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = np.asarray(image).copy()
    nearly_white = np.all(arr[:, :, :3] > 248, axis=2)
    arr[nearly_white, 3] = 0
    return arr


def place_molecule(fig: plt.Figure, box, smiles: str, label: str, color: str,
                   width: int = 900, height: int = 500, show_label: bool = True) -> None:
    ax = fig.add_axes(box)
    ax.imshow(mol_image(smiles, width, height))
    ax.axis("off")
    if show_label:
        ax.text(
            0.5, -0.01, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.0, color=color, weight="bold", clip_on=False,
        )


def read_inputs(data_dir: Path):
    variants = pd.read_csv(data_dir / "counterfactual_variant_scores.csv")
    candidate_scores = pd.read_csv(data_dir / "counterfactual_candidate_scores.csv.gz")
    peaks = pd.read_csv(data_dir / "query_peak_support.csv")
    candidates = pd.read_csv(data_dir / "same_formula_candidates.csv")
    magma = json.loads((data_dir / "same_formula_magma_annotations.json").read_text())
    magma_by_role = {
        sample["spec_id"].split("sample_545225_", 1)[1]: sample for sample in magma["samples"]
    }
    baseline = variants.loc[variants.variant_id.eq("baseline")].copy()
    baseline_candidates = candidate_scores.loc[candidate_scores.variant_id.eq("baseline")].copy()
    if len(baseline_candidates) != 124:
        raise ValueError("Expected 62 baseline candidates for each model")
    return variants, baseline, baseline_candidates, peaks, candidates, magma_by_role


def plot_spectrum(ax: plt.Axes, peaks: pd.DataFrame, support_ids: list[int]) -> None:
    base = peaks.sort_values("peak_id")
    ax.vlines(base.mz, 0, base.relative_intensity, color=LIGHT, lw=1.0, zorder=1)
    selected = base.loc[base.peak_id.isin(support_ids)].copy()
    selected["order"] = selected.peak_id.map({pid: i + 1 for i, pid in enumerate(support_ids)})
    ax.vlines(selected.mz, 0, selected.relative_intensity, color=PEAK_BLUE, lw=1.8, zorder=3)
    for row in selected.itertuples():
        y = float(row.relative_intensity)
        ax.text(
            row.mz, y + 0.035, str(int(row.order)), ha="center", va="bottom",
            fontsize=6.3, color=PEAK_BLUE, weight="bold", clip_on=False,
        )
    ax.set_xlim(50, 345)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_title("Negative-ion spectrum", loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.set_yticks([0, 0.5, 1.0])


def plot_rank_braid(ax: plt.Axes, baseline_candidates: pd.DataFrame) -> pd.DataFrame:
    baseline_candidates = baseline_candidates.copy()
    baseline_candidates["connectivity_key"] = baseline_candidates.candidate_smiles.map(
        lambda smiles: Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]
    )
    true_key = baseline_candidates.loc[
        baseline_candidates.is_true.eq(1), "connectivity_key"
    ].iloc[0]
    decoy_key = baseline_candidates.loc[
        baseline_candidates.is_fixed_decoy.eq(1), "connectivity_key"
    ].iloc[0]
    collapsed = (
        baseline_candidates.groupby(["model", "connectivity_key"], as_index=False)
        .agg(score=("score", "max"), n_entries=("candidate_smiles", "size"))
    )
    collapsed["is_true"] = collapsed.connectivity_key.eq(true_key).astype(int)
    collapsed["is_fixed_decoy"] = collapsed.connectivity_key.eq(decoy_key).astype(int)
    collapsed["rank"] = collapsed.groupby("model").score.rank(
        method="first", ascending=False
    ).astype(int)
    ultra = collapsed.loc[collapsed.model.eq("rt_only_d11")].copy()
    dreams = collapsed.loc[collapsed.model.eq("dreams")].copy()
    u = ultra.set_index("connectivity_key")
    d = dreams.set_index("connectivity_key")
    merged = u[["rank", "score", "is_true", "is_fixed_decoy"]].join(
        d[["rank", "score"]], lsuffix="_ultra", rsuffix="_dreams", how="inner"
    ).reset_index()
    for row in merged.itertuples():
        is_true = bool(row.is_true)
        is_decoy = bool(row.is_fixed_decoy)
        color = TRUE if is_true else DECOY if is_decoy else "#C6CCD0"
        lw = 2.0 if (is_true or is_decoy) else 0.35
        alpha = 1.0 if (is_true or is_decoy) else 0.35
        z = 4 if (is_true or is_decoy) else 1
        ax.plot([0, 1], [row.rank_ultra, row.rank_dreams], color=color, lw=lw,
                alpha=alpha, zorder=z)
        ax.scatter([0, 1], [row.rank_ultra, row.rank_dreams], s=18 if z == 4 else 4,
                   color=color, edgecolor="white" if z == 4 else "none", lw=0.5, zorder=z + 1)
    ax.set_xlim(-0.2, 1.2)
    n_connectivity = len(ultra)
    ax.set_ylim(n_connectivity + 1, 0)
    ax.set_xticks([0, 1], ["UltraMS", "DreaMS"])
    ax.get_xticklabels()[0].set_color(ULTRA)
    ax.get_xticklabels()[1].set_color(DREAMS)
    ax.set_yticks([1, 10, 20, 40, n_connectivity])
    ax.set_ylabel("Candidate rank")
    ax.set_title("Same-formula rank braid", loc="left", pad=6)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.grid(axis="y", color=GRID, lw=0.55)
    true_row = merged.loc[merged.is_true.eq(1)].iloc[0]
    decoy_row = merged.loc[merged.is_fixed_decoy.eq(1)].iloc[0]
    ax.text(0.04, true_row.rank_ultra - 2.2, f"true  #{int(true_row.rank_ultra)}",
            color=TRUE, fontsize=6.4, ha="left")
    ax.text(0.96, true_row.rank_dreams + 3.0, f"true  #{int(true_row.rank_dreams)}",
            color=TRUE, fontsize=6.4, ha="right")
    ax.text(0.96, decoy_row.rank_dreams + 3.0, f"decoy  #{int(decoy_row.rank_dreams)}",
            color=DECOY, fontsize=6.4, ha="right")
    return merged


def fragment_rows(magma_by_role: dict, support_ids: list[int]) -> pd.DataFrame:
    rows = []
    roles = ["true", "dreams_top1_decoy"]
    for order, peak_id in enumerate(support_ids[:4], start=1):
        for role in roles:
            sample = magma_by_role[role]
            ann = {int(x["peak_id"]): x for x in sample["annotations"]}[peak_id]
            rows.append(
                {
                    "support_order": order,
                    "peak_id": peak_id,
                    "candidate_role": role,
                    "mz": float(ann["mz_observed"]),
                    "fragment_formula": ann["frag_formula"],
                    "fragment_smiles": ann["frag_smiles"],
                    "assigned": int(ann["assigned"]),
                    "ppm_diff": ann["ppm_diff"],
                }
            )
    return pd.DataFrame(rows)


def plot_fragment_ledger(fig: plt.Figure, fragments: pd.DataFrame) -> None:
    x0, width, gap = 0.055, 0.214, 0.018
    for order in range(1, 5):
        group = fragments.loc[fragments.support_order.eq(order)]
        true_row = group.loc[group.candidate_role.eq("true")].iloc[0]
        decoy_row = group.loc[group.candidate_role.eq("dreams_top1_decoy")].iloc[0]
        left = x0 + (order - 1) * (width + gap)
        fig.text(
            left, 0.405, f"{order}   m/z {true_row.mz:.2f}",
            ha="left", va="bottom", color=PEAK_BLUE, fontsize=7.0, weight="bold",
        )
        fig.text(
            left + width * 0.5, 0.385, str(true_row.fragment_formula),
            ha="center", va="bottom", color=MID, fontsize=6.1,
        )
        place_molecule(
            fig, [left, 0.25, width * 0.48, 0.125], str(true_row.fragment_smiles),
            "true fragment", TRUE, 500, 340, show_label=False,
        )
        place_molecule(
            fig, [left + width * 0.52, 0.25, width * 0.48, 0.125],
            str(decoy_row.fragment_smiles), "decoy fragment", DECOY, 500, 340,
            show_label=False,
        )
        fig.text(left + width * 0.24, 0.228, "true fragment", ha="center", va="center",
                 fontsize=6.1, color=TRUE, weight="bold")
        fig.text(left + width * 0.76, 0.228, "decoy fragment", ha="center", va="center",
                 fontsize=6.1, color=DECOY, weight="bold")
        ax_line = fig.add_axes([left, 0.19, width, 0.025])
        ax_line.plot([0.02, 0.98], [0.5, 0.5], color=LIGHT, lw=0.7)
        ax_line.scatter([0.02, 0.98], [0.5, 0.5], s=12, color=[TRUE, DECOY], zorder=2)
        ax_line.axis("off")


def main() -> None:
    args = parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, baseline, baseline_candidates, peaks, candidates, magma_by_role = read_inputs(args.data_dir)
    support_ids = (
        peaks.loc[peaks.is_selected_support.eq(1)]
        .sort_values("support_order").peak_id.astype(int).tolist()
    )
    fragments = fragment_rows(magma_by_role, support_ids)
    true_smiles = str(candidates.loc[candidates.is_true.eq(1), "candidate_smiles"].iloc[0])
    decoy_smiles = str(candidates.loc[candidates.is_fixed_decoy.eq(1), "candidate_smiles"].iloc[0])
    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(true_smiles))

    fig = plt.figure(figsize=(13.2, 7.15), dpi=300, facecolor="white")
    spectrum_ax = fig.add_axes([0.055, 0.535, 0.29, 0.34])
    plot_spectrum(spectrum_ax, peaks, support_ids)

    fig.text(0.385, 0.888, "Two connectivity isomers, one elemental formula", fontsize=8.2,
             color=INK, ha="left", va="bottom", weight="bold")
    fig.text(0.52, 0.842, formula, fontsize=8.0, color=MID, ha="center", va="center")
    fig.text(0.52, 0.815, "60 connectivity structures  |  62 benchmark entries", fontsize=6.6, color=MID,
             ha="center", va="center")
    place_molecule(fig, [0.36, 0.57, 0.165, 0.245], true_smiles, "measured molecule", TRUE, 900, 650)
    place_molecule(fig, [0.535, 0.57, 0.165, 0.245], decoy_smiles, "DreaMS rank-1 isomer", DECOY, 900, 650)

    rank_ax = fig.add_axes([0.755, 0.535, 0.19, 0.34])
    rank_source = plot_rank_braid(rank_ax, baseline_candidates)
    u = baseline.loc[baseline.model.eq("rt_only_d11")].iloc[0]
    d = baseline.loc[baseline.model.eq("dreams")].iloc[0]
    fig.text(0.755, 0.495, f"UltraMS margin  {u.fixed_decoy_margin:+.3f}", color=ULTRA,
             fontsize=6.4, ha="left")
    fig.text(0.885, 0.495, f"DreaMS  {d.fixed_decoy_margin:+.3f}", color=DREAMS,
             fontsize=6.4, ha="left")

    fig.text(0.055, 0.455, "Peak-resolved fragment constellation", fontsize=8.2,
             color=INK, ha="left", va="bottom", weight="bold")
    fig.text(
        0.055, 0.432,
        "Four intense, cross-CE peaks with equal fragment formula but different candidate fragment connectivity",
        fontsize=6.3, color=MID, ha="left", va="bottom",
    )
    plot_fragment_ledger(fig, fragments)
    fig.text(
        0.055, 0.115,
        "All candidates share the same molecular formula; fragment structures are candidate interpretations.",
        fontsize=6.1, color=MID, ha="left",
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(args.output_dir / f"{FIGURE_STEM}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    fragments.to_csv(args.output_dir / f"{FIGURE_STEM}_fragment_source.csv", index=False)
    rank_source.to_csv(args.output_dir / f"{FIGURE_STEM}_rank_source.csv", index=False)
    peaks.to_csv(args.output_dir / f"{FIGURE_STEM}_peak_source.csv", index=False)
    baseline.to_csv(args.output_dir / f"{FIGURE_STEM}_decision_source.csv", index=False)
    audit = {
        "figure": FIGURE_STEM,
        "sample_idx": 545225,
        "adduct": "[M-H]-",
        "formula": formula,
        "n_candidates": 62,
        "n_connectivity_structures": int(rank_source.connectivity_key.nunique()),
        "support_peak_ids": support_ids,
        "n_fragment_pairs_drawn": 4,
        "baseline": {
            row.model: {
                "true_rank": int(row.true_rank),
                "fixed_decoy_margin": float(row.fixed_decoy_margin),
            }
            for row in baseline.itertuples()
        },
        "guardrails": [
            "Candidate set is the frozen Benchmark-10 mass-search set.",
            "All 62 benchmark entries share the query formula and collapse to 60 first-block InChIKeys.",
            "MAGMa structures are candidate fragment interpretations.",
            "Peak selection is model-independent: shared MAGMa formula, different fragment connectivity, CE20 repetition, then intensity rank.",
        ],
    }
    (args.output_dir / f"{FIGURE_STEM}_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
