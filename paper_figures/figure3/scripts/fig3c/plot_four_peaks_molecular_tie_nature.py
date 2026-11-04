#!/usr/bin/env python
"""Draw a peak-to-fragment-to-molecule evidence story for one [M-H]- spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, PathPatch
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from svgpath2mpl import parse_path

from plot_same_formula_decision_story import (
    DREAMS,
    GRID,
    INK,
    LIGHT,
    MID,
    TRUE,
    ULTRA,
    fragment_rows,
    read_inputs,
    set_style,
)


ULTRA = "#3965BB"
DREAMS = "#E9A0AA"
TRUE = "#238B8E"
ISOMER = "#D97732"
PEAK_COLORS = ["#2C7FB8", "#C69214", "#7A6AA6", "#238B8E"]
STEM = "four_peaks_break_molecular_tie_nature"
PUBLIC_STEM = "fig3e_sp_1"
DISPLAY_TOP_K = 20
# Presentation order only: place the two adjacent C7O5H3 assignments together.
# The frozen peak identities and all measured/model values remain unchanged.
DISPLAY_PEAK_IDS = (18, 21, 11, 51)
PEAK_RETRIEVAL_ID = 51
PEAK_RETRIEVAL_FRAGMENT_KEY = "YRRAGUMVDQQZIY"
OLD_HEIGHT_MM = 94.0
NEW_HEIGHT_MM = 82.0
CANVAS_WIDTH_MM = 205.0
UPPER_ANCHOR_Y = 0.325
UPPER_SCALE = 0.80


def layout_y(y: float) -> float:
    """Preserve lower-panel millimetres and compress the upper story by 20%."""
    y_mm = y * OLD_HEIGHT_MM
    anchor_mm = UPPER_ANCHOR_Y * OLD_HEIGHT_MM
    if y >= UPPER_ANCHOR_Y:
        y_mm = anchor_mm + UPPER_SCALE * (y_mm - anchor_mm)
    return y_mm / NEW_HEIGHT_MM


def layout_height(height: float, *, upper: bool) -> float:
    scale = UPPER_SCALE if upper else 1.0
    return height * OLD_HEIGHT_MM * scale / NEW_HEIGHT_MM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _style_dict(style: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for item in style.split(";") if ":" in item
        for key, value in [item.split(":", 1)]
    }


def molecule_svg(smiles: str, width: int, height: int, bond_width: float,
                 output_path: Path) -> str:
    """Create and persist a transparent RDKit SVG depiction."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid molecule SMILES: {smiles}")
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = bond_width
    options.multipleBondOffset = 0.19
    options.padding = 0.045
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("stroke-linecap:butt", "stroke-linecap:round")
    svg = svg.replace("stroke-linejoin:miter", "stroke-linejoin:round")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return svg


def draw_molecule_svg(ax: plt.Axes, smiles: str, width: int, height: int,
                      bond_width: float, output_path: Path,
                      content_box: tuple[float, float, float, float] = (
                          0.0, 0.0, 1.0, 1.0
                      )) -> None:
    """Render RDKit SVG paths as Matplotlib vector artists."""
    svg = molecule_svg(smiles, width, height, bond_width, output_path)
    root = ET.fromstring(svg)

    position = ax.get_position()
    content_left, content_bottom, content_width, content_height = content_box
    box_width_pt = (
        position.width * ax.figure.get_figwidth() * 72 * content_width
    )
    box_height_pt = (
        position.height * ax.figure.get_figheight() * 72 * content_height
    )
    box_ratio = box_width_pt / box_height_pt
    image_ratio = width / height
    if box_ratio >= image_ratio:
        y_fraction = 1.0
        x_fraction = image_ratio / box_ratio
        x_offset = content_left + (content_width - x_fraction * content_width) / 2.0
        y_offset = 0.0
    else:
        x_fraction = 1.0
        y_fraction = box_ratio / image_ratio
        x_offset = 0.0
        y_offset = (content_height - y_fraction * content_height) / 2.0

    x_fraction *= content_width
    y_fraction *= content_height
    if box_ratio >= image_ratio:
        y_offset = content_bottom
    else:
        x_offset = content_left
        y_offset += content_bottom

    transform = (
        Affine2D()
        .scale(x_fraction / width, -y_fraction / height)
        .translate(x_offset, y_offset + y_fraction)
        + ax.transAxes
    )
    for element in root.iter():
        if not element.tag.endswith("path") or not element.attrib.get("d"):
            continue
        style = _style_dict(element.attrib.get("style", ""))
        fill = element.attrib.get("fill", style.get("fill", "none"))
        stroke = element.attrib.get("stroke", style.get("stroke", "none"))
        # RDKit specifies stroke widths in source-canvas pixels. Once the path
        # is fitted into a small publication panel, carrying that number over
        # as points makes the bonds unnaturally heavy. Use the journal-scale
        # minimum directly so all structures remain crisp at final size.
        linewidth = 0.5 if stroke != "none" else 0.0
        patch = PathPatch(
            parse_path(element.attrib["d"]),
            transform=transform,
            facecolor="none" if fill == "none" else fill,
            edgecolor="none" if stroke == "none" else stroke,
            linewidth=linewidth,
            joinstyle="round",
            capstyle="round",
            clip_on=True,
        )
        ax.add_patch(patch)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")


def add_molecule(fig: plt.Figure, box, smiles: str, label: str, color: str,
                 asset_path: Path,
                 width: int = 850, height: int = 520,
                 label_size: float = 6.7) -> plt.Axes:
    ax = fig.add_axes(box)
    draw_molecule_svg(ax, smiles, width, height, 1.25, asset_path)
    ax.axis("off")
    ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=label_size, color=color, weight="bold", clip_on=False,
            linespacing=1.05)
    return ax


def plot_query_spectrum(ax: plt.Axes, peaks: pd.DataFrame, support_ids: list[int]) -> pd.DataFrame:
    base = peaks.sort_values("peak_id")
    ax.vlines(base.mz, 0, base.relative_intensity, color=LIGHT, lw=1.0, zorder=1)
    selected = base.loc[base.peak_id.isin(support_ids[:4])].copy()
    order_map = {peak_id: order for order, peak_id in enumerate(support_ids[:4], start=1)}
    selected["story_order"] = selected.peak_id.map(order_map)
    selected = selected.sort_values("story_order")
    for row in selected.itertuples(index=False):
        color = PEAK_COLORS[int(row.story_order) - 1]
        ax.vlines(row.mz, 0, row.relative_intensity, color=color, lw=2.8, zorder=3)
        ax.scatter([row.mz], [row.relative_intensity], s=26, color=color,
                   edgecolor="white", lw=0.7, zorder=4)
        ax.text(row.mz, row.relative_intensity + 0.045, str(int(row.story_order)),
                ha="center", va="bottom", color=color, fontsize=7.0, weight="bold")
    ax.set_xlim(50, 345)
    ax.axhline(0, color=INK, lw=0.55)
    ax.set_ylim(0, 1.14)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.55)
    ax.set_yticks([0, 0.5, 1.0])
    return selected


def _as_bool(series: pd.Series) -> pd.Series:
    """Read boolean evidence robustly from either typed or text CSV columns."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def load_peak_retrieval_evidence(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load the candidate-blind top topology and its three reference records."""
    analysis_root = data_dir / "peak_embedding_cross_parent_20260823"
    blind_dir = analysis_root / "stage12_blind_topology_selection"
    blind_path = blind_dir / "blind_top_topology_per_peak.csv"
    revealed_path = blind_dir / "revealed_pairwise_compatibility.csv"
    audit_path = blind_dir / "audit.json"
    source_path = (
        analysis_root
        / "stage10_blind_candidate_scoring"
        / "blind_source_valid_neighbor_annotations.csv"
    )
    blind = pd.read_csv(blind_path)
    revealed = pd.read_csv(revealed_path)
    blind_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selected = blind.loc[blind.peak_id.eq(PEAK_RETRIEVAL_ID)].copy()
    revealed_selected = revealed.loc[revealed.peak_id.eq(PEAK_RETRIEVAL_ID)].copy()
    if len(selected) != 1 or len(revealed_selected) != 1:
        raise ValueError("Candidate-blind peak 51 selection is not unique.")
    selected_row = selected.iloc[0]
    revealed_row = revealed_selected.iloc[0]
    if str(selected_row.frag_connectivity_key) != PEAK_RETRIEVAL_FRAGMENT_KEY:
        raise ValueError("Candidate-blind peak 51 topology changed.")
    if not bool(revealed_row.target_compatible) or bool(
        revealed_row.pleurostimin_compatible
    ):
        raise ValueError("Candidate-blind peak 51 compatibility changed.")
    blindness = blind_audit.get("blindness_audit", {})
    if blindness.get("candidate_identity_used_to_select_topology") is not False:
        raise ValueError("Candidate identity entered the topology selection.")
    if blindness.get("candidate_compatibility_used_to_select_topology") is not False:
        raise ValueError("Candidate compatibility entered the topology selection.")

    evidence = pd.read_csv(source_path)
    rows = evidence.loc[
        evidence.peak_id.eq(PEAK_RETRIEVAL_ID)
        & evidence.frag_connectivity_key.eq(PEAK_RETRIEVAL_FRAGMENT_KEY)
    ].copy()
    checks = {
        "n_rows": len(rows) == 3,
        "n_reference_spectra": rows.reference_store_index.nunique() == 3,
        "n_reference_parent_connectivities": rows.reference_parent_key.nunique() == 3,
        "query_peak_identity": rows.peak_id.eq(PEAK_RETRIEVAL_ID).all(),
        "fragment_connectivity": rows.frag_connectivity_key.eq(
            PEAK_RETRIEVAL_FRAGMENT_KEY
        ).all(),
        "selected_family_is_highest_cosine": np.isclose(
            float(selected_row.max_cosine), float(rows.peak_embedding_cosine.max())
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(
            "Peak-retrieval evidence failed frozen validation: " + ", ".join(failed)
        )
    rows = rows.sort_values(
        ["peak_embedding_cosine", "reference_store_index"], ascending=[False, True]
    ).reset_index(drop=True)
    rows["candidate_blind_topology_rank"] = 1
    rows["selection_used_candidate_identity"] = False
    rows["selection_used_candidate_compatibility"] = False
    rows["target_compatible_after_reveal"] = True
    rows["pleurostimin_compatible_after_reveal"] = False
    summary = {
        "peak_id": PEAK_RETRIEVAL_ID,
        "query_mz": float(rows.query_mz.iloc[0]),
        "fragment_connectivity_key": PEAK_RETRIEVAL_FRAGMENT_KEY,
        "selection": "highest-cosine source-valid topology before candidate reveal",
        "selection_used_candidate_identity": False,
        "selection_used_candidate_compatibility": False,
        "n_reference_spectra": int(rows.reference_store_index.nunique()),
        "n_reference_parent_connectivities": int(rows.reference_parent_key.nunique()),
        "target_compatible": int(rows.reference_store_index.nunique()),
        "fixed_isomer_compatible": 0,
        "maximum_cosine": float(selected_row.max_cosine),
        "cosine_range": [
            float(rows.peak_embedding_cosine.min()),
            float(rows.peak_embedding_cosine.max()),
        ],
        "delta_ppm_range": [float(rows.delta_ppm.min()), float(rows.delta_ppm.max())],
        "blind_selection_file": str(blind_path),
        "revealed_compatibility_file": str(revealed_path),
        "blind_audit_file": str(audit_path),
        "source_file": str(source_path),
    }
    return rows, summary


def fragment_row(fig: plt.Figure, left: float, bottom: float, width: float, height: float,
                 true_row: pd.Series, decoy_row: pd.Series, order: int,
                 asset_dir: Path, *, key_evidence: bool = False) -> dict:
    color = PEAK_COLORS[order - 1]
    if not np.isclose(float(true_row.mz), float(decoy_row.mz), rtol=0.0, atol=1e-9):
        raise ValueError(f"Paired fragment rows must share measured m/z (row {order}).")
    if str(true_row.fragment_formula) != str(decoy_row.fragment_formula):
        raise ValueError(f"Paired fragment rows must share candidate composition (row {order}).")
    anchor = fig.add_axes([left, bottom, width * 0.09, height])
    anchor.scatter([0.5], [0.5], s=62, color=color, edgecolor="white", lw=0.8, zorder=3)
    anchor.text(0.5, 0.5, str(order), ha="center", va="center", color="white",
                weight="bold", fontsize=6.3, zorder=4)
    anchor.set_xlim(0, 1)
    anchor.set_ylim(0, 1)
    anchor.axis("off")

    molecule_bottom = bottom - height * 0.06 if key_evidence else bottom + height * 0.02
    molecule_height = height * 1.05 if key_evidence else height * 0.88
    molecule_width = width * 0.45 if key_evidence else width * 0.42
    # Keep the key-evidence frame clear of the numbered peak anchor while
    # preserving the original molecule scale and column alignment.
    true_left = left + width * 0.05 if key_evidence else left + width * 0.04
    decoy_left = left + width * 0.525 if key_evidence else left + width * 0.54
    true_ax = fig.add_axes([
        true_left, molecule_bottom, molecule_width, molecule_height
    ])
    if key_evidence:
        true_ax.add_patch(FancyBboxPatch(
            (0.06, 0.06), 0.73, 0.84,
            boxstyle="round,pad=0.008,rounding_size=0.035",
            transform=true_ax.transAxes,
            facecolor="#F2F6FC", edgecolor=ULTRA, linewidth=0.8,
            zorder=0, clip_on=False,
        ))
        true_ax.text(
            0.085, 0.84, "blind\nannotation", transform=true_ax.transAxes,
            ha="left", va="top", fontsize=4.5, linespacing=0.88,
            color=ULTRA, weight="bold",
            zorder=5,
        )
    draw_molecule_svg(
        true_ax, str(true_row.fragment_smiles), 560, 330, 1.2,
        asset_dir / f"peak_{order}_measured_candidate.svg",
        content_box=(0.0, 0.0, 1.0, 1.0),
    )
    true_ax.axis("off")
    decoy_ax = fig.add_axes([
        decoy_left, molecule_bottom, molecule_width, molecule_height
    ])
    draw_molecule_svg(decoy_ax, str(decoy_row.fragment_smiles), 560, 330, 1.2,
                      asset_dir / f"peak_{order}_same_formula_isomer.svg")
    decoy_ax.axis("off")

    label_x = left + width * 0.50
    label_y = bottom + height * (1.15 if key_evidence else 1.03)
    branch_y = bottom + height * (1.035 if key_evidence else 0.925)
    branch_drop_y = bottom + height * (0.955 if key_evidence else 0.885)
    true_center_x = left + width * 0.26
    decoy_center_x = left + width * 0.74
    label_gap = width * 0.135
    fig.text(label_x, label_y,
             f"m/z {true_row.mz:.3f}   {true_row.fragment_formula}",
             ha="center", va="top", fontsize=6.2, color=color, weight="bold")
    # Keep the shared label visually isolated: two quiet leader segments map
    # the one measured mass/composition to the two candidate connectivities.
    for x0, x1 in [
        (true_center_x, label_x - label_gap),
        (label_x + label_gap, decoy_center_x),
    ]:
        fig.add_artist(Line2D(
            [x0, x1], [branch_y, branch_y],
            transform=fig.transFigure, color=color, alpha=0.24, lw=0.50,
            solid_capstyle="round", zorder=1,
        ))
    for center_x in [true_center_x, decoy_center_x]:
        fig.add_artist(Line2D(
            [center_x, center_x], [branch_y, branch_drop_y],
            transform=fig.transFigure, color=color, alpha=0.24, lw=0.50,
            solid_capstyle="round", zorder=1,
        ))
    return {"anchor": anchor, "true": true_ax, "decoy": decoy_ax}


def peak_retrieval_strip(fig: plt.Figure, left: float, bottom: float, width: float,
                         evidence: dict) -> None:
    """Add the candidate-blind topology selection beneath m/z 316."""
    if evidence["target_compatible"] != evidence["n_reference_spectra"]:
        raise ValueError("Displayed target-compatible count is not complete.")
    if evidence["fixed_isomer_compatible"] != 0:
        raise ValueError("Displayed fixed-isomer-compatible count must be zero.")
    method_y = bottom + layout_height(0.014, upper=True)
    result_y = bottom - layout_height(0.007, upper=True)
    fig.text(
        left + width * 0.50, method_y,
        f"10-ppm m/z match  →  highest peak similarity  (cosine {evidence['maximum_cosine']:.3f})",
        ha="center", va="bottom", fontsize=4.75, color=MID,
    )
    fig.add_artist(Line2D(
        [left + width * 0.14, left + width * 0.86],
        [method_y - layout_height(0.004, upper=True)] * 2,
        transform=fig.transFigure, color=ULTRA, alpha=0.28, lw=0.55,
        solid_capstyle="round", zorder=1,
    ))
    fig.text(
        left + width * 0.26, result_y,
        f"target-compatible  ·  {evidence['target_compatible']}/{evidence['n_reference_spectra']} refs",
        ha="center", va="top", fontsize=5.25, color=TRUE, weight="bold",
    )
    fig.text(
        left + width * 0.74, result_y,
        f"Pleurostimin-compatible  ·  {evidence['fixed_isomer_compatible']}/{evidence['n_reference_spectra']} refs",
        ha="center", va="top", fontsize=5.25, color=ISOMER, weight="bold",
    )


def connect_peak_to_fragment(fig: plt.Figure, spectrum_ax: plt.Axes,
                             peak_row, target_ax: plt.Axes) -> None:
    order = int(peak_row.story_order)
    spectrum_y = float(peak_row.relative_intensity) / 1.14
    patch = ConnectionPatch(
        xyA=(1.0, spectrum_y),
        coordsA=spectrum_ax.transAxes,
        axesA=spectrum_ax,
        xyB=(0.5, 0.5),
        coordsB=target_ax.transAxes,
        axesB=target_ax,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={0.055 * (2.5 - order)}",
        color=PEAK_COLORS[order - 1],
        alpha=0.45,
        lw=1.05,
        zorder=0,
        clip_on=False,
    )
    fig.add_artist(patch)


def rank_source(baseline_candidates: pd.DataFrame) -> pd.DataFrame:
    frame = baseline_candidates.copy()
    frame["connectivity_key"] = frame.candidate_smiles.map(
        lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(str(x))).split("-")[0]
    )
    collapsed = frame.groupby(["model", "connectivity_key"], as_index=False).agg(
        score=("score", "max"),
        is_true=("is_true", "max"),
        is_fixed_decoy=("is_fixed_decoy", "max"),
    )
    collapsed["rank"] = collapsed.groupby("model").score.rank(
        method="first", ascending=False
    ).astype(int)
    return collapsed


def verdict_panel(ax: plt.Axes, ranks: pd.DataFrame) -> None:
    model_x = {"rt_only_d11": 0.0, "dreams": 1.0}
    for model, group in ranks.groupby("model"):
        x = model_x[model]
        top = group.loc[group["rank"].le(15)]
        ax.scatter(np.full(len(top), x), top["rank"], s=7, color="#CCD2D5", alpha=0.75)
        for role, color, marker in [
            ("is_true", TRUE, "o"),
            ("is_fixed_decoy", ISOMER, "D"),
        ]:
            row = group.loc[group[role].eq(1)].iloc[0]
            ax.scatter([x], [row["rank"]], s=62, color=color, marker=marker,
                       edgecolor="white", lw=0.8, zorder=4)
            label = "target" if role == "is_true" else "isomer"
            label_x = x + 0.14 if model == "rt_only_d11" else x - 0.14
            label_ha = "left" if model == "rt_only_d11" else "right"
            label_y = float(row["rank"])
            if int(row["rank"]) == 1:
                label_y = 1.75 if model == "rt_only_d11" else 0.45
            ax.text(label_x, label_y, f"{label}  #{int(row['rank'])}",
                    ha=label_ha, va="center", fontsize=6.0,
                    color=color, weight="bold")
    ax.set_xlim(-0.58, 1.58)
    # Reserve headroom for rank-1 markers and labels instead of relying on
    # figure-level padding, which would make the complete panel taller.
    ax.set_ylim(16, -1.2)
    ax.set_xticks([0, 1], ["UltraMS", "DreaMS"])
    ax.get_xticklabels()[0].set_color(ULTRA)
    ax.get_xticklabels()[1].set_color(DREAMS)
    ax.set_yticks([1, 5, 10, 15])
    ax.set_ylabel("Rank (60 structures)", fontsize=6.0, labelpad=2)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.5)


def candidate_profile_source(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_scores.loc[candidate_scores.variant_id.eq("baseline")].copy()
    frame["connectivity_key"] = frame.candidate_smiles.map(
        lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(str(x))).split("-")[0]
    )
    scores = frame.groupby(["model", "connectivity_key"], as_index=False).agg(
        score=("score", "max"),
        is_true=("is_true", "max"),
        is_fixed_decoy=("is_fixed_decoy", "max"),
    )
    representatives = (
        frame.sort_values(
            ["model", "connectivity_key", "is_true", "is_fixed_decoy", "score"],
            ascending=[True, True, False, False, False],
        )
        .drop_duplicates(["model", "connectivity_key"])
        [["model", "connectivity_key", "candidate_index", "candidate_smiles"]]
    )
    collapsed = scores.merge(representatives, on=["model", "connectivity_key"], how="left")
    collapsed["rank"] = collapsed.groupby("model").score.rank(
        method="first", ascending=False
    ).astype(int)

    ultra = collapsed.loc[collapsed.model.eq("rt_only_d11")].set_index("connectivity_key")
    dreams = collapsed.loc[collapsed.model.eq("dreams")].set_index("connectivity_key")
    joined = ultra[["score", "rank", "candidate_index", "candidate_smiles", "is_true",
                    "is_fixed_decoy"]].join(
        dreams[["score", "rank"]], lsuffix="_ultra", rsuffix="_dreams", how="inner"
    ).reset_index()

    true_smiles = str(joined.loc[joined.is_true.eq(1), "candidate_smiles"].iloc[0])
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    true_fp = fp_generator.GetFingerprint(Chem.MolFromSmiles(true_smiles))
    phenolic_oh_pattern = Chem.MolFromSmarts("[OX2H][c]")
    carbonyl_pattern = Chem.MolFromSmarts(
        "[$([CX3]=[OX1]),$([cX3]=[OX1])]"
    )
    rows = []
    for row in joined.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.candidate_smiles))
        rows.append({
            "connectivity_key": row.connectivity_key,
            "candidate_index": int(row.candidate_index),
            "candidate_smiles": row.candidate_smiles,
            "is_true": int(row.is_true),
            "is_fixed_decoy": int(row.is_fixed_decoy),
            "ultrams_score": float(row.score_ultra),
            "dreams_score": float(row.score_dreams),
            "ultrams_rank": int(row.rank_ultra),
            "dreams_rank": int(row.rank_dreams),
            "tpsa_a2": float(rdMolDescriptors.CalcTPSA(mol)),
            "clogp": float(Descriptors.MolLogP(mol)),
            "h_bond_donors": int(Lipinski.NumHDonors(mol)),
            "h_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
            "phenolic_oh_groups": int(
                len(mol.GetSubstructMatches(phenolic_oh_pattern))
            ),
            "carbonyl_groups": int(
                len(mol.GetSubstructMatches(carbonyl_pattern))
            ),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
            "stereocentres": int(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
            "morgan_tanimoto_to_target": float(
                DataStructs.TanimotoSimilarity(true_fp, fp_generator.GetFingerprint(mol))
            ),
        })
    profiles = pd.DataFrame(rows)
    target = profiles.loc[profiles.is_true.eq(1)]
    fixed_decoy = profiles.loc[profiles.is_fixed_decoy.eq(1)]
    other = profiles.loc[profiles.is_true.eq(0) & profiles.is_fixed_decoy.eq(0)]
    ultra_top = other.sort_values("ultrams_rank").head(3)
    dreams_top = profiles.sort_values("dreams_rank").head(3)
    selected = pd.concat([target, fixed_decoy, ultra_top, dreams_top], ignore_index=True)
    selected = selected.drop_duplicates("connectivity_key", keep="first").reset_index(drop=True)
    labels = []
    for row in selected.itertuples(index=False):
        if row.is_true:
            labels.append("Target")
        elif row.is_fixed_decoy:
            labels.append("DreaMS #1")
        elif row.ultrams_rank <= 3 and row.dreams_rank <= 3:
            labels.append(f"UltraMS #{row.ultrams_rank}\nDreaMS #{row.dreams_rank}")
        elif row.ultrams_rank <= 4:
            labels.append(f"UltraMS #{row.ultrams_rank}")
        else:
            labels.append(f"DreaMS #{row.dreams_rank}")
    selected["display_label"] = labels

    descriptor_columns = [
        "morgan_tanimoto_to_target", "tpsa_a2", "clogp", "h_bond_donors",
        "h_bond_acceptors", "phenolic_oh_groups", "carbonyl_groups",
        "rotatable_bonds", "aromatic_rings", "fraction_csp3", "stereocentres",
    ]
    for column in descriptor_columns:
        values = profiles[column]
        selected[f"{column}_percentile"] = selected[column].map(
            lambda value: float((values.le(value).mean()))
        )
    return selected, profiles


def candidate_profile_panel(
    fig: plt.Figure, all_profiles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        ("tpsa_a2", "TPSA (Å²)", ".1f"),
        ("clogp", "cLogP", ".2f"),
        ("phenolic_oh_groups", "Phenolic OH groups", ".0f"),
        ("carbonyl_groups", "Carbonyl groups", ".0f"),
        ("rotatable_bonds", "Rotatable bonds", ".0f"),
        ("fraction_csp3", "sp³ carbon fraction", ".2f"),
    ]
    ranked = pd.concat([
        all_profiles.sort_values("ultrams_rank").head(DISPLAY_TOP_K).assign(
            display_model="UltraMS", display_rank=lambda x: x.ultrams_rank
        ),
        all_profiles.sort_values("dreams_rank").head(DISPLAY_TOP_K).assign(
            display_model="DreaMS", display_rank=lambda x: x.dreams_rank
        ),
    ], ignore_index=True)
    fig.text(0.04, layout_y(0.258), "Top-20 candidate properties", fontsize=6.5,
             color=INK, ha="left", va="top")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=ULTRA, markeredgecolor="white", markeredgewidth=0.5,
               label="UltraMS"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=4.2,
               markerfacecolor=DREAMS, markeredgecolor="white", markeredgewidth=0.5,
               label="DreaMS"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=ULTRA, markeredgecolor=INK, markeredgewidth=1.0,
               label="measured target"),
        Line2D([0], [0], color=INK, lw=1.0, alpha=0.75,
               label="non-target distribution"),
    ]
    fig.legend(handles=legend_handles, loc="upper left",
               bbox_to_anchor=(0.20, layout_y(0.267)),
               frameon=False, ncol=4, handletextpad=0.3, columnspacing=0.65,
               borderaxespad=0, fontsize=6.0)
    fig.text(0.755, layout_y(0.258), "darker = higher rank", fontsize=6.0,
             color=MID, ha="left", va="top")

    left, right, gap = 0.07, 0.98, 0.018
    width = (right - left - gap * (len(metrics) - 1)) / len(metrics)
    density_rows = []
    for i, (column, label, number_format) in enumerate(metrics):
        ax = fig.add_axes([
            left + i * (width + gap),
            layout_y(0.055),
            width,
            layout_height(0.14, upper=False),
        ])
        values = all_profiles[column].to_numpy(dtype=float)
        data_min, data_max = float(values.min()), float(values.max())
        span = max(data_max - data_min, 1.0)
        pad = span * 0.06
        ax.hlines([1, 0], data_min, data_max, color=GRID, lw=0.55, zorder=0)
        grid = np.linspace(data_min - pad, data_max + pad, 240)
        for model, color, marker, y, direction in [
            ("UltraMS", ULTRA, "o", 1.0, 1.0),
            ("DreaMS", DREAMS, "D", 0.0, -1.0),
        ]:
            subset = ranked.loc[ranked.display_model.eq(model)].sort_values(
                "display_rank", ascending=False
            ).copy()
            # Separate coincident descriptor values vertically without changing x.
            subset["plot_y"] = y
            for _, duplicate_group in subset.groupby(column, sort=False):
                offsets = np.linspace(-0.09, 0.09, len(duplicate_group))
                subset.loc[duplicate_group.index, "plot_y"] = y + offsets
            non_target = subset.loc[subset.is_true.eq(0), column].to_numpy(dtype=float)
            std = float(np.std(non_target, ddof=1))
            q25, q75 = np.percentile(non_target, [25, 75])
            robust_scale = min(std, float((q75 - q25) / 1.349))
            if not np.isfinite(robust_scale) or robust_scale <= 0:
                robust_scale = std if np.isfinite(std) and std > 0 else span / 8
            bandwidth = max(
                0.9 * robust_scale * len(non_target) ** (-0.2),
                span * 0.035,
                1e-6,
            )
            if column in {"phenolic_oh_groups", "carbonyl_groups", "rotatable_bonds"}:
                bandwidth = max(bandwidth, 0.35)
            z = (grid[:, None] - non_target[None, :]) / bandwidth
            density = np.exp(-0.5 * z ** 2).mean(axis=1) / (
                bandwidth * np.sqrt(2 * np.pi)
            )
            area = float(np.trapezoid(density, grid))
            if area > 0:
                density = density / area
            scaled = density / max(float(density.max()), 1e-12)
            ridge_y = y + direction * scaled * 0.27
            ax.fill_between(grid, y, ridge_y, color=color, alpha=0.13,
                            linewidth=0, zorder=1)
            ax.plot(grid, ridge_y, color=color, alpha=0.9, lw=0.8, zorder=2)
            for x_value, density_value, scaled_value in zip(grid, density, scaled):
                density_rows.append({
                    "display_model": model,
                    "descriptor": column,
                    "x": float(x_value),
                    "density": float(density_value),
                    "scaled_density": float(scaled_value),
                    "bandwidth": float(bandwidth),
                    "n_candidates": int(len(non_target)),
                    "target_value": float(
                        subset.loc[subset.is_true.eq(1), column].iloc[0]
                    ),
                })
            base_rgb = np.asarray(mpl.colors.to_rgb(color))
            for row in subset.itertuples(index=False):
                strength = 1.0 - (
                    (float(row.display_rank) - 1.0)
                    * (0.72 / max(DISPLAY_TOP_K - 1, 1))
                )
                face = tuple(1.0 - (1.0 - base_rgb) * strength)
                is_target = bool(row.is_true)
                ax.scatter([getattr(row, column)], [row.plot_y],
                           s=29.0 if row.display_rank == 1 else 18.0,
                           marker=marker, facecolor=face,
                           edgecolor=INK if is_target else "white",
                           linewidth=1.0 if is_target else 0.4,
                           zorder=4 if row.display_rank == 1 else 3)
        ax.set_xlim(data_min - pad, data_max + pad)
        ax.set_ylim(-0.55, 1.55)
        ax.set_title(label, fontsize=6.0, weight="normal", pad=2)
        ax.set_xticks([data_min, data_max])
        ax.set_xticklabels([
            format(data_min, number_format), format(data_max, number_format)
        ])
        ax.set_yticks([1, 0])
        if i == 0:
            ax.set_yticklabels(["UltraMS", "DreaMS"])
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
        if i != 0:
            ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#7C858B")
        ax.spines["bottom"].set_linewidth(0.55)
        ax.tick_params(axis="x", length=2, width=0.5, pad=1.5, labelsize=6.0)
        ax.tick_params(axis="y", length=2 if i == 0 else 0, width=0.5,
                       pad=1.5, labelsize=6.0)
    return ranked, pd.DataFrame(density_rows)


def main() -> None:
    args = parse_args()
    set_style()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 6.5,
        "axes.titlesize": 6.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        # Outline SVG glyphs so Quick Look, browsers and Illustrator cannot
        # reinterpret CSS pixel sizes or substitute a metrically different font.
        # The PDF remains searchable/selectable through pdf.fonttype=42.
        "svg.fonttype": "path",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = args.output_dir / "molecule_svgs"
    asset_dir.mkdir(parents=True, exist_ok=True)
    stem = "four_peaks_break_molecular_tie_nature"
    baseline = pd.read_csv(args.data_dir / f"{stem}_decision_source.csv")
    candidate_scores = pd.read_csv(args.data_dir / "counterfactual_candidate_scores.csv.gz")
    peaks = pd.read_csv(args.data_dir / "query_peak_support.csv")
    frozen_peak_source = pd.read_csv(args.data_dir / f"{stem}_peak_source.csv")
    frozen_support_ids = frozen_peak_source.peak_id.astype(int).tolist()
    if set(frozen_support_ids) != set(DISPLAY_PEAK_IDS):
        raise ValueError(
            "Frozen support peaks no longer match the registered Figure 3c display set."
        )
    support_ids = list(DISPLAY_PEAK_IDS)
    support_order = dict(zip(support_ids, range(1, len(support_ids) + 1)))
    peaks["is_selected_support"] = peaks.peak_id.isin(support_ids).astype(int)
    peaks["support_order"] = peaks.peak_id.map(support_order)
    frozen_effects = frozen_peak_source.set_index("peak_id")
    for column in ["fixed_margin_drop", "global_margin_drop"]:
        peaks[column] = peaks.peak_id.map(frozen_effects[column])
    fragments = pd.read_csv(args.data_dir / f"{stem}_fragment_source.csv")
    candidates = pd.read_csv(args.data_dir / f"{stem}_candidate_source.csv")
    ranks = pd.read_csv(args.data_dir / f"{stem}_rank_source.csv")
    peak_retrieval_rows, peak_retrieval_summary = load_peak_retrieval_evidence(
        args.data_dir
    )
    profiles, all_profiles = candidate_profile_source(candidate_scores)
    true_smiles = str(candidates.loc[candidates.is_true.eq(1), "candidate_smiles"].iloc[0])
    decoy_smiles = str(candidates.loc[candidates.is_fixed_decoy.eq(1), "candidate_smiles"].iloc[0])
    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(true_smiles))

    mm = 1 / 25.4
    # Full-width Nature layout.  The established normalized panel map is kept
    # so the extra 25 mm is distributed across the measured-spectrum,
    # fragment-connectivity, ranking and six descriptor panels instead of
    # becoming unused outer whitespace.
    fig = plt.figure(
        figsize=(CANVAS_WIDTH_MM * mm, NEW_HEIGHT_MM * mm),
        dpi=300,
        facecolor="white",
    )
    fig.text(0.035, layout_y(0.965), "1  Measured spectrum", fontsize=6.5,
             weight="bold", color=INK, va="top")
    fig.text(0.035, layout_y(0.927), f"[M-H]-   {formula}", fontsize=6.0,
             color=MID, va="top")
    spectrum_ax = fig.add_axes([
        0.075,
        layout_y(0.57),
        0.205,
        layout_height(0.285, upper=True),
    ])
    selected = plot_query_spectrum(spectrum_ax, peaks, support_ids)
    add_molecule(fig, [
        0.065,
        layout_y(0.325),
        0.205,
        layout_height(0.205, upper=True),
    ], true_smiles,
                 "8-methoxyquercetin\n7-O-glucoside", TRUE,
                 asset_dir / "measured_molecule.svg", 900, 650, 6.0)

    fig.text(0.315, layout_y(0.965), "2  Candidate fragment connectivity", fontsize=6.5,
             weight="bold", color=INK, va="top")
    fig.text(0.417, layout_y(0.913), "Measured candidate", fontsize=6.0,
             color=TRUE, ha="center", va="top", weight="bold")
    fig.text(0.593, layout_y(0.913), "Pleurostimin", fontsize=6.0,
             color=ISOMER, ha="center", va="top", weight="bold")
    row_bottoms = [0.765, 0.635, 0.505, 0.375]
    row_axes = []
    for order, bottom in enumerate(row_bottoms, start=1):
        group = fragments.loc[fragments.peak_id.eq(support_ids[order - 1])]
        true_row = group.loc[group.candidate_role.eq("true")].iloc[0]
        decoy_row = group.loc[group.candidate_role.eq("dreams_top1_decoy")].iloc[0]
        row_axes.append(fragment_row(
            fig, 0.31, layout_y(bottom), 0.39,
            layout_height(0.105, upper=True),
            true_row, decoy_row, order, asset_dir,
            key_evidence=int(true_row.peak_id) == PEAK_RETRIEVAL_ID,
        ))

    peak_retrieval_strip(
        fig, 0.31, layout_y(0.335), 0.39, peak_retrieval_summary
    )

    for peak_row in selected.itertuples(index=False):
        connect_peak_to_fragment(
            fig, spectrum_ax, peak_row,
            row_axes[int(peak_row.story_order) - 1]["anchor"],
        )

    fig.text(0.725, layout_y(0.965), "3  Same-formula candidate ranking", fontsize=6.5,
             weight="bold", color=INK, va="top")
    add_molecule(fig, [
        0.715,
        layout_y(0.695),
        0.12,
        layout_height(0.18, upper=True),
    ], true_smiles,
                 "measured", TRUE, asset_dir / "ranking_measured_molecule.svg",
                 780, 580)
    add_molecule(fig, [
        0.855,
        layout_y(0.695),
        0.12,
        layout_height(0.18, upper=True),
    ], decoy_smiles,
                 "Pleurostimin", ISOMER,
                 asset_dir / "ranking_same_formula_isomer.svg", 780, 580)

    verdict_ax = fig.add_axes([
        0.735,
        layout_y(0.37),
        0.225,
        layout_height(0.245, upper=True),
    ])
    verdict_panel(verdict_ax, ranks)
    ranked_profiles, density_profiles = candidate_profile_panel(fig, all_profiles)
    for suffix in ["png", "pdf", "svg"]:
        for output_stem in [STEM, PUBLIC_STEM]:
            fig.savefig(args.output_dir / f"{output_stem}.{suffix}", dpi=600)
    plt.close(fig)

    fragments.to_csv(args.output_dir / f"{STEM}_fragment_source.csv", index=False)
    ranks.to_csv(args.output_dir / f"{STEM}_rank_source.csv", index=False)
    selected.to_csv(args.output_dir / f"{STEM}_peak_source.csv", index=False)
    raw_spectrum_source = peaks.copy()
    raw_spectrum_source["is_highlighted_story_peak"] = (
        raw_spectrum_source.peak_id.isin(support_ids[:4]).astype(int)
    )
    raw_spectrum_source.to_csv(
        args.output_dir / f"{STEM}_raw_spectrum_source.csv", index=False
    )
    candidates.loc[candidates.is_true.eq(1) | candidates.is_fixed_decoy.eq(1)].to_csv(
        args.output_dir / f"{STEM}_candidate_source.csv", index=False
    )
    profile_name = f"{STEM}_candidate_profile_source.csv"
    profiles.to_csv(args.output_dir / profile_name, index=False)
    distribution_name = f"{STEM}_candidate_descriptor_distribution_source.csv"
    all_profiles.to_csv(args.output_dir / distribution_name, index=False)
    ranked_name = f"{STEM}_top20_candidate_descriptor_source.csv"
    ranked_profiles.to_csv(args.output_dir / ranked_name, index=False)
    density_name = f"{STEM}_top20_non_target_density_source.csv"
    density_profiles.to_csv(args.output_dir / density_name, index=False)
    descriptor_decisions = {
        "tpsa_a2": ("retained", "Required polarity and exposed-surface context."),
        "clogp": ("retained", "Required hydrophobicity context."),
        "phenolic_oh_groups": (
            "added", "Separates phenolic deprotonation sites from total H-bond donors."
        ),
        "carbonyl_groups": (
            "added", "Distinguishes oxygen allocation and carbonyl-bearing scaffolds."
        ),
        "rotatable_bonds": ("retained", "Non-redundant molecular-flexibility context."),
        "fraction_csp3": ("retained", "Non-redundant saturated-framework context."),
        "h_bond_donors": ("removed", "Strongly redundant with TPSA in this candidate set."),
        "aromatic_rings": ("removed", "Highly concentrated and coarse for positional isomers."),
    }
    decision_rows = []
    ultra_top20 = all_profiles.nsmallest(DISPLAY_TOP_K, "ultrams_rank")
    dreams_top20 = all_profiles.nsmallest(DISPLAY_TOP_K, "dreams_rank")
    for column, (status, rationale) in descriptor_decisions.items():
        decision_rows.append({
            "descriptor": column,
            "status": status,
            "n_unique_all_60": int(all_profiles[column].nunique()),
            "n_unique_ultrams_top20": int(ultra_top20[column].nunique()),
            "n_unique_dreams_top20": int(dreams_top20[column].nunique()),
            "minimum_all_60": float(all_profiles[column].min()),
            "maximum_all_60": float(all_profiles[column].max()),
            "ultrams_top20_mean": float(ultra_top20[column].mean()),
            "dreams_top20_mean": float(dreams_top20[column].mean()),
            "spearman_with_tpsa": float(all_profiles[column].corr(
                all_profiles.tpsa_a2, method="spearman"
            )),
            "spearman_with_clogp": float(all_profiles[column].corr(
                all_profiles.clogp, method="spearman"
            )),
            "rationale": rationale,
        })
    descriptor_audit = pd.DataFrame(decision_rows)
    descriptor_audit_name = f"{STEM}_descriptor_selection_audit.csv"
    descriptor_audit.to_csv(args.output_dir / descriptor_audit_name, index=False)
    peak_retrieval_name = f"{STEM}_peak_embedding_evidence_source.csv"
    peak_retrieval_rows.to_csv(args.output_dir / peak_retrieval_name, index=False)
    if args.data_dir.resolve() != args.output_dir.resolve():
        profiles.to_csv(args.data_dir / profile_name, index=False)
        all_profiles.to_csv(args.data_dir / distribution_name, index=False)
        ranked_profiles.to_csv(args.data_dir / ranked_name, index=False)
        density_profiles.to_csv(args.data_dir / density_name, index=False)
        descriptor_audit.to_csv(args.data_dir / descriptor_audit_name, index=False)
        peak_retrieval_rows.to_csv(args.data_dir / peak_retrieval_name, index=False)
    baseline.to_csv(args.output_dir / f"{STEM}_decision_source.csv", index=False)
    display_peak_source = frozen_peak_source.copy()
    display_peak_source["display_order"] = display_peak_source.peak_id.map(
        {peak_id: order for order, peak_id in enumerate(support_ids, start=1)}
    )
    display_peak_source = display_peak_source.sort_values("display_order")
    display_order_name = f"{STEM}_display_order_source.csv"
    display_peak_source.to_csv(args.output_dir / display_order_name, index=False)
    if args.data_dir.resolve() != args.output_dir.resolve():
        display_peak_source.to_csv(args.data_dir / display_order_name, index=False)
    audit = {
        "figure": STEM,
        "canvas_mm": [CANVAS_WIDTH_MM, NEW_HEIGHT_MM],
        "full_width_layout": {
            "target_width_mm": CANVAS_WIDTH_MM,
            "width_distribution": "existing normalized panel map; extra width shared across all evidence panels",
            "vertical_height_unchanged": True,
        },
        "upper_story_height_scale": UPPER_SCALE,
        "lower_descriptor_physical_height_preserved": True,
        "molecule_rendering": {
            "format": "RDKit SVG paths",
            "n_svg_assets": len(list(asset_dir.glob("*.svg"))),
            "embedded_raster_images_in_main_svg": 0,
            "bond_linewidth_pt": 0.5,
            "multiple_bond_offset": 0.19,
            "bond_cap_and_join": "round",
        },
        "sample_idx": 545225,
        "adduct": "[M-H]-",
        "formula": formula,
        "n_benchmark_entries": 62,
        "n_connectivity_structures": 60,
        "support_peak_ids": support_ids[:4],
        "peak_embedding_retrieval": peak_retrieval_summary,
        "descriptor_panel_candidates": [
            f"UltraMS top {DISPLAY_TOP_K}", f"DreaMS top {DISPLAY_TOP_K}"
        ],
        "descriptor_axis_reference": "range across the same 60 connectivity candidates",
        "descriptor_distribution": {
            "definition": "Gaussian-kernel-smoothed empirical distribution within each model's top 20 after excluding the measured target",
            "n_candidates_per_model": DISPLAY_TOP_K - 1,
            "bandwidth": "Silverman-style robust scale with a 3.5% full-range floor; discrete count descriptors additionally use a 0.35-unit floor",
        },
        "descriptor_panel_metrics": [
            "TPSA", "cLogP", "phenolic OH groups", "carbonyl groups",
            "rotatable bonds", "fraction Csp3",
        ],
        "peak_selection_uses_model_outputs": False,
        "baseline": {
            row.model: {
                "true_rank": int(row.true_rank),
                "fixed_decoy_margin": float(row.fixed_decoy_margin),
            }
            for row in baseline.itertuples(index=False)
        },
        "formats_complete": all((args.output_dir / f"{STEM}.{suffix}").exists()
                                for suffix in ["png", "pdf", "svg"]),
        "guardrails": [
            "All benchmark candidates share the query formula.",
            "Peak selection is model-independent: shared formula, different connectivity, CE20 repetition, then intensity.",
            "MAGMa structures are candidate fragment interpretations.",
            "Display order places the two adjacent C7O5H3 assignments together; it does not relabel them as isotopes.",
            "For m/z 316, the highest-cosine source-valid topology was frozen before target/Pleurostimin compatibility was evaluated.",
            "This is a frozen illustrative case, not a population effect estimate.",
        ],
    }
    audit_text = json.dumps(audit, indent=2)
    (args.output_dir / f"{STEM}_audit.json").write_text(audit_text)
    if args.data_dir.resolve() != args.output_dir.resolve():
        (args.data_dir / f"{STEM}_audit.json").write_text(audit_text)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
