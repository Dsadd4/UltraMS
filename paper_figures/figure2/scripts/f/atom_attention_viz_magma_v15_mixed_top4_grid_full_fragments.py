"""
Compact 2x2 mixed atom-query panel for Figure 2 refinement.

This redraws four approved cases from existing complete plot-data caches. It
does not run model inference or MAGMa. The default cards cover S, F, Cl and Br:
  26580:S, 28053:F, 20940:Cl, 7960:Br
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from atom_attention_viz_magma_v2 import _PROJECT_ROOT, ATOM_COLORS, DEFAULT_COLOR
from atom_attention_viz_magma_v12_br_candidates import (
    ATTENTION_STICK,
    BASELINE,
    NEUTRAL_PEAK,
    TEXT,
    composite_on_white,
    draw_fragment_molecule,
    draw_parent_molecule,
    estimate_bar_width,
    load_plot_data_cache,
    molecular_formula,
    normalize_attention,
    write_peak_table_csv,
)

OUTPUT_BASENAME = "atom_attention_magma_ultra_mixed_top4_grid_full_fragments_v15"
DEFAULT_CARDS = ["26580:S", "28053:F", "20940:Cl", "7960:Br"]
DEFAULT_CACHE_JSONS = [
    os.path.join(
        _PROJECT_ROOT,
        "main_figure",
        "fig2",
        "output",
        "case_study_v11_lower_inset_more",
        "atom_attention_magma_ultra_lower_inset_more_v11.plot_data.json",
    ),
    os.path.join(
        _PROJECT_ROOT,
        "main_figure",
        "fig2",
        "output",
        "case_study_br_candidates_v12",
        "atom_attention_magma_ultra_br_candidates_v12.plot_data.json",
    ),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(_PROJECT_ROOT) / path


def parse_card_specs(values):
    specs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Card spec must look like global_index:atom, got {value!r}")
        global_idx, atom = value.split(":", 1)
        specs.append({"global_index": int(global_idx), "atom": atom})
    if len(specs) != 4:
        raise ValueError("This compact grid expects exactly four cards.")
    return specs


def build_cache_lookup(caches):
    lookup = {}
    for cache_i, cache in enumerate(caches):
        for pos, global_idx in enumerate(cache["global_indices"]):
            lookup[int(global_idx)] = (cache_i, pos)
    return lookup


def add_parent_inset(ax, smiles, atom, card_i):
    y0 = 0.58 if card_i == 0 else 0.48
    mol_ax = ax.inset_axes([0.60, y0, 0.34, 0.43], zorder=12)
    mol_ax.axis("off")
    mol_ax.patch.set_visible(True)
    mol_ax.patch.set_facecolor("white")
    mol_ax.patch.set_alpha(0.96)
    image = draw_parent_molecule(smiles, [atom])
    if image is not None:
        mol_ax.imshow(composite_on_white(image))


def plot_atom_spectrum(ax, sample, weights, magma_mask, callouts, global_idx, atom, card_i):
    spec = sample["spectrum"]
    mz = np.asarray(spec[:, 0], dtype=float)
    inten = np.asarray(spec[:, 1], dtype=float)
    n = min(len(mz), len(inten), len(weights), len(magma_mask))
    mz = mz[:n]
    inten = inten[:n]
    weights = np.asarray(weights[:n], dtype=float)
    magma_mask = np.asarray(magma_mask[:n], dtype=bool)
    atom_color = ATOM_COLORS.get(atom, DEFAULT_COLOR)
    attn_norm, _ = normalize_attention(weights)
    attn_depth = 0.42 * attn_norm
    xlim = (float(np.min(mz)) - 8, float(np.max(mz)) + 8)
    cap_width = estimate_bar_width(mz) * 0.30

    non_hit = ~magma_mask
    ax.axhline(0, color=BASELINE, linewidth=0.72, zorder=0)
    ax.vlines(mz[non_hit], 0, inten[non_hit], color=NEUTRAL_PEAK, linewidth=0.58, zorder=1)
    ax.scatter(
        mz[non_hit],
        inten[non_hit],
        s=7,
        color="#f0f0f0",
        edgecolor=NEUTRAL_PEAK,
        linewidth=0.32,
        zorder=2,
    )
    ax.vlines(mz[magma_mask], 0, inten[magma_mask], color=atom_color, linewidth=1.05, zorder=3)
    ax.scatter(
        mz[magma_mask],
        inten[magma_mask],
        s=12,
        color=atom_color,
        edgecolor="#303030",
        linewidth=0.32,
        zorder=4,
    )
    ax.vlines(
        mz,
        0,
        -attn_depth,
        color=mcolors.to_rgba(ATTENTION_STICK, 0.34),
        linewidth=0.55,
        zorder=1,
    )
    for idx in np.argsort(-attn_norm)[: min(12, len(attn_norm))]:
        x = float(mz[idx])
        y = -float(attn_depth[idx])
        alpha = 0.84 if magma_mask[idx] else 0.45
        ax.hlines(
            y,
            x - cap_width,
            x + cap_width,
            color=mcolors.to_rgba(ATTENTION_STICK, alpha),
            linewidth=0.82,
            zorder=3,
        )

    for callout in callouts:
        peak_id = int(callout["peak_id"])
        if peak_id >= len(mz):
            continue
        x = float(mz[peak_id])
        y = min(float(inten[peak_id]) + 0.12, 1.13)
        ax.text(
            x,
            y,
            str(callout["number"]),
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color="white",
            zorder=10,
            bbox=dict(
                boxstyle="circle,pad=0.17",
                facecolor=atom_color,
                edgecolor="white",
                linewidth=0.55,
            ),
            clip_on=True,
        )
        ax.hlines(
            -float(attn_depth[peak_id]),
            x - cap_width * 1.35,
            x + cap_width * 1.35,
            color=ATTENTION_STICK,
            linewidth=1.05,
            zorder=5,
        )

    ax.text(
        0.012,
        0.91,
        atom,
        transform=ax.transAxes,
        color=atom_color,
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    formula = molecular_formula(sample["smiles"])
    ax.text(
        0.0,
        1.08,
        f"Case {global_idx} | {atom} | {formula}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        color=TEXT,
    )
    add_parent_inset(ax, sample["smiles"], atom, card_i)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.48, 1.20)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"])
    ax.set_ylabel("Intensity", fontsize=6.5)
    ax.set_xlabel("m/z", fontsize=6.5)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(length=2.0, width=0.6, color="#303030", pad=1.4)


def draw_fragment_grid(ax, callouts, atom):
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "MAGMa fragments",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=TEXT,
    )
    n_cols = 2
    n_rows = 3
    top = 0.91
    bottom = 0.02
    row_h = (top - bottom) / n_rows
    col_w = 0.49
    atom_color = ATOM_COLORS.get(atom, DEFAULT_COLOR)
    for i, callout in enumerate(callouts[:6]):
        col = i % n_cols
        row = i // n_cols
        x0 = 0.01 + col * 0.50
        y0 = top - (row + 1) * row_h + row_h * 0.02
        if row > 0 and col == 0:
            ax.plot(
                [0.0, 0.98],
                [y0 + row_h * 0.98, y0 + row_h * 0.98],
                transform=ax.transAxes,
                color="#ededed",
                linewidth=0.45,
                clip_on=False,
            )
        ax.text(
            x0 + 0.030,
            y0 + row_h * 0.18,
            str(callout["number"]),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="circle,pad=0.18",
                facecolor=atom_color,
                edgecolor="white",
                linewidth=0.55,
            ),
        )
        img_ax = ax.inset_axes([x0 + 0.038, y0 + row_h * 0.30, col_w * 0.88, row_h * 0.64])
        img_ax.axis("off")
        image = draw_fragment_molecule(callout.get("frag_smiles"), atom)
        if image is not None:
            img_ax.imshow(composite_on_white(image))
        else:
            img_ax.text(
                0.5,
                0.5,
                callout.get("frag_formula") or "fragment",
                ha="center",
                va="center",
                fontsize=6,
                color=TEXT,
            )
        ax.text(
            x0 + 0.075,
            y0 + row_h * 0.18,
            f"{atom} | m/z {callout['mz']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color=TEXT,
        )
        ax.text(
            x0 + 0.075,
            y0 + row_h * 0.075,
            callout.get("frag_formula") or "",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.55,
            color="#555555",
        )


def renumber_callouts(callouts):
    out = []
    for number, callout in enumerate(sorted(callouts, key=lambda c: int(c["number"])), 1):
        item = dict(callout)
        item["number"] = int(number)
        out.append(item)
    return out


def get_plot_record(cache, global_idx):
    for rec in cache["plot_records"]:
        if int(rec["global_index"]) == int(global_idx):
            return rec
    raise ValueError(f"Global index {global_idx} not found in cache records.")


def build_top_atom_callouts(cache, pos, global_idx, atom, max_callouts=6):
    rec = get_plot_record(cache, global_idx)
    mzs = rec["peaks"]["mz"]
    intensities = rec["peaks"]["intensity"]
    attention = rec["attention_by_atom"][atom]
    hits = rec["magma_hit_by_atom"][atom]
    details = rec.get("magma_details_by_peak", {})

    candidates = []
    for peak_id, (mz, intensity, attn, hit) in enumerate(zip(mzs, intensities, attention, hits)):
        det = details.get(str(peak_id), {})
        if not hit or not det.get("frag_smiles"):
            continue
        candidates.append({
            "global_index": int(global_idx),
            "case_idx": int(pos),
            "atom": atom,
            "peak_id": int(peak_id),
            "mz": float(mz),
            "intensity": float(intensity),
            "attention": float(attn),
            "frag_formula": det.get("frag_formula"),
            "frag_smiles": det.get("frag_smiles"),
        })

    candidates = sorted(candidates, key=lambda c: (-c["attention"], c["mz"]))
    selected = []
    seen = set()
    for candidate in candidates:
        key = (candidate["peak_id"], candidate.get("frag_smiles"))
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        if len(selected) >= max_callouts:
            break

    for number, candidate in enumerate(selected, 1):
        candidate["number"] = int(number)
    return selected


def find_record(caches, lookup, global_idx):
    if int(global_idx) not in lookup:
        raise ValueError(f"Global index {global_idx} not found in provided caches.")
    cache_i, pos = lookup[int(global_idx)]
    return caches[cache_i], pos


def make_selected_records(caches, lookup, card_specs):
    selected_records = []
    selected_callouts = []
    seen_records = set()
    for spec in card_specs:
        cache, pos = find_record(caches, lookup, spec["global_index"])
        global_idx = int(spec["global_index"])
        if global_idx not in seen_records:
            rec = next(
                rec for rec in cache["plot_records"]
                if int(rec["global_index"]) == global_idx
            )
            selected_records.append(rec)
            seen_records.add(global_idx)
        selected_callouts.extend(
            build_top_atom_callouts(cache, pos, global_idx, spec["atom"], max_callouts=6)
        )
    return selected_records, selected_callouts


def render_grid(caches, lookup, card_specs, out_prefix):
    fig = plt.figure(figsize=(12.8, 9.0))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.985,
        top=0.955,
        bottom=0.06,
        wspace=0.10,
        hspace=0.18,
    )

    for card_i, spec in enumerate(card_specs):
        cache, pos = find_record(caches, lookup, spec["global_index"])
        atom = spec["atom"]
        atom_idx = cache["atom_names"].index(atom)
        sample = cache["samples"][pos]
        global_idx = int(cache["global_indices"][pos])
        callouts = build_top_atom_callouts(cache, pos, global_idx, atom, max_callouts=6)
        row = card_i // 2
        col = card_i % 2
        sub = grid[row, col].subgridspec(1, 2, width_ratios=[1.44, 1.05], wspace=0.055)
        spec_ax = fig.add_subplot(sub[0, 0])
        frag_ax = fig.add_subplot(sub[0, 1])
        weights = cache["attn_maps"][pos, atom_idx, :len(sample["spectrum"])]
        mask = cache["magma_annots"][pos][:len(sample["spectrum"]), atom_idx]
        plot_atom_spectrum(spec_ax, sample, weights, mask, callouts, global_idx, atom, card_i)
        draw_fragment_grid(frag_ax, callouts, atom)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-json", nargs="+", default=DEFAULT_CACHE_JSONS)
    parser.add_argument("--card", nargs="+", default=DEFAULT_CARDS)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "main_figure", "fig2", "output", "case_study_mixed_top4_grid_full_fragments_v15"),
    )
    args = parser.parse_args()

    cache_paths = [resolve_project_path(value) for value in args.cache_json]
    caches = [load_plot_data_cache(path) for path in cache_paths]
    lookup = build_cache_lookup(caches)
    card_specs = parse_card_specs(args.card)
    out_dir = resolve_project_path(args.out_dir)
    out_prefix = out_dir / OUTPUT_BASENAME

    render_grid(caches, lookup, card_specs, out_prefix)
    selected_records, selected_callouts = make_selected_records(caches, lookup, card_specs)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "fig2_case_study_mixed_top4_grid_full_fragments_v15",
        "source_cache_jsons": [str(path) for path in cache_paths],
        "cards": card_specs,
        "selected_callouts": selected_callouts,
    }
    (out_dir / f"{OUTPUT_BASENAME}.summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{OUTPUT_BASENAME}.plot_data.json").write_text(
        json.dumps({
            "stage": "fig2_case_study_mixed_top4_grid_full_fragments_v15_plot_data",
            "source_cache_jsons": [str(path) for path in cache_paths],
            "cards": card_specs,
            "selected_callouts": selected_callouts,
            "records": selected_records,
        }, indent=2),
        encoding="utf-8",
    )
    write_peak_table_csv(
        out_dir / f"{OUTPUT_BASENAME}.peak_table.csv",
        selected_records,
        caches[0]["atom_names"],
        selected_callouts,
    )
    print(f"Saved PNG/PDF and data under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
