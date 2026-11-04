"""
One-row four-column version of the mixed S/F/Cl/Br atom-attention case panel.

This uses the same cached plot data and selected cases as v15, but redraws the
cards as a 1x4 strip for figure assembly.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atom_attention_viz_magma_v15_mixed_top4_grid_full_fragments import (
    DEFAULT_CACHE_JSONS,
    DEFAULT_CARDS,
    OUTPUT_BASENAME as V15_BASENAME,
    build_cache_lookup,
    build_top_atom_callouts,
    draw_fragment_grid,
    find_record,
    make_selected_records,
    parse_card_specs,
    plot_atom_spectrum,
    resolve_project_path,
)
from atom_attention_viz_magma_v12_br_candidates import write_peak_table_csv
from atom_attention_viz_magma_v2 import _PROJECT_ROOT


OUTPUT_BASENAME = "atom_attention_magma_ultra_mixed_top4_row4_v16"


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


def render_row4(caches, lookup, card_specs, out_prefix):
    fig = plt.figure(figsize=(15.6, 5.2))
    grid = fig.add_gridspec(
        1,
        4,
        left=0.035,
        right=0.995,
        top=0.93,
        bottom=0.08,
        wspace=0.12,
    )

    for card_i, spec in enumerate(card_specs):
        cache, pos = find_record(caches, lookup, spec["global_index"])
        atom = spec["atom"]
        atom_idx = cache["atom_names"].index(atom)
        sample = cache["samples"][pos]
        global_idx = int(cache["global_indices"][pos])
        callouts = build_top_atom_callouts(cache, pos, global_idx, atom, max_callouts=6)
        sub = grid[0, card_i].subgridspec(2, 1, height_ratios=[1.00, 1.02], hspace=0.12)
        spec_ax = fig.add_subplot(sub[0, 0])
        frag_ax = fig.add_subplot(sub[1, 0])
        weights = cache["attn_maps"][pos, atom_idx, :len(sample["spectrum"])]
        mask = cache["magma_annots"][pos][:len(sample["spectrum"]), atom_idx]
        plot_atom_spectrum(spec_ax, sample, weights, mask, callouts, global_idx, atom, card_i)
        if card_i > 0:
            spec_ax.set_ylabel("")
            spec_ax.set_yticklabels([])
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
        default=os.path.join(_PROJECT_ROOT, "main_figure", "fig2", "output", "case_study_mixed_top4_row4_v16"),
    )
    args = parser.parse_args()

    cache_paths = [resolve_project_path(value) for value in args.cache_json]
    from atom_attention_viz_magma_v12_br_candidates import load_plot_data_cache

    caches = [load_plot_data_cache(path) for path in cache_paths]
    lookup = build_cache_lookup(caches)
    card_specs = parse_card_specs(args.card)
    out_dir = resolve_project_path(args.out_dir)
    out_prefix = out_dir / OUTPUT_BASENAME

    render_row4(caches, lookup, card_specs, out_prefix)
    selected_records, selected_callouts = make_selected_records(caches, lookup, card_specs)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "fig2_case_study_mixed_top4_row4_v16",
        "source_script": V15_BASENAME,
        "source_cache_jsons": [str(path) for path in cache_paths],
        "cards": card_specs,
        "selected_callouts": selected_callouts,
    }
    (out_dir / f"{OUTPUT_BASENAME}.summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{OUTPUT_BASENAME}.plot_data.json").write_text(
        json.dumps(
            {
                "stage": "fig2_case_study_mixed_top4_row4_v16_plot_data",
                "source_cache_jsons": [str(path) for path in cache_paths],
                "cards": card_specs,
                "selected_callouts": selected_callouts,
                "records": selected_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_peak_table_csv(
        out_dir / f"{OUTPUT_BASENAME}.peak_table.csv",
        selected_records,
        caches[0]["atom_names"],
        selected_callouts,
    )
    print(f"Saved row4 PNG/PDF and data under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
