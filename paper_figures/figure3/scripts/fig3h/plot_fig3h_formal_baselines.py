#!/usr/bin/env python3
"""Add formal baselines to the existing Figure 3h panel types."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
PANEL = SCRIPT.parents[1]
SOURCE = PANEL / "source_data" / "formal_seven_raw_matchms"
OTHER_SOURCE = PANEL / "source_data" / "other_strict_adduct_v5_s2s_raw_seven_methods"
OUTPUT = PANEL / "output"
MAIN_OUTPUT = PANEL.parents[1] / "output"
COLORS = {
    "UltraMS": "#3965BB",
    "DreaMS": "#E9A0AA",
    "Linear": "#A7A7A7",
    "DeepSets": "#C48A4A",
    "Fourier": "#8767AD",
    "Codebook": "#5AA69A",
    "Raw cosine": "#4A4A4A",
}
LINESTYLES = {
    "UltraMS": "-",
    "DreaMS": "--",
    "Linear": (0, (1, 1)),
    "DeepSets": (0, (4, 1.5)),
    "Fourier": ":",
    "Codebook": "-.",
    "Raw cosine": (0, (6, 1.5, 1.2, 1.5)),
}
ADDUCTS = {"pos": "[M+H]+", "neg": "[M-H]-"}
ADDITIONAL_ADDUCTS = ("[M+NH4]+", "[M+Cl]-", "[M+FA]-", "[M+H-H2O]+", "[M+Na]+")
ADDITIONAL_GROUPS = {
    "Additional\npositive": ("[M+H-H2O]+", "[M+NH4]+", "[M+Na]+"),
    "Additional\nnegative": ("[M+Cl]-", "[M+FA]-"),
}
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
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 5.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
        }
    )


def load_formal() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    marker = json.loads((SOURCE / "DONE.json").read_text())
    if (
        marker.get("status") != "complete"
        or marker.get("formal_method_completeness_enforced") is not True
        or marker.get("primary_model_seed") != 0
    ):
        raise RuntimeError("formal Figure 3h completion marker is required")
    curves = pd.read_csv(SOURCE / "fdr_envelopes_primary_seed0.csv.gz")
    working = pd.read_csv(SOURCE / "working_points_primary_seed0.csv")
    low = pd.read_csv(SOURCE / "low_fdr_mean_recall_primary_seed0.csv")
    curves = curves[curves.split == "test"].copy()
    working = working[working.split == "test"].copy()
    low = low[low.split == "test"].copy()
    required = set(COLORS)
    for frame in (curves, working, low):
        if set(frame.method) != required or set(frame["mode"]) != {"pos", "neg"}:
            raise ValueError("formal source table is missing a displayed method or adduct")
    # One stable legend order, ranked by the mean low-FDR recall over both
    # strict-adduct tasks. Bars are independently sorted within each task.
    order = (
        low.groupby("method").mean_identification_recall.mean()
        .sort_values(kind="stable").index.tolist()
    )
    return curves, working, low, order


def build_additional_micro_summary(full: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, adducts in ADDITIONAL_GROUPS.items():
        frame = full[full.adduct.isin(adducts)]
        for method in COLORS:
            selected = frame[frame.method == method]
            n_query = int(selected.n_query.sum())
            accepted = int(selected.accepted.sum())
            correct = int(selected.correct.sum())
            errors = accepted - correct
            rows.append(
                {
                    "group": group,
                    "method": method,
                    "adducts": ";".join(adducts),
                    "n_adducts": len(adducts),
                    "n_query": n_query,
                    "sum_separate_library_sizes": int(selected.n_library.sum()),
                    "accepted": accepted,
                    "correct": correct,
                    "errors": errors,
                    "pooled_empirical_fdr": errors / accepted,
                    "micro_identification_recall": correct / n_query,
                    "mean_micro_identification_recall_fdr_0_5pct": float(
                        np.average(
                            selected.mean_identification_recall_fdr_0_5pct,
                            weights=selected.n_query,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_additional_adducts() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = json.loads((OTHER_SOURCE / "AUDIT.json").read_text())
    summary_path = OTHER_SOURCE / "other_strict_adduct_low_fdr_summary.csv"
    if audit.get("status") != "complete" or audit.get("target_fdr") != 0.05:
        raise RuntimeError("completed 5% FDR other-adduct audit is required")
    expected_hash = audit["outputs"][summary_path.name]["sha256"]
    if sha256(summary_path) != expected_hash:
        raise RuntimeError("other-adduct summary hash does not match its audit")
    full = pd.read_csv(summary_path)
    required_methods = set(COLORS)
    expected_adducts = {"[M+H-H2O]+", "[M+NH4]+", "[M+Na]+", "[M+Cl]-", "[M+FA]-"}
    if set(full.method) != required_methods or set(full.adduct) != expected_adducts:
        raise ValueError("other-adduct source table is incomplete")
    display = full[full.adduct.isin(ADDITIONAL_ADDUCTS)].copy()
    if len(display) != len(COLORS) * len(ADDITIONAL_ADDUCTS):
        raise ValueError("displayed additional-adduct rows are incomplete")
    micro = build_additional_micro_summary(display)
    if len(micro) != len(COLORS) * len(ADDITIONAL_GROUPS):
        raise ValueError("additional-adduct micro summary is incomplete")
    if not (micro.pooled_empirical_fdr <= 0.05 + 1e-12).all():
        raise ValueError("pooled additional-adduct FDR exceeds 5%")
    return micro, full


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def draw_fdr(
    ax: plt.Axes,
    curves: pd.DataFrame,
    working: pd.DataFrame,
    mode: str,
    methods: list[str],
    *,
    show_legend: bool,
) -> None:
    for method in methods:
        frame = curves[(curves.method == method) & (curves["mode"] == mode)].sort_values("fdr_budget")
        ax.plot(
            100 * frame.fdr_budget,
            100 * frame.identification_recall_envelope,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.0,
            label=method,
        )
        point = working[(working.method == method) & (working["mode"] == mode)].iloc[0]
        ax.scatter(
            [5.0], [100 * point.identification_recall], s=14,
            color=COLORS[method], edgecolor="white", linewidth=0.35, zorder=5,
        )
    ax.axvline(5.0, color="#737B82", linestyle=":", linewidth=0.7)
    xmax = 5.5 if mode == "pos" else 7.5
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 102)
    ax.set_xlabel("False discovery rate (%)")
    ax.set_ylabel("Recall (%)")
    ax.set_title(ADDUCTS[mode], pad=3, fontweight="normal")
    ticks = [0, 2.5, 5] if mode == "pos" else [0, 2.5, 5, 7.5]
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    ax.set_yticks([0, 25, 50, 75, 100])
    if show_legend:
        ax.legend(loc="lower right", ncol=2, handlelength=1.15,
                  handletextpad=0.3, columnspacing=0.55, labelspacing=0.2)
    clean_axis(ax)


def draw_bars(ax: plt.Axes, low: pd.DataFrame, methods: list[str], *, show_legend: bool) -> None:
    x = np.arange(2, dtype=float)
    width = 0.12
    # Sort low-to-high separately inside each adduct group, as in Figure 3a.
    for mode_index, mode in enumerate(("pos", "neg")):
        frame = low[low["mode"] == mode].set_index("method")
        ranked = sorted(methods, key=lambda method: float(frame.loc[method, "mean_identification_recall"]))
        offsets = width * (np.arange(len(ranked)) - (len(ranked) - 1) / 2)
        for method, offset in zip(ranked, offsets):
            value = 100 * float(frame.loc[method, "mean_identification_recall"])
            bar = ax.bar(
                x[mode_index] + offset, value, width=width, color=COLORS[method], alpha=0.92,
                edgecolor="white", linewidth=0.35, label=method if mode_index == 0 else None,
            )[0]
            if method in {"DreaMS", "UltraMS"}:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.0f}",
                    ha="center", va="bottom", fontsize=4.8, color=COLORS[method],
                )
    ax.set_xticks(x, [ADDUCTS[mode] for mode in ("pos", "neg")])
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Mean recall, FDR 0–5% (%)")
    if show_legend:
        ax.legend(loc="upper right", ncol=2, handlelength=1.0,
                  handletextpad=0.3, columnspacing=0.55, labelspacing=0.2)
    clean_axis(ax)


def draw_additional_adduct_bars(
    ax: plt.Axes,
    micro: pd.DataFrame,
    ordered: list[str] | None = None,
    *,
    show_ylabel: bool = True,
) -> None:
    if ordered is None:
        ordered = list(ADDITIONAL_GROUPS)
    pivot = micro.pivot(
        index="group",
        columns="method",
        values="mean_micro_identification_recall_fdr_0_5pct",
    )
    x = np.arange(len(ordered), dtype=float)
    width = 0.14
    for group_index, group in enumerate(ordered):
        ranked = sorted(COLORS, key=lambda method: float(pivot.loc[group, method]))
        offsets = width * (np.arange(len(ranked)) - (len(ranked) - 1) / 2)
        for method, offset in zip(ranked, offsets):
            value = 100 * float(pivot.loc[group, method])
            bar = ax.bar(
                x[group_index] + offset,
                value,
                width=width,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )[0]
            if method in {"DreaMS", "UltraMS"}:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.1,
                    f"{value:.1f}",
                    color=COLORS[method],
                    fontsize=4.8,
                    ha="center",
                    va="bottom",
                )
    ax.set_xticks(x, ordered)
    ax.set_xlim(-0.55, len(ordered) - 0.45)
    ax.set_ylim(0, 43)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Mean micro recall, FDR 0–5% (%)" if show_ylabel else "")
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


def validate_combined_layout(
    fig: plt.Figure,
    curve_axes: list[plt.Axes],
    bar_axes: list[plt.Axes],
) -> dict[str, str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox

    def visible_texts(ax: plt.Axes) -> list[plt.Text]:
        candidates = [
            ax.xaxis.label,
            ax.yaxis.label,
            ax.title,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
        ]
        return [text for text in candidates if text.get_visible() and text.get_text().strip()]

    all_texts = [
        *visible_texts(curve_axes[0]),
        *visible_texts(curve_axes[1]),
        *(text for axis in bar_axes for text in visible_texts(axis)),
    ]
    outside = []
    for text in all_texts:
        box = text.get_window_extent(renderer=renderer)
        if box.x0 < figure_box.x0 - 0.5 or box.y0 < figure_box.y0 - 0.5:
            outside.append(text.get_text())
        if box.x1 > figure_box.x1 + 0.5 or box.y1 > figure_box.y1 + 0.5:
            outside.append(text.get_text())
    if outside:
        raise RuntimeError(f"combined Figure 3h has clipped text: {sorted(set(outside))}")

    top_texts = [*visible_texts(curve_axes[0]), *visible_texts(curve_axes[1])]
    bottom_texts = [text for axis in bar_axes for text in visible_texts(axis)]
    cross_panel_overlap = []
    for top_text in top_texts:
        top_box = top_text.get_window_extent(renderer=renderer)
        for bottom_text in bottom_texts:
            bottom_box = bottom_text.get_window_extent(renderer=renderer)
            if top_box.overlaps(bottom_box):
                cross_panel_overlap.append((top_text.get_text(), bottom_text.get_text()))
    if cross_panel_overlap:
        raise RuntimeError(f"combined Figure 3h has cross-panel text overlap: {cross_panel_overlap}")

    for bar_axis in bar_axes:
        bar_text_boxes = [text.get_window_extent(renderer=renderer) for text in bar_axis.texts]
        for index, left in enumerate(bar_text_boxes):
            for right in bar_text_boxes[index + 1 :]:
                if left.overlaps(right):
                    raise RuntimeError("additional-adduct numeric labels overlap")
        for text, text_box in zip(bar_axis.texts, bar_text_boxes):
            for patch in bar_axis.patches:
                patch_box = patch.get_window_extent(renderer=renderer)
                if text_box.overlaps(patch_box):
                    raise RuntimeError(f"bar annotation overlaps a bar: {text.get_text()}")
    return {
        "text_clipping_check": "passed",
        "cross_panel_text_overlap_check": "passed",
        "bar_annotation_overlap_check": "passed",
    }


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, object] = {}
        if suffix == "png":
            kwargs["dpi"] = 450
        fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def main() -> None:
    set_style()
    curves, working, low, methods = load_formal()
    additional_micro, additional_full = load_additional_adducts()
    for mode, suffix in (("pos", "1"), ("neg", "2")):
        fig, ax = plt.subplots(figsize=(42 * MM, 42 * MM))
        draw_fdr(ax, curves, working, mode, methods, show_legend=True)
        fig.subplots_adjust(left=0.25, right=0.965, bottom=0.25, top=0.86)
        save(fig, OUTPUT / f"fig3h_{suffix}")
    fig_bar, ax_bar = plt.subplots(1, 2, figsize=(102.5 * MM, 34 * MM), sharey=True)
    draw_additional_adduct_bars(ax_bar[0], additional_micro, ["Additional\npositive"])
    draw_additional_adduct_bars(
        ax_bar[1], additional_micro, ["Additional\nnegative"], show_ylabel=False
    )
    ax_bar[1].tick_params(axis="y", labelleft=False)
    fig_bar.subplots_adjust(left=0.12, right=0.985, bottom=0.30, top=0.93, wspace=0.27)
    save(fig_bar, OUTPUT / "fig3h_3")

    combined = plt.figure(figsize=(102.5 * MM, 68 * MM))
    axes = [
        combined.add_axes((0.13, 0.55, 0.37, 0.24)),
        combined.add_axes((0.60, 0.55, 0.37, 0.24)),
    ]
    draw_fdr(axes[0], curves, working, "pos", methods, show_legend=False)
    draw_fdr(axes[1], curves, working, "neg", methods, show_legend=False)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    bar_axes = [
        combined.add_axes((0.13, 0.18, 0.37, 0.20)),
        combined.add_axes((0.60, 0.18, 0.37, 0.20)),
    ]
    draw_additional_adduct_bars(
        bar_axes[0], additional_micro, ["Additional\npositive"]
    )
    draw_additional_adduct_bars(
        bar_axes[1], additional_micro, ["Additional\nnegative"], show_ylabel=False
    )
    bar_axes[1].tick_params(axis="y", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    combined.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
        borderaxespad=0.12,
    )
    layout_checks = validate_combined_layout(combined, axes, bar_axes)
    save(combined, OUTPUT / "fig3h")
    MAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        (MAIN_OUTPUT / f"fig3h.{extension}").write_bytes(
            (OUTPUT / f"fig3h.{extension}").read_bytes()
        )

    curves.to_csv(OUTPUT / "fig3h_fdr_curve_data.csv", index=False)
    working.to_csv(OUTPUT / "fig3h_working_point_data.csv", index=False)
    low.to_csv(OUTPUT / "fig3h_mean_recall_data.csv", index=False)
    additional_full.to_csv(OUTPUT / "fig3h_additional_adduct_data.csv", index=False)
    additional_micro.to_csv(OUTPUT / "fig3h_additional_adduct_micro_summary.csv", index=False)
    audit = {
        "status": "complete",
        "figure_size_mm": [102.5, 68.0],
        "height_matched_to": "Figure 3b",
        "render_layout_checks": layout_checks,
        "visual_types": [
            "strict [M+H]+ FDR-recall curve",
            "strict [M-H]- FDR-recall curve",
            "two aligned bar panels for query-micro-averaged mean recall over FDR 0–5% in additional positive and negative groups",
        ],
        "additional_adduct_metric": "query-micro-averaged exact-envelope mean identification recall over FDR 0–5% across all included exact-adduct tasks",
        "additional_adduct_displayed": list(ADDITIONAL_ADDUCTS),
        "additional_adduct_groups": {
            group.replace("\n", " "): list(adducts)
            for group, adducts in ADDITIONAL_GROUPS.items()
        },
        "additional_adduct_training": "all four controls were trained on distinct same-molecule spectrum pairs from the frozen composite-polarity training compounds; each displayed additional adduct contributes its own training spectra",
        "learned_control_supervision": "symmetric same-molecule spectrum-pair InfoNCE; no molecular embedding target and no ChemBERTa",
        "raw_cosine_training": "not applicable; MatchMS CosineGreedy is evaluated directly on the frozen peak lists",
        "primary_model_seed": 0,
        "methods": methods,
        "excluded_from_plot": ["Spec2Vec", "Binned cosine"],
        "exclusion_scope": "display only; formal result artifacts are retained",
        "strict_adducts": ADDUCTS,
        "sources": {
            path.name: sha256(path)
            for path in (
                SOURCE / "fdr_envelopes_primary_seed0.csv.gz",
                SOURCE / "working_points_primary_seed0.csv",
                SOURCE / "low_fdr_mean_recall_primary_seed0.csv",
                OTHER_SOURCE / "other_strict_adduct_low_fdr_summary.csv",
                OTHER_SOURCE / "other_strict_adduct_per_query.csv.gz",
                OTHER_SOURCE / "other_strict_adduct_exact_fdr_frontiers.csv.gz",
                OTHER_SOURCE / "other_strict_adduct_task_rows.csv.gz",
                OTHER_SOURCE / "AUDIT.json",
                PANEL / "source_data" / "msnlib_adduct_inventory_20260823.csv",
                OUTPUT / "fig3h_additional_adduct_micro_summary.csv",
                OUTPUT / "fig3h_adduct_sample_sizes.csv",
            )
        },
    }
    (OUTPUT / "fig3h_formal_plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
