#!/usr/bin/env python
"""Render peak-neighbor fidelity curves from saved plot-ready data.

This script intentionally does not recompute embeddings, nearest neighbors,
classical predictors, fingerprints, or metrics. It only reads the saved JSON
metrics table and redraws figures for style/layout iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


DEFAULT_MODEL_ORDER = ["UltraMS", "DreaMS", "Peak features", "Linear SVM", "Random forest"]
COLORS = {
    "UltraMS": "#e94b3c",
    "DreaMS": "#4db9d1",
    "Peak features": "#4d4d4d",
    "Linear SVM": "#8e63b7",
    "Random forest": "#3c8d40",
}
MARKERS = {
    "UltraMS": "o",
    "DreaMS": "s",
    "Peak features": "D",
    "Linear SVM": "^",
    "Random forest": "v",
}


def parse_ylim(text: str | None) -> tuple[float, float] | None:
    if text is None or text.strip().lower() in {"", "none", "auto"}:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"ylim must be 'low,high', got {text!r}")
    return parts[0], parts[1]


def model_order(metrics: dict, models_arg: str | None) -> list[str]:
    available = list(metrics["curves"].keys())
    if models_arg:
        requested = [m.strip() for m in models_arg.split(",") if m.strip()]
        missing = [m for m in requested if m not in available]
        if missing:
            raise ValueError(f"Requested models not found in metrics JSON: {missing}")
        return requested
    ordered = [m for m in DEFAULT_MODEL_ORDER if m in available]
    ordered.extend([m for m in available if m not in ordered])
    return ordered


def filter_metrics(metrics: dict, models: list[str], max_k: int | None) -> dict:
    out = deepcopy(metrics)
    out["curves"] = {m: deepcopy(metrics["curves"][m]) for m in models}
    for model in models:
        curve = out["curves"][model]
        k = np.asarray(curve["k"], dtype=int)
        keep = np.ones(len(k), dtype=bool) if max_k is None else k <= max_k
        for key, vals in list(curve.items()):
            if isinstance(vals, list) and len(vals) == len(k):
                curve[key] = [vals[i] for i in np.where(keep)[0]]
    out["plotting"] = out.get("plotting", {})
    out["plotting"]["models"] = models
    out["plotting"]["max_k"] = max_k
    return out


def write_plot_csv(metrics: dict, output_path: Path, models: list[str]) -> None:
    fieldnames = [
        "model",
        "k",
        "formula_hit",
        "fragment_family_hit",
        "best_fp_tanimoto",
        "best_mz_mae_da",
        "best_mz_median_da",
        "n_valid_neighbors",
        "n_valid_fp",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in models:
            curve = metrics["curves"][model]
            for pos, k in enumerate(curve["k"]):
                row = {"model": model, "k": int(k)}
                for key in fieldnames[2:]:
                    row[key] = curve.get(key, [None] * len(curve["k"]))[pos]
                writer.writerow(row)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.2, length=2.8, width=0.75)


def plot_metric(
    ax,
    metrics: dict,
    models: list[str],
    metric: str,
    title: str,
    ylabel: str,
    y_log: bool,
    ylim: tuple[float, float] | None,
) -> None:
    k = metrics["curves"][models[0]]["k"]
    for model in models:
        y = np.asarray(metrics["curves"][model][metric], dtype=float)
        if y_log:
            y = np.maximum(y, 1e-4)
        ax.plot(
            k,
            y,
            color=COLORS.get(model, "#737373"),
            marker=MARKERS.get(model, "o"),
            markersize=2.7,
            linewidth=1.15,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_xlim(min(k) * 0.82, max(k) * 1.16)
    if y_log:
        ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlabel("Top-k cross-spectrum neighbors", fontsize=6.6)
    ax.set_ylabel(ylabel, fontsize=6.6)
    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif not y_log:
        ax.set_ylim(0, 1.02)
    style_axes(ax)


def render(metrics: dict, output_dir: Path, output_prefix: str, plot_tag: str, models: list[str], args) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    specs = [
        ("formula_hit", "Formula agreement", "hit@k", False, parse_ylim(args.formula_ylim), f"fig2_curve_formula_hit_{plot_tag}"),
        (
            "fragment_family_hit",
            "Fragment family agreement",
            "hit@k",
            False,
            parse_ylim(args.family_ylim),
            f"fig2_curve_family_hit_{plot_tag}",
        ),
        (
            "best_fp_tanimoto",
            "Fragment structure similarity",
            "best Tanimoto@k",
            False,
            parse_ylim(args.fp_ylim),
            f"fig2_curve_fp_tanimoto_{plot_tag}",
        ),
        ("best_mz_mae_da", "Mass fidelity", "best m/z error@k (Da)", True, parse_ylim(args.mz_ylim), f"fig2_curve_mz_error_{plot_tag}"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, ylabel, y_log, ylim, stem in specs:
        fig, ax = plt.subplots(figsize=(args.single_width, args.single_height), dpi=args.dpi)
        plot_metric(ax, metrics, models, metric, title, ylabel, y_log, ylim)
        ax.legend(frameon=False, fontsize=5.3, loc="best", handlelength=1.1)
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=args.dpi)
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height), dpi=args.dpi)
    for ax, (metric, title, ylabel, y_log, ylim, _) in zip(axes.ravel(), specs):
        plot_metric(ax, metrics, models, metric, title, ylabel, y_log, ylim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.7,
        loc="upper center",
        ncol=min(args.legend_cols, len(models)),
        bbox_to_anchor=(0.5, 1.012),
        handlelength=1.2,
        columnspacing=0.9,
    )
    n_query = metrics.get("n_query_peaks")
    n_index = metrics.get("n_peaks")
    footer = f"{n_query:,} held-out query peaks; {n_index:,} MAGMa peaks in index" if n_query and n_index else ""
    if footer:
        fig.text(0.5, -0.005, footer, ha="center", va="top", fontsize=6.8)
    fig.tight_layout(pad=1.1, rect=(0, 0.02, 1, 0.945))
    fig.savefig(output_dir / f"{output_prefix}_2x2.png", bbox_inches="tight", dpi=args.dpi)
    fig.savefig(output_dir / f"{output_prefix}_2x2.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-prefix", default="fig2_peak_neighbor_curves_from_data")
    ap.add_argument("--plot-tag", default="from_data")
    ap.add_argument("--max-k", type=int, default=100)
    ap.add_argument("--models", default=None, help="Comma-separated model list. Defaults to all known models in display order.")
    ap.add_argument("--plot-data-csv", default=None)
    ap.add_argument("--formula-ylim", default="0.0,0.82")
    ap.add_argument("--family-ylim", default="0.32,1.02")
    ap.add_argument("--fp-ylim", default="0.12,0.74")
    ap.add_argument("--mz-ylim", default="0.035,75.0")
    ap.add_argument("--width", type=float, default=5.85)
    ap.add_argument("--height", type=float, default=4.45)
    ap.add_argument("--single-width", type=float, default=2.55)
    ap.add_argument("--single-height", type=float, default=2.08)
    ap.add_argument("--legend-cols", type=int, default=5)
    ap.add_argument("--dpi", type=int, default=350)
    args = ap.parse_args()

    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    models = model_order(metrics, args.models)
    metrics = filter_metrics(metrics, models, args.max_k)
    metrics["plotting"].update({
        "source_metrics_json": str(args.metrics_json),
        "output_prefix": args.output_prefix,
        "plot_tag": args.plot_tag,
        "formula_ylim": args.formula_ylim,
        "family_ylim": args.family_ylim,
        "fp_ylim": args.fp_ylim,
        "mz_ylim": args.mz_ylim,
    })

    plot_csv = args.plot_data_csv or f"{args.output_prefix}_plot_data.csv"
    write_plot_csv(metrics, args.output_dir / plot_csv, models)
    with open(args.output_dir / f"{args.output_prefix}_plot_config.json", "w", encoding="utf-8") as f:
        json.dump(metrics["plotting"], f, indent=2)
    render(metrics, args.output_dir, args.output_prefix, args.plot_tag, models, args)
    print(f"wrote {args.output_dir / (args.output_prefix + '_2x2.png')}")
    print(f"wrote {args.output_dir / plot_csv}")


if __name__ == "__main__":
    main()
