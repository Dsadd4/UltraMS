#!/usr/bin/env python3
"""Render the formal validation-to-test FDR-transfer supplement.

The script is deliberately plot-only: it consumes the frozen, repeated
molecule-disjoint transfer table produced by ``aggregate_fig3h_results.py``.
No threshold selection or metric recomputation is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["Binned cosine", "DreaMS", "UltraMS"]
COLORS = {
    "UltraMS": "#3965BB",
    "DreaMS": "#E9A0AA",
    "Binned cosine": "#666666",
    "Spec2Vec": "#5AA69A",
}
MARKERS = {"UltraMS": "o", "DreaMS": "D", "Binned cosine": "s", "Spec2Vec": "^"}
INK = "#25282B"
MID = "#737B82"
GRID = "#D9DEE2"
MM = 1 / 25.4


def sha256_file(path: Path) -> str:
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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, object] = {"bbox_inches": None}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(output.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.34, 0.23, 0.34, "Calibration\nhalf"),
        (0.385, 0.34, 0.23, 0.34, "Lock score\nthreshold"),
        (0.75, 0.34, 0.23, 0.34, "Held-out\nhalf"),
    ]
    for x, y, width, height, label in boxes:
        patch = plt.Rectangle(
            (x, y), width, height, facecolor="#F5F6F7", edgecolor="#B9C0C5", linewidth=0.65
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=INK)
    for start, end in ((0.255, 0.375), (0.62, 0.74)):
        ax.annotate("", xy=(end, 0.51), xytext=(start, 0.51),
                    arrowprops={"arrowstyle": "-|>", "color": MID, "lw": 0.7})
    ax.text(0.315, 0.41, "Select", ha="center", va="top", color=MID, fontsize=5.8)
    ax.text(0.68, 0.41, "Apply", ha="center", va="top", color=MID, fontsize=5.8)
    ax.text(0.5, 0.80, "Calibration FDR ≤ 5%", ha="center", va="center", color=MID, fontsize=6.1)
    ax.set_title("Molecule-disjoint threshold transfer", pad=3)


def distribution_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    *,
    target_line: float | None = None,
    ylim: tuple[float, float],
) -> list[dict[str, object]]:
    centers = {"pos": 0.0, "neg": 1.0}
    offsets = dict(zip(METHODS, np.linspace(-0.24, 0.24, len(METHODS))))
    summary_rows: list[dict[str, object]] = []
    for mode in ("pos", "neg"):
        for method in METHODS:
            values = 100 * table.loc[
                (table["mode"] == mode) & (table.method == method), metric
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) != 200:
                raise ValueError(f"expected 200 formal repeats for {method}/{mode}; got {len(values)}")
            position = centers[mode] + offsets[method]
            violin = ax.violinplot(
                [values], positions=[position], widths=0.105,
                showmeans=False, showmedians=False, showextrema=False,
                quantiles=[[0.025, 0.975]],
            )
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[method])
                body.set_edgecolor("none")
                body.set_alpha(0.22)
            violin["cquantiles"].set_color(COLORS[method])
            violin["cquantiles"].set_linewidth(0.7)
            median = float(np.median(values))
            q025, q975 = np.quantile(values, [0.025, 0.975])
            ax.scatter(
                position, median, marker=MARKERS[method], s=10,
                color=COLORS[method], edgecolor="white", linewidth=0.35, zorder=4,
                label=method if mode == "pos" else None,
            )
            summary_rows.append(
                {
                    "method": method,
                    "mode": mode,
                    "metric": metric,
                    "median_percent": median,
                    "q025_percent": float(q025),
                    "q975_percent": float(q975),
                    "n_repeats": len(values),
                }
            )
    if target_line is not None:
        ax.axhline(target_line, color=MID, linestyle=":", linewidth=0.7, zorder=1)
    ax.set_xticks([0, 1], ["[M+H]+", "[M-H]-"])
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=3)
    clean_axis(ax)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--alias-stem",
        help="optional package-facing alias stem (for example fig3h_sp_1)",
    )
    args = parser.parse_args()
    done_path = args.aggregate_dir / "DONE.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("status") != "complete" or done.get("formal_method_completeness_enforced") is not True:
            raise RuntimeError(f"formal completion marker required: {done_path}")
        if done.get("primary_model_seed") != 0:
            raise RuntimeError(f"primary seed-0 policy required: {done_path}")
    else:
        # The compact formal-result bundle preserves the authoritative transfer
        # summary rather than the top-level aggregate marker.  Validate the
        # producer's table hash and protocol fields before plotting it.
        done_path = args.aggregate_dir / "threshold_transfer_summary.json"
        done = json.loads(done_path.read_text())
        if (
            done.get("status") != "complete"
            or done.get("n_repeats") != 200
            or done.get("target_fdr") != 0.05
        ):
            raise RuntimeError(f"formal transfer summary required: {done_path}")

    primary_source = args.aggregate_dir / "transfer_repeats_primary_seed0.csv.gz"
    if primary_source.exists():
        source = primary_source
        table = pd.read_csv(source)
    else:
        source = args.aggregate_dir / "threshold_transfer_repeats.csv.gz"
        if done.get("table_sha256") != sha256_file(source):
            raise RuntimeError(f"transfer table hash mismatch: {source}")
        table = pd.read_csv(source)
        # The formal source also retains the archived Spec2Vec audit.  The
        # current publication display is intentionally limited to the three
        # author-selected methods.
        table = table[table.method.isin(METHODS)].copy()
    if set(table.method) != set(METHODS) or set(table["mode"]) != {"pos", "neg"}:
        raise ValueError("formal three-method, two-adduct transfer table required")
    cell_sizes = table.groupby(["method", "mode"], dropna=False).size()
    if len(table) != 1_200 or not (cell_sizes == 200).all():
        raise ValueError(f"expected 200 primary repeats in each of 6 cells; got {cell_sizes.to_dict()}")

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(179 * MM, 53 * MM), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    schematic(axes[0])
    rows = distribution_panel(
        axes[1], table, "test_fdr", "Test FDR (%)", "Transferred error rate",
        target_line=5.0, ylim=(0, 16),
    )
    rows += distribution_panel(
        axes[2], table, "identification_recall", "Identification recall (%)", "Held-out identification recall",
        ylim=(0, 100),
    )
    handles, labels = axes[1].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="lower left", ncol=2, handletextpad=0.3,
                   columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2)
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.20, top=0.88, wspace=0.36)
    output = args.output_dir / "s4_fdr_threshold_transfer"
    save(fig, output)

    aliases: list[Path] = []
    if args.alias_stem:
        for suffix in ("png", "pdf", "svg"):
            source_path = output.with_suffix(f".{suffix}")
            alias_path = args.output_dir / f"{args.alias_stem}.{suffix}"
            shutil.copy2(source_path, alias_path)
            aliases.append(alias_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "threshold_transfer_plot_source.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "threshold_transfer_plot_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "source_id": f"{source.parent.name}/{source.name}",
        "source_sha256": sha256_file(source),
        "aggregate_done_sha256": sha256_file(done_path),
        "n_rows": int(len(table)),
        "methods": METHODS,
        "adducts": {"pos": "[M+H]+", "neg": "[M-H]-"},
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(
                [*args.output_dir.glob("s4_fdr_threshold_transfer.*"), *aliases]
            )
        },
    }
    (args.output_dir / "PLOT_AUDIT.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
