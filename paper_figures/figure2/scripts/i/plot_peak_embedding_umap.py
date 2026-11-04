#!/usr/bin/env python
"""Plot Fig. 2 peak-level embedding UMAP and quantitative metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = {
    "Halogenated": "#ee6a7c",
    "N-containing": "#2f6fdd",
    "S/P-containing": "#8b6fc7",
    "O-rich": "#24a99a",
    "Hydrocarbon": "#f2a64a",
    "Other": "#8b8f98",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def balanced_sample(meta: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    if len(meta) <= max_points:
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    labels = meta["frag_label"].to_numpy()
    groups = [np.where(labels == lab)[0] for lab in sorted(set(labels))]
    per = max(1, max_points // len(groups))
    idx = []
    for g in groups:
        take = min(len(g), per)
        idx.extend(rng.choice(g, size=take, replace=False).tolist())
    if len(idx) < max_points:
        rest = np.setdiff1d(np.arange(len(meta)), np.asarray(idx), assume_unique=False)
        extra = rng.choice(rest, size=min(len(rest), max_points - len(idx)), replace=False)
        idx.extend(extra.tolist())
    return np.asarray(idx, dtype=int)


def nearest_neighbor_metrics(x: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    from sklearn.neighbors import NearestNeighbors
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = []
        for smi in meta.get("frag_smiles", pd.Series([""] * len(meta))).fillna("").to_numpy():
            mol = Chem.MolFromSmiles(str(smi)) if smi else None
            fps.append(generator.GetFingerprint(mol) if mol is not None else None)
    except Exception:
        DataStructs = None
        fps = [None] * len(meta)

    x = l2_normalize(x)
    nn = NearestNeighbors(n_neighbors=min(len(x), 101), metric="cosine")
    nn.fit(x)
    neigh = nn.kneighbors(x, return_distance=False)

    mz = meta["mz"].to_numpy()
    formula = meta["frag_formula"].to_numpy()
    labels = meta["frag_label"].to_numpy()
    groups = meta["spec_id"].to_numpy()
    mz_err = []
    formula_hit = []
    family_hit = []
    tanimoto = []
    for i, row in enumerate(neigh):
        picked = [j for j in row if j != i and groups[j] != groups[i]]
        if not picked:
            picked = [j for j in row if j != i]
        if not picked:
            continue
        j = picked[0]
        mz_err.append(abs(float(mz[i]) - float(mz[j])))
        formula_hit.append(formula[i] == formula[j])
        family_hit.append(labels[i] == labels[j])
        if DataStructs is not None and fps[i] is not None and fps[j] is not None:
            tanimoto.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))
    return {
        "nn_mz_mae_da": float(np.mean(mz_err)),
        "nn_mz_median_da": float(np.median(mz_err)),
        "formula_hit_at1": float(np.mean(formula_hit)),
        "fragment_family_hit_at1": float(np.mean(family_hit)),
        "fragment_fp_tanimoto_at1": float(np.mean(tanimoto)) if tanimoto else float("nan"),
    }


def fit_umap(x: np.ndarray, seed: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.18,
        spread=1.0,
        metric="cosine",
        init="random",
        n_epochs=350,
        random_state=seed,
        low_memory=True,
    )
    return reducer.fit_transform(l2_normalize(x))


def draw_mol_insets(ax, coords: np.ndarray, meta: pd.DataFrame, max_mols: int = 3) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    except Exception:
        return

    smiles_col = "frag_smiles" if "frag_smiles" in meta.columns else "smiles"
    candidates = (
        meta[meta[smiles_col].fillna("").astype(str) != ""]
        .groupby(smiles_col)
        .agg(n=(smiles_col, "size"), match_rate=("match_rate", "mean"))
        .sort_values(["n", "match_rate"], ascending=False)
        .head(max_mols)
        .index
        .tolist()
    )
    used = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        idx = np.where(meta[smiles_col].astype(str).to_numpy() == smi)[0]
        if len(idx) < 5:
            continue
        xy = np.median(coords[idx], axis=0)
        # Suppress nearly overlapping molecule labels.
        if any(np.linalg.norm(xy - prev) < 1.6 for prev in used):
            continue
        used.append(xy)
        img = Draw.MolToImage(mol, size=(150, 105), kekulize=True)
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.20),
            xy,
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=5,
        )
        ax.add_artist(ab)
        if len(used) >= max_mols:
            break


def plot_panel(ax, coords: np.ndarray, meta: pd.DataFrame, title: str, add_mols: bool) -> None:
    labels = list(PALETTE)
    for lab in labels:
        idx = np.where(meta["frag_label"].to_numpy() == lab)[0]
        if not len(idx):
            continue
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=2.0,
            c=PALETTE[lab],
            alpha=0.58,
            linewidths=0,
            rasterized=True,
            label=lab,
        )
    if add_mols:
        draw_mol_insets(ax, coords, meta)
    ax.set_title(title, fontsize=8.2, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(0.18, 0.08), xytext=(0.04, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    ax.annotate("", xy=(0.04, 0.22), xytext=(0.04, 0.08), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    ax.text(0.19, 0.055, "UMAP 1", transform=ax.transAxes, fontsize=6.5, va="top")
    ax.text(0.018, 0.23, "UMAP 2", transform=ax.transAxes, fontsize=6.5, rotation=90, va="bottom")


def main() -> None:
    root = project_root()
    data_dir = root / "main_figure/fig2/data/task5_peak_embedding_umap"
    out_dir = root / "main_figure/fig2/output/task5_peak_embedding_umap"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=data_dir)
    ap.add_argument("--output-dir", type=Path, default=out_dir)
    ap.add_argument("--max-points", type=int, default=18000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    meta_all = pd.read_csv(args.data_dir / "peak_embedding_metadata.csv")
    ultra_all = np.load(args.data_dir / "peak_embeddings_ultra.npy")
    dreams_all = np.load(args.data_dir / "peak_embeddings_dreams.npy")

    keep_labels = [lab for lab, n in meta_all["frag_label"].value_counts().items() if n >= 30]
    keep = meta_all["frag_label"].isin(keep_labels).to_numpy()
    meta_all = meta_all[keep].reset_index(drop=True)
    ultra_all = ultra_all[keep]
    dreams_all = dreams_all[keep]

    idx = balanced_sample(meta_all, args.max_points, args.seed)
    meta = meta_all.iloc[idx].reset_index(drop=True)
    ultra = ultra_all[idx]
    dreams = dreams_all[idx]

    metrics = {
        "UltraMS": nearest_neighbor_metrics(ultra, meta),
        "DreaMS": nearest_neighbor_metrics(dreams, meta),
        "n_peaks_plotted": int(len(meta)),
        "fragment_label_counts": meta["frag_label"].value_counts().to_dict(),
    }

    print("fitting UMAP UltraMS")
    u_xy = fit_umap(ultra, args.seed)
    print("fitting UMAP DreaMS")
    d_xy = fit_umap(dreams, args.seed + 1)

    coords = meta.copy()
    coords["ultra_umap1"] = u_xy[:, 0]
    coords["ultra_umap2"] = u_xy[:, 1]
    coords["dreams_umap1"] = d_xy[:, 0]
    coords["dreams_umap2"] = d_xy[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coords.to_csv(args.output_dir / "peak_embedding_umap_coords.csv", index=False)
    with open(args.output_dir / "peak_embedding_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig = plt.figure(figsize=(7.1, 3.0), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.18, 0.74], wspace=0.08)
    ax_u = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[0, 2])

    plot_panel(ax_u, u_xy, meta, "UltraMS peak embeddings", add_mols=True)
    plot_panel(ax_d, d_xy, meta, "DreaMS peak embeddings", add_mols=False)

    handles, labels = ax_u.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.43, -0.02),
        ncol=3, frameon=False, fontsize=6.6, handletextpad=0.25, columnspacing=0.8,
        markerscale=3.4,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    metric_names = ["formula_hit_at1", "fragment_fp_tanimoto_at1"]
    short = ["Formula\nhit@1", "FP Tanimoto\n@1"]
    x = np.arange(len(metric_names))
    width = 0.34
    ultra_vals = [metrics["UltraMS"][m] for m in metric_names]
    dreams_vals = [metrics["DreaMS"][m] for m in metric_names]
    ax_m.bar(x - width / 2, ultra_vals, width, color="#e94b3c", label="UltraMS")
    ax_m.bar(x + width / 2, dreams_vals, width, color="#4db9d1", label="DreaMS")
    for xpos, val in zip(x - width / 2, ultra_vals):
        ax_m.text(xpos, val + 0.018, f"{val*100:.0f}%", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
    for xpos, val in zip(x + width / 2, dreams_vals):
        ax_m.text(xpos, val + 0.018, f"{val*100:.0f}%", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
    ax_m.set_title("Peak-neighbor fidelity", fontsize=8.2, fontweight="bold", pad=3)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(short, fontsize=6.5)
    ax_m.set_ylim(0, 1.03)
    ax_m.set_ylabel("hit@1", fontsize=6.8)
    ax_m.tick_params(axis="y", labelsize=6.5, length=2.5)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.text(-0.45, 0.99, "Top-1 m/z error (Da) ↓",
              ha="left", va="top", fontsize=6.6, fontweight="bold")
    ax_m.text(-0.45, 0.91, f"UltraMS {metrics['UltraMS']['nn_mz_mae_da']:.2f}",
              color="#e94b3c", fontsize=6.8, fontweight="bold")
    ax_m.text(-0.45, 0.83, f"DreaMS {metrics['DreaMS']['nn_mz_mae_da']:.2f}",
              color="#2498b6", fontsize=6.8, fontweight="bold")
    fig.savefig(args.output_dir / "fig2_peak_embedding_umap.png", bbox_inches="tight", dpi=300)
    fig.savefig(args.output_dir / "fig2_peak_embedding_umap.pdf", bbox_inches="tight")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output_dir / 'fig2_peak_embedding_umap.png'}")


if __name__ == "__main__":
    main()
