#!/usr/bin/env python
"""Single figure for peak-neighbor fidelity metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def balanced_sample(meta: pd.DataFrame, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(meta) <= max_points:
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    labels = meta["frag_label"].to_numpy()
    groups = [np.where(labels == lab)[0] for lab in sorted(set(labels))]
    per_group = max(1, max_points // len(groups))
    idx: list[int] = []
    for group in groups:
        idx.extend(rng.choice(group, size=min(len(group), per_group), replace=False).tolist())
    if len(idx) < max_points:
        rest = np.setdiff1d(np.arange(len(meta)), np.asarray(idx), assume_unique=False)
        idx.extend(rng.choice(rest, size=min(len(rest), max_points - len(idx)), replace=False).tolist())
    return np.asarray(idx, dtype=int)


def faiss_neighbors(
    x: np.ndarray,
    n_neighbors: int,
    method: str,
    chunk_size: int,
    hnsw_m: int,
    ef_search: int,
) -> np.ndarray:
    import faiss

    x = np.ascontiguousarray(l2_normalize(x).astype(np.float32))
    k = min(len(x), n_neighbors)
    if method == "faiss_flat":
        index = faiss.IndexFlatIP(x.shape[1])
    elif method == "faiss_hnsw":
        try:
            index = faiss.IndexHNSWFlat(x.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(x.shape[1], hnsw_m)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = max(ef_search, 80)
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unsupported FAISS method: {method}")
    index.add(x)

    rows = []
    for start in range(0, len(x), chunk_size):
        _, neigh = index.search(x[start:start + chunk_size], k)
        rows.append(neigh)
    return np.vstack(rows)


def sklearn_neighbors(x: np.ndarray, n_neighbors: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    x = l2_normalize(x)
    nn = NearestNeighbors(n_neighbors=min(len(x), n_neighbors), metric="cosine")
    nn.fit(x)
    return nn.kneighbors(x, return_distance=False)


def nearest_neighbor_metrics(
    x: np.ndarray,
    meta: pd.DataFrame,
    method: str,
    n_neighbors: int,
    chunk_size: int,
    hnsw_m: int,
    ef_search: int,
) -> dict[str, float]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in meta["frag_smiles"].fillna("").astype(str).to_numpy():
        mol = Chem.MolFromSmiles(smi) if smi and smi.lower() != "nan" else None
        fps.append(generator.GetFingerprint(mol) if mol is not None else None)

    if method.startswith("faiss"):
        neigh = faiss_neighbors(x, n_neighbors, method, chunk_size, hnsw_m, ef_search)
    elif method == "sklearn":
        neigh = sklearn_neighbors(x, n_neighbors)
    else:
        raise ValueError(f"Unsupported --nn-method {method}")

    mz = meta["mz"].to_numpy()
    formula = meta["frag_formula"].to_numpy()
    family = meta["frag_label"].to_numpy()
    spec = meta["spec_id"].to_numpy()
    mz_err = []
    formula_hit = []
    family_hit = []
    tanimoto = []
    for i, row in enumerate(neigh):
        picked = [j for j in row if j != i and spec[j] != spec[i]]
        if not picked:
            continue
        j = picked[0]
        mz_err.append(abs(float(mz[i]) - float(mz[j])))
        formula_hit.append(formula[i] == formula[j])
        family_hit.append(family[i] == family[j])
        if fps[i] is not None and fps[j] is not None:
            tanimoto.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

    return {
        "nn_mz_mae_da": float(np.mean(mz_err)),
        "nn_mz_median_da": float(np.median(mz_err)),
        "formula_hit_at1": float(np.mean(formula_hit)),
        "fragment_family_hit_at1": float(np.mean(family_hit)),
        "fragment_fp_tanimoto_at1": float(np.mean(tanimoto)),
        "n_queries": int(len(mz_err)),
        "n_missing_cross_spectrum_neighbor": int(len(x) - len(mz_err)),
    }


def main() -> None:
    root = project_root()
    default_data = root / "main_figure/fig2/data/task5_peak_embedding_umap/large"
    default_out = root / "main_figure/fig2/output/task5_peak_embedding_umap"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--output-dir", type=Path, default=default_out)
    ap.add_argument("--output-stem", default="fig2_peak_neighbor_fidelity")
    ap.add_argument("--metrics-name", default="peak_neighbor_fidelity_metrics.json")
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nn-method", choices=["faiss_hnsw", "faiss_flat", "sklearn"], default="faiss_hnsw")
    ap.add_argument("--n-neighbors", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=128)
    args = ap.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    meta = pd.read_csv(args.data_dir / "peak_embedding_metadata.csv")
    ultra = np.load(args.data_dir / "peak_embeddings_ultra.npy")
    dreams = np.load(args.data_dir / "peak_embeddings_dreams.npy")
    idx = balanced_sample(meta, args.max_points, args.seed)
    meta = meta.iloc[idx].reset_index(drop=True)
    ultra = ultra[idx]
    dreams = dreams[idx]

    metrics = {
        "UltraMS": nearest_neighbor_metrics(
            ultra, meta, args.nn_method, args.n_neighbors, args.chunk_size, args.hnsw_m, args.ef_search
        ),
        "DreaMS": nearest_neighbor_metrics(
            dreams, meta, args.nn_method, args.n_neighbors, args.chunk_size, args.hnsw_m, args.ef_search
        ),
        "n_peaks": int(len(meta)),
        "n_spectra": int(meta["spec_id"].nunique()),
        "fragment_label_counts": meta["frag_label"].value_counts().to_dict(),
        "sampling": {
            "max_points": int(args.max_points),
            "seed": int(args.seed),
            "source_data_dir": str(args.data_dir),
        },
        "nearest_neighbor": {
            "method": args.nn_method,
            "n_neighbors": int(args.n_neighbors),
            "chunk_size": int(args.chunk_size),
            "hnsw_m": int(args.hnsw_m),
            "ef_search": int(args.ef_search),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / args.metrics_name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    fig, (ax_hit, ax_mz) = plt.subplots(
        1, 2, figsize=(4.2, 2.35), dpi=350, gridspec_kw={"width_ratios": [1.45, 0.85]}
    )
    colors = {"UltraMS": "#e94b3c", "DreaMS": "#4db9d1"}
    metric_keys = ["formula_hit_at1", "fragment_fp_tanimoto_at1", "fragment_family_hit_at1"]
    labels = ["Formula\nhit@1", "FP Tanimoto\n@1", "Family\nhit@1"]
    x = np.arange(len(metric_keys))
    width = 0.34
    for offset, model in [(-width / 2, "UltraMS"), (width / 2, "DreaMS")]:
        vals = [metrics[model][k] for k in metric_keys]
        ax_hit.bar(x + offset, vals, width, color=colors[model], label=model)
        for xpos, val in zip(x + offset, vals):
            txt = f"{val*100:.0f}%" if val <= 1 else f"{val:.1f}"
            ax_hit.text(xpos, val + 0.025, txt, ha="center", va="bottom", fontsize=6.2, fontweight="bold")
    ax_hit.set_title("Fragment-neighbor agreement", fontsize=8.2, fontweight="bold", pad=4)
    ax_hit.set_xticks(x)
    ax_hit.set_xticklabels(labels, fontsize=6.2)
    ax_hit.set_ylim(0, 0.88)
    ax_hit.set_ylabel("score", fontsize=6.8)
    ax_hit.tick_params(axis="y", labelsize=6.2, length=2.5)
    ax_hit.spines["top"].set_visible(False)
    ax_hit.spines["right"].set_visible(False)
    ax_hit.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=1.1)

    models = ["UltraMS", "DreaMS"]
    mz_vals = [metrics[m]["nn_mz_mae_da"] for m in models]
    y = np.arange(len(models))
    ax_mz.barh(y, mz_vals, color=[colors[m] for m in models], height=0.46)
    for yi, val in zip(y, mz_vals):
        ax_mz.text(val * 1.05, yi, f"{val:.2f}", va="center", ha="left", fontsize=6.6, fontweight="bold")
    ax_mz.set_xscale("log")
    ax_mz.set_yticks(y)
    ax_mz.set_yticklabels(models, fontsize=6.7, fontweight="bold")
    ax_mz.invert_yaxis()
    ax_mz.set_xlabel("Top-1 m/z error (Da)", fontsize=6.8)
    ax_mz.set_title("Mass fidelity", fontsize=8.2, fontweight="bold", pad=4)
    ax_mz.tick_params(axis="x", labelsize=6.1, length=2.5)
    ax_mz.spines["top"].set_visible(False)
    ax_mz.spines["right"].set_visible(False)
    ax_mz.text(0.02, -0.33, f"{len(meta):,} MAGMa peaks, {meta['spec_id'].nunique():,} spectra",
               transform=ax_mz.transAxes, ha="left", va="top", fontsize=6.2)

    fig.savefig(args.output_dir / f"{args.output_stem}.png", bbox_inches="tight", dpi=350)
    fig.savefig(args.output_dir / f"{args.output_stem}.pdf", bbox_inches="tight")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output_dir / f'{args.output_stem}.png'}")


if __name__ == "__main__":
    main()
