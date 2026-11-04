#!/usr/bin/env python
"""Top-k peak-neighbor fidelity curves for UltraMS and DreaMS embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLORS = {"UltraMS": "#e94b3c", "DreaMS": "#4db9d1"}
MARKERS = {"UltraMS": "o", "DreaMS": "s"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_k_values(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("--k-values must contain positive integers")
    return values


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


def make_fingerprints(smiles: Iterable[str]):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in smiles:
        smi = "" if pd.isna(smi) else str(smi)
        mol = Chem.MolFromSmiles(smi) if smi and smi.lower() != "nan" else None
        fps.append(generator.GetFingerprint(mol) if mol is not None else None)
    return fps


def cross_spectrum_neighbors(neigh: np.ndarray, spec: np.ndarray, max_k: int) -> list[np.ndarray]:
    filtered = []
    for i, row in enumerate(neigh):
        picked = [int(j) for j in row if j >= 0 and j != i and spec[j] != spec[i]]
        filtered.append(np.asarray(picked[:max_k], dtype=np.int64))
    return filtered


def curve_metrics(
    x: np.ndarray,
    meta: pd.DataFrame,
    k_values: list[int],
    method: str,
    n_neighbors: int,
    chunk_size: int,
    hnsw_m: int,
    ef_search: int,
) -> dict:
    from rdkit import DataStructs

    search_neighbors = max(n_neighbors, max(k_values) + 1)
    if method.startswith("faiss"):
        neigh = faiss_neighbors(x, search_neighbors, method, chunk_size, hnsw_m, ef_search)
    elif method == "sklearn":
        neigh = sklearn_neighbors(x, search_neighbors)
    else:
        raise ValueError(f"Unsupported --nn-method {method}")

    mz = meta["mz"].to_numpy(dtype=float)
    formula = meta["frag_formula"].fillna("").astype(str).to_numpy()
    family = meta["frag_label"].fillna("").astype(str).to_numpy()
    spec = meta["spec_id"].fillna("").astype(str).to_numpy()
    fps = make_fingerprints(meta["frag_smiles"].to_numpy())
    top = cross_spectrum_neighbors(neigh, spec, max(k_values))

    out = {
        "k": [int(k) for k in k_values],
        "formula_hit": [],
        "fragment_family_hit": [],
        "best_fp_tanimoto": [],
        "best_mz_mae_da": [],
        "best_mz_median_da": [],
        "n_valid_neighbors": [],
        "n_valid_fp": [],
    }

    for k in k_values:
        formula_hits = []
        family_hits = []
        mz_errors = []
        tanimoto_best = []
        for i, picked in enumerate(top):
            if len(picked) < k:
                continue
            cand = picked[:k]
            formula_hits.append(bool(np.any(formula[cand] == formula[i])))
            family_hits.append(bool(np.any(family[cand] == family[i])))
            mz_errors.append(float(np.min(np.abs(mz[cand] - mz[i]))))

            if fps[i] is not None:
                cand_fps = [fps[j] for j in cand if fps[j] is not None]
                if cand_fps:
                    sims = DataStructs.BulkTanimotoSimilarity(fps[i], cand_fps)
                    tanimoto_best.append(float(max(sims)))

        out["formula_hit"].append(float(np.mean(formula_hits)) if formula_hits else float("nan"))
        out["fragment_family_hit"].append(float(np.mean(family_hits)) if family_hits else float("nan"))
        out["best_fp_tanimoto"].append(float(np.mean(tanimoto_best)) if tanimoto_best else float("nan"))
        out["best_mz_mae_da"].append(float(np.mean(mz_errors)) if mz_errors else float("nan"))
        out["best_mz_median_da"].append(float(np.median(mz_errors)) if mz_errors else float("nan"))
        out["n_valid_neighbors"].append(int(len(mz_errors)))
        out["n_valid_fp"].append(int(len(tanimoto_best)))

    out["n_missing_cross_spectrum_neighbor_at_max_k"] = int(sum(len(p) < max(k_values) for p in top))
    return out


def write_curve_tables(metrics: dict, output_dir: Path, csv_name: str, json_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / json_name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

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
    with open(output_dir / csv_name, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in ["UltraMS", "DreaMS"]:
            curve = metrics["curves"][model]
            for pos, k in enumerate(curve["k"]):
                row = {"model": model, "k": int(k)}
                for key in fieldnames[2:]:
                    row[key] = curve[key][pos]
                writer.writerow(row)


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6.4, length=2.8, width=0.75)


def plot_metric(ax, metrics: dict, metric: str, title: str, ylabel: str, y_log: bool = False) -> None:
    k = metrics["curves"]["UltraMS"]["k"]
    for model in ["UltraMS", "DreaMS"]:
        y = np.asarray(metrics["curves"][model][metric], dtype=float)
        if y_log:
            y = np.maximum(y, 1e-4)
        ax.plot(
            k,
            y,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=3.0,
            linewidth=1.35,
            label=model,
        )
    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlabel("Top-k cross-spectrum neighbors", fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)
    if not y_log:
        ax.set_ylim(0, 1.02)
    style_axes(ax)


def plot_all(metrics: dict, output_dir: Path, output_prefix: str) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    specs = [
        ("formula_hit", "Formula agreement", "hit@k", False, "fig2_curve_formula_hit_x10"),
        ("fragment_family_hit", "Fragment family agreement", "hit@k", False, "fig2_curve_family_hit_x10"),
        ("best_fp_tanimoto", "Fragment structure similarity", "best Tanimoto@k", False, "fig2_curve_fp_tanimoto_x10"),
        ("best_mz_mae_da", "Mass fidelity", "best m/z error@k (Da)", True, "fig2_curve_mz_error_x10"),
    ]

    for metric, title, ylabel, y_log, stem in specs:
        fig, ax = plt.subplots(figsize=(2.35, 2.05), dpi=350)
        plot_metric(ax, metrics, metric, title, ylabel, y_log)
        ax.legend(frameon=False, fontsize=6.5, loc="best", handlelength=1.3)
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=350)
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(4.95, 4.15), dpi=350)
    for ax, (metric, title, ylabel, y_log, _) in zip(axes.ravel(), specs):
        plot_metric(ax, metrics, metric, title, ylabel, y_log)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.0,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1.4,
    )
    fig.text(
        0.5,
        -0.005,
        f"{metrics['n_peaks']:,} MAGMa peaks, {metrics['n_spectra']:,} spectra",
        ha="center",
        va="top",
        fontsize=7.0,
    )
    fig.tight_layout(pad=1.1, rect=(0, 0.02, 1, 0.95))
    fig.savefig(output_dir / f"{output_prefix}_2x2.png", bbox_inches="tight", dpi=350)
    fig.savefig(output_dir / f"{output_prefix}_2x2.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = project_root()
    default_data = root / "main_figure/fig2/data/task5_peak_embedding_umap/x10"
    default_out = root / "main_figure/fig2/output/task5_peak_embedding_umap/x10"
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=default_data)
    ap.add_argument("--output-dir", type=Path, default=default_out)
    ap.add_argument("--output-prefix", default="fig2_peak_neighbor_curves_x10")
    ap.add_argument("--csv-name", default="peak_neighbor_curves_x10.csv")
    ap.add_argument("--json-name", default="peak_neighbor_curves_x10.json")
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-values", default="1,2,5,10,20,50,100,200")
    ap.add_argument("--nn-method", choices=["faiss_hnsw", "faiss_flat", "sklearn"], default="faiss_hnsw")
    ap.add_argument("--n-neighbors", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=128)
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    meta = pd.read_csv(args.data_dir / "peak_embedding_metadata.csv")
    ultra = np.load(args.data_dir / "peak_embeddings_ultra.npy")
    dreams = np.load(args.data_dir / "peak_embeddings_dreams.npy")
    idx = balanced_sample(meta, args.max_points, args.seed)
    meta = meta.iloc[idx].reset_index(drop=True)
    ultra = ultra[idx]
    dreams = dreams[idx]

    metrics = {
        "curves": {
            "UltraMS": curve_metrics(
                ultra, meta, k_values, args.nn_method, args.n_neighbors, args.chunk_size, args.hnsw_m, args.ef_search
            ),
            "DreaMS": curve_metrics(
                dreams, meta, k_values, args.nn_method, args.n_neighbors, args.chunk_size, args.hnsw_m, args.ef_search
            ),
        },
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
            "n_neighbors": int(max(args.n_neighbors, max(k_values) + 1)),
            "chunk_size": int(args.chunk_size),
            "hnsw_m": int(args.hnsw_m),
            "ef_search": int(args.ef_search),
        },
    }

    write_curve_tables(metrics, args.output_dir, args.csv_name, args.json_name)
    plot_all(metrics, args.output_dir, args.output_prefix)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.output_dir / (args.output_prefix + '_2x2.png')}")


if __name__ == "__main__":
    main()
