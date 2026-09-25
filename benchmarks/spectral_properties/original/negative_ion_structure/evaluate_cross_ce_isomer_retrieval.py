"""Evaluate cross-energy isomer retrieval and freeze UMAP source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize


DISPLAY_NAMES = {
    "raw": "Raw spectrum",
    "dreams_supervised": "DreaMS",
    "dreams_supervised_fusion": "DreaMS",
    "ultrams_ssl": "UltraMS (pretrained)",
    "ultrams_ssl_fusion": "UltraMS (pretrained)",
    "ultrams_supervised": "UltraMS",
    "ultrams_supervised_fusion": "UltraMS",
}
METRICS = ["top1", "mrr", "auroc", "margin"]


def raw_features(peaks_path: Path, bin_width: float = 0.05, max_mz: float = 1000.0):
    with np.load(peaks_path) as peaks:
        mzs = peaks["mzs"]
        intensities = peaks["intensities"]
        lengths = peaks["lengths"]
    n_rows, n_peaks = mzs.shape
    n_bins = int(np.ceil(max_mz / bin_width)) + 1
    rows, cols, data = [], [], []
    for i in range(n_rows):
        length = int(lengths[i])
        mz = mzs[i, :length]
        intensity = np.sqrt(intensities[i, :length])
        bins = np.rint(mz / bin_width).astype(int)
        valid = (bins >= 0) & (bins < n_bins)
        rows.extend([i] * int(valid.sum()))
        cols.extend(bins[valid].tolist())
        data.extend(intensity[valid].tolist())
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_bins), dtype=np.float32)
    matrix.sum_duplicates()
    return normalize(matrix, norm="l2", axis=1)


def unit_norm(values):
    if sparse.issparse(values):
        return normalize(values, norm="l2", axis=1)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def centroid(values, indices: np.ndarray):
    if sparse.issparse(values):
        result = np.asarray(values[indices].mean(axis=0), dtype=np.float32).reshape(1, -1)
    else:
        result = values[indices].mean(axis=0, keepdims=True)
    return unit_norm(result)


def similarity(query, references):
    result = query @ references.T
    return result.toarray() if sparse.issparse(result) else np.asarray(result)


def evaluate_model(metadata: pd.DataFrame, features, model: str):
    features = unit_norm(features)
    query_rows = []
    for formula, formula_frame in metadata.groupby("formula", sort=True):
        structures = sorted(formula_frame["connectivity"].unique())
        reference_centroids = []
        for structure in structures:
            indices = formula_frame.index[
                (formula_frame["connectivity"] == structure)
                & (formula_frame["energy_role"] == "reference_60eV")
            ].to_numpy()
            reference_centroids.append(centroid(features, indices))
        references = np.vstack(reference_centroids)
        query_indices = formula_frame.index[
            formula_frame["energy_role"] == "query_20eV"
        ].to_numpy()
        scores = similarity(features[query_indices], references)
        for local_i, source_i in enumerate(query_indices):
            truth = structures.index(metadata.at[source_i, "connectivity"])
            order = np.argsort(-scores[local_i], kind="stable")
            rank = int(np.flatnonzero(order == truth)[0]) + 1
            wrong = np.delete(scores[local_i], truth)
            query_rows.append({
                "model": model,
                "spectrum_id": metadata.at[source_i, "spectrum_id"],
                "formula": formula,
                "connectivity": metadata.at[source_i, "connectivity"],
                "n_candidates": len(structures),
                "rank": rank,
                "top1": float(rank == 1),
                "mrr": 1.0 / rank,
                "positive_similarity": float(scores[local_i, truth]),
                "max_isomer_similarity": float(wrong.max()),
                "margin": float(scores[local_i, truth] - wrong.max()),
                "negative_similarities": ";".join(f"{v:.7g}" for v in wrong),
            })
    query = pd.DataFrame(query_rows)
    structure = query.groupby(["model", "formula", "connectivity"], as_index=False).agg(
        top1=("top1", "mean"), mrr=("mrr", "mean"), margin=("margin", "mean")
    )
    formula_rank = structure.groupby(["model", "formula"], as_index=False).agg(
        top1=("top1", "mean"), mrr=("mrr", "mean"), margin=("margin", "mean")
    )
    auroc_rows = []
    for formula, group in query.groupby("formula"):
        positive = group["positive_similarity"].to_numpy()
        negative = np.concatenate([
            np.fromstring(value, sep=";") for value in group["negative_similarities"]
        ])
        labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
        scores = np.r_[positive, negative]
        auroc_rows.append({"formula": formula, "auroc": roc_auc_score(labels, scores)})
    formula = formula_rank.merge(pd.DataFrame(auroc_rows), on="formula", validate="one_to_one")
    return query, structure, formula


def bootstrap_summary(formula: pd.DataFrame, rng: np.random.Generator, n_boot: int):
    output = []
    for model, frame in formula.groupby("model"):
        values = frame[METRICS].to_numpy(float)
        indices = rng.integers(0, len(values), size=(n_boot, len(values)))
        boot = values[indices].mean(axis=1)
        for j, metric in enumerate(METRICS):
            output.append({
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "metric": metric,
                "value": float(values[:, j].mean()),
                "ci_low": float(np.quantile(boot[:, j], 0.025)),
                "ci_high": float(np.quantile(boot[:, j], 0.975)),
                "n_formula_groups": len(values),
            })
    return pd.DataFrame(output)


def paired_differences(formula: pd.DataFrame, rng: np.random.Generator, n_boot: int):
    wide = formula.pivot(index="formula", columns="model", values=METRICS)
    rows = []
    models = sorted(formula["model"].unique())
    for left in models:
        for right in models:
            if left >= right or left not in wide.columns.levels[1] or right not in wide.columns.levels[1]:
                continue
            delta = np.column_stack([wide[(metric, right)] - wide[(metric, left)] for metric in METRICS])
            indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
            boot = delta[indices].mean(axis=1)
            for j, metric in enumerate(METRICS):
                rows.append({
                    "reference_model": left, "comparison_model": right, "metric": metric,
                    "difference": float(delta[:, j].mean()),
                    "ci_low": float(np.quantile(boot[:, j], 0.025)),
                    "ci_high": float(np.quantile(boot[:, j], 0.975)),
                })
    return pd.DataFrame(rows)


def make_umap(metadata, embedding_dir: Path, selected_dreams: str, selected_ultra: str, out_dir: Path):
    import umap

    frames = []
    for model in (selected_dreams, selected_ultra):
        values = np.load(embedding_dir / f"test_{model}.npy")
        reducer = umap.UMAP(
            n_neighbors=50, min_dist=0.12, n_components=2, metric="cosine",
            random_state=42, transform_seed=42,
        )
        coords = reducer.fit_transform(unit_norm(values))
        frame = metadata.copy()
        frame["model"] = model
        frame["display_name"] = DISPLAY_NAMES[model]
        frame["umap1"] = coords[:, 0]
        frame["umap2"] = coords[:, 1]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(out_dir / "umap_coordinates.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--input-peaks", type=int, default=60)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    selection = {}
    all_summaries = []
    for split in ("val", "test"):
        metadata = pd.read_csv(args.benchmark_dir / f"{split}_metadata.csv")
        feature_sets = {
            "raw": raw_features(args.benchmark_dir / f"{split}_top{args.input_peaks}_peaks.npz"),
            "dreams_supervised": np.load(args.embedding_dir / f"{split}_dreams_supervised.npy"),
            "dreams_supervised_fusion": np.load(
                args.embedding_dir / f"{split}_dreams_supervised_fusion.npy"
            ),
            "ultrams_ssl": np.load(args.embedding_dir / f"{split}_ultrams_ssl.npy"),
            "ultrams_ssl_fusion": np.load(
                args.embedding_dir / f"{split}_ultrams_ssl_fusion.npy"
            ),
            "ultrams_supervised": np.load(args.embedding_dir / f"{split}_ultrams_supervised.npy"),
            "ultrams_supervised_fusion": np.load(
                args.embedding_dir / f"{split}_ultrams_supervised_fusion.npy"
            ),
        }
        query_frames, structure_frames, formula_frames = [], [], []
        for model, values in feature_sets.items():
            query, structure, formula = evaluate_model(metadata, values, model)
            query_frames.append(query)
            structure_frames.append(structure)
            formula_frames.append(formula)
        query = pd.concat(query_frames, ignore_index=True)
        structure = pd.concat(structure_frames, ignore_index=True)
        formula = pd.concat(formula_frames, ignore_index=True)
        summary = bootstrap_summary(formula, rng, args.bootstrap)
        differences = paired_differences(formula, rng, args.bootstrap)
        query.to_csv(args.out_dir / f"{split}_query_metrics.csv", index=False)
        structure.to_csv(args.out_dir / f"{split}_structure_metrics.csv", index=False)
        formula.to_csv(args.out_dir / f"{split}_formula_metrics.csv", index=False)
        summary.to_csv(args.out_dir / f"{split}_summary.csv", index=False)
        differences.to_csv(args.out_dir / f"{split}_paired_differences.csv", index=False)
        summary.insert(0, "split", split)
        all_summaries.append(summary)
        if split == "val":
            primary = summary[summary["metric"] == "top1"].set_index("model")["value"]
            ultra_candidates = [
                "ultrams_ssl", "ultrams_ssl_fusion",
                "ultrams_supervised", "ultrams_supervised_fusion",
            ]
            dreams_candidates = ["dreams_supervised", "dreams_supervised_fusion"]
            selected = max(ultra_candidates, key=lambda name: (primary[name], name))
            selected_dreams = max(dreams_candidates, key=lambda name: (primary[name], name))
            selection = {
                "rule": "highest formula-macro cross-energy Top-1 on validation; ties use MRR",
                "selected_ultrams": selected,
                "selected_dreams": selected_dreams,
                "validation_top1": {
                    name: float(primary[name]) for name in ultra_candidates + dreams_candidates
                },
            }
    pd.concat(all_summaries, ignore_index=True).to_csv(args.out_dir / "all_summary.csv", index=False)
    (args.out_dir / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    test_metadata = pd.read_csv(args.benchmark_dir / "test_metadata.csv")
    make_umap(
        test_metadata, args.embedding_dir, selection["selected_dreams"],
        selection["selected_ultrams"], args.out_dir,
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
