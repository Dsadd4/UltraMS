"""Create audited metrics and paired bootstrap CIs for the locked family task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)

from evaluate_cross_ce_isomer_retrieval import raw_features, unit_norm


MODELS = {"raw": "Raw spectrum", "dreams_supervised": "DreaMS", "ultrams_supervised": "UltraMS"}
METRICS = ("nmi", "ari", "silhouette")


def fit_metrics(values, labels, n_clusters):
    values = unit_norm(values)
    predicted = KMeans(n_clusters=n_clusters, n_init=50, random_state=42).fit_predict(values)
    silhouettes = silhouette_samples(values, labels, metric="cosine")
    return predicted, silhouettes


def metric_values(labels, predicted, silhouettes, indices=None):
    if indices is None:
        indices = np.arange(len(labels))
    return np.array([
        normalized_mutual_info_score(labels[indices], predicted[indices], average_method="arithmetic"),
        adjusted_rand_score(labels[indices], predicted[indices]),
        float(silhouettes[indices].mean()),
    ])


def stratified_bootstrap_indices(labels, n_boot, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    return [
        np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for _ in range(n_boot)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-peaks", type=int, default=150)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, difference_rows = [], []
    for split in ("val", "test"):
        full = pd.read_csv(args.benchmark_dir / f"{split}_metadata.csv")
        selected = pd.read_csv(args.locked_dir / f"{split}_selected_metadata.csv")
        index_by_connectivity = pd.Series(full.index, index=full.connectivity).to_dict()
        indices = np.asarray([index_by_connectivity[value] for value in selected.connectivity], int)
        labels = selected.structure_cluster.to_numpy(int)
        n_clusters = len(np.unique(labels))
        features = {
            "raw": raw_features(args.benchmark_dir / f"{split}_top{args.input_peaks}_peaks.npz")[indices],
            "dreams_supervised": np.load(args.embedding_dir / f"{split}_dreams_supervised.npy")[indices],
            "ultrams_supervised": np.load(args.embedding_dir / f"{split}_ultrams_supervised.npy")[indices],
        }
        fitted = {}
        prediction_frame = selected[[
            "spectrum_id", "connectivity", "canonical_smiles", "precursor_mz", "structure_cluster"
        ]].copy()
        for model, values in features.items():
            predicted, silhouettes = fit_metrics(values, labels, n_clusters)
            fitted[model] = (predicted, silhouettes)
            prediction_frame[f"{model}_cluster"] = predicted
            prediction_frame[f"{model}_silhouette"] = silhouettes
        prediction_frame.to_csv(args.out_dir / f"{split}_per_sample_metrics.csv", index=False)

        boot_indices = stratified_bootstrap_indices(labels, args.bootstrap, seed=3407)
        model_boot = {}
        for model, (predicted, silhouettes) in fitted.items():
            point = metric_values(labels, predicted, silhouettes)
            boot = np.vstack([
                metric_values(labels, predicted, silhouettes, sample) for sample in boot_indices
            ])
            model_boot[model] = boot
            for j, metric in enumerate(METRICS):
                summary_rows.append({
                    "split": split, "model": model, "display_name": MODELS[model],
                    "metric": metric, "value": point[j],
                    "ci_low": np.quantile(boot[:, j], 0.025),
                    "ci_high": np.quantile(boot[:, j], 0.975),
                    "n_spectra": len(labels), "n_structure_families": n_clusters,
                })
        for reference in ("raw", "dreams_supervised"):
            delta_point = metric_values(labels, *fitted["ultrams_supervised"]) - metric_values(
                labels, *fitted[reference]
            )
            delta_boot = model_boot["ultrams_supervised"] - model_boot[reference]
            for j, metric in enumerate(METRICS):
                difference_rows.append({
                    "split": split, "reference_model": reference,
                    "comparison_model": "ultrams_supervised", "metric": metric,
                    "difference": delta_point[j],
                    "ci_low": np.quantile(delta_boot[:, j], 0.025),
                    "ci_high": np.quantile(delta_boot[:, j], 0.975),
                })
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "metric_summary.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(args.out_dir / "paired_differences.csv", index=False)
    (args.out_dir / "bootstrap_protocol.json").write_text(json.dumps({
        "clustering": "KMeans(k=6, n_init=50, seed=42) in the original high-dimensional representation",
        "bootstrap": f"{args.bootstrap} stratified resamples within fixed structure-family labels",
        "uncertainty_scope": "sample uncertainty conditional on the frozen clustering protocol",
        "pairing": "identical bootstrap indices for Raw, DreaMS and UltraMS",
        "silhouette": "mean of per-sample cosine silhouette values",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
