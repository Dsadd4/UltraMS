"""Evaluate supervised CLS-derived embeddings on fixed Butina structure families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from evaluate_cross_ce_isomer_retrieval import raw_features, unit_norm


MODELS = {"raw": "Raw spectrum", "dreams_supervised": "DreaMS", "ultrams_supervised": "UltraMS"}


def butina_labels(smiles: list[str], cutoff: float, top_clusters: int):
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]
    distances = []
    for i in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend(1.0 - value for value in similarities)
    clusters = sorted(
        Butina.ClusterData(distances, len(fps), cutoff, isDistData=True, reordering=True),
        key=len, reverse=True,
    )[:top_clusters]
    selected = np.concatenate([np.asarray(cluster, dtype=int) for cluster in clusters])
    labels = np.concatenate([
        np.full(len(cluster), cluster_id, dtype=int) for cluster_id, cluster in enumerate(clusters)
    ])
    return selected, labels, [len(cluster) for cluster in clusters]


def evaluate(values, labels, n_clusters: int):
    values = unit_norm(values)
    predicted = KMeans(n_clusters=n_clusters, n_init=50, random_state=42).fit_predict(values)
    return {
        "nmi": normalized_mutual_info_score(labels, predicted, average_method="arithmetic"),
        "ari": adjusted_rand_score(labels, predicted),
        "silhouette": silhouette_score(values, labels, metric="cosine"),
    }


def make_umap(metadata, dreams, ultra, out_dir: Path):
    import umap

    frames = []
    for model, values in (("dreams_supervised", dreams), ("ultrams_supervised", ultra)):
        reducer = umap.UMAP(
            n_neighbors=30, min_dist=0.08, metric="cosine", random_state=42, transform_seed=42,
        )
        coords = reducer.fit_transform(unit_norm(values))
        frame = metadata.copy()
        frame["model"] = model
        frame["display_name"] = MODELS[model]
        frame["umap1"], frame["umap2"] = coords[:, 0], coords[:, 1]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "test_umap_coordinates.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--distance-cutoff", type=float, default=0.75)
    parser.add_argument("--top-clusters", type=int, default=10)
    parser.add_argument("--input-peaks", type=int, default=150)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    test_data = None
    for split in ("val", "test"):
        metadata = pd.read_csv(args.benchmark_dir / f"{split}_metadata.csv")
        selected, labels, sizes = butina_labels(
            metadata.canonical_smiles.tolist(), args.distance_cutoff, args.top_clusters
        )
        selected_metadata = metadata.iloc[selected].copy().reset_index(drop=True)
        selected_metadata["structure_cluster"] = labels
        selected_metadata.to_csv(args.out_dir / f"{split}_selected_metadata.csv", index=False)
        raw = raw_features(args.benchmark_dir / f"{split}_top{args.input_peaks}_peaks.npz")[selected]
        dreams = np.load(args.embedding_dir / f"{split}_dreams_supervised.npy")[selected]
        ultra = np.load(args.embedding_dir / f"{split}_ultrams_supervised.npy")[selected]
        for model, values in (("raw", raw), ("dreams_supervised", dreams), ("ultrams_supervised", ultra)):
            result = evaluate(values, labels, args.top_clusters)
            for metric, value in result.items():
                summary_rows.append({
                    "split": split, "model": model, "display_name": MODELS[model],
                    "metric": metric, "value": value, "n_spectra": len(selected),
                    "cluster_sizes": ";".join(map(str, sizes)),
                })
        if split == "test":
            test_data = (selected_metadata, dreams, ultra)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "protocol.json").write_text(json.dumps({
        "distance_cutoff": args.distance_cutoff,
        "minimum_pair_similarity": 1 - args.distance_cutoff,
        "top_clusters": args.top_clusters,
        "selection_source": "validation molecular-fingerprint compactness and coverage only",
        "embedding_metrics": ["NMI", "ARI", "cosine silhouette"],
        "umap": "cosine; n_neighbors=30; min_dist=0.08; seed=42; display only",
    }, indent=2) + "\n")
    assert test_data is not None
    make_umap(*test_data, args.out_dir)


if __name__ == "__main__":
    main()
