#!/usr/bin/env python
"""Train peak-feature baselines and evaluate cross-spectrum nearest neighbors.

Computations follow the original Figure 2 experiment; no plotting code is included.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_ORDER = ["UltraMS", "DreaMS", "Peak features", "Linear SVM", "Random forest"]


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


def faiss_search(
    pool: np.ndarray,
    query: np.ndarray,
    n_neighbors: int,
    method: str,
    chunk_size: int,
    hnsw_m: int,
    ef_search: int,
) -> np.ndarray:
    import faiss

    pool = np.ascontiguousarray(l2_normalize(pool).astype(np.float32))
    query = np.ascontiguousarray(l2_normalize(query).astype(np.float32))
    k = min(len(pool), n_neighbors)
    if method == "faiss_flat":
        index = faiss.IndexFlatIP(pool.shape[1])
    elif method == "faiss_hnsw":
        try:
            index = faiss.IndexHNSWFlat(pool.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexHNSWFlat(pool.shape[1], hnsw_m)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = max(ef_search, 80)
        index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unsupported FAISS method: {method}")
    index.add(pool)

    rows = []
    for start in range(0, len(query), chunk_size):
        _, neigh = index.search(query[start:start + chunk_size], k)
        rows.append(neigh)
    return np.vstack(rows)


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


def add_spectrum_features(meta: pd.DataFrame, subset_csv: Path) -> pd.DataFrame:
    subset = pd.read_csv(
        subset_csv,
        usecols=["spec_id", "precursor_mz", "collision_energy", "n_peaks", "adduct", "prec_type"],
    )
    subset = subset.drop_duplicates("spec_id")
    out = meta.merge(subset, on="spec_id", how="left", suffixes=("", "_subset"))
    if "adduct_subset" in out.columns:
        out["adduct"] = out["adduct"].fillna(out["adduct_subset"])
        out = out.drop(columns=["adduct_subset"])
    out["precursor_mz"] = out["precursor_mz"].fillna(out["mz"])
    out["neutral_loss"] = out["precursor_mz"] - out["mz"]
    out["mz_ratio"] = out["mz"] / np.maximum(out["precursor_mz"], 1e-6)
    out["log_intensity"] = np.log1p(np.maximum(out["intensity"].fillna(0).astype(float), 0.0))
    out["collision_energy"] = out["collision_energy"].fillna(out["collision_energy"].median())
    out["n_peaks"] = out["n_peaks"].fillna(out["n_peaks"].median())
    out["ion_mode"] = np.where(out["adduct"].fillna("").astype(str).str.contains(r"\+", regex=True), "positive", "negative")
    return out


def make_classical_features(meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    from sklearn.preprocessing import StandardScaler

    numeric_cols = [
        "mz",
        "log_intensity",
        "precursor_mz",
        "neutral_loss",
        "mz_ratio",
        "collision_energy",
        "n_peaks",
    ]
    numeric = meta[numeric_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median())
    numeric_scaled = StandardScaler().fit_transform(numeric)

    adduct = pd.get_dummies(meta["adduct"].fillna("unknown").astype(str), prefix="adduct")
    ion = pd.get_dummies(meta["ion_mode"].fillna("unknown").astype(str), prefix="ion")
    categorical = pd.concat([adduct, ion], axis=1).astype(float)
    feature_names = numeric_cols + categorical.columns.tolist()
    x = np.hstack([numeric_scaled, categorical.to_numpy(dtype=float)]).astype(np.float32)
    return x, feature_names


def spectrum_group_split(spec_ids: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rows = np.arange(len(spec_ids))
    train_idx, test_idx = next(splitter.split(rows, groups=spec_ids))
    return train_idx.astype(int), test_idx.astype(int)


def train_classical_predictors(
    x: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    rf_trees: int,
) -> tuple[dict[str, np.ndarray], dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import LinearSVC

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    train_y = y[train_idx]

    svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=6000, random_state=seed)
    svm.fit(x[train_idx], train_y)
    svm_scores = svm.decision_function(x)
    if svm_scores.ndim == 1:
        svm_scores = np.column_stack([-svm_scores, svm_scores])

    rf = RandomForestClassifier(
        n_estimators=rf_trees,
        max_depth=16,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(x[train_idx], train_y)
    rf_scores = rf.predict_proba(x)

    train_summary = {
        "label_classes": encoder.classes_.tolist(),
        "linear_svm": {
            "C": 0.5,
            "class_weight": "balanced",
            "max_iter": 6000,
            "train_balanced_accuracy": float(balanced_accuracy_score(train_y, svm.predict(x[train_idx]))),
        },
        "random_forest": {
            "n_estimators": int(rf_trees),
            "max_depth": 16,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "train_balanced_accuracy": float(balanced_accuracy_score(train_y, rf.predict(x[train_idx]))),
        },
    }
    return {
        "Linear SVM": np.asarray(svm_scores, dtype=np.float32),
        "Random forest": np.asarray(rf_scores, dtype=np.float32),
    }, train_summary


def curve_metrics_for_queries(
    pool_x: np.ndarray,
    query_idx: np.ndarray,
    meta: pd.DataFrame,
    fps: list,
    k_values: list[int],
    method: str,
    n_neighbors: int,
    chunk_size: int,
    hnsw_m: int,
    ef_search: int,
) -> dict:
    from rdkit import DataStructs

    max_k = max(k_values)
    neigh = faiss_search(
        pool_x,
        pool_x[query_idx],
        max(n_neighbors, max_k + 64),
        method,
        chunk_size,
        hnsw_m,
        ef_search,
    )

    mz = meta["mz"].to_numpy(dtype=float)
    formula = meta["frag_formula"].fillna("").astype(str).to_numpy()
    family = meta["frag_label"].fillna("").astype(str).to_numpy()
    spec = meta["spec_id"].fillna("").astype(str).to_numpy()

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

    filtered: list[np.ndarray] = []
    for row_pos, row in enumerate(neigh):
        qi = int(query_idx[row_pos])
        picked = [int(j) for j in row if j >= 0 and j != qi and spec[j] != spec[qi]]
        filtered.append(np.asarray(picked[:max_k], dtype=np.int64))

    for k in k_values:
        formula_hits = []
        family_hits = []
        mz_errors = []
        tanimoto_best = []
        for row_pos, picked in enumerate(filtered):
            if len(picked) < k:
                continue
            qi = int(query_idx[row_pos])
            cand = picked[:k]
            formula_hits.append(bool(np.any(formula[cand] == formula[qi])))
            family_hits.append(bool(np.any(family[cand] == family[qi])))
            mz_errors.append(float(np.min(np.abs(mz[cand] - mz[qi]))))

            if fps[qi] is not None:
                cand_fps = [fps[j] for j in cand if fps[j] is not None]
                if cand_fps:
                    sims = DataStructs.BulkTanimotoSimilarity(fps[qi], cand_fps)
                    tanimoto_best.append(float(max(sims)))

        out["formula_hit"].append(float(np.mean(formula_hits)) if formula_hits else float("nan"))
        out["fragment_family_hit"].append(float(np.mean(family_hits)) if family_hits else float("nan"))
        out["best_fp_tanimoto"].append(float(np.mean(tanimoto_best)) if tanimoto_best else float("nan"))
        out["best_mz_mae_da"].append(float(np.mean(mz_errors)) if mz_errors else float("nan"))
        out["best_mz_median_da"].append(float(np.median(mz_errors)) if mz_errors else float("nan"))
        out["n_valid_neighbors"].append(int(len(mz_errors)))
        out["n_valid_fp"].append(int(len(tanimoto_best)))

    out["n_missing_cross_spectrum_neighbor_at_max_k"] = int(sum(len(p) < max_k for p in filtered))
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
        for model in MODEL_ORDER:
            curve = metrics["curves"][model]
            for pos, k in enumerate(curve["k"]):
                row = {"model": model, "k": int(k)}
                for key in fieldnames[2:]:
                    row[key] = curve[key][pos]
                writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--subset-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--csv-name", default="peak_neighbor_curves.csv")
    ap.add_argument("--json-name", default="peak_neighbor_curves.json")
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-values", default="1,2,5,10,20,50,100")
    ap.add_argument("--nn-method", choices=["faiss_hnsw", "faiss_flat"], default="faiss_hnsw")
    ap.add_argument("--n-neighbors", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--ef-search", type=int, default=128)
    ap.add_argument("--rf-trees", type=int, default=240)
    args = ap.parse_args()

    k_values = parse_k_values(args.k_values)
    meta = pd.read_csv(args.data_dir / "peak_embedding_metadata.csv")
    ultra = np.load(args.data_dir / "peak_embeddings_ultra.npy")
    dreams = np.load(args.data_dir / "peak_embeddings_dreams.npy")
    idx = balanced_sample(meta, args.max_points, args.seed)
    meta = meta.iloc[idx].reset_index(drop=True)
    meta = add_spectrum_features(meta, args.subset_csv)
    ultra = ultra[idx]
    dreams = dreams[idx]

    peak_features, feature_names = make_classical_features(meta)
    train_idx, query_idx = spectrum_group_split(meta["spec_id"].to_numpy(), args.test_size, args.seed)
    predictor_scores, training = train_classical_predictors(
        peak_features,
        meta["frag_label"].to_numpy(),
        train_idx,
        args.seed,
        args.rf_trees,
    )
    fps = make_fingerprints(meta["frag_smiles"].to_numpy())

    representations = {
        "UltraMS": ultra,
        "DreaMS": dreams,
        "Peak features": peak_features,
        **predictor_scores,
    }
    curves = {}
    for model in MODEL_ORDER:
        curves[model] = curve_metrics_for_queries(
            representations[model],
            query_idx,
            meta,
            fps,
            k_values,
            args.nn_method,
            args.n_neighbors,
            args.chunk_size,
            args.hnsw_m,
            args.ef_search,
        )

    metrics = {
        "curves": curves,
        "n_peaks": int(len(meta)),
        "n_spectra": int(meta["spec_id"].nunique()),
        "n_train_peaks": int(len(train_idx)),
        "n_query_peaks": int(len(query_idx)),
        "n_train_spectra": int(meta.iloc[train_idx]["spec_id"].nunique()),
        "n_query_spectra": int(meta.iloc[query_idx]["spec_id"].nunique()),
        "fragment_label_counts": meta["frag_label"].value_counts().to_dict(),
        "feature_names": feature_names,
        "sampling": {
            "max_points": int(args.max_points),
            "seed": int(args.seed),
            "test_size": float(args.test_size),
            "source_data_dir": str(args.data_dir),
        },
        "training": training,
        "nearest_neighbor": {
            "method": args.nn_method,
            "n_neighbors": int(max(args.n_neighbors, max(k_values) + 64)),
            "chunk_size": int(args.chunk_size),
            "hnsw_m": int(args.hnsw_m),
            "ef_search": int(args.ef_search),
        },
        "classical_baseline_note": (
            "Peak features use mz, intensity, precursor_mz, neutral_loss, mz_ratio, "
            "collision_energy, spectrum peak count, adduct and ion mode. MAGMa fragment "
            "labels/formulas/smiles are evaluation targets only; RF/SVM train on "
            "spectrum-held-out training peaks and are evaluated on held-out query peaks."
        ),
    }

    write_curve_tables(metrics, args.output_dir, args.csv_name, args.json_name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
