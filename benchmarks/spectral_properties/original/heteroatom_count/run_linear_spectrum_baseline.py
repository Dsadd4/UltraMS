"""Fit the Figure 2h classical baselines on the frozen MassSpecGym folds.

The baseline uses the same raw-spectrum representation as the Figure 2g
classical baseline: maximum normalized intensity in fixed 1-Da m/z bins. A
linear support-vector and random-forest regressors are fitted on the existing
training fold and evaluated once on the requested frozen evaluation fold. The script
writes predictions, metrics and a complete run configuration so the plotted
values are auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR


ELEMENTS = ["S", "Cl", "F", "Br"]
TARGET_COLUMNS = [f"cnt_{element}" for element in ELEMENTS]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binned_spectra(
    frame: pd.DataFrame,
    n_bins: int,
    bin_size: float,
) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for row_index, row in enumerate(frame.itertuples(index=False)):
        spectrum = np.frombuffer(row.spec_bytes, dtype=np.float32).reshape(150, 2)
        spectrum = spectrum[: int(row.attn_len)]
        if spectrum.size == 0:
            continue

        mz = spectrum[:, 0].astype(np.float64, copy=False)
        intensity = np.maximum(spectrum[:, 1].astype(np.float64, copy=False), 0.0)
        maximum = float(intensity.max(initial=0.0))
        if maximum <= 0:
            continue
        intensity = intensity / maximum

        bins = np.floor(mz / bin_size).astype(np.int64)
        valid = (bins >= 0) & (bins < n_bins) & np.isfinite(intensity)
        bins = bins[valid]
        intensity = intensity[valid]
        if not len(bins):
            continue

        unique_bins = np.unique(bins)
        for bin_index in unique_bins:
            rows.append(row_index)
            columns.append(int(bin_index))
            values.append(float(intensity[bins == bin_index].max()))

    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(frame), n_bins),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mz-max", type=float, default=1500.0)
    parser.add_argument("--bin-size", type=float, default=1.0)
    parser.add_argument("--svm-c", type=float, default=0.5)
    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--evaluation-fold", choices=("val", "test"), default="test")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_bins = int(np.ceil(args.mz_max / args.bin_size)) + 1
    columns = ["fold", "spec_bytes", "attn_len", *TARGET_COLUMNS]
    data = pd.read_parquet(args.parquet, columns=columns)
    train = data[data["fold"].eq("train")].reset_index(drop=True)
    evaluation = data[data["fold"].eq(args.evaluation_fold)].reset_index(drop=True)

    print(f"Building train matrix: n={len(train):,}, bins={n_bins}", flush=True)
    x_train = binned_spectra(train, n_bins=n_bins, bin_size=args.bin_size)
    print(f"Building {args.evaluation_fold} matrix: n={len(evaluation):,}", flush=True)
    x_evaluation = binned_spectra(evaluation, n_bins=n_bins, bin_size=args.bin_size)
    y_train = train[TARGET_COLUMNS].to_numpy(np.float64)
    y_evaluation = evaluation[TARGET_COLUMNS].to_numpy(np.float64)

    estimators = {
        "Linear SVM": lambda: LinearSVR(
            C=args.svm_c,
            epsilon=0.0,
            loss="squared_epsilon_insensitive",
            max_iter=6000,
            random_state=42,
        ),
        "Random forest": lambda: RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=14,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
    }
    metrics = []
    prediction_columns: dict[str, np.ndarray] = {
        f"true_{element}": y_evaluation[:, index]
        for index, element in enumerate(ELEMENTS)
    }
    for method, estimator_factory in estimators.items():
        for index, element in enumerate(ELEMENTS):
            print(f"Fitting {method}: {element}", flush=True)
            model = estimator_factory()
            model.fit(x_train, y_train[:, index])
            predictions = model.predict(x_evaluation)
            score = float(r2_score(y_evaluation[:, index], predictions))
            metrics.append(
                {
                    "element": element,
                    "model": method,
                    "r2": score,
                    "evaluation_fold": args.evaluation_fold,
                    "evaluation_n": int(len(evaluation)),
                    "train_n": int(len(train)),
                }
            )
            prediction_columns[f"pred_{method.lower().replace(' ', '_')}_{element}"] = predictions
            print(f"{method} {element}: R2={score:.6f}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output_dir / "classical_regression_metrics.csv", index=False)

    prediction_frame = pd.DataFrame(prediction_columns)
    prediction_frame.to_csv(
        args.output_dir / f"classical_regression_{args.evaluation_fold}_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    config = {
        "task": "element count regression",
        "source_parquet": str(args.parquet.resolve()),
        "source_parquet_sha256": file_sha256(args.parquet),
        "train_fold": "train",
        "evaluation_fold": args.evaluation_fold,
        "train_n": int(len(train)),
        "evaluation_n": int(len(evaluation)),
        "elements": ELEMENTS,
        "representation": "maximum base-peak-normalized intensity in fixed m/z bins",
        "mz_max": args.mz_max,
        "bin_size": args.bin_size,
        "n_bins": n_bins,
        "estimators": {
            "Linear SVM": {
                "class": "sklearn.svm.LinearSVR",
                "C": args.svm_c,
                "epsilon": 0.0,
                "loss": "squared_epsilon_insensitive",
                "max_iter": 6000,
                "random_state": 42,
            },
            "Random forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "n_estimators": args.rf_estimators,
                "max_depth": 14,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "random_state": 42,
            },
        },
        "metric": "per-element R2",
        "metrics": metrics,
    }
    with (args.output_dir / "classical_regression_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
