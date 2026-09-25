"""Recalculate the Figure 2 results from the released evaluation outputs.

Run from any directory with ``python benchmarks/spectral_properties/reproduce.py``.
The command writes a compact CSV beside this script by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROWS: list[dict[str, object]] = []


def add(panel: str, task: str, dataset: str, model: str, metric: str, value: float,
        n: int | str, level: str) -> None:
    ROWS.append(dict(panel=panel, task=task, dataset=dataset, model=model,
                     metric=metric, value=float(value), n=n, level=level))


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def reconstruction() -> None:
    for dataset, stem in (("GeMS-A10", "gems"), ("MassSpecGym", "massgym")):
        with np.load(DATA / f"reconstruction_{stem}.npz") as arrays:
            for model, prefix in (("UltraMS", "v9"), ("DreaMS", "dreams")):
                error = np.abs(arrays[f"{prefix}_mz_gt"] - arrays[f"{prefix}_mz_pred"])
                add("2a", "Masked peak m/z reconstruction", dataset, model,
                    "accuracy_0.05_Da", np.mean(error <= 0.05), len(error), "peak predictions")


def ion_mode() -> None:
    with (DATA / "ion_mode.json").open() as handle:
        results = json.load(handle)["results"]
    for key, model in (("rt_only_d", "UltraMS"), ("dreams", "DreaMS")):
        values = results[key]["mlp_test"]
        for metric in ("accuracy", "f1_neg", "f1_pos"):
            add("2b", "Ion mode recognition", "MSnLib", model,
                metric, values[metric], values["n"], "saved evaluation")


def structural_families() -> None:
    rows = read_csv("structure_families_test.csv")
    labels = [row["structure_cluster"] for row in rows]
    for model, prefix in (("UltraMS", "ultrams_supervised"),
                          ("DreaMS", "dreams_supervised"), ("Raw spectrum", "raw")):
        clusters = [row[f"{prefix}_cluster"] for row in rows]
        add("2c", "Negative ion structural families", "SpectraVerse", model,
            "NMI", normalized_mutual_info_score(labels, clusters), len(rows), "spectrum labels")
        add("2c", "Negative ion structural families", "SpectraVerse", model,
            "ARI", adjusted_rand_score(labels, clusters), len(rows), "spectrum labels")
        add("2c", "Negative ion structural families", "SpectraVerse", model,
            "silhouette", np.mean([float(row[f"{prefix}_silhouette"]) for row in rows]),
            len(rows), "spectrum scores")


def neutral_losses() -> None:
    labels = ("H2O", "NH3", "CO2", "HF", "HCl", "CO", "CH3")
    displayed = ("H2O", "NH3", "CO2", "CO", "CH3")
    for model, stem in (("UltraMS", "ultrams"), ("DreaMS", "dreams")):
        targets = np.load(DATA / f"neutral_loss_{stem}_targets.npy")
        probs = np.load(DATA / f"neutral_loss_{stem}_probabilities.npy")
        if targets.shape != probs.shape or targets.shape[1] != len(labels):
            raise ValueError(f"Unexpected neutral-loss arrays for {model}")
        aucs = {}
        aps = {}
        for index, label in enumerate(labels):
            aucs[label] = roc_auc_score(targets[:, index], probs[:, index])
            aps[label] = average_precision_score(targets[:, index], probs[:, index])
            add("2d-e", "Neutral loss prediction", "MassSpecGym", model,
                f"{label}_ROC_AUC", aucs[label], len(targets), "test probabilities")
            add("2d-e", "Neutral loss prediction", "MassSpecGym", model,
                f"{label}_AP", aps[label], len(targets), "test probabilities")
        for group, names in (("displayed_five", displayed), ("all_seven", labels)):
            add("2d-e", "Neutral loss prediction", "MassSpecGym", model,
                f"{group}_macro_ROC_AUC", np.mean([aucs[name] for name in names]),
                len(targets), "test probabilities")
            add("2d-e", "Neutral loss prediction", "MassSpecGym", model,
                f"{group}_macro_AP", np.mean([aps[name] for name in names]),
                len(targets), "test probabilities")


def element_peak_localization() -> None:
    for row in read_csv("element_peak_hit_rate.csv"):
        add("2g", "Element-bearing peak localization", "MassSpecGym",
            row["method"], f"{row['atom']}_top{row['k']}_hit_rate", float(row["mean"]),
            int(row["n"]), "saved repeat summary")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(1 - np.sum((y_true - y_pred) ** 2) /
                 np.sum((y_true - np.mean(y_true)) ** 2))


def heteroatom_counts() -> None:
    for filename, methods in (
        ("heteroatom_deep_predictions.csv.gz", (("UltraMS", "ultrams"), ("DreaMS", "dreams"))),
        ("heteroatom_classical_predictions.csv.gz", (("Linear SVM", "linear_svm"),
                                                      ("Random forest", "random_forest"))),
    ):
        import gzip
        with gzip.open(DATA / filename, "rt", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for element in ("S", "Cl", "F", "Br"):
            true = np.array([float(row[f"true_{element}"]) for row in rows])
            for model, prefix in methods:
                pred = np.array([float(row[f"pred_{prefix}_{element}"]) for row in rows])
                add("2h", "Heteroatom count prediction", "MassSpecGym", model,
                    f"{element}_R2", r2_score(true, pred), len(rows), "test predictions")


def peak_neighbors() -> None:
    for row in read_csv("peak_neighbor_fidelity.csv"):
        for metric in ("formula_hit", "fragment_family_hit", "best_fp_tanimoto", "best_mz_mae_da"):
            add("2j", "Peak-neighbor chemical fidelity", "MSnLib/MAGMa", row["model"],
                f"top{row['k']}_{metric}", float(row[metric]), int(row["n_valid_neighbors"]),
                "saved neighbor summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results.csv")
    args = parser.parse_args()
    reconstruction()
    ion_mode()
    structural_families()
    neutral_losses()
    element_peak_localization()
    heteroatom_counts()
    peak_neighbors()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("panel", "task", "dataset", "model", "metric", "value", "n", "level"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"Wrote {len(ROWS)} results to {args.output}")


if __name__ == "__main__":
    main()
