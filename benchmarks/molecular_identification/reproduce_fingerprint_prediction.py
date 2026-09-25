"""Calculate fingerprint-based identification accuracy from saved test queries."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
TASKS = {
    "NPLIB1": ("nplib1", "nplib1_metrics.csv", {"UltraMS": "u5_cls_projmean_rawproj", "DreaMS": "dreams_global_iw"}),
    "MassSpecGym mass": ("massspecgym_mass", "massspecgym_mass_metrics.csv", {"UltraMS": "u5_cls_projmean", "DreaMS": "dreams_global_iw"}),
    "MassSpecGym formula": ("massspecgym_formula", "massspecgym_formula_metrics.csv", {"UltraMS": "u5_cls_projmean", "DreaMS": "dreams_global_iw"}),
}


def read_csv(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream)


def calculate() -> dict:
    results = {}
    for task, (file_stem, summary_name, models) in TASKS.items():
        counts = defaultdict(lambda: {"n_queries": 0, "top1": 0, "top5": 0, "top20": 0})
        for row in read_csv(DATA / "fingerprint_prediction" / f"{file_stem}_test_queries.csv.gz"):
            if row["phase"] != "test":
                raise ValueError(f"{task}: unexpected phase {row['phase']}")
            rank = int(row["rank"])
            if rank < 1:
                raise ValueError(f"{task}: invalid rank {rank}")
            item = counts[row["model"]]
            item["n_queries"] += 1
            for k in (1, 5, 20):
                hit = int(rank <= k)
                if int(row[f"hit{k}"]) != hit:
                    raise ValueError(f"{task}: inconsistent hit{k} for rank {rank}")
                item[f"top{k}"] += hit

        summaries = list(read_csv(DATA / summary_name))
        for display_name, model_name in models.items():
            matching = [row for row in summaries if row["model"] == model_name and (
                (task == "NPLIB1" and row["scope"] == "aggregate" and row["split"] == "all")
                or (task != "NPLIB1" and row["scenario"] == "all" and row["group"] == "ensembles")
            )]
            if len(matching) != 1:
                raise ValueError(f"{task}/{display_name}: expected one saved summary")
            source = matching[0]
            item = counts[model_name]
            n = item["n_queries"]
            if n != int(source["n_queries"]):
                raise ValueError(f"{task}/{display_name}: query count differs from source")
            metrics = {f"top{k}": 100 * item[f"top{k}"] / n for k in (1, 5, 20)}
            for metric, value in metrics.items():
                if not math.isclose(value, float(source[metric]), abs_tol=1e-9):
                    raise ValueError(f"{task}/{display_name}: {metric} differs from source")
            results[f"{task}/{display_name}"] = {"n_queries": n, **metrics}
    return results


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2))
