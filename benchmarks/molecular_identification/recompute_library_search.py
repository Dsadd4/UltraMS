"""Recompute Figure 3e same-adduct FDR results from saved top-1 scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source_code" / "original"))
from baseline_common import exact_tie_aware_fdr_curve, fdr_working_point  # noqa: E402
from evaluate_exact_fdr import exact_low_fdr_average  # noqa: E402


def table(name: str) -> dict:
    with (HERE / "data" / name).open(newline="") as handle:
        return {(row["method"], row["mode"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "library_search_from_scores.json")
    args = parser.parse_args()
    manifest = json.loads((HERE / "data" / "strict_adduct_manifest_summary.json").read_text())
    working = table("working_points_primary_seed0.csv")
    low_fdr = table("low_fdr_mean_recall_primary_seed0.csv")
    output = {}
    for path in sorted((HERE / "data" / "strict_adduct_scores").glob("top1_*_test_seed_*.npz")):
        parts = path.stem.split("_")
        method = {
            "ultrams": "UltraMS", "dreams": "DreaMS", "deepsets": "DeepSets",
            "codebook": "Codebook", "linear": "Linear", "fourier": "Fourier",
        }[parts[1]]
        mode = parts[2]
        with np.load(path, allow_pickle=False) as data:
            score = data["top1_score"]
            correct = data["top1_correct"]
            eligible = int(data["has_library_positive"].sum()) if "has_library_positive" in data else int(
                manifest["modes"][mode]["test"]["n_queries_with_library_positive"])
        curve = exact_tie_aware_fdr_curve(score, correct, denominator=eligible, min_hits=10)
        point = fdr_working_point(curve, .05)
        mean = exact_low_fdr_average(curve, .05)["average_recall"]
        reference = working[(method, mode)]
        expected_mean = low_fdr[(method, mode)]
        for key, value in (("identification_recall", point["recall"]),
                           ("empirical_fdr", point["fdr"]),
                           ("threshold", point["threshold"])):
            if not math.isclose(float(reference[key]), value, abs_tol=1e-10, rel_tol=0):
                raise AssertionError(f"{method}/{mode} {key}: {value} != {reference[key]}")
        if not math.isclose(mean, float(expected_mean["mean_identification_recall"]), abs_tol=1e-10, rel_tol=0):
            raise AssertionError(f"{method}/{mode}: low-FDR average differs")
        output[f"{method}/{mode}"] = {
            "n_test_queries": len(score), "n_eligible_queries": eligible,
            "unthresholded_recall": float(np.sum(correct) / eligible),
            "unthresholded_fdr": float((len(score) - np.sum(correct)) / len(score)),
            "recall_at_fdr_5pct": point["recall"],
            "mean_recall_fdr_0_to_5pct": mean,
        }
    if len(output) != 12:
        raise AssertionError(f"expected 12 method/mode results; found {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Recomputed all 12 six-method results from top-1 scores: {args.output}")


if __name__ == "__main__":
    main()
