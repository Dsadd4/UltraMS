"""Recalculate the molecular-identification results shown in Figure 3.

Run from any directory: ``python benchmarks/molecular_identification/reproduce.py``.
Only the Python standard library is needed for this saved-result replay.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
METHODS = ("DeepSets", "Codebook", "Linear", "Fourier", "DreaMS", "UltraMS")


def rows(name: str):
    path = DATA / name
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def topk_from_rank(rank: int, k: int) -> int:
    return int(0 < rank <= k)


def single_spectrum() -> dict:
    counts = defaultdict(lambda: {"n_spectra": 0, "top1": 0, "top5": 0, "top20": 0, "top100": 0})
    for row in rows("single_spectrum_query_ranks.csv.gz"):
        rank = int(row["rank"])
        item = counts[row["method"]]
        item["n_spectra"] += 1
        for k in (1, 5, 20, 100):
            item[f"top{k}"] += topk_from_rank(rank, k)
    return {method: {"n_spectra": item["n_spectra"], **{
        f"top{k}": item[f"top{k}"] / item["n_spectra"] for k in (1, 5, 20, 100)
    }} for method, item in counts.items()}


def multiple_spectrum_voting() -> dict:
    counts = defaultdict(lambda: {"n_molecules": 0, "top1": 0, "top5": 0, "top20": 0})
    for row in rows("multiple_spectrum_voting_per_group.csv.gz"):
        method = row["method"]
        rank = int(row["consensus_rank"])
        item = counts[method]
        item["n_molecules"] += 1
        for k in (1, 5, 20):
            item[f"top{k}"] += topk_from_rank(rank, k)
    return {method: {"n_molecules": item["n_molecules"], **{
        f"top{k}": item[f"top{k}"] / item["n_molecules"] for k in (1, 5, 20)
    }} for method, item in counts.items()}


def label_blind_voting() -> dict:
    counts = defaultdict(lambda: {"n_spectra": 0, "single_hits": 0, "voting_hits": 0, "groups": set()})
    query_rows = list(rows("label_blind_per_query.csv.gz"))
    group_molecules = defaultdict(set)
    for row in query_rows:
        item = counts[row["method"]]
        item["n_spectra"] += 1
        item["single_hits"] += int(row["single_top1_correct"])
        item["voting_hits"] += int(row["consensus_top1_correct"])
        item["groups"].add(row["group_id"])
        group_molecules[row["group_id"]].add(row["hidden_target_smiles"])
    assignment_n = 0
    assignment_groups = set()
    for assignment in rows("label_blind_group_assignments_test.csv.gz"):
        assignment_n += 1
        assignment_groups.add(assignment["group_id"])
    if assignment_n != 52_070 or len(assignment_groups) != 15_541:
        raise AssertionError("label-blind test grouping differs from saved experiment")
    selected = [row for row in rows("label_blind_threshold_selection_val.csv") if row["selected"] == "1"]
    if len(selected) != 1 or float(selected[0]["similarity_threshold"]) != 0.5:
        raise AssertionError("label-blind validation threshold differs")
    output = {}
    interval = {row["model"]: row for row in rows("paired_group_bootstrap.csv")
                if row["metric"] == "top1_delta"}
    strata = defaultdict(lambda: {"n_spectra": 0, "single_hits": 0, "voting_hits": 0})
    for row in query_rows:
        stratum = "same_molecule_groups" if len(group_molecules[row["group_id"]]) == 1 else "mixed_molecule_groups"
        item = strata[(row["method"], stratum)]
        item["n_spectra"] += 1
        item["single_hits"] += int(row["single_top1_correct"])
        item["voting_hits"] += int(row["consensus_top1_correct"])
    for method, item in counts.items():
        if item["groups"] != assignment_groups:
            raise AssertionError(f"{method}: grouped test spectra differ")
        n = item["n_spectra"]
        single = item["single_hits"] / n
        voting = item["voting_hits"] / n
        output[method] = {"n_spectra": n, "n_groups": len(item["groups"]),
                          "single_top1": single, "voting_top1": voting,
                          "gain_percentage_points": 100 * (voting - single)}
        for stratum in ("same_molecule_groups", "mixed_molecule_groups"):
            part = strata[(method, stratum)]
            output[method][stratum] = {
                "n_spectra": part["n_spectra"],
                "gain_percentage_points": 100 * (part["voting_hits"] - part["single_hits"]) / part["n_spectra"],
            }
        if method in interval:
            close(voting - single, float(interval[method]["estimate"]), f"label-blind gain {method}")
            output[method]["group_bootstrap_ci_percentage_points"] = [
                100 * float(interval[method]["ci_low"]),
                100 * float(interval[method]["ci_high"]),
            ]
    return output


def fdr_summary(points: list[dict], target: float = .05) -> dict:
    """Same exact step-envelope integration as the original Figure 3e evaluator."""
    available = [(float(row["empirical_fdr"]), float(row["identification_recall"]))
                 for row in points if row["empirical_fdr"]]
    available.sort()
    breaks = sorted({0.0, target, *(fdr for fdr, _ in available if 0 <= fdr <= target)})
    area = 0.0
    for left, right in zip(breaks[:-1], breaks[1:]):
        area += (right - left) * max((recall for fdr, recall in available if fdr <= left), default=0.0)
    eligible = [row for row in points if row["empirical_fdr"] and float(row["empirical_fdr"]) <= target]
    best = sorted(eligible, key=lambda row: (-float(row["identification_recall"]),
                                                  float(row["empirical_fdr"]),
                                                  -float(row["threshold"])))[0] if eligible else None
    endpoint = max(points, key=lambda row: int(row["accepted"]))
    return {
        "n_test_queries": int(endpoint["accepted"]),
        "n_eligible_queries": int(endpoint.get("n_eligible_queries") or 0),
        "unthresholded_recall": float(endpoint["identification_recall"]),
        "unthresholded_fdr": float(endpoint["empirical_fdr"]),
        "recall_at_fdr_5pct": float(best["identification_recall"]) if best else 0.0,
        "mean_recall_fdr_0_to_5pct": area / target,
    }


def strict_adduct() -> dict:
    grouped = defaultdict(list)
    for row in rows("strict_adduct_fdr_frontiers.csv.gz"):
        if row["split"] == "test" and row["method"] in METHODS:
            grouped[(row["method"], row["mode"])].append(row)
    return {f"{method}/{mode}": fdr_summary(points)
            for (method, mode), points in grouped.items()}


def additional_adducts() -> dict:
    grouped = defaultdict(list)
    for row in rows("other_strict_adduct_exact_fdr_frontiers.csv.gz"):
        if row["method"] in METHODS:
            grouped[(row["method"], row["mode"], row["adduct"])].append(row)
    reference = {(row["method"], row["mode"], row["adduct"]): row
                 for row in rows("other_strict_adduct_low_fdr_summary.csv")}
    sums = defaultdict(lambda: [0.0, 0])
    for key, points in grouped.items():
        method, mode, _ = key
        mean = fdr_summary(points)["mean_recall_fdr_0_to_5pct"]
        close(mean, float(reference[key]["mean_identification_recall_fdr_0_5pct"]),
              f"additional adduct {method}/{mode}/{key[2]}")
        n = int(reference[key]["n_query"])
        sums[(method, mode)][0] += n * mean
        sums[(method, mode)][1] += n
    return {f"{method}/{mode}": {"n_queries": n, "mean_micro_recall_fdr_0_to_5pct": total / n}
            for (method, mode), (total, n) in sums.items()}


def collision_energy() -> dict:
    labels = {"all_compounds": None,
              "stereochemically_complex": "stereochemically_complex",
              "multi_ring_aromatics": "multi_ring_aromatics",
              "carboxylic_acids": "carboxylic_acids",
              "halogenated_compounds": "halogenated_compounds"}
    counts = defaultdict(lambda: [0, 0])
    for row in rows("collision_energy_queries.csv.gz"):
        for label, flag in labels.items():
            if flag is None or row[flag] == "1":
                item = counts[(row["method"], row["mode"], label)]
                item[0] += int(row["top1_correct"])
                item[1] += 1
    return {f"{method}/{mode}/{label}": {"n_molecules": n, "recall_at_1": hits / n}
            for (method, mode, label), (hits, n) in counts.items()}


def additional_molecule_identification() -> dict:
    sources = {
        "NPLIB1": ("nplib1_metrics.csv", {"UltraMS": "u5_cls_projmean_rawproj", "DreaMS": "dreams_global_iw"}),
        "MassSpecGym mass": ("massspecgym_mass_metrics.csv", {"UltraMS": "u5_cls_projmean", "DreaMS": "dreams_global_iw"}),
        "MassSpecGym formula": ("massspecgym_formula_metrics.csv", {"UltraMS": "u5_cls_projmean", "DreaMS": "dreams_global_iw"}),
    }
    output = {}
    for task, (filename, models) in sources.items():
        source_rows = list(rows(filename))
        for public_name, source_name in models.items():
            matches = [row for row in source_rows if row["model"] == source_name and (
                (task == "NPLIB1" and row["scope"] == "aggregate" and row["split"] == "all")
                or (task != "NPLIB1" and row["scenario"] == "all" and row["group"] == "ensembles"))]
            if len(matches) != 1:
                raise ValueError(f"{task}/{public_name}: expected one saved result")
            row = matches[0]
            output[f"{task}/{public_name}"] = {
                "n_queries": int(row["n_queries"]),
                **{f"top{k}": float(row[f"top{k}"]) / 100 for k in (1, 5, 20)},
            }
    return output


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise AssertionError(f"{label}: {actual} differs from source {expected}")


def validate(result: dict) -> None:
    single = result["single_spectrum"]
    for row in rows("single_method_summary.csv"):
        if single[row["method"]]["n_spectra"] != int(row["n_queries"]):
            raise AssertionError(f"single {row['method']}: query count differs")
        for k in (1, 5, 20):
            close(single[row["method"]][f"top{k}"], float(row[f"top{k}_primary"]), f"single {row['method']}@{k}")
    for row in rows("single_spectrum_rank_cdf.csv.gz"):
        k = int(row["candidate_rank_k"])
        if k in (1, 5, 20, 100):
            close(single[row["method"]][f"top{k}"], float(row["cdf_percent"]) / 100,
                  f"single CDF {row['method']}@{k}")
    multiple = result["multiple_spectrum_voting"]
    for row in rows("multiple_spectrum_voting_summary.csv"):
        if multiple[row["method"]]["n_molecules"] != int(row["n_molecules"]):
            raise AssertionError(f"voting {row['method']}: molecule count differs")
        for k in (1, 5, 20):
            close(multiple[row["method"]][f"top{k}"], float(row[f"top{k}"]),
                  f"voting {row['method']}@{k}")
    blind = result["label_blind_voting"]
    for row in rows("model_performance.csv"):
        if row["stratum"] == "all_grouped":
            close(blind[row["model"]]["single_top1"], float(row["single_top1"]), f"blind single {row['model']}")
            close(blind[row["model"]]["voting_top1"], float(row["consensus_top1"]), f"blind voting {row['model']}")
        elif row["stratum"] in ("pure_groups", "impure_groups"):
            stratum = "same_molecule_groups" if row["stratum"] == "pure_groups" else "mixed_molecule_groups"
            part = blind[row["model"]][stratum]
            if part["n_spectra"] != int(row["n_spectra"]):
                raise AssertionError(f"{row['model']} {stratum}: spectrum count differs")
            close(part["gain_percentage_points"] / 100, float(row["top1_delta"]),
                  f"{row['model']} {stratum} gain")
    strict = result["same_adduct_library_search"]
    for row in rows("working_points_primary_seed0.csv"):
        key = f"{row['method']}/{row['mode']}"
        close(strict[key]["recall_at_fdr_5pct"], float(row["identification_recall"]), f"strict {key}")
    for row in rows("low_fdr_mean_recall_primary_seed0.csv"):
        key = f"{row['method']}/{row['mode']}"
        close(strict[key]["mean_recall_fdr_0_to_5pct"], float(row["mean_identification_recall"]), f"strict mean {key}")
    supplementary = result["supplementary_molecule_identification"]
    for row in rows("additional_molecule_identification.csv"):
        if row["display_model"] in ("UltraMS", "DreaMS"):
            key = f"{row['panel']}/{row['display_model']}"
            metric = row["metric"].lower().replace("-", "")
            close(supplementary[key][metric], float(row["retrieval_percent"]) / 100, f"supplementary {key}/{metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "results.json")
    args = parser.parse_args()
    result = {
        "single_spectrum": single_spectrum(),
        "multiple_spectrum_voting": multiple_spectrum_voting(),
        "label_blind_voting": label_blind_voting(),
        "same_adduct_library_search": strict_adduct(),
        "additional_adduct_library_search": additional_adducts(),
        "collision_energy_search": collision_energy(),
        "supplementary_molecule_identification": additional_molecule_identification(),
    }
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for task, methods in result.items():
        print(f"{task}: {len(methods)} result rows")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
