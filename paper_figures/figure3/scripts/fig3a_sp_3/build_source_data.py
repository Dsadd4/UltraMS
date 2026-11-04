#!/usr/bin/env python3
"""Package and audit frozen S7 per-query results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import binomtest


HERE = Path(__file__).resolve()
S7_DIR = HERE.parents[1]
PROJECT_ROOT = HERE.parents[5]
CONFIG_PATH = S7_DIR / "config" / "analysis_config.json"
SOURCE_DIR = S7_DIR / "source_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse target SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def outcome(global_rank: pd.Series, fusion_rank: pd.Series) -> np.ndarray:
    return np.select(
        [
            (global_rank > 1) & (fusion_rank == 1),
            (global_rank == 1) & (fusion_rank == 1),
            (global_rank == 1) & (fusion_rank > 1),
        ],
        ["rescued", "preserved", "harmed"],
        default="unresolved",
    )


def read_rank_view(
    path: Path,
    prefix: str,
    input_paths: set[Path],
    keep_metadata: bool = False,
) -> pd.DataFrame:
    input_paths.add(path)
    frame = pd.read_csv(path)
    required = {
        "spec_id",
        "rank",
        "rr",
        "hit1",
        "hit5",
        "hit10",
        "hit20",
        "top1_smiles",
        "gt_score",
        "top1_score",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["spec_id"].duplicated().any():
        raise ValueError(f"Duplicate spec_id rows in {path}")

    metadata = [
        "spec_id",
        "parent_spec_id",
        "smiles",
        "adduct",
        "block_type",
        "n_peaks_model",
        "n_peaks_original",
        "n_peaks_above_precursor_plus1",
        "candidate_count",
        "candidate_count_raw",
        "candidate_count_scored",
    ]
    values = [
        "rank",
        "rr",
        "hit1",
        "hit5",
        "hit10",
        "hit20",
        "top1_smiles",
        "gt_score",
        "top1_score",
    ]
    columns = metadata + values if keep_metadata else ["spec_id"] + values
    columns = [column for column in columns if column in frame.columns]
    result = frame[columns].copy()
    return result.rename(
        columns={column: f"{prefix}_{column}" for column in values if column in result}
    )


def merge_views(base: pd.DataFrame, view: pd.DataFrame, source: str) -> pd.DataFrame:
    before = len(base)
    merged = base.merge(view, on="spec_id", how="inner", validate="one_to_one")
    if len(merged) != before:
        raise ValueError(f"Query mismatch after merging {source}: {before} -> {len(merged)}")
    return merged


def add_molecular_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = []
    heavy_atoms = []
    rings = []
    bertz = []
    for smiles in frame["smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit could not parse target SMILES: {smiles}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        heavy_atoms.append(mol.GetNumHeavyAtoms())
        rings.append(rdMolDescriptors.CalcNumRings(mol))
        bertz.append(Descriptors.BertzCT(mol))
    result = frame.copy()
    result["canonical_smiles"] = canonical
    result["heavy_atoms"] = heavy_atoms
    result["ring_count"] = rings
    result["bertz_complexity"] = bertz
    return result


def apply_strata(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = frame.copy()
    specs = [
        ("candidate_stratum", "candidate_count", "candidate_competition"),
        ("peak_stratum", "n_peaks_model", "retained_peak_evidence"),
        ("heavy_atom_stratum", "heavy_atoms", "molecular_size"),
    ]
    for output, value_column, config_key in specs:
        result[output] = "unassigned"
        for lower, upper, label in config["difficulty_strata"][config_key]:
            mask = result[value_column].between(lower, upper, inclusive="both")
            result.loc[mask, output] = label
        if (result[output] == "unassigned").any():
            raise ValueError(f"Unassigned rows in {output}")
    return result


def clustered_bootstrap(
    frame: pd.DataFrame,
    value_columns: Iterable[str],
    cluster_column: str,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    value_columns = list(value_columns)
    grouped = frame.groupby(cluster_column, sort=True, observed=True)
    sums = grouped[value_columns].sum().to_numpy(dtype=float)
    counts = grouped.size().to_numpy(dtype=float)
    if len(counts) == 0:
        return {column: (np.nan, np.nan, np.nan) for column in value_columns}
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, len(value_columns)), dtype=float)
    batch = 100
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        sampled = rng.integers(0, len(counts), size=(stop - start, len(counts)))
        numerator = sums[sampled].sum(axis=1)
        denominator = counts[sampled].sum(axis=1)[:, None]
        estimates[start:stop] = numerator / denominator
    point = frame[value_columns].mean().to_numpy(dtype=float)
    lower, upper = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        column: (float(point[i]), float(lower[i]), float(upper[i]))
        for i, column in enumerate(value_columns)
    }


def view_metrics(frame: pd.DataFrame, dataset: str, task: str, mapping: dict[str, str]) -> pd.DataFrame:
    rows = []
    for view, rank_column in mapping.items():
        ranks = frame[rank_column].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "view": view,
                "n_queries": len(frame),
                "n_molecules": frame["canonical_smiles"].nunique(),
                "top1_hits": int((ranks == 1).sum()),
                "top1_percent": float((ranks == 1).mean() * 100),
                "top5_percent": float((ranks <= 5).mean() * 100),
                "top10_percent": float((ranks <= 10).mean() * 100),
                "mrr": float(np.mean(1.0 / ranks)),
                "median_rank": float(np.median(ranks)),
            }
        )
    return pd.DataFrame(rows)


def outcome_record(
    frame: pd.DataFrame,
    dataset: str,
    task: str,
    model: str,
    comparison: str,
    global_column: str,
    fusion_column: str,
    replicates: int,
    seed: int,
) -> dict:
    work = frame[["canonical_smiles", global_column, fusion_column]].copy()
    work["rescued"] = ((work[global_column] > 1) & (work[fusion_column] == 1)).astype(float)
    work["preserved"] = ((work[global_column] == 1) & (work[fusion_column] == 1)).astype(float)
    work["harmed"] = ((work[global_column] == 1) & (work[fusion_column] > 1)).astype(float)
    work["unresolved"] = ((work[global_column] > 1) & (work[fusion_column] > 1)).astype(float)
    work["net_change"] = (work[fusion_column] == 1).astype(float) - (
        work[global_column] == 1
    ).astype(float)
    work["rr_change"] = 1.0 / work[fusion_column] - 1.0 / work[global_column]
    boot = clustered_bootstrap(
        work,
        ["rescued", "preserved", "harmed", "unresolved", "net_change", "rr_change"],
        "canonical_smiles",
        replicates,
        seed,
    )
    counts = {name: int(work[name].sum()) for name in ["rescued", "preserved", "harmed", "unresolved"]}
    discordant = counts["rescued"] + counts["harmed"]
    p_value = float(
        binomtest(counts["rescued"], discordant, 0.5, alternative="two-sided").pvalue
    ) if discordant else 1.0
    record = {
        "dataset": dataset,
        "task": task,
        "model": model,
        "comparison": comparison,
        "n_queries": len(work),
        "n_molecules": work["canonical_smiles"].nunique(),
        **counts,
        "global_top1_hits": int((work[global_column] == 1).sum()),
        "fusion_top1_hits": int((work[fusion_column] == 1).sum()),
        "global_top1_percent": float((work[global_column] == 1).mean() * 100),
        "fusion_top1_percent": float((work[fusion_column] == 1).mean() * 100),
        "rescue_percent": boot["rescued"][0] * 100,
        "rescue_ci_low": boot["rescued"][1] * 100,
        "rescue_ci_high": boot["rescued"][2] * 100,
        "harm_percent": boot["harmed"][0] * 100,
        "harm_ci_low": boot["harmed"][1] * 100,
        "harm_ci_high": boot["harmed"][2] * 100,
        "net_top1_gain_pp": boot["net_change"][0] * 100,
        "net_gain_ci_low": boot["net_change"][1] * 100,
        "net_gain_ci_high": boot["net_change"][2] * 100,
        "mrr_change": boot["rr_change"][0],
        "mrr_change_ci_low": boot["rr_change"][1],
        "mrr_change_ci_high": boot["rr_change"][2],
        "mcnemar_exact_p": p_value,
    }
    return record


def build_nplib1(config: dict, input_paths: set[Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    package = PROJECT_ROOT / config["nplib1"]["promoted_package"]
    view_paths = config["nplib1"]["views"]
    merged_splits = []
    for split in config["nplib1"]["splits"]:
        global_path = package / view_paths["global"].format(split=split)
        frame = read_rank_view(global_path, "global", input_paths, keep_metadata=True)
        for key, prefix in [
            ("peak", "peak"),
            ("raw_projection", "rawproj"),
            ("global_peak_fusion", "global_peak"),
            ("final_fusion", "final"),
            ("dreams_global", "dreams_global"),
            ("dreams_peak", "dreams_peak"),
            ("dreams_fusion", "dreams_final"),
        ]:
            path = package / view_paths[key].format(split=split)
            frame = merge_views(frame, read_rank_view(path, prefix, input_paths), str(path))
        frame["split"] = split
        merged_splits.append(frame)
    result = pd.concat(merged_splits, ignore_index=True)
    result["query_key"] = result["split"].astype(str).radd("split") + ":" + result["spec_id"]
    if len(result) != 3699 or result["query_key"].duplicated().any():
        raise ValueError(f"Expected 3,699 unique split-query rows, found {len(result)}")
    result = add_molecular_descriptors(result)
    result = apply_strata(result, config)
    result["global_to_peakfusion_outcome"] = outcome(result["global_rank"], result["global_peak_rank"])
    result["peakfusion_to_final_outcome"] = outcome(result["global_peak_rank"], result["final_rank"])
    result["ultrams_outcome"] = outcome(result["global_rank"], result["final_rank"])
    result["dreams_outcome"] = outcome(result["dreams_global_rank"], result["dreams_final_rank"])
    result["ultrams_rank_change"] = result["global_rank"] - result["final_rank"]
    result["dreams_rank_change"] = result["dreams_global_rank"] - result["dreams_final_rank"]

    weights = []
    for split in config["nplib1"]["splits"]:
        for model in ["u5_cls_projmean", "u5_cls_projmean_rawproj", "dreams_global_iw"]:
            relative = config["nplib1"]["weight_grid_pattern"].format(split=split, model=model)
            path = package / relative
            input_paths.add(path)
            grid = pd.read_csv(path)
            selected = grid.loc[grid["rank"] == 1]
            if len(selected) != 1:
                raise ValueError(f"Expected one rank-1 weight row in {path}")
            row = selected.iloc[0].to_dict()
            row.update({"split": split, "model": model, "validation_n": int(row["n_queries"])})
            weights.append(row)
    weights_frame = pd.DataFrame(weights)

    metrics = view_metrics(
        result,
        "NPLIB1",
        "ms2peaks_min50",
        {
            "UltraMS global": "global_rank",
            "UltraMS projected peak": "peak_rank",
            "UltraMS raw projection": "rawproj_rank",
            "UltraMS global + projected peak": "global_peak_rank",
            "UltraMS final fusion": "final_rank",
            "DreaMS global": "dreams_global_rank",
            "DreaMS peak": "dreams_peak_rank",
            "DreaMS final fusion": "dreams_final_rank",
        },
    )

    comparisons = [
        ("UltraMS", "global_to_global_plus_peak", "global_rank", "global_peak_rank"),
        ("UltraMS", "global_plus_peak_to_final", "global_peak_rank", "final_rank"),
        ("UltraMS", "global_to_final", "global_rank", "final_rank"),
        ("DreaMS", "global_to_final", "dreams_global_rank", "dreams_final_rank"),
        ("UltraMS", "projected_peak_to_final", "peak_rank", "final_rank"),
        ("UltraMS", "raw_projection_to_final", "rawproj_rank", "final_rank"),
    ]
    outcome_rows = [
        outcome_record(
            result,
            "NPLIB1",
            "ms2peaks_min50",
            model,
            comparison,
            left,
            right,
            config["bootstrap_replicates"],
            config["seed"] + index,
        )
        for index, (model, comparison, left, right) in enumerate(comparisons)
    ]
    outcome_frame = pd.DataFrame(outcome_rows)

    stratum_rows = []
    axes = [
        ("Candidate competition", "candidate_stratum", [item[2] for item in config["difficulty_strata"]["candidate_competition"]]),
        ("Retained peak evidence", "peak_stratum", [item[2] for item in config["difficulty_strata"]["retained_peak_evidence"]]),
        ("Molecular size", "heavy_atom_stratum", [item[2] for item in config["difficulty_strata"]["molecular_size"]]),
    ]
    model_columns = [
        ("UltraMS", "global_rank", "final_rank"),
        ("DreaMS", "dreams_global_rank", "dreams_final_rank"),
    ]
    row_index = 0
    for axis_name, column, levels in axes:
        for level in levels:
            subset = result.loc[result[column] == level].copy()
            for model, global_column, fusion_column in model_columns:
                record = outcome_record(
                    subset,
                    "NPLIB1",
                    "ms2peaks_min50",
                    model,
                    "global_to_final",
                    global_column,
                    fusion_column,
                    config["bootstrap_replicates"],
                    config["seed"] + 100 + row_index,
                )
                record.update({"stratum_axis": axis_name, "stratum": level})
                stratum_rows.append(record)
                row_index += 1
    stratum_frame = pd.DataFrame(stratum_rows)

    return result, {
        "weights": weights_frame,
        "metrics": metrics,
        "outcomes": outcome_frame,
        "strata": stratum_frame,
    }


def select_cases(frame: pd.DataFrame, config: dict, input_paths: set[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_config = config["case_selection"]
    eligible = frame.loc[
        (frame["candidate_count"] == case_config["eligible_candidate_count"])
        & (frame["n_peaks_model"] == case_config["eligible_peak_count"])
    ].copy()
    rules = {
        "rescued": ("ultrams_rank_change", case_config["rescued_metric"]),
        "preserved": ("peak_rank", case_config["preserved_metric"]),
        "harmed": ("final_rank", case_config["harmed_metric"]),
    }
    selected_rows = []
    for lane, (column, label) in rules.items():
        lane_frame = eligible.loc[eligible["ultrams_outcome"] == lane].copy()
        if lane_frame.empty:
            raise ValueError(f"No eligible case for {lane}")
        median = float(lane_frame[column].median())
        lane_frame["selection_distance"] = (lane_frame[column] - median).abs()
        lane_frame = lane_frame.sort_values(["selection_distance", "split", "spec_id"])
        row = lane_frame.iloc[0].copy()
        row["case_lane"] = lane
        row["eligible_lane_n"] = len(lane_frame)
        row["selection_metric"] = label
        row["selection_median"] = median
        row["selection_rule"] = case_config["rule"]
        selected_rows.append(row)
    cases = pd.DataFrame(selected_rows)

    spectra = []
    for _, case in cases.iterrows():
        path = PROJECT_ROOT / config["nplib1"]["dataset_pattern"].format(split=int(case["split"]))
        input_paths.add(path)
        source = pd.read_csv(
            path,
            usecols=["spec_id", "mzs", "intensities", "precursor_mz", "formula", "fold"],
        )
        match = source.loc[source["spec_id"] == case["spec_id"]]
        if len(match) != 1:
            raise ValueError(f"Expected one source spectrum for {case['spec_id']} in {path}")
        row = match.iloc[0]
        mzs = np.fromstring(str(row["mzs"]), sep=",")
        intensities = np.fromstring(str(row["intensities"]), sep=",")
        if len(mzs) != len(intensities) or len(mzs) < 150:
            raise ValueError(f"Unexpected peak arrays for {case['spec_id']}")
        strongest = np.argsort(intensities)[-150:]
        order = strongest[np.argsort(mzs[strongest])]
        rel = intensities[order] / intensities[order].max()
        for peak_order, source_index in enumerate(order, start=1):
            spectra.append(
                {
                    "case_lane": case["case_lane"],
                    "split": int(case["split"]),
                    "spec_id": case["spec_id"],
                    "peak_order_mz": peak_order,
                    "source_peak_index": int(source_index),
                    "mz": float(mzs[source_index]),
                    "raw_intensity": float(intensities[source_index]),
                    "relative_intensity": float(rel[peak_order - 1]),
                    "precursor_mz": float(row["precursor_mz"]),
                    "formula": row["formula"],
                    "fold": row["fold"],
                }
            )
    return cases, pd.DataFrame(spectra)


def build_massspecgym(config: dict, input_paths: set[Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames = []
    metric_frames = []
    outcome_rows = []
    view_paths = config["massspecgym"]["views"]
    for task_index, task in enumerate(["mass", "formula"]):
        task_config = config["massspecgym"][task]
        source_dir = PROJECT_ROOT / task_config["source_run"]
        fusion_dir = PROJECT_ROOT / task_config["fusion_run"]
        global_path = source_dir / view_paths["global"].format(task=task)
        frame = read_rank_view(global_path, "global", input_paths, keep_metadata=True)
        for key, prefix, directory in [
            ("peak", "peak", source_dir),
            ("fusion", "final", fusion_dir),
            ("dreams_global", "dreams_global", source_dir),
            ("dreams_peak", "dreams_peak", source_dir),
            ("dreams_fusion", "dreams_final", fusion_dir),
        ]:
            path = directory / view_paths[key].format(task=task)
            frame = merge_views(frame, read_rank_view(path, prefix, input_paths), str(path))
        if len(frame) != config["massspecgym"]["n_queries_per_task"]:
            raise ValueError(f"Expected 16,440 MassSpecGym {task} queries, found {len(frame)}")
        frame = add_molecular_descriptors(frame)
        frame["task"] = task
        frame["ultrams_outcome"] = outcome(frame["global_rank"], frame["final_rank"])
        frame["dreams_outcome"] = outcome(frame["dreams_global_rank"], frame["dreams_final_rank"])
        frames.append(frame)
        metric_frames.append(
            view_metrics(
                frame,
                "MassSpecGym",
                task,
                {
                    "UltraMS global": "global_rank",
                    "UltraMS projected peak": "peak_rank",
                    "UltraMS global + peak fusion": "final_rank",
                    "DreaMS global": "dreams_global_rank",
                    "DreaMS peak": "dreams_peak_rank",
                    "DreaMS global + peak fusion": "dreams_final_rank",
                },
            )
        )
        for model_index, (model, left, right) in enumerate(
            [
                ("UltraMS", "global_rank", "final_rank"),
                ("DreaMS", "dreams_global_rank", "dreams_final_rank"),
            ]
        ):
            outcome_rows.append(
                outcome_record(
                    frame,
                    "MassSpecGym",
                    task,
                    model,
                    "global_to_final",
                    left,
                    right,
                    config["bootstrap_replicates"],
                    config["seed"] + 500 + task_index * 10 + model_index,
                )
            )
    return pd.concat(frames, ignore_index=True), {
        "metrics": pd.concat(metric_frames, ignore_index=True),
        "outcomes": pd.DataFrame(outcome_rows),
    }


def paired_model_difference(frame: pd.DataFrame, config: dict) -> dict:
    work = frame[["canonical_smiles", "global_rank", "final_rank", "dreams_global_rank", "dreams_final_rank"]].copy()
    work["ultra_net"] = (work["final_rank"] == 1).astype(float) - (work["global_rank"] == 1).astype(float)
    work["dreams_net"] = (work["dreams_final_rank"] == 1).astype(float) - (work["dreams_global_rank"] == 1).astype(float)
    work["difference"] = work["ultra_net"] - work["dreams_net"]
    boot = clustered_bootstrap(
        work,
        ["difference"],
        "canonical_smiles",
        config["bootstrap_replicates"],
        config["seed"] + 900,
    )["difference"]
    return {
        "comparison": "UltraMS minus DreaMS fusion-attributable Top1 gain",
        "n_queries": len(work),
        "n_molecules": work["canonical_smiles"].nunique(),
        "difference_pp": boot[0] * 100,
        "ci_low_pp": boot[1] * 100,
        "ci_high_pp": boot[2] * 100,
    }


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text())
    input_paths: set[Path] = {CONFIG_PATH}

    nplib, nplib_tables = build_nplib1(config, input_paths)
    cases, case_spectra = select_cases(nplib, config, input_paths)
    massspecgym, massspecgym_tables = build_massspecgym(config, input_paths)

    outputs = {
        "nplib1_query_decomposition.csv": nplib,
        "nplib1_split_weights.csv": nplib_tables["weights"],
        "nplib1_view_metrics.csv": nplib_tables["metrics"],
        "nplib1_outcome_stats.csv": nplib_tables["outcomes"],
        "nplib1_stratum_stats.csv": nplib_tables["strata"],
        "nplib1_case_selection.csv": cases,
        "nplib1_case_spectra.csv": case_spectra,
        "massspecgym_query_decomposition.csv": massspecgym,
        "massspecgym_view_metrics.csv": massspecgym_tables["metrics"],
        "massspecgym_outcome_stats.csv": massspecgym_tables["outcomes"],
    }
    for filename, frame in outputs.items():
        frame.to_csv(SOURCE_DIR / filename, index=False)

    pd.DataFrame(columns=["dataset", "spec_id", "reason"]).to_csv(
        SOURCE_DIR / "exclusion_log.csv", index=False
    )

    nplib_primary = nplib_tables["outcomes"].loc[
        nplib_tables["outcomes"]["comparison"] == "global_to_final"
    ]
    stats = {
        "analysis_id": config["analysis_id"],
        "nplib1": {
            "n_queries": len(nplib),
            "n_unique_spectra": nplib["spec_id"].nunique(),
            "n_molecules": nplib["canonical_smiles"].nunique(),
            "models": nplib_primary.to_dict(orient="records"),
            "paired_model_difference": paired_model_difference(nplib, config),
            "hard_case_pool_n": int(
                ((nplib["candidate_count"] == 50) & (nplib["n_peaks_model"] == 150)).sum()
            ),
        },
        "massspecgym": {
            "n_queries_per_task": config["massspecgym"]["n_queries_per_task"],
            "models": massspecgym_tables["outcomes"].to_dict(orient="records"),
        },
        "definitions": config["outcome_definition"],
        "limitations": [
            "NPLIB1 block-micro counts comprise 3,699 held-out fold-query evaluations from 3,352 unique spectra; confidence intervals resample target-molecule clusters.",
            "The final NPLIB1 fusion uses fold-dependent sparse validation weights rather than a uniformly non-zero three-source mixture.",
            "MassSpecGym provides a two-view global-plus-peak replication only and is not evidence for raw-projection transfer.",
            "Deterministically selected cases illustrate population-defined outcome lanes and are not independent tests.",
        ],
    }
    (SOURCE_DIR / "statistics.json").write_text(json.dumps(stats, indent=2))

    provenance = {
        "analysis_id": config["analysis_id"],
        "project_root": str(PROJECT_ROOT),
        "builder": str(HERE),
        "config": str(CONFIG_PATH),
        "seed": config["seed"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "upstream_inputs": [
            {
                "path": str(path),
                "relative_path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(input_paths)
        ],
    }
    (SOURCE_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    manifest_rows = []
    for path in sorted(SOURCE_DIR.iterdir()):
        if path.is_file() and path.name != "source_manifest.csv":
            manifest_rows.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    pd.DataFrame(manifest_rows).to_csv(SOURCE_DIR / "source_manifest.csv", index=False)

    print(f"Packaged NPLIB1 queries: {len(nplib)}")
    print(f"Packaged MassSpecGym queries: {len(massspecgym)}")
    print(nplib_tables["metrics"].to_string(index=False))
    print(nplib_tables["outcomes"].to_string(index=False))
    print(f"Source data: {SOURCE_DIR}")


if __name__ == "__main__":
    main()
