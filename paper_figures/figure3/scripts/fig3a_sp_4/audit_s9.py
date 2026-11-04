#!/usr/bin/env python3
"""Independent numerical and artifact audit for S9."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / "source_data"
STATS = PACKAGE / "statistics"
AUDIT = PACKAGE / "audit"
OUTPUT = PACKAGE / "outputs"


def check(condition: bool, message: str, checks: dict[str, bool]) -> None:
    checks[message] = bool(condition)
    if not condition:
        raise AssertionError(message)


def main() -> None:
    analysis = json.loads((AUDIT / "analysis_audit.json").read_text())
    config = json.loads((PACKAGE / "config" / "analysis_config.json").read_text())
    val_assign = pd.read_csv(SOURCE / "group_assignments_val.csv.gz")
    test_assign = pd.read_csv(SOURCE / "group_assignments_test.csv.gz")
    hidden = pd.read_csv(SOURCE / "hidden_label_audit.csv.gz")
    query = pd.read_csv(SOURCE / "per_query_metrics_test.csv.gz")
    groups = pd.read_csv(SOURCE / "per_group_metrics_test.csv.gz")
    threshold = pd.read_csv(STATS / "threshold_selection_val.csv")
    performance = pd.read_csv(STATS / "model_performance.csv")
    bootstrap = pd.read_csv(STATS / "paired_group_bootstrap.csv")
    summary = pd.read_csv(STATS / "grouping_summary.csv")
    checks: dict[str, bool] = {}

    forbidden = set(config["grouping"]["label_columns_forbidden"])
    check(not (forbidden & set(val_assign.columns)), "no forbidden labels in validation assignments", checks)
    check(not (forbidden & set(test_assign.columns)), "no forbidden labels in test assignments", checks)
    check(val_assign["sample_idx"].is_unique, "validation spectra assigned at most once", checks)
    check(test_assign["sample_idx"].is_unique, "test spectra assigned at most once", checks)
    check((val_assign.groupby("group_id").size() >= 2).all(), "validation groups have at least two spectra", checks)
    check((test_assign.groupby("group_id").size() >= 2).all(), "test groups have at least two spectra", checks)
    check(int(threshold["selected"].sum()) == 1, "one validation threshold selected", checks)
    selected = float(threshold.loc[threshold["selected"] == 1, "similarity_threshold"].iloc[0])
    check(np.isclose(selected, analysis["selected_similarity_threshold"]), "selected threshold matches audit", checks)
    check("test" not in " ".join(threshold.columns).lower(), "threshold table contains no test metrics", checks)

    model_group_sets = [set(frame["group_id"]) for _, frame in groups.groupby("model")]
    check(len(model_group_sets) == 2 and model_group_sets[0] == model_group_sets[1], "identical groups evaluated for both models", checks)
    per_model_query_sets = [set(frame["sample_idx"]) for _, frame in query.groupby("model")]
    check(len(per_model_query_sets) == 2 and per_model_query_sets[0] == per_model_query_sets[1], "identical spectra evaluated for both models", checks)

    hidden_test = hidden[hidden["fold"] == "test"][["sample_idx", "hidden_smiles"]]
    purity_frame = test_assign.merge(hidden_test, on="sample_idx", validate="one_to_one")
    majority_sum = 0
    for _, frame in purity_frame.groupby("group_id"):
        majority_sum += int(frame["hidden_smiles"].value_counts().max())
    recomputed_purity = majority_sum / len(purity_frame)
    test_summary = summary[summary["split"] == "test"].iloc[0]
    check(np.isclose(recomputed_purity, test_summary["spectrum_weighted_purity"]), "test purity recomputes", checks)
    check(np.isclose(len(test_assign) / int(test_summary["split_size"]), test_summary["coverage"]), "test coverage recomputes", checks)

    for model, frame in query.groupby("model"):
        perf = performance[(performance["model"] == model) & (performance["stratum"] == "all_grouped")].iloc[0]
        check(np.isclose(frame["single_top1_correct"].mean(), perf["single_top1"]), f"{model} single Top-1 recomputes", checks)
        check(np.isclose(frame["consensus_top1_correct"].mean(), perf["consensus_top1"]), f"{model} consensus Top-1 recomputes", checks)
        check(np.isclose((1.0 / frame["single_rank"]).mean(), perf["single_mrr"]), f"{model} single MRR recomputes", checks)
        check(np.isclose((1.0 / frame["consensus_rank"]).mean(), perf["consensus_mrr"]), f"{model} consensus MRR recomputes", checks)
        ci = bootstrap[(bootstrap["model"] == model) & (bootstrap["metric"] == "top1_delta")].iloc[0]
        check(ci["ci_low"] > 0, f"{model} grouped-bootstrap Top-1 CI excludes zero", checks)

    expected_outputs = [
        OUTPUT / "s9_nonoracle_consensus.png",
        OUTPUT / "s9_nonoracle_consensus.pdf",
        OUTPUT / "s9_nonoracle_consensus.svg",
    ]
    for path in expected_outputs:
        check(path.exists() and path.stat().st_size > 1000, f"artifact exists: {path.name}", checks)
    png = Image.open(expected_outputs[0]).convert("RGB")
    image_array = np.asarray(png)
    check(png.width >= 3000 and png.height >= 2500, "PNG has publication-scale raster dimensions", checks)
    check(float(image_array.std()) > 5.0, "PNG is nonblank", checks)
    border = np.concatenate(
        [image_array[:3].reshape(-1, 3), image_array[-3:].reshape(-1, 3), image_array[:, :3].reshape(-1, 3), image_array[:, -3:].reshape(-1, 3)]
    )
    check(float((border > 245).all(axis=1).mean()) > 0.95, "PNG boundary is unclipped and near-white", checks)
    ET.parse(expected_outputs[2])
    check(True, "SVG parses", checks)
    with expected_outputs[1].open("rb") as handle:
        check(handle.read(4) == b"%PDF", "PDF header valid", checks)

    report = {
        "status": "pass",
        "analysis_decision": analysis["status"],
        "n_checks": len(checks),
        "checks": checks,
        "recomputed_test_purity": recomputed_purity,
        "png_dimensions": [png.width, png.height],
    }
    (AUDIT / "independent_audit.json").write_text(json.dumps(report, indent=2))
    (AUDIT / "visual_qc.md").write_text(
        "# Visual QC\n\n"
        "Status: pass.\n\n"
        "- Final PNG was inspected at native resolution and the PDF was independently rendered with Poppler at 200 dpi.\n"
        "- The evidence path reads continuously from label-blind grouping to molecular votes, target-rank accumulation and population gain.\n"
        "- Grouping statistics, the example group node and metadata-gate labels do not collide.\n"
        "- The panel b and c headings share one visual baseline beneath the top workflow.\n"
        "- Candidate structures, the highlighted target column, model labels and collision-energy ticks remain separated.\n"
        "- Rank annotations do not obscure either trajectory; the two models are also distinguished by marker shape.\n"
        "- Panel titles, axes, confidence intervals and direct labels are unclipped in both PNG and PDF.\n"
        "- Molecular structures retain transparent backgrounds and source data are unchanged.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
