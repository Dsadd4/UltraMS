#!/usr/bin/env python3
"""Numerical and artifact checks for S7."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve()
S7_DIR = HERE.parents[1]
SOURCE_DIR = S7_DIR / "source_data"
OUTPUT_DIR = S7_DIR / "output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, found {actual}")


def main() -> None:
    required_source = [
        "nplib1_query_decomposition.csv",
        "nplib1_split_weights.csv",
        "nplib1_view_metrics.csv",
        "nplib1_outcome_stats.csv",
        "nplib1_stratum_stats.csv",
        "nplib1_case_selection.csv",
        "nplib1_case_spectra.csv",
        "massspecgym_query_decomposition.csv",
        "massspecgym_view_metrics.csv",
        "massspecgym_outcome_stats.csv",
        "statistics.json",
        "provenance.json",
        "source_manifest.csv",
        "exclusion_log.csv",
    ]
    required_output = [
        "s7_source_fusion_decision.png",
        "s7_source_fusion_decision.pdf",
        "s7_source_fusion_decision.svg",
        "s7_caption_draft.txt",
    ]
    for filename in required_source:
        if not (SOURCE_DIR / filename).is_file():
            raise FileNotFoundError(SOURCE_DIR / filename)
    for filename in required_output:
        if not (OUTPUT_DIR / filename).is_file():
            raise FileNotFoundError(OUTPUT_DIR / filename)

    manifest = pd.read_csv(SOURCE_DIR / "source_manifest.csv")
    for _, row in manifest.iterrows():
        path = SOURCE_DIR / row["file"]
        if path.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"Byte-size mismatch: {path}")
        if sha256(path) != row["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {path}")

    nplib = pd.read_csv(SOURCE_DIR / "nplib1_query_decomposition.csv")
    if len(nplib) != 3699 or nplib["query_key"].nunique() != 3699:
        raise AssertionError("NPLIB1 denominator is not 3,699 unique split-query evaluations")
    if nplib["spec_id"].nunique() != 3352:
        raise AssertionError("NPLIB1 unique-spectrum denominator is not 3,352")
    if nplib["canonical_smiles"].nunique() != 1844:
        raise AssertionError("NPLIB1 target-molecule denominator is not 1,844")
    if nplib[["global_rank", "peak_rank", "rawproj_rank", "global_peak_rank", "final_rank"]].isna().any().any():
        raise AssertionError("Missing NPLIB1 rank values")

    metrics = pd.read_csv(SOURCE_DIR / "nplib1_view_metrics.csv").set_index("view")
    expected_hits = {
        "UltraMS global": 1100,
        "UltraMS projected peak": 1077,
        "UltraMS raw projection": 1084,
        "UltraMS global + projected peak": 1206,
        "UltraMS final fusion": 1220,
        "DreaMS global": 966,
        "DreaMS peak": 934,
        "DreaMS final fusion": 1033,
    }
    for view, hits in expected_hits.items():
        if int(metrics.loc[view, "top1_hits"]) != hits:
            raise AssertionError(f"Unexpected top-1 hits for {view}")

    outcomes = pd.read_csv(SOURCE_DIR / "nplib1_outcome_stats.csv")
    primary = outcomes.loc[outcomes["comparison"] == "global_to_final"].set_index("model")
    expected_outcomes = {
        "UltraMS": {"rescued": 187, "preserved": 1033, "harmed": 67, "unresolved": 2412},
        "DreaMS": {"rescued": 160, "preserved": 873, "harmed": 93, "unresolved": 2573},
    }
    for model, counts in expected_outcomes.items():
        for outcome_name, count in counts.items():
            if int(primary.loc[model, outcome_name]) != count:
                raise AssertionError(f"Unexpected {model} {outcome_name} count")
        if sum(counts.values()) != 3699:
            raise AssertionError(f"Outcome partition does not sum to 3,699 for {model}")

    cases = pd.read_csv(SOURCE_DIR / "nplib1_case_selection.csv")
    spectra = pd.read_csv(SOURCE_DIR / "nplib1_case_spectra.csv")
    if set(cases["case_lane"]) != {"rescued", "preserved", "harmed"}:
        raise AssertionError("Case lanes are incomplete")
    if len(spectra) != 450 or not (spectra.groupby("case_lane").size() == 150).all():
        raise AssertionError("Each case must contain exactly 150 packaged peaks")
    if not (spectra.groupby("case_lane")["relative_intensity"].max().round(12) == 1.0).all():
        raise AssertionError("Case spectra are not max normalized")

    mass = pd.read_csv(SOURCE_DIR / "massspecgym_query_decomposition.csv")
    if len(mass) != 32880 or not (mass.groupby("task").size() == 16440).all():
        raise AssertionError("MassSpecGym denominators are not 16,440 per task")
    mass_metrics = pd.read_csv(SOURCE_DIR / "massspecgym_view_metrics.csv")
    promoted = {
        ("mass", "UltraMS global + peak fusion"): 17.092457420924575,
        ("mass", "DreaMS global + peak fusion"): 13.242092457420926,
        ("formula", "UltraMS global + peak fusion"): 14.07542579075426,
        ("formula", "DreaMS global + peak fusion"): 11.411192214111923,
    }
    for (task, view), expected in promoted.items():
        match = mass_metrics.loc[(mass_metrics["task"] == task) & (mass_metrics["view"] == view), "top1_percent"]
        if len(match) != 1:
            raise AssertionError(f"Missing promoted MassSpecGym row: {task}, {view}")
        assert_close(float(match.iloc[0]), expected, tolerance=1e-10)

    png_path = OUTPUT_DIR / "s7_source_fusion_decision.png"
    image = Image.open(png_path).convert("RGB")
    if image.width < 3500 or image.height < 4500:
        raise AssertionError(f"PNG is too small for publication preview: {image.size}")
    pixels = pd.Series(list(image.resize((400, 600)).getdata()))
    nonwhite = (pixels.apply(lambda rgb: min(rgb) < 245)).mean()
    if not 0.02 < nonwhite < 0.60:
        raise AssertionError(f"Unexpected non-white pixel fraction: {nonwhite:.4f}")

    svg_path = OUTPUT_DIR / "s7_source_fusion_decision.svg"
    ET.parse(svg_path)
    if svg_path.stat().st_size < 100_000:
        raise AssertionError("SVG output is unexpectedly small")
    pdf_path = OUTPUT_DIR / "s7_source_fusion_decision.pdf"
    if pdf_path.read_bytes()[:4] != b"%PDF" or pdf_path.stat().st_size < 50_000:
        raise AssertionError("PDF output is invalid or unexpectedly small")

    output_manifest = []
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            output_manifest.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    pd.DataFrame(output_manifest).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)

    report = {
        "status": "machine_checks_passed",
        "nplib1_queries": 3699,
        "nplib1_unique_spectra": 3352,
        "nplib1_molecules": 1844,
        "massspecgym_queries_per_task": 16440,
        "png_pixels": [image.width, image.height],
        "canvas_mm": [179.07, 196.09],
        "width_to_height_ratio": image.width / image.height,
        "nonwhite_fraction_low_resolution": nonwhite,
        "source_manifest_rows_verified": len(manifest),
        "manual_visual_qc": "recorded separately after rendered preview inspection",
    }
    (OUTPUT_DIR / "machine_qc.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
