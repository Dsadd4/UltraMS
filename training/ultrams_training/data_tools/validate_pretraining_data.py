#!/usr/bin/env python3
"""Full, streaming readiness audit for UltraMS pretraining Parquet data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from ._audit_utils import (
        MANIFEST_VERSION,
        add_stats,
        atomic_write_json,
        empty_stats,
        fingerprint,
        schema_fingerprint,
        sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from _audit_utils import (  # type: ignore
        MANIFEST_VERSION,
        add_stats,
        atomic_write_json,
        empty_stats,
        fingerprint,
        schema_fingerprint,
        sha256_file,
    )


SHA256_RE = re.compile(r"[0-9a-f]{64}")
MZ_COLUMNS = ("spectrum_mz", "mz")
INTENSITY_COLUMNS = ("spectrum_intensity", "intensity")
REQUIRED_SCALARS = ("precursor_mz", "polarity", "RT")
DEFAULT_EXPECTED_LIST_LENGTH = 500
DEFAULT_MIN_POSITIVE_MZ = 3
DEFAULT_MAX_PRECURSOR_MZ = 1000.0
DEFAULT_MAX_FRAGMENT_MZ = 1500.0
DEFAULT_MAX_INTENSITY: float | None = None


def _error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _list_value_type(data_type: pa.DataType) -> pa.DataType | None:
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return data_type.value_type
    if pa.types.is_fixed_size_list(data_type):
        return data_type.value_type
    return None


def _resolve_peak_columns(
    schema: pa.Schema, report: dict[str, Any]
) -> tuple[str, str] | None:
    mz_names = [name for name in MZ_COLUMNS if name in schema.names]
    intensity_names = [name for name in INTENSITY_COLUMNS if name in schema.names]
    if len(mz_names) != 1:
        _error(report, f"schema must contain exactly one m/z column from {MZ_COLUMNS}")
    if len(intensity_names) != 1:
        _error(
            report,
            f"schema must contain exactly one intensity column from {INTENSITY_COLUMNS}",
        )
    if len(mz_names) != 1 or len(intensity_names) != 1:
        return None
    return mz_names[0], intensity_names[0]


def _validate_schema(
    schema: pa.Schema, role: str, report: dict[str, Any]
) -> tuple[str, str] | None:
    initial_errors = len(report["errors"])
    for name in REQUIRED_SCALARS:
        if name not in schema.names:
            _error(report, f"schema is missing required column {name!r}")
    if "precursor_mz" in schema.names and not pa.types.is_floating(
        schema.field("precursor_mz").type
    ):
        _error(report, "precursor_mz must have a floating-point Arrow type")
    if "polarity" in schema.names and not pa.types.is_integer(
        schema.field("polarity").type
    ):
        _error(report, "polarity must have an integer Arrow type")
    if "RT" in schema.names and not pa.types.is_floating(schema.field("RT").type):
        _error(report, "RT must have a floating-point Arrow type")

    peak_columns = _resolve_peak_columns(schema, report)
    if peak_columns is not None:
        for name in peak_columns:
            value_type = _list_value_type(schema.field(name).type)
            if value_type is None or not pa.types.is_floating(value_type):
                _error(report, f"{name} must be a list of floating-point values")
    if role == "polarity" and "polarity" not in schema.names:
        _error(report, "polarity derivative requires a polarity column")
    return peak_columns if len(report["errors"]) == initial_errors else None


def _as_flat_list_values(array: pa.Array) -> tuple[np.ndarray, np.ndarray, int]:
    if array.null_count:
        raise ValueError("list column contains null rows")
    lengths = (
        pc.list_value_length(array)
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )
    values = pc.list_flatten(array)
    if values.null_count:
        raise ValueError("list column contains null elements")
    return values.to_numpy(zero_copy_only=False), lengths, len(array)


def _positive_peak_counts(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    if not len(lengths):
        return np.zeros(0, dtype=np.int64)
    if np.all(lengths == lengths[0]):
        width = int(lengths[0])
        if width == 0:
            return np.zeros(len(lengths), dtype=np.int64)
        return (values.reshape(len(lengths), width) > 0).sum(axis=1)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    result = np.zeros(len(lengths), dtype=np.int64)
    nonempty = lengths > 0
    starts = offsets[:-1][nonempty]
    result[nonempty] = np.add.reduceat((values > 0).astype(np.int64), starts)
    return result


def _empty_content_validation(
    expected_list_length: int,
    min_positive_mz: int,
    enforce_min_positive_mz: bool,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    max_fragment_mz: float = DEFAULT_MAX_FRAGMENT_MZ,
    max_intensity: float | None = DEFAULT_MAX_INTENSITY,
) -> dict[str, Any]:
    return {
        "contract": {
            "expected_list_length": expected_list_length,
            "min_positive_mz": min_positive_mz,
            "min_positive_mz_is_hard_error": enforce_min_positive_mz,
            "max_precursor_mz": max_precursor_mz,
            "max_fragment_mz": max_fragment_mz,
            "stored_intensity": "finite_nonnegative",
            "max_intensity": max_intensity,
            "max_intensity_is_hard_error": max_intensity is not None,
            "positive_mz_sorted_nondecreasing": True,
            "zero_padding_tail_only": True,
            "mz_intensity_padding_synchronized": True,
        },
        "rows": 0,
        "list_lengths": {
            "mz_expected": 0,
            "mz_unexpected": 0,
            "mz_unscannable": 0,
            "intensity_expected": 0,
            "intensity_unexpected": 0,
            "intensity_unscannable": 0,
            "paired_equal": 0,
            "paired_unequal": 0,
            "paired_unscannable": 0,
        },
        "peak_rows": {
            "at_least_minimum": 0,
            "below_minimum": 0,
            "unscannable": 0,
        },
        "structural_rows": {
            "precursor_above_maximum": 0,
            "fragment_above_maximum": 0,
            "intensity_above_maximum": 0,
            "padding_mismatch": 0,
            "interleaved_padding": 0,
            "unsorted_positive_mz": 0,
            "fully_scanned": 0,
            "unscannable": 0,
        },
        "values": {
            "mz_total": 0,
            "mz_finite": 0,
            "mz_nonfinite": 0,
            "mz_nonnegative_finite": 0,
            "mz_negative_finite": 0,
            "intensity_total": 0,
            "intensity_finite": 0,
            "intensity_nonfinite": 0,
            "intensity_nonnegative_finite": 0,
            "intensity_negative_finite": 0,
            "positive_mz": 0,
            "positive_mz_paired": 0,
            "positive_mz_unpaired": 0,
            "positive_mz_with_valid_intensity": 0,
            "positive_mz_with_invalid_intensity": 0,
        },
    }


def _positive_mz_intensity_pair_counts(
    mz_values: np.ndarray,
    mz_lengths: np.ndarray,
    intensity_values: np.ndarray,
    intensity_lengths: np.ndarray,
) -> tuple[int, int, int]:
    """Return paired-valid, paired-invalid and unpaired positive-m/z counts."""
    if np.array_equal(mz_lengths, intensity_lengths):
        positive = np.isfinite(mz_values) & (mz_values > 0)
        intensity_valid = np.isfinite(intensity_values) & (intensity_values > 0)
        valid = int(np.count_nonzero(positive & intensity_valid))
        invalid = int(np.count_nonzero(positive & ~intensity_valid))
        return valid, invalid, 0

    mz_offsets = np.concatenate(([0], np.cumsum(mz_lengths, dtype=np.int64)))
    intensity_offsets = np.concatenate(
        ([0], np.cumsum(intensity_lengths, dtype=np.int64))
    )
    valid = 0
    invalid = 0
    unpaired = 0
    for row_index, (mz_length, intensity_length) in enumerate(
        zip(mz_lengths, intensity_lengths, strict=True)
    ):
        mz_row = mz_values[mz_offsets[row_index] : mz_offsets[row_index + 1]]
        positive = np.isfinite(mz_row) & (mz_row > 0)
        positive_count = int(np.count_nonzero(positive))
        if mz_length != intensity_length:
            unpaired += positive_count
            continue
        intensity_row = intensity_values[
            intensity_offsets[row_index] : intensity_offsets[row_index + 1]
        ]
        intensity_valid = np.isfinite(intensity_row) & (intensity_row > 0)
        valid += int(np.count_nonzero(positive & intensity_valid))
        invalid += int(np.count_nonzero(positive & ~intensity_valid))
    return valid, invalid, unpaired


def _validate_content_closure(report: dict[str, Any]) -> None:
    validation = report["content_validation"]
    rows = validation["rows"]
    lengths = validation["list_lengths"]
    peaks = validation["peak_rows"]
    values = validation["values"]

    closures = {
        "m/z list-row": lengths["mz_expected"]
        + lengths["mz_unexpected"]
        + lengths["mz_unscannable"],
        "intensity list-row": lengths["intensity_expected"]
        + lengths["intensity_unexpected"]
        + lengths["intensity_unscannable"],
        "paired list-row": lengths["paired_equal"]
        + lengths["paired_unequal"]
        + lengths["paired_unscannable"],
        "positive-peak row": peaks["at_least_minimum"]
        + peaks["below_minimum"]
        + peaks["unscannable"],
    }
    for name, closed_rows in closures.items():
        if closed_rows != rows:
            _error(report, f"content summary does not close for {name} counts")

    value_closures = {
        "m/z finite": values["mz_finite"] + values["mz_nonfinite"],
        "m/z sign": values["mz_nonnegative_finite"] + values["mz_negative_finite"],
        "intensity finite": values["intensity_finite"] + values["intensity_nonfinite"],
        "intensity sign": values["intensity_nonnegative_finite"]
        + values["intensity_negative_finite"],
        "positive m/z pairing": values["positive_mz_paired"]
        + values["positive_mz_unpaired"],
        "paired positive m/z intensity": values["positive_mz_with_valid_intensity"]
        + values["positive_mz_with_invalid_intensity"],
    }
    expected = {
        "m/z finite": values["mz_total"],
        "m/z sign": values["mz_finite"],
        "intensity finite": values["intensity_total"],
        "intensity sign": values["intensity_finite"],
        "positive m/z pairing": values["positive_mz"],
        "paired positive m/z intensity": values["positive_mz_paired"],
    }
    for name, closed_values in value_closures.items():
        if closed_values != expected[name]:
            _error(report, f"content summary does not close for {name} counts")

    model_input = report["model_input"]
    if (
        model_input["eligible_rows"]
        + model_input["skipped_rows"]
        + model_input["unscannable_rows"]
        != rows
    ):
        _error(report, "model-input row summary does not close")


def _scan_batch(
    batch: pa.RecordBatch,
    peak_columns: tuple[str, str],
    role: str,
    shard_name: str,
    batch_index: int,
    report: dict[str, Any],
    expected_list_length: int,
    min_positive_mz: int,
    enforce_min_positive_mz: bool,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    max_fragment_mz: float = DEFAULT_MAX_FRAGMENT_MZ,
    max_intensity: float | None = DEFAULT_MAX_INTENSITY,
) -> dict[str, Any]:
    stats = empty_stats()
    stats["rows"] = batch.num_rows
    prefix = f"{shard_name} batch {batch_index}"
    validation = report["content_validation"]
    validation["rows"] += batch.num_rows

    polarity = batch.column(batch.schema.get_field_index("polarity"))
    polarity_values = polarity.to_numpy(zero_copy_only=False)
    negative = int(np.count_nonzero(polarity_values == 0))
    positive = int(np.count_nonzero(polarity_values == 1))
    other = batch.num_rows - negative - positive
    stats["polarity"] = {
        "negative_0": negative,
        "positive_1": positive,
        "other_or_null": other,
    }
    if role == "polarity" and other:
        _error(report, f"{prefix}: {other} non-binary or null polarity values")

    rt = batch.column(batch.schema.get_field_index("RT"))
    if rt.null_count:
        _error(report, f"{prefix}: {rt.null_count} null RT values")
    rt_values = rt.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    finite_rt = np.isfinite(rt_values)
    if not finite_rt.all():
        _error(report, f"{prefix}: {int((~finite_rt).sum())} non-finite RT values")
    positive_rt = finite_rt & (rt_values > 0)
    accepted_rt = positive_rt & (rt_values <= 1500)
    above_rt = finite_rt & (rt_values > 1500)
    stats["rt"] = {
        "positive": int(positive_rt.sum()),
        "accepted_0_to_1500": int(accepted_rt.sum()),
        "above_1500": int(above_rt.sum()),
        "nonpositive_nan_or_null": int(batch.num_rows - positive_rt.sum()),
    }

    precursor = batch.column(batch.schema.get_field_index("precursor_mz"))
    if precursor.null_count:
        _error(report, f"{prefix}: {precursor.null_count} null precursor_mz values")
    precursor_values = precursor.to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    valid_precursor = np.isfinite(precursor_values) & (precursor_values > 0)
    if not valid_precursor.all():
        _error(
            report,
            f"{prefix}: {int((~valid_precursor).sum())} non-finite or non-positive precursor_mz values",
        )
    precursor_above_maximum = int(
        np.count_nonzero(valid_precursor & (precursor_values > max_precursor_mz))
    )
    validation["structural_rows"]["precursor_above_maximum"] += precursor_above_maximum
    if precursor_above_maximum:
        _error(
            report,
            f"{prefix}: {precursor_above_maximum} precursor_mz values exceed "
            f"{max_precursor_mz:g}",
        )

    mz = batch.column(batch.schema.get_field_index(peak_columns[0]))
    intensity = batch.column(batch.schema.get_field_index(peak_columns[1]))
    mz_result: tuple[np.ndarray, np.ndarray, int] | None = None
    intensity_result: tuple[np.ndarray, np.ndarray, int] | None = None
    try:
        mz_result = _as_flat_list_values(mz)
    except ValueError as exc:
        _error(report, f"{prefix}: m/z {exc}")
        validation["list_lengths"]["mz_unscannable"] += batch.num_rows
        validation["peak_rows"]["unscannable"] += batch.num_rows
    try:
        intensity_result = _as_flat_list_values(intensity)
    except ValueError as exc:
        _error(report, f"{prefix}: intensity {exc}")
        validation["list_lengths"]["intensity_unscannable"] += batch.num_rows

    if mz_result is None or intensity_result is None:
        validation["structural_rows"]["unscannable"] += batch.num_rows
        validation["list_lengths"]["paired_unscannable"] += batch.num_rows
        if mz_result is None and intensity_result is not None:
            _, intensity_lengths, _ = intensity_result
            expected_count = int(
                np.count_nonzero(intensity_lengths == expected_list_length)
            )
            validation["list_lengths"]["intensity_expected"] += expected_count
            validation["list_lengths"]["intensity_unexpected"] += (
                batch.num_rows - expected_count
            )
        if intensity_result is None and mz_result is not None:
            mz_values, mz_lengths, _ = mz_result
            expected_count = int(np.count_nonzero(mz_lengths == expected_list_length))
            validation["list_lengths"]["mz_expected"] += expected_count
            validation["list_lengths"]["mz_unexpected"] += (
                batch.num_rows - expected_count
            )
            peak_counts = _positive_peak_counts(mz_values, mz_lengths)
            below = int(np.count_nonzero(peak_counts < min_positive_mz))
            validation["peak_rows"]["below_minimum"] += below
            validation["peak_rows"]["at_least_minimum"] += batch.num_rows - below
        model_input = report["model_input"]
        model_input["unscannable_rows"] += batch.num_rows
        return stats

    mz_values, mz_lengths, _ = mz_result
    intensity_values, intensity_lengths, _ = intensity_result
    mz_expected = int(np.count_nonzero(mz_lengths == expected_list_length))
    intensity_expected = int(
        np.count_nonzero(intensity_lengths == expected_list_length)
    )
    validation["list_lengths"]["mz_expected"] += mz_expected
    validation["list_lengths"]["mz_unexpected"] += batch.num_rows - mz_expected
    validation["list_lengths"]["intensity_expected"] += intensity_expected
    validation["list_lengths"]["intensity_unexpected"] += (
        batch.num_rows - intensity_expected
    )
    if mz_expected != batch.num_rows:
        _error(
            report,
            f"{prefix}: {batch.num_rows - mz_expected} m/z rows do not have "
            f"the expected list length {expected_list_length}",
        )
    if intensity_expected != batch.num_rows:
        _error(
            report,
            f"{prefix}: {batch.num_rows - intensity_expected} intensity rows do not "
            f"have the expected list length {expected_list_length}",
        )

    unequal = int(np.count_nonzero(mz_lengths != intensity_lengths))
    validation["list_lengths"]["paired_unequal"] += unequal
    validation["list_lengths"]["paired_equal"] += batch.num_rows - unequal
    if unequal:
        _error(
            report, f"{prefix}: {unequal} rows have unequal m/z and intensity lengths"
        )

    finite_mz = np.isfinite(mz_values)
    negative_mz = finite_mz & (mz_values < 0)
    validation["values"]["mz_total"] += len(mz_values)
    validation["values"]["mz_finite"] += int(finite_mz.sum())
    validation["values"]["mz_nonfinite"] += int((~finite_mz).sum())
    validation["values"]["mz_nonnegative_finite"] += int(
        np.count_nonzero(finite_mz & (mz_values >= 0))
    )
    validation["values"]["mz_negative_finite"] += int(negative_mz.sum())
    if not finite_mz.all() or negative_mz.any():
        count = int(np.count_nonzero(~finite_mz | negative_mz))
        _error(report, f"{prefix}: {count} invalid m/z values")
    fragment_above_maximum = int(
        np.count_nonzero(finite_mz & (mz_values > max_fragment_mz))
    )
    validation["structural_rows"]["fragment_above_maximum"] += fragment_above_maximum
    if fragment_above_maximum:
        _error(
            report,
            f"{prefix}: {fragment_above_maximum} fragment m/z values exceed "
            f"{max_fragment_mz:g}",
        )

    finite_intensity = np.isfinite(intensity_values)
    negative_intensity = finite_intensity & (intensity_values < 0)
    validation["values"]["intensity_total"] += len(intensity_values)
    validation["values"]["intensity_finite"] += int(finite_intensity.sum())
    validation["values"]["intensity_nonfinite"] += int((~finite_intensity).sum())
    validation["values"]["intensity_nonnegative_finite"] += int(
        np.count_nonzero(finite_intensity & (intensity_values >= 0))
    )
    validation["values"]["intensity_negative_finite"] += int(negative_intensity.sum())
    if not finite_intensity.all() or negative_intensity.any():
        count = int(np.count_nonzero(~finite_intensity | negative_intensity))
        _error(report, f"{prefix}: {count} invalid intensity values")
    if max_intensity is not None:
        intensity_above_maximum = int(
            np.count_nonzero(finite_intensity & (intensity_values > max_intensity))
        )
        validation["structural_rows"][
            "intensity_above_maximum"
        ] += intensity_above_maximum
        if intensity_above_maximum:
            _error(
                report,
                f"{prefix}: {intensity_above_maximum} intensity values exceed "
                f"{max_intensity:g}",
            )

    fully_shaped = bool(
        mz_expected == batch.num_rows
        and intensity_expected == batch.num_rows
        and unequal == 0
    )
    if fully_shaped:
        mz_matrix = mz_values.reshape(batch.num_rows, expected_list_length)
        intensity_matrix = intensity_values.reshape(
            batch.num_rows, expected_list_length
        )
        positive_mask = mz_matrix > 0
        intensity_present = intensity_matrix > 0
        padding_mismatch = np.any(positive_mask != intensity_present, axis=1)
        seen_padding = np.maximum.accumulate(~positive_mask, axis=1)
        interleaved_padding = np.any(seen_padding & positive_mask, axis=1)
        adjacent_positive = positive_mask[:, :-1] & positive_mask[:, 1:]
        unsorted = np.any(
            adjacent_positive & (mz_matrix[:, 1:] < mz_matrix[:, :-1]), axis=1
        )
        structural = validation["structural_rows"]
        structural["fully_scanned"] += batch.num_rows
        structural["padding_mismatch"] += int(np.count_nonzero(padding_mismatch))
        structural["interleaved_padding"] += int(np.count_nonzero(interleaved_padding))
        structural["unsorted_positive_mz"] += int(np.count_nonzero(unsorted))
        if padding_mismatch.any():
            _error(
                report,
                f"{prefix}: {int(np.count_nonzero(padding_mismatch))} rows have "
                "mismatched m/z and intensity padding",
            )
        if interleaved_padding.any():
            _error(
                report,
                f"{prefix}: {int(np.count_nonzero(interleaved_padding))} rows have "
                "non-tail m/z padding",
            )
        if unsorted.any():
            _error(
                report,
                f"{prefix}: {int(np.count_nonzero(unsorted))} rows have unsorted "
                "positive fragment m/z values",
            )
    else:
        validation["structural_rows"]["unscannable"] += batch.num_rows

    positive_mz = int(np.count_nonzero(finite_mz & (mz_values > 0)))
    valid_pairs, invalid_pairs, unpaired = _positive_mz_intensity_pair_counts(
        mz_values, mz_lengths, intensity_values, intensity_lengths
    )
    validation["values"]["positive_mz"] += positive_mz
    validation["values"]["positive_mz_paired"] += valid_pairs + invalid_pairs
    validation["values"]["positive_mz_unpaired"] += unpaired
    validation["values"]["positive_mz_with_valid_intensity"] += valid_pairs
    validation["values"]["positive_mz_with_invalid_intensity"] += invalid_pairs
    if invalid_pairs:
        _error(
            report,
            f"{prefix}: {invalid_pairs} positive m/z values do not have a "
            "finite positive corresponding intensity",
        )

    # Zero m/z values are padding. Every represented peak is therefore > 0.
    peak_counts = _positive_peak_counts(mz_values, mz_lengths)
    model_input = report["model_input"]
    count_0 = int(np.count_nonzero(peak_counts == 0))
    count_1 = int(np.count_nonzero(peak_counts == 1))
    count_2 = int(np.count_nonzero(peak_counts == 2))
    eligible = int(np.count_nonzero(peak_counts >= 3))
    below_minimum = int(np.count_nonzero(peak_counts < min_positive_mz))
    accepted = batch.num_rows - below_minimum
    model_input["peak_count_0"] += count_0
    model_input["peak_count_1"] += count_1
    model_input["peak_count_2"] += count_2
    model_input["peak_count_3_or_more"] += eligible
    model_input["eligible_rows"] += accepted
    model_input["skipped_rows"] += below_minimum
    validation["peak_rows"]["at_least_minimum"] += accepted
    validation["peak_rows"]["below_minimum"] += below_minimum
    if below_minimum and enforce_min_positive_mz:
        _error(
            report,
            f"{prefix}: {below_minimum} rows have fewer than "
            f"{min_positive_mz} positive-m/z peaks",
        )
    return stats


def _stable_file_records(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (record["name"], int(record["rows"]), record["sha256"]) for record in records
    ]


def _recompute_dataset_fingerprint(manifest: dict[str, Any]) -> str:
    payload: dict[str, Any] = {
        "run_identity": manifest["run_identity"],
        "files": _stable_file_records(manifest["output"]["file_records"]),
    }
    if "selection" in manifest:
        payload["selection"] = manifest["selection"]["fingerprint"]
    return fingerprint(payload)


def _source_inventory_fingerprint(
    schema_digest: str, records: list[dict[str, Any]]
) -> str:
    global_start = 0
    files = []
    for index, record in enumerate(records):
        rows = int(record["rows"])
        files.append(
            {
                "index": index,
                "name": record["name"],
                "bytes": int(record["bytes"]),
                "rows": rows,
                "global_start": global_start,
                "global_end_exclusive": global_start + rows,
                "sha256": record["sha256"],
                "schema_fingerprint": record["schema_fingerprint"],
            }
        )
        global_start += rows
    stable = {
        "manifest_version": MANIFEST_VERSION,
        "ordering": "lexical_basename",
        "schema_fingerprint": schema_digest,
        "files": files,
        "total_files": len(files),
        "total_rows": global_start,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    return fingerprint(stable)


def _validate_hash_manifest(
    dataset_dir: Path, records: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    path = dataset_dir / "files.sha256"
    if not path.is_file():
        _error(report, "missing files.sha256")
        return
    expected = [f"{record['sha256']}  {record['name']}" for record in records]
    actual = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if actual != expected:
        _error(report, "files.sha256 does not exactly match manifest file_records")


def audit_dataset(
    dataset_dir: Path,
    role: str,
    *,
    expected_dataset_name: str | None = None,
    expected_rows: int | None = None,
    expected_list_length: int = DEFAULT_EXPECTED_LIST_LENGTH,
    min_positive_mz: int = DEFAULT_MIN_POSITIVE_MZ,
    enforce_min_positive_mz: bool = True,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    max_fragment_mz: float = DEFAULT_MAX_FRAGMENT_MZ,
    max_intensity: float | None = DEFAULT_MAX_INTENSITY,
    expected_schema_fingerprint: str | None = None,
    batch_size: int = 65_536,
    progress_every: int = 25,
) -> dict[str, Any]:
    if expected_list_length <= 0:
        raise ValueError("expected_list_length must be positive")
    if min_positive_mz <= 0:
        raise ValueError("min_positive_mz must be positive")
    if expected_rows is not None and expected_rows <= 0:
        raise ValueError("expected_rows must be positive when provided")
    if min(max_precursor_mz, max_fragment_mz) <= 0:
        raise ValueError("m/z maxima must be positive")
    if max_intensity is not None and max_intensity <= 0:
        raise ValueError("max_intensity must be positive when provided")
    if expected_schema_fingerprint is not None and (
        SHA256_RE.fullmatch(expected_schema_fingerprint.lower()) is None
    ):
        raise ValueError("expected_schema_fingerprint must be a SHA-256 value")
    dataset_dir = dataset_dir.resolve()
    report: dict[str, Any] = {
        "role": role,
        "path": str(dataset_dir),
        "status": "failed",
        "errors": [],
        "files": [],
        "stats": empty_stats(),
        "content_validation": _empty_content_validation(
            expected_list_length,
            min_positive_mz,
            enforce_min_positive_mz,
            max_precursor_mz,
            max_fragment_mz,
            max_intensity,
        ),
        "model_input": {
            "loader_contract": (
                f"rows require at least {min_positive_mz} positive-m/z peaks"
            ),
            "eligible_rows": 0,
            "skipped_rows": 0,
            "unscannable_rows": 0,
            "peak_count_0": 0,
            "peak_count_1": 0,
            "peak_count_2": 0,
            "peak_count_3_or_more": 0,
        },
    }
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        _error(report, "missing manifest.json")
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _error(report, f"cannot read manifest.json: {exc}")
        return report
    report["manifest"] = {
        "dataset_name": manifest.get("dataset_name"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "input_content_fingerprint": manifest.get("input_content_fingerprint"),
        "schema_fingerprint": manifest.get("schema_fingerprint"),
    }
    for field in (
        "dataset_name",
        "run_identity",
        "input_content_fingerprint",
        "dataset_fingerprint",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            _error(report, f"manifest is missing {field}")
    for field in (
        "run_identity",
        "input_content_fingerprint",
        "dataset_fingerprint",
        "schema_fingerprint",
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            _error(report, f"manifest {field} is not a SHA-256 value")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        _error(
            report, f"unsupported manifest_version {manifest.get('manifest_version')!r}"
        )
    if manifest.get("status") != "complete":
        _error(report, "manifest status is not complete")
    dataset_name = manifest.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name:
        _error(report, "manifest dataset_name is missing")
    elif expected_dataset_name is not None and dataset_name != expected_dataset_name:
        _error(report, f"manifest dataset_name differs from expected {role} dataset")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        _error(report, "manifest config is missing")
    elif isinstance(manifest.get("input_content_fingerprint"), str):
        identity_config = dict(manifest_config)
        identity_config.pop("source_dir", None)
        identity_config.pop("output_dir", None)
        expected_run_identity = fingerprint(
            {
                "config": identity_config,
                "input": manifest["input_content_fingerprint"],
            }
        )
        if manifest.get("run_identity") != expected_run_identity:
            _error(report, "run_identity does not match config and input fingerprint")

    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("file_records"), list):
        _error(report, "manifest output.file_records is missing")
        return report
    records = sorted(output["file_records"], key=lambda item: str(item.get("name", "")))
    parquet_paths = sorted(dataset_dir.glob("*.parquet"), key=lambda path: path.name)
    if [record.get("name") for record in records] != [
        path.name for path in parquet_paths
    ]:
        _error(report, "manifest file set does not exactly match local Parquet files")
        return report
    if not parquet_paths:
        _error(report, "dataset contains no Parquet files")
        return report

    common_schema: pa.Schema | None = None
    peak_columns: tuple[str, str] | None = None
    actual_records: list[dict[str, Any]] = []
    for index, (path, declared) in enumerate(zip(parquet_paths, records, strict=True)):
        if progress_every and (index == 0 or (index + 1) % progress_every == 0):
            print(
                f"[{role}] auditing shard {index + 1}/{len(parquet_paths)}",
                file=sys.stderr,
            )
        try:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
        except Exception as exc:  # pragma: no cover - backend-specific exceptions
            _error(report, f"cannot open {path.name}: {exc}")
            continue
        schema_matches = common_schema is None
        if common_schema is None:
            common_schema = schema
            peak_columns = _validate_schema(schema, role, report)
            schema_matches = peak_columns is not None
        else:
            schema_matches = common_schema.equals(schema, check_metadata=True)
            if not schema_matches:
                _error(report, f"{path.name}: schema differs from the first shard")
        shard_stats = empty_stats()
        if (
            schema_matches
            and peak_columns is not None
            and all(name in schema.names for name in (*REQUIRED_SCALARS, *peak_columns))
        ):
            columns = [*REQUIRED_SCALARS, *peak_columns]
            try:
                for batch_index, batch in enumerate(
                    parquet.iter_batches(batch_size=batch_size, columns=columns)
                ):
                    add_stats(
                        shard_stats,
                        _scan_batch(
                            batch,
                            peak_columns,
                            role,
                            path.name,
                            batch_index,
                            report,
                            expected_list_length,
                            min_positive_mz,
                            enforce_min_positive_mz,
                            max_precursor_mz,
                            max_fragment_mz,
                            max_intensity,
                        ),
                    )
            except Exception as exc:  # malformed Arrow content must remain JSON-visible
                _error(report, f"{path.name}: content scan failed: {exc}")
        try:
            actual = {
                "name": path.name,
                "bytes": path.stat().st_size,
                "rows": int(parquet.metadata.num_rows),
                "sha256": sha256_file(path),
                "schema_fingerprint": schema_fingerprint(schema),
                "stats": shard_stats,
            }
        except OSError as exc:
            _error(report, f"cannot fingerprint {path.name}: {exc}")
            continue
        actual_records.append(actual)
        report["files"].append(actual)
        add_stats(report["stats"], shard_stats)
        for field in ("bytes", "rows", "sha256", "schema_fingerprint"):
            if declared.get(field) != actual[field]:
                _error(
                    report, f"{path.name}: manifest {field} does not match actual file"
                )
        if declared.get("stats") != shard_stats:
            _error(
                report, f"{path.name}: manifest stats do not match full content scan"
            )

    if common_schema is not None:
        actual_schema_fingerprint = schema_fingerprint(common_schema)
        report["schema_fingerprint"] = actual_schema_fingerprint
        if manifest.get("schema_fingerprint") != actual_schema_fingerprint:
            _error(report, "manifest schema_fingerprint does not match Parquet schema")
        if (
            expected_schema_fingerprint is not None
            and actual_schema_fingerprint != expected_schema_fingerprint.lower()
        ):
            _error(
                report,
                "Parquet schema fingerprint does not match the frozen expected schema",
            )
    declared_stats = output.get("stats")
    if declared_stats != report["stats"]:
        _error(report, "manifest aggregate stats do not match full content scan")
    if output.get("rows") not in (None, report["stats"]["rows"]):
        _error(report, "manifest output.rows does not match full content scan")
    if output.get("files") != len(actual_records):
        _error(report, "manifest output.files does not match actual shard count")
    if output.get("bytes") != sum(record["bytes"] for record in actual_records):
        _error(report, "manifest output.bytes does not match actual files")

    metadata_rows = sum(record["rows"] for record in actual_records)
    if report["stats"]["rows"] != metadata_rows:
        _error(report, "full content scan row total does not match Parquet metadata")
    if expected_rows is not None and metadata_rows != expected_rows:
        _error(
            report,
            f"audited row total {metadata_rows} does not match expected_rows "
            f"{expected_rows}",
        )
    _validate_content_closure(report)

    if len(actual_records) == len(records):
        audit_manifest = dict(manifest)
        audit_manifest["output"] = dict(output)
        audit_manifest["output"]["file_records"] = actual_records
        recomputed = _recompute_dataset_fingerprint(audit_manifest)
        report["recomputed_dataset_fingerprint"] = recomputed
        if manifest.get("dataset_fingerprint") != recomputed:
            _error(
                report,
                "dataset_fingerprint does not match run identity and actual files",
            )
        _validate_hash_manifest(dataset_dir, actual_records, report)
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def audit_pretraining_data(
    clean_dir: Path,
    polarity_dir: Path,
    *,
    expected_clean_name: str | None = None,
    expected_polarity_name: str | None = None,
    expected_clean_rows: int | None = None,
    expected_polarity_rows: int | None = None,
    expected_list_length: int = DEFAULT_EXPECTED_LIST_LENGTH,
    min_positive_mz: int = DEFAULT_MIN_POSITIVE_MZ,
    enforce_min_positive_mz: bool = True,
    max_precursor_mz: float = DEFAULT_MAX_PRECURSOR_MZ,
    max_fragment_mz: float = DEFAULT_MAX_FRAGMENT_MZ,
    max_intensity: float | None = DEFAULT_MAX_INTENSITY,
    expected_schema_fingerprint: str | None = None,
    batch_size: int = 65_536,
    progress_every: int = 25,
) -> dict[str, Any]:
    clean = audit_dataset(
        clean_dir,
        "clean",
        expected_dataset_name=expected_clean_name,
        expected_rows=expected_clean_rows,
        expected_list_length=expected_list_length,
        min_positive_mz=min_positive_mz,
        enforce_min_positive_mz=enforce_min_positive_mz,
        max_precursor_mz=max_precursor_mz,
        max_fragment_mz=max_fragment_mz,
        max_intensity=max_intensity,
        expected_schema_fingerprint=expected_schema_fingerprint,
        batch_size=batch_size,
        progress_every=progress_every,
    )
    polarity = audit_dataset(
        polarity_dir,
        "polarity",
        expected_dataset_name=expected_polarity_name,
        expected_rows=expected_polarity_rows,
        expected_list_length=expected_list_length,
        min_positive_mz=min_positive_mz,
        enforce_min_positive_mz=enforce_min_positive_mz,
        max_precursor_mz=max_precursor_mz,
        max_fragment_mz=max_fragment_mz,
        max_intensity=max_intensity,
        expected_schema_fingerprint=expected_schema_fingerprint,
        batch_size=batch_size,
        progress_every=progress_every,
    )
    errors: list[str] = []
    if (
        clean["status"] == "passed"
        and polarity.get("files")
        and polarity.get("manifest")
    ):
        clean_records = clean["files"]
        clean_schema = clean["schema_fingerprint"]
        expected_parent = _source_inventory_fingerprint(clean_schema, clean_records)
        polarity_parent = polarity["manifest"]["input_content_fingerprint"]
        if polarity_parent != expected_parent:
            errors.append(
                "polarity input_content_fingerprint does not identify the audited clean dataset"
            )
        polarity_source_counts = json.loads(
            (Path(polarity["path"]) / "manifest.json").read_text(encoding="utf-8")
        ).get("source_polarity_counts")
        expected_source_counts = {
            "rows": clean["stats"]["rows"],
            **clean["stats"]["polarity"],
        }
        if polarity_source_counts != expected_source_counts:
            errors.append(
                "polarity source_polarity_counts do not match audited clean polarity counts"
            )
    report = {
        "audit_version": 1,
        "status": "passed"
        if clean["status"] == polarity["status"] == "passed" and not errors
        else "failed",
        "errors": errors,
        "datasets": {"clean": clean, "polarity": polarity},
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--polarity-dir", type=Path)
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--expected-clean-name")
    parser.add_argument("--expected-polarity-name")
    parser.add_argument("--expected-clean-rows", type=int)
    parser.add_argument("--expected-polarity-rows", type=int)
    parser.add_argument(
        "--expected-list-length", type=int, default=DEFAULT_EXPECTED_LIST_LENGTH
    )
    parser.add_argument("--min-positive-mz", type=int, default=DEFAULT_MIN_POSITIVE_MZ)
    parser.add_argument(
        "--max-precursor-mz", type=float, default=DEFAULT_MAX_PRECURSOR_MZ
    )
    parser.add_argument(
        "--max-fragment-mz", type=float, default=DEFAULT_MAX_FRAGMENT_MZ
    )
    parser.add_argument("--max-intensity", type=float, default=DEFAULT_MAX_INTENSITY)
    parser.add_argument("--expected-schema-fingerprint")
    parser.add_argument(
        "--allow-short-spectra",
        action="store_true",
        help="report rows below --min-positive-mz without making them a hard error",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.batch_size <= 0
        or args.progress_every < 0
        or args.expected_list_length <= 0
        or args.min_positive_mz <= 0
        or args.max_precursor_mz <= 0
        or args.max_fragment_mz <= 0
        or (args.max_intensity is not None and args.max_intensity <= 0)
    ):
        raise SystemExit(
            "batch-size, expected-list-length and min-positive-mz must be positive; "
            "progress-every must be nonnegative"
        )
    try:
        if args.clean_only:
            clean = audit_dataset(
                args.clean_dir,
                "clean",
                expected_dataset_name=args.expected_clean_name,
                expected_rows=args.expected_clean_rows,
                expected_list_length=args.expected_list_length,
                min_positive_mz=args.min_positive_mz,
                enforce_min_positive_mz=not args.allow_short_spectra,
                max_precursor_mz=args.max_precursor_mz,
                max_fragment_mz=args.max_fragment_mz,
                max_intensity=args.max_intensity,
                expected_schema_fingerprint=args.expected_schema_fingerprint,
                batch_size=args.batch_size,
                progress_every=args.progress_every,
            )
            report = {
                "audit_version": 1,
                "status": clean["status"],
                "errors": [],
                "datasets": {"clean": clean},
            }
        else:
            if args.polarity_dir is None:
                raise ValueError(
                    "--polarity-dir is required unless --clean-only is set"
                )
            report = audit_pretraining_data(
                args.clean_dir,
                args.polarity_dir,
                expected_clean_name=args.expected_clean_name,
                expected_polarity_name=args.expected_polarity_name,
                expected_clean_rows=args.expected_clean_rows,
                expected_polarity_rows=args.expected_polarity_rows,
                expected_list_length=args.expected_list_length,
                min_positive_mz=args.min_positive_mz,
                enforce_min_positive_mz=not args.allow_short_spectra,
                max_precursor_mz=args.max_precursor_mz,
                max_fragment_mz=args.max_fragment_mz,
                max_intensity=args.max_intensity,
                expected_schema_fingerprint=args.expected_schema_fingerprint,
                batch_size=args.batch_size,
                progress_every=args.progress_every,
            )
    except Exception as exc:  # preserve a machine-readable failure at the CLI boundary
        report = {
            "audit_version": 1,
            "status": "failed",
            "errors": [f"audit aborted: {type(exc).__name__}: {exc}"],
            "datasets": {},
        }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
