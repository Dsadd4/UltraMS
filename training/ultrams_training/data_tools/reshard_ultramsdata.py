#!/usr/bin/env python3
"""Safely reshuffle UltraMSdata Parquet shards without the PyArrow 4-GiB list bug.

The two peak columns are promoted to ``large_list<double>`` before any
``Table.take`` operation. Bounded output shards are restored to the exact input
schema before they are atomically committed. Recovery deliberately replays the
deterministic source stream from the beginning; no RNG or leftover buffer is
silently reconstructed from incomplete state.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from ._audit_utils import (
        MANIFEST_VERSION,
        atomic_write_json,
        discover_parquet_files,
        fingerprint,
        fsync_directory,
        load_json,
        schema_fingerprint,
        sha256_file,
        write_hash_manifest,
    )
except ImportError:  # pragma: no cover - direct script execution
    from _audit_utils import (  # type: ignore
        MANIFEST_VERSION,
        atomic_write_json,
        discover_parquet_files,
        fingerprint,
        fsync_directory,
        load_json,
        schema_fingerprint,
        sha256_file,
        write_hash_manifest,
    )


PEAK_COLUMNS = ("spectrum_mz", "spectrum_intensity")
SOURCE_ROW_ID_COLUMN = "__ultrams_source_row_id"
RESUME_STRATEGY = "deterministic_replay_from_source_start"
PRODUCER_VERSION = 3
GATHER_VERIFICATION_MODES = {"row_ids", "all_values"}


@dataclass(frozen=True)
class SafeReshardConfig:
    source_dir: Path
    output_dir: Path
    expected_total_rows: int
    expected_schema_fingerprint: str
    expected_input_files: int | None = None
    target_rows_per_shard: int = 200_000
    seed: int = 42
    expected_list_length: int = 500
    minimum_positive_mz: int = 3
    gather_verification: str = "all_values"
    compression: str = "snappy"
    input_sha256_manifest: Path | None = None

    def stable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_dir"] = str(self.source_dir.resolve())
        result["output_dir"] = str(self.output_dir.resolve())
        result["expected_schema_fingerprint"] = self.expected_schema_fingerprint.lower()
        result["input_sha256_manifest"] = (
            str(self.input_sha256_manifest.resolve())
            if self.input_sha256_manifest
            else None
        )
        return result


def _validate_config(config: SafeReshardConfig) -> None:
    if config.source_dir.resolve() == config.output_dir.resolve():
        raise ValueError("source and output directories must differ")
    positive = (
        config.expected_total_rows,
        config.target_rows_per_shard,
        config.expected_list_length,
        config.minimum_positive_mz,
    )
    if any(value <= 0 for value in positive):
        raise ValueError(
            "row, shard, list-length and positive-m/z gates must be positive"
        )
    if config.minimum_positive_mz > config.expected_list_length:
        raise ValueError("minimum_positive_mz cannot exceed expected_list_length")
    if config.gather_verification not in GATHER_VERIFICATION_MODES:
        raise ValueError(
            "gather_verification must be one of "
            f"{sorted(GATHER_VERIFICATION_MODES)}"
        )
    peak_child_bytes = config.target_rows_per_shard * config.expected_list_length * 8
    if peak_child_bytes >= 2**32:
        raise ValueError(
            "target_rows_per_shard produces a >=4-GiB peak child buffer; "
            "use smaller output shards"
        )
    if config.expected_input_files is not None and config.expected_input_files <= 0:
        raise ValueError("expected_input_files must be positive")
    value = config.expected_schema_fingerprint.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected_schema_fingerprint must be a SHA-256 hex digest")
    if not 0 <= config.seed <= 2**32 - 1:
        raise ValueError("seed must be in NumPy RandomState's uint32 range")


def _parse_strict_sha256_manifest(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            digest = parts[0].lower() if parts else ""
            if (
                len(parts) != 2
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(
                    f"invalid SHA-256 manifest line {line_number}: {raw!r}"
                )
            name = Path(parts[1].lstrip("* ")).name
            if name in hashes:
                raise ValueError(f"duplicate SHA-256 manifest entry: {name}")
            hashes[name] = digest
    if not hashes:
        raise ValueError(f"empty SHA-256 manifest: {path}")
    return hashes


def _validate_peak_schema(schema: pa.Schema) -> None:
    for name in PEAK_COLUMNS:
        index = schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"missing required peak column {name!r}")
        data_type = schema.field(index).type
        if not (
            pa.types.is_list(data_type)
            or pa.types.is_large_list(data_type)
            or pa.types.is_fixed_size_list(data_type)
        ):
            raise ValueError(f"{name} must be a list column, got {data_type}")
        if not pa.types.is_float64(data_type.value_type):
            raise ValueError(
                f"{name} values must be float64, got {data_type.value_type}"
            )


def _build_or_validate_input_manifest(
    config: SafeReshardConfig, audit_dir: Path
) -> dict[str, Any]:
    source_dir = config.source_dir.resolve()
    manifest_path = audit_dir / "input_manifest.json"
    cached = load_json(manifest_path) if manifest_path.exists() else None
    cached_by_name = (
        {item["name"]: item for item in cached.get("files", [])} if cached else {}
    )
    files = discover_parquet_files(source_dir)
    if (
        config.expected_input_files is not None
        and len(files) != config.expected_input_files
    ):
        raise RuntimeError(
            f"input file gate failed: expected {config.expected_input_files}, got {len(files)}"
        )

    supplied_hashes = (
        _parse_strict_sha256_manifest(config.input_sha256_manifest.resolve())
        if config.input_sha256_manifest
        else {}
    )
    names = {path.name for path in files}
    if supplied_hashes and set(supplied_hashes) != names:
        missing = sorted(names - set(supplied_hashes))
        extra = sorted(set(supplied_hashes) - names)
        raise RuntimeError(
            f"SHA-256 manifest coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    common_schema: pa.Schema | None = None
    items: list[dict[str, Any]] = []
    global_start = 0
    for index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        _validate_peak_schema(schema)
        if common_schema is None:
            common_schema = schema
        elif not common_schema.equals(schema, check_metadata=True):
            raise RuntimeError(f"input schema mismatch: {path}")
        rows = int(parquet.metadata.num_rows)
        stat = path.stat()
        actual_sha256: str | None = None
        if supplied_hashes:
            cached_item = cached_by_name.get(path.name)
            cache_matches_file = bool(
                cached_item
                and cached_item.get("bytes") == stat.st_size
                and cached_item.get("rows") == rows
                and cached_item.get("mtime_ns") == stat.st_mtime_ns
                and cached_item.get("device") == stat.st_dev
                and cached_item.get("inode") == stat.st_ino
                and cached_item.get("schema_fingerprint") == schema_fingerprint(schema)
                and cached_item.get("sha256") == supplied_hashes[path.name]
            )
            actual_sha256 = (
                cached_item["sha256"] if cache_matches_file else sha256_file(path)
            )
            if actual_sha256 != supplied_hashes[path.name]:
                raise RuntimeError(f"input SHA-256 mismatch: {path}")
        items.append(
            {
                "index": index,
                "name": path.name,
                "bytes": stat.st_size,
                "rows": rows,
                "global_start": global_start,
                "global_end_exclusive": global_start + rows,
                "mtime_ns": stat.st_mtime_ns,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "schema_fingerprint": schema_fingerprint(schema),
                "sha256": actual_sha256,
            }
        )
        global_start += rows

    assert common_schema is not None
    actual_schema_fingerprint = schema_fingerprint(common_schema)
    if global_start != config.expected_total_rows:
        raise RuntimeError(
            f"input row gate failed: expected {config.expected_total_rows}, got {global_start}"
        )
    if actual_schema_fingerprint != config.expected_schema_fingerprint.lower():
        raise RuntimeError(
            "input schema fingerprint gate failed: "
            f"expected {config.expected_schema_fingerprint.lower()}, "
            f"got {actual_schema_fingerprint}"
        )

    stable_files = []
    for item in items:
        stable_item = {
            "index": item["index"],
            "name": item["name"],
            "bytes": item["bytes"],
            "rows": item["rows"],
            "global_start": item["global_start"],
            "global_end_exclusive": item["global_end_exclusive"],
            "schema_fingerprint": item["schema_fingerprint"],
            "sha256": item["sha256"],
        }
        if not supplied_hashes:
            stable_item.update(
                {
                    "mtime_ns": item["mtime_ns"],
                    "device": item["device"],
                    "inode": item["inode"],
                }
            )
        stable_files.append(stable_item)
    stable = {
        "ordering": "lexical_basename",
        "schema_fingerprint": actual_schema_fingerprint,
        "sha256_verified": bool(supplied_hashes),
        "files": stable_files,
        "total_files": len(items),
        "total_rows": global_start,
        "total_bytes": sum(item["bytes"] for item in items),
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source_dir": str(source_dir),
        "schema": str(common_schema),
        **stable,
        # Keep filesystem identity for inexpensive verified resumes. These
        # fields are operational metadata and are intentionally excluded from
        # the stable content fingerprint above.
        "files": items,
        "content_fingerprint": fingerprint(stable),
    }
    if cached:
        if cached.get("content_fingerprint") != manifest["content_fingerprint"]:
            raise RuntimeError(
                "input inventory changed after output initialization; use a new output directory"
            )
        if cached != manifest:
            atomic_write_json(manifest_path, manifest)
    else:
        atomic_write_json(manifest_path, manifest)
    return manifest


def _field_with_type(field: pa.Field, data_type: pa.DataType) -> pa.Field:
    return pa.field(
        field.name,
        data_type,
        nullable=field.nullable,
        metadata=field.metadata,
    )


def promote_peak_lists(table: pa.Table) -> pa.Table:
    """Promote peak lists to 64-bit offsets before any gather operation."""
    output = table
    for name in PEAK_COLUMNS:
        index = output.schema.get_field_index(name)
        field = output.schema.field(index)
        column = output.column(index)
        if pa.types.is_large_list(column.type):
            continue
        target_type = pa.large_list(column.type.value_field)
        promoted = pa.chunked_array(
            [chunk.cast(target_type) for chunk in column.chunks], type=target_type
        )
        output = output.set_column(
            index, _field_with_type(field, target_type), promoted
        )
    return output


def restore_peak_lists(table: pa.Table, original_schema: pa.Schema) -> pa.Table:
    """Restore the exact original peak-column fields on a bounded shard."""
    output = table
    for name in PEAK_COLUMNS:
        index = output.schema.get_field_index(name)
        original_field = original_schema.field(original_schema.get_field_index(name))
        column = output.column(index)
        restored = pa.chunked_array(
            [chunk.cast(original_field.type) for chunk in column.chunks],
            type=original_field.type,
        )
        output = output.set_column(index, original_field, restored)
    output = output.replace_schema_metadata(original_schema.metadata)
    if not output.schema.equals(original_schema, check_metadata=True):
        raise RuntimeError("failed to restore the exact input schema")
    return output


def safe_take(table: pa.Table, permutation: np.ndarray) -> pa.Table:
    """Gather rows only after both vulnerable list columns use 64-bit offsets."""
    for name in PEAK_COLUMNS:
        if not pa.types.is_large_list(table[name].type):
            raise RuntimeError(f"unsafe take refused: {name} is {table[name].type}")
    return table.take(pa.array(permutation, type=pa.int64()))


def _logical_list_matrix(
    column: pa.ChunkedArray, *, rows: int, expected_length: int
) -> np.ndarray:
    flattened = pc.list_flatten(column)
    if isinstance(flattened, pa.ChunkedArray):
        flattened = flattened.combine_chunks()
    values = flattened.to_numpy(zero_copy_only=False)
    expected_values = rows * expected_length
    if values.size != expected_values:
        raise RuntimeError(
            f"logical list size mismatch: expected {expected_values}, got {values.size}"
        )
    return values.reshape(rows, expected_length)


def _exact_column_equal(left: pa.ChunkedArray, right: pa.ChunkedArray) -> bool:
    if left.type != right.type or len(left) != len(right):
        return False
    left_array = left.combine_chunks()
    right_array = right.combine_chunks()
    if pa.types.is_floating(left.type):
        left_values = left_array.to_numpy(zero_copy_only=False)
        right_values = right_array.to_numpy(zero_copy_only=False)
        return bool(np.array_equal(left_values, right_values, equal_nan=True))
    return left_array.equals(right_array)


def verify_tables_exact(
    expected: pa.Table,
    actual: pa.Table,
    *,
    expected_list_length: int,
    compare_batch_rows: int = 8192,
) -> None:
    """Compare every field and every value while treating paired NaNs as equal."""
    if len(expected) != len(actual) or expected.column_names != actual.column_names:
        raise RuntimeError("exact table audit shape or columns differ")
    if not expected.schema.equals(actual.schema, check_metadata=True):
        raise RuntimeError("exact table audit schemas differ")
    for start in range(0, len(expected), compare_batch_rows):
        rows = min(compare_batch_rows, len(expected) - start)
        expected_batch = expected.slice(start, rows)
        actual_batch = actual.slice(start, rows)
        for name in expected.column_names:
            if name in PEAK_COLUMNS:
                left = _logical_list_matrix(
                    expected_batch[name],
                    rows=rows,
                    expected_length=expected_list_length,
                )
                right = _logical_list_matrix(
                    actual_batch[name],
                    rows=rows,
                    expected_length=expected_list_length,
                )
                equal = np.array_equal(left, right, equal_nan=True)
            else:
                equal = _exact_column_equal(expected_batch[name], actual_batch[name])
            if not equal:
                raise RuntimeError(
                    f"exact table audit failed for {name} in rows "
                    f"[{start}, {start + rows})"
                )


def verify_exact_gather(
    source: pa.Table,
    shuffled: pa.Table,
    permutation: np.ndarray,
    *,
    expected_list_length: int,
    compare_batch_rows: int = 8192,
) -> None:
    """Prove that every gathered peak value remains attached to its raw row."""
    if len(source) != len(shuffled) or len(source) != len(permutation):
        raise RuntimeError("exact gather audit row counts do not match")
    if SOURCE_ROW_ID_COLUMN not in source.column_names:
        raise RuntimeError("exact gather audit is missing source row IDs")
    source_ids = source[SOURCE_ROW_ID_COLUMN].combine_chunks().to_numpy()
    shuffled_ids = shuffled[SOURCE_ROW_ID_COLUMN].combine_chunks().to_numpy()
    if not np.array_equal(shuffled_ids, source_ids[permutation]):
        raise RuntimeError("source row IDs changed during gather")

    rows = len(source)
    for start in range(0, rows, compare_batch_rows):
        end = min(rows, start + compare_batch_rows)
        batch_permutation = permutation[start:end]
        expected_batch = safe_take(source, batch_permutation)
        actual_batch = shuffled.slice(start, end - start)
        try:
            verify_tables_exact(
                expected_batch,
                actual_batch,
                expected_list_length=expected_list_length,
                compare_batch_rows=compare_batch_rows,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"exact gather audit failed in output rows [{start}, {end})"
            ) from error


def verify_gather_row_ids(
    source: pa.Table,
    shuffled: pa.Table,
    permutation: np.ndarray,
) -> None:
    """Verify the full row permutation without repeating the peak gather."""
    if len(source) != len(shuffled) or len(source) != len(permutation):
        raise RuntimeError("row-id gather audit row counts do not match")
    if SOURCE_ROW_ID_COLUMN not in source.column_names:
        raise RuntimeError("row-id gather audit is missing source row IDs")
    source_ids = source[SOURCE_ROW_ID_COLUMN].combine_chunks().to_numpy()
    shuffled_ids = shuffled[SOURCE_ROW_ID_COLUMN].combine_chunks().to_numpy()
    if not np.array_equal(shuffled_ids, source_ids[permutation]):
        raise RuntimeError("source row IDs changed during gather")


def _gather_verification_record(
    mode: str,
    *,
    flushes: int,
    rows: int,
    peak_child_bytes_max: int,
) -> dict[str, Any]:
    if mode == "all_values":
        method = "source-row-id plus exact all-column value comparison"
        columns = "all"
    else:
        method = "exact source-row-id permutation"
        columns = "source_row_id"
    return {
        "status": "passed",
        "mode": mode,
        "method": method,
        "columns": columns,
        "flushes": flushes,
        "rows_compared_across_operations": rows,
        "peak_child_buffer_bytes_max": peak_child_bytes_max,
    }


def _list_lengths_and_positive_counts(
    column: pa.ChunkedArray,
    *,
    expected_length: int,
    count_positive: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    length_parts: list[np.ndarray] = []
    positive_parts: list[np.ndarray] = []
    for chunk in column.chunks:
        if chunk.null_count:
            raise RuntimeError(f"null list rows in {column.type}")
        if pa.types.is_fixed_size_list(chunk.type):
            lengths = np.full(len(chunk), chunk.type.list_size, dtype=np.int64)
        else:
            offsets = chunk.offsets.to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
            lengths = np.diff(offsets)
        if len(lengths) and not np.all(lengths == expected_length):
            raise RuntimeError(
                f"list-length audit failed: expected {expected_length}, "
                f"observed [{int(lengths.min())}, {int(lengths.max())}]"
            )
        length_parts.append(lengths)
        if count_positive:
            flat = chunk.flatten()
            if flat.null_count:
                raise RuntimeError("null values in spectrum_mz")
            values = flat.to_numpy(zero_copy_only=False)
            if values.size != len(chunk) * expected_length:
                raise RuntimeError("flattened spectrum_mz size is inconsistent")
            positive_parts.append(
                np.count_nonzero(
                    values.reshape(len(chunk), expected_length) > 0, axis=1
                )
            )
        else:
            flat = chunk.flatten()
            if flat.null_count:
                raise RuntimeError("null values in spectrum_intensity")
    lengths = (
        np.concatenate(length_parts) if length_parts else np.empty(0, dtype=np.int64)
    )
    positives = (
        np.concatenate(positive_parts)
        if count_positive and positive_parts
        else np.empty(0, dtype=np.int64)
        if count_positive
        else None
    )
    return lengths, positives


def audit_shard_semantics(
    table: pa.Table,
    *,
    expected_schema: pa.Schema,
    expected_rows: int,
    expected_list_length: int,
    minimum_positive_mz: int,
) -> dict[str, Any]:
    if table.num_rows != expected_rows:
        raise RuntimeError(
            f"shard row audit failed: expected {expected_rows}, got {table.num_rows}"
        )
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise RuntimeError("shard schema audit failed")
    mz_lengths, positive_counts = _list_lengths_and_positive_counts(
        table["spectrum_mz"],
        expected_length=expected_list_length,
        count_positive=True,
    )
    intensity_lengths, _ = _list_lengths_and_positive_counts(
        table["spectrum_intensity"],
        expected_length=expected_list_length,
        count_positive=False,
    )
    if not np.array_equal(mz_lengths, intensity_lengths):
        raise RuntimeError("m/z and intensity list lengths differ")
    assert positive_counts is not None
    failing = np.flatnonzero(positive_counts < minimum_positive_mz)
    if len(failing):
        raise RuntimeError(
            f"positive-m/z audit failed for {len(failing)} rows; "
            f"first row={int(failing[0])}, count={int(positive_counts[failing[0]])}"
        )
    return {
        "rows": table.num_rows,
        "list_length": expected_list_length,
        "mz_length_min": int(mz_lengths.min()) if len(mz_lengths) else None,
        "mz_length_max": int(mz_lengths.max()) if len(mz_lengths) else None,
        "intensity_length_min": (
            int(intensity_lengths.min()) if len(intensity_lengths) else None
        ),
        "intensity_length_max": (
            int(intensity_lengths.max()) if len(intensity_lengths) else None
        ),
        "minimum_positive_mz_required": minimum_positive_mz,
        "minimum_positive_mz_observed": (
            int(positive_counts.min()) if len(positive_counts) else None
        ),
        "rows_below_minimum_positive_mz": 0,
        "null_list_rows": 0,
        "null_peak_values": 0,
    }


def _shard_sizes(total_rows: int, target_rows_per_shard: int) -> list[int]:
    number = max(1, (total_rows + target_rows_per_shard - 1) // target_rows_per_shard)
    base, remainder = divmod(total_rows, number)
    return [base + (index < remainder) for index in range(number)]


def _validate_output_record(
    output_dir: Path,
    record: dict[str, Any],
    expected_schema_fingerprint: str,
) -> None:
    path = output_dir / record["name"]
    if not path.exists():
        raise RuntimeError(f"committed output is missing: {path}")
    stat = path.stat()
    if stat.st_size != record["bytes"]:
        raise RuntimeError(f"committed output size changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"committed output SHA-256 changed: {path}")
    parquet = pq.ParquetFile(path)
    if int(parquet.metadata.num_rows) != record["rows"]:
        raise RuntimeError(f"committed output row count changed: {path}")
    actual_schema = schema_fingerprint(parquet.schema_arrow)
    if actual_schema != expected_schema_fingerprint:
        raise RuntimeError(f"committed output schema changed: {path}")


def _assert_input_item_unchanged(
    path: Path, item: dict[str, Any], *, verify_sha256: bool
) -> None:
    stat = path.stat()
    current = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    expected = {key: item[key] for key in current}
    if current != expected:
        raise RuntimeError(f"input file identity changed during build: {path}")
    parquet = pq.ParquetFile(path)
    if int(parquet.metadata.num_rows) != item["rows"]:
        raise RuntimeError(f"input row count changed during build: {path}")
    if schema_fingerprint(parquet.schema_arrow) != item["schema_fingerprint"]:
        raise RuntimeError(f"input schema changed during build: {path}")
    if verify_sha256:
        expected_digest = item.get("sha256")
        if not expected_digest or sha256_file(path) != expected_digest:
            raise RuntimeError(f"input SHA-256 changed during build: {path}")


def _load_progress(
    output_dir: Path,
    audit_dir: Path,
    run_identity: str,
    expected_schema_fingerprint: str,
) -> list[dict[str, Any]]:
    paths = sorted((audit_dir / "progress").glob("shard_*.json"))
    records: list[dict[str, Any]] = []
    for expected_index, path in enumerate(paths):
        progress = load_json(path)
        if progress.get("run_identity") != run_identity:
            raise RuntimeError(f"progress identity mismatch: {path}")
        if progress.get("resume_strategy") != RESUME_STRATEGY:
            raise RuntimeError(f"progress resume strategy mismatch: {path}")
        if progress.get("output_index") != expected_index:
            raise RuntimeError("progress is not a contiguous shard prefix")
        record = progress["output"]
        if record["name"] != f"shard_{expected_index:04d}.parquet":
            raise RuntimeError(f"unexpected output name in {path}")
        _validate_output_record(output_dir, record, expected_schema_fingerprint)
        records.append(record)
    return records


def _validate_output_inventory(output_dir: Path, committed: int) -> None:
    names = sorted(path.name for path in output_dir.glob("shard_*.parquet"))
    tracked = [f"shard_{index:04d}.parquet" for index in range(committed)]
    allowed_orphan = f"shard_{committed:04d}.parquet"
    if names == tracked or names == tracked + [allowed_orphan]:
        return
    raise RuntimeError(
        "output directory contains non-contiguous or multiple untracked shards: "
        f"{names[:10]}"
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


@contextmanager
def _exclusive_run_lock(audit_dir: Path):
    path = audit_dir / "run.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another reshard producer holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_and_audit_temporary(
    table: pa.Table,
    temporary: Path,
    config: SafeReshardConfig,
    original_schema: pa.Schema,
    expected_rows: int,
    in_memory_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporary.unlink(missing_ok=True)
    pq.write_table(
        table,
        temporary,
        compression=config.compression,
        use_dictionary=True,
        write_statistics=True,
    )
    _fsync_file(temporary)
    parquet = pq.ParquetFile(temporary)
    if not parquet.schema_arrow.equals(original_schema, check_metadata=True):
        raise RuntimeError("written Parquet schema differs from input schema")
    if int(parquet.metadata.num_rows) != expected_rows:
        raise RuntimeError("written Parquet row count differs from planned shard size")
    readback = pq.read_table(temporary)
    verify_tables_exact(
        table,
        readback,
        expected_list_length=config.expected_list_length,
    )
    readback_audit = audit_shard_semantics(
        readback,
        expected_schema=original_schema,
        expected_rows=expected_rows,
        expected_list_length=config.expected_list_length,
        minimum_positive_mz=config.minimum_positive_mz,
    )
    if readback_audit != in_memory_audit:
        raise RuntimeError("written Parquet semantic audit differs from memory")
    stat = temporary.stat()
    file_record = {
        "bytes": stat.st_size,
        "rows": expected_rows,
        "sha256": sha256_file(temporary),
        "schema_fingerprint": schema_fingerprint(parquet.schema_arrow),
    }
    return file_record, readback_audit


def _commit_or_compare_shard(
    *,
    table: pa.Table,
    output_index: int,
    expected_rows: int,
    config: SafeReshardConfig,
    original_schema: pa.Schema,
    output_dir: Path,
    audit_dir: Path,
    run_identity: str,
    committed_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    name = f"shard_{output_index:04d}.parquet"
    output = output_dir / name
    temporary = output_dir / f".{name}.partial.{os.getpid()}"
    in_memory_audit = audit_shard_semantics(
        table,
        expected_schema=original_schema,
        expected_rows=expected_rows,
        expected_list_length=config.expected_list_length,
        minimum_positive_mz=config.minimum_positive_mz,
    )
    file_values, readback_audit = _write_and_audit_temporary(
        table,
        temporary,
        config,
        original_schema,
        expected_rows,
        in_memory_audit,
    )
    record = {
        "name": name,
        **file_values,
        "semantic_audit": in_memory_audit,
        "readback_semantic_audit": readback_audit,
    }

    if output_index < len(committed_records):
        committed = committed_records[output_index]
        if any(
            committed.get(key) != record.get(key)
            for key in ("name", "bytes", "rows", "sha256", "schema_fingerprint")
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"deterministic replay differs at {name}")
        if committed.get("semantic_audit") != in_memory_audit:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"semantic audit differs during replay at {name}")
        if committed.get("readback_semantic_audit") != readback_audit:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"readback semantic audit differs during replay at {name}"
            )
        temporary.unlink()
        return committed, False

    if output_index != len(committed_records):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("attempted to commit a non-contiguous output shard")
    if output.exists():
        if (
            output.stat().st_size != record["bytes"]
            or sha256_file(output) != record["sha256"]
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"orphan output conflicts with deterministic replay: {output}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, output)
        fsync_directory(output_dir)
    output_stat = output.stat()
    record["mtime_ns"] = output_stat.st_mtime_ns
    progress = {
        "manifest_version": MANIFEST_VERSION,
        "run_identity": run_identity,
        "resume_strategy": RESUME_STRATEGY,
        "output_index": output_index,
        "output": record,
    }
    atomic_write_json(
        audit_dir / "progress" / f"shard_{output_index:04d}.json", progress
    )
    committed_records.append(record)
    return record, True


def _partial_result(
    run_identity: str,
    input_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    total_shards: int,
    *,
    gather_verification_flushes: int,
    gather_verification_rows: int,
    gather_verification_peak_child_bytes_max: int,
    gather_verification_mode: str,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "status": "in_progress",
        "run_identity": run_identity,
        "resume_strategy": RESUME_STRATEGY,
        "input_content_fingerprint": input_manifest["content_fingerprint"],
        "completed_output_shards": len(records),
        "planned_output_shards": total_shards,
        "completed_output_rows": sum(record["rows"] for record in records),
        "exact_gather_verification": _gather_verification_record(
            gather_verification_mode,
            flushes=gather_verification_flushes,
            rows=gather_verification_rows,
            peak_child_bytes_max=gather_verification_peak_child_bytes_max,
        ),
    }


def _validate_complete_manifest(
    manifest: dict[str, Any],
    *,
    run_identity: str,
    committed_records: list[dict[str, Any]],
    expected_total_rows: int,
    input_content_fingerprint: str,
    output_dir: Path,
    expected_gather_verification: str,
    expected_sha256_verified: bool,
) -> None:
    if (
        manifest.get("status") != "complete"
        or manifest.get("run_identity") != run_identity
    ):
        raise RuntimeError("final manifest status or identity mismatch")
    output = manifest.get("output", {})
    if manifest.get("input_content_fingerprint") != input_content_fingerprint:
        raise RuntimeError("final manifest input fingerprint changed")
    if output.get("file_records") != committed_records:
        raise RuntimeError("final manifest records differ from committed progress")
    exact = manifest.get("exact_gather_verification", {})
    if (
        exact.get("status") != "passed"
        or exact.get("mode") != expected_gather_verification
        or exact.get("flushes") != len(committed_records)
        or exact.get("rows_compared_across_operations", 0) < expected_total_rows
    ):
        raise RuntimeError("final exact-gather evidence is incomplete")
    reverification = manifest.get("final_input_reverification", {})
    expected_reverification_status = (
        "passed" if expected_sha256_verified else "not_requested"
    )
    if (
        reverification.get("status") != expected_reverification_status
        or reverification.get("sha256_verified") is not expected_sha256_verified
        or reverification.get("files", 0) <= 0
    ):
        raise RuntimeError("final input SHA-256 reverification is incomplete")
    rows = sum(record["rows"] for record in committed_records)
    if output.get("rows") != rows or rows != expected_total_rows:
        raise RuntimeError("final manifest output rows do not close")
    closure = manifest.get("row_closure", {})
    if closure != {
        "expected": expected_total_rows,
        "input": expected_total_rows,
        "output": expected_total_rows,
        "closed": True,
    }:
        raise RuntimeError("final manifest row closure changed")
    expected_hash_lines = [
        f"{record['sha256']}  {record['name']}" for record in committed_records
    ]
    hash_path = output_dir / "files.sha256"
    if (
        not hash_path.exists()
        or hash_path.read_text().strip().splitlines() != expected_hash_lines
    ):
        raise RuntimeError("output SHA-256 manifest differs from committed progress")
    expected_dataset_fingerprint = fingerprint(
        {
            "run_identity": run_identity,
            "files": [
                (record["name"], record["rows"], record["sha256"])
                for record in committed_records
            ],
            "exact_gather_verification": exact,
            "final_input_reverification": reverification,
        }
    )
    if manifest.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise RuntimeError("final dataset fingerprint changed")


def _reshard_ultramsdata_locked(
    config: SafeReshardConfig,
    *,
    max_new_shards_this_run: int | None = None,
) -> dict[str, Any]:
    source_dir = config.source_dir.resolve()
    output_dir = config.output_dir.resolve()
    audit_dir = output_dir / "_audit"
    for stale in output_dir.glob(".shard_*.parquet.partial.*"):
        stale.unlink()

    input_manifest = _build_or_validate_input_manifest(config, audit_dir)
    first_source = source_dir / input_manifest["files"][0]["name"]
    original_schema = pq.ParquetFile(first_source).schema_arrow
    producer_sha256 = sha256_file(Path(__file__).resolve())
    stable_config = config.stable_dict()
    identity_config = dict(stable_config)
    identity_config.pop("source_dir", None)
    identity_config.pop("output_dir", None)
    identity_config.pop("input_sha256_manifest", None)
    run_identity = fingerprint(
        {
            "producer_version": PRODUCER_VERSION,
            "producer_sha256": producer_sha256,
            "pyarrow_version": pa.__version__,
            "numpy_version": np.__version__,
            "config": identity_config,
            "input": input_manifest["content_fingerprint"],
        }
    )
    identity_path = audit_dir / "run_identity.json"
    if identity_path.exists():
        if load_json(identity_path).get("run_identity") != run_identity:
            raise RuntimeError("configuration changed; use a new output directory")
    else:
        atomic_write_json(
            identity_path,
            {
                "manifest_version": MANIFEST_VERSION,
                "producer_version": PRODUCER_VERSION,
                "producer_sha256": producer_sha256,
                "pyarrow_version": pa.__version__,
                "numpy_version": np.__version__,
                "run_identity": run_identity,
                "resume_strategy": RESUME_STRATEGY,
                "config": stable_config,
                "status": "in_progress",
            },
        )

    committed_records = _load_progress(
        output_dir,
        audit_dir,
        run_identity,
        input_manifest["schema_fingerprint"],
    )
    _validate_output_inventory(output_dir, len(committed_records))
    shard_sizes = _shard_sizes(
        input_manifest["total_rows"], config.target_rows_per_shard
    )
    if len(committed_records) > len(shard_sizes):
        raise RuntimeError("progress contains more shards than the current plan")

    final_manifest_path = output_dir / "manifest.json"
    if final_manifest_path.exists():
        final_manifest = load_json(final_manifest_path)
        if len(committed_records) != len(shard_sizes):
            raise RuntimeError(
                "final manifest exists before all shard progress records"
            )
        _validate_complete_manifest(
            final_manifest,
            run_identity=run_identity,
            committed_records=committed_records,
            expected_total_rows=config.expected_total_rows,
            input_content_fingerprint=input_manifest["content_fingerprint"],
            output_dir=output_dir,
            expected_gather_verification=config.gather_verification,
            expected_sha256_verified=config.input_sha256_manifest is not None,
        )
        complete_state = {
            "manifest_version": MANIFEST_VERSION,
            "status": "complete",
            "run_identity": run_identity,
            "resume_strategy": RESUME_STRATEGY,
            "completed_output_shards": len(committed_records),
            "completed_output_rows": config.expected_total_rows,
            "dataset_fingerprint": final_manifest["dataset_fingerprint"],
        }
        atomic_write_json(audit_dir / "run_state.json", complete_state)
        identity = load_json(identity_path)
        identity["status"] = "complete"
        atomic_write_json(identity_path, identity)
        return final_manifest

    rng = np.random.RandomState(config.seed)
    buffer_tables: list[pa.Table] = []
    buffer_rows = 0
    output_index = 0
    files_read = 0
    rows_read = 0
    newly_committed = 0
    stopped = False
    gather_verification_flushes = 0
    gather_verification_rows = 0
    gather_verification_peak_child_bytes_max = 0

    def flush() -> bool:
        nonlocal buffer_tables, buffer_rows, output_index, newly_committed, stopped
        nonlocal gather_verification_flushes, gather_verification_rows
        nonlocal gather_verification_peak_child_bytes_max
        target_rows = shard_sizes[output_index]
        combined = pa.concat_tables(buffer_tables)
        if len(combined) != buffer_rows:
            raise RuntimeError("buffer row accounting mismatch")
        permutation = rng.permutation(buffer_rows)
        shuffled = safe_take(combined, permutation)
        if config.gather_verification == "all_values":
            verify_exact_gather(
                combined,
                shuffled,
                permutation,
                expected_list_length=config.expected_list_length,
            )
        else:
            verify_gather_row_ids(combined, shuffled, permutation)
        gather_verification_flushes += 1
        gather_verification_rows += len(shuffled)
        gather_verification_peak_child_bytes_max = max(
            gather_verification_peak_child_bytes_max,
            len(combined) * config.expected_list_length * 8,
        )
        output_large = shuffled.slice(0, target_rows)
        leftover = shuffled.slice(target_rows)
        output_table = restore_peak_lists(
            output_large.drop([SOURCE_ROW_ID_COLUMN]), original_schema
        )
        _, was_new = _commit_or_compare_shard(
            table=output_table,
            output_index=output_index,
            expected_rows=target_rows,
            config=config,
            original_schema=original_schema,
            output_dir=output_dir,
            audit_dir=audit_dir,
            run_identity=run_identity,
            committed_records=committed_records,
        )
        output_index += 1
        if was_new:
            newly_committed += 1
        buffer_tables = [leftover] if len(leftover) else []
        buffer_rows = len(leftover)
        if (
            max_new_shards_this_run is not None
            and newly_committed >= max_new_shards_this_run
        ):
            stopped = True
        return stopped

    for item in input_manifest["files"]:
        source_path = source_dir / item["name"]
        _assert_input_item_unchanged(source_path, item, verify_sha256=False)
        table = pq.read_table(source_path)
        if len(table) != item["rows"]:
            raise RuntimeError(f"source row count changed while reading: {source_path}")
        if not table.schema.equals(original_schema, check_metadata=True):
            raise RuntimeError(f"source schema changed while reading: {source_path}")
        source_row_ids = pa.array(
            np.arange(
                item["global_start"], item["global_end_exclusive"], dtype=np.int64
            )
        )
        table = table.append_column(SOURCE_ROW_ID_COLUMN, source_row_ids)
        table = promote_peak_lists(table)
        _assert_input_item_unchanged(source_path, item, verify_sha256=False)
        buffer_tables.append(table)
        buffer_rows += len(table)
        rows_read += len(table)
        files_read += 1
        while (
            output_index < len(shard_sizes) and buffer_rows >= shard_sizes[output_index]
        ):
            if flush():
                break
        if stopped:
            break

    if stopped:
        result = _partial_result(
            run_identity,
            input_manifest,
            committed_records,
            len(shard_sizes),
            gather_verification_flushes=gather_verification_flushes,
            gather_verification_rows=gather_verification_rows,
            gather_verification_peak_child_bytes_max=(
                gather_verification_peak_child_bytes_max
            ),
            gather_verification_mode=config.gather_verification,
        )
        atomic_write_json(audit_dir / "run_state.json", result)
        return result

    if files_read != input_manifest["total_files"]:
        raise RuntimeError(
            f"not all input files were read: {files_read}/{input_manifest['total_files']}"
        )
    if rows_read != input_manifest["total_rows"]:
        raise RuntimeError(
            f"input rows read do not close: {rows_read}/{input_manifest['total_rows']}"
        )
    while output_index < len(shard_sizes) and buffer_rows >= shard_sizes[output_index]:
        flush()
    if output_index != len(shard_sizes) or buffer_rows != 0 or buffer_tables:
        raise RuntimeError(
            f"final buffer did not close: output={output_index}/{len(shard_sizes)} "
            f"leftover_rows={buffer_rows}"
        )
    if len(committed_records) != len(shard_sizes):
        raise RuntimeError("committed shard count does not match the plan")
    output_rows = sum(record["rows"] for record in committed_records)
    if (
        output_rows != input_manifest["total_rows"]
        or output_rows != config.expected_total_rows
    ):
        raise RuntimeError(
            f"final row closure failed: output={output_rows}, input={input_manifest['total_rows']}, "
            f"expected={config.expected_total_rows}"
        )
    if [record["rows"] for record in committed_records] != shard_sizes:
        raise RuntimeError("committed shard sizes differ from the deterministic plan")

    for item in input_manifest["files"]:
        _assert_input_item_unchanged(
            source_dir / item["name"],
            item,
            verify_sha256=input_manifest["sha256_verified"],
        )

    write_hash_manifest(output_dir / "files.sha256", committed_records)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "producer_version": PRODUCER_VERSION,
        "producer_sha256": producer_sha256,
        "status": "complete",
        "run_identity": run_identity,
        "resume_strategy": RESUME_STRATEGY,
        "config": stable_config,
        "input_manifest": "_audit/input_manifest.json",
        "input_content_fingerprint": input_manifest["content_fingerprint"],
        "input": {
            "files": input_manifest["total_files"],
            "rows": input_manifest["total_rows"],
            "bytes": input_manifest["total_bytes"],
            "sha256_verified": input_manifest["sha256_verified"],
            "schema_fingerprint": input_manifest["schema_fingerprint"],
        },
        "output": {
            "files": len(committed_records),
            "rows": output_rows,
            "bytes": sum(record["bytes"] for record in committed_records),
            "planned_shard_rows": shard_sizes,
            "schema_fingerprint": input_manifest["schema_fingerprint"],
            "sha256_manifest": "files.sha256",
            "file_records": committed_records,
        },
        "row_closure": {
            "expected": config.expected_total_rows,
            "input": input_manifest["total_rows"],
            "output": output_rows,
            "closed": True,
        },
        "exact_gather_verification": _gather_verification_record(
            config.gather_verification,
            flushes=gather_verification_flushes,
            rows=gather_verification_rows,
            peak_child_bytes_max=gather_verification_peak_child_bytes_max,
        ),
        "final_input_reverification": {
            "status": (
                "passed"
                if input_manifest["sha256_verified"]
                else "not_requested"
            ),
            "files": input_manifest["total_files"],
            "sha256_verified": input_manifest["sha256_verified"],
        },
    }
    manifest["dataset_fingerprint"] = fingerprint(
        {
            "run_identity": run_identity,
            "files": [
                (record["name"], record["rows"], record["sha256"])
                for record in committed_records
            ],
            "exact_gather_verification": manifest["exact_gather_verification"],
            "final_input_reverification": manifest["final_input_reverification"],
        }
    )
    atomic_write_json(final_manifest_path, manifest)
    state = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "run_identity": run_identity,
        "resume_strategy": RESUME_STRATEGY,
        "completed_output_shards": len(committed_records),
        "completed_output_rows": output_rows,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
    }
    atomic_write_json(audit_dir / "run_state.json", state)
    identity = load_json(identity_path)
    identity["status"] = "complete"
    atomic_write_json(identity_path, identity)
    return manifest


def reshard_ultramsdata(
    config: SafeReshardConfig,
    *,
    max_new_shards_this_run: int | None = None,
) -> dict[str, Any]:
    """Build or safely resume an audited deterministic Ae3 reshuffle."""
    _validate_config(config)
    if max_new_shards_this_run is not None and max_new_shards_this_run <= 0:
        raise ValueError("max_new_shards_this_run must be positive")
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "_audit"
    (audit_dir / "progress").mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(audit_dir):
        return _reshard_ultramsdata_locked(
            config, max_new_shards_this_run=max_new_shards_this_run
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--expected-total-rows", type=int, required=True)
    parser.add_argument("--expected-input-files", type=int)
    parser.add_argument("--expected-schema-fingerprint", required=True)
    parser.add_argument("--target-rows-per-shard", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-list-length", type=int, default=500)
    parser.add_argument("--minimum-positive-mz", type=int, default=3)
    parser.add_argument(
        "--gather-verification",
        choices=sorted(GATHER_VERIFICATION_MODES),
        default="all_values",
    )
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--input-sha256-manifest", type=Path)
    parser.add_argument("--max-new-shards-this-run", type=int)
    args = parser.parse_args()
    config = SafeReshardConfig(
        source_dir=args.src,
        output_dir=args.dst,
        expected_total_rows=args.expected_total_rows,
        expected_input_files=args.expected_input_files,
        expected_schema_fingerprint=args.expected_schema_fingerprint,
        target_rows_per_shard=args.target_rows_per_shard,
        seed=args.seed,
        expected_list_length=args.expected_list_length,
        minimum_positive_mz=args.minimum_positive_mz,
        gather_verification=args.gather_verification,
        compression=args.compression,
        input_sha256_manifest=args.input_sha256_manifest,
    )
    result = reshard_ultramsdata(
        config, max_new_shards_this_run=args.max_new_shards_this_run
    )
    print(f"status={result['status']} output={config.output_dir}")
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
