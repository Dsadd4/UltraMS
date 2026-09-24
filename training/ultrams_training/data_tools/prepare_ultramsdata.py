#!/usr/bin/env python3
"""Materialize an auditable UltraMSdata dataset.

Inputs are read in lexical basename order as one global row stream. The
default audited rule removes global 200,000-row partitions 798, 800, 801,
802, 810 and 811, then removes rows with ``precursor_mz > 1000``. Kept rows
retain global order. A formal run uses the audited kept-row count to create
804 near-equal shards, giving all six DDP ranks exactly the same row count.

Recovery checkpoints are committed only after an output shard is atomically
renamed. Each checkpoint records the exact source file/row cursor and all
cumulative statistics, so an interrupted partial shard is safely replayed.
"""

from __future__ import annotations

import argparse
import os
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
        add_stats,
        atomic_write_json,
        build_or_validate_input_manifest,
        empty_stats,
        fingerprint,
        fsync_directory,
        load_json,
        schema_fingerprint,
        sha256_file,
        summarize_arrow,
        validate_completed_output,
        write_hash_manifest,
    )
except ImportError:  # pragma: no cover - direct script execution
    from _audit_utils import (  # type: ignore
        MANIFEST_VERSION,
        add_stats,
        atomic_write_json,
        build_or_validate_input_manifest,
        empty_stats,
        fingerprint,
        fsync_directory,
        load_json,
        schema_fingerprint,
        sha256_file,
        summarize_arrow,
        validate_completed_output,
        write_hash_manifest,
    )


DEFAULT_BAD_PARTITIONS = (798, 800, 801, 802, 810, 811)


@dataclass(frozen=True)
class CleanConfig:
    source_dir: Path
    output_dir: Path
    dataset_name: str = "ae3_clean_reconstructed_v1"
    partition_size: int = 200_000
    drop_partitions: tuple[int, ...] = DEFAULT_BAD_PARTITIONS
    precursor_column: str = "precursor_mz"
    max_precursor_mz: float = 1000.0
    output_rows_per_shard: int = 200_000
    output_shards: int | None = None
    expected_output_rows: int | None = None
    world_size: int = 6
    require_divisible_shards: bool = True
    batch_size: int = 65_536
    compression: str = "zstd"
    input_sha256_manifest: Path | None = None
    rehash_inputs: bool = False

    def stable_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_dir"] = str(self.source_dir.resolve())
        result["output_dir"] = str(self.output_dir.resolve())
        result["input_sha256_manifest"] = (
            str(self.input_sha256_manifest.resolve())
            if self.input_sha256_manifest
            else None
        )
        result["drop_partitions"] = list(self.drop_partitions)
        return result


def _validate_config(config: CleanConfig) -> None:
    if config.dataset_name == "shards_clean_v6":
        raise ValueError(
            "the reconstructed dataset must not reuse the historical shards_clean_v6 name"
        )
    positive = (
        config.partition_size,
        config.output_rows_per_shard,
        config.world_size,
        config.batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError(
            "partition, output-shard, world-size and batch sizes must be positive"
        )
    if (config.output_shards is None) != (config.expected_output_rows is None):
        raise ValueError(
            "output_shards and expected_output_rows must be provided together"
        )
    if config.output_shards is not None:
        if config.output_shards <= 0 or config.expected_output_rows <= 0:
            raise ValueError("output_shards and expected_output_rows must be positive")
        if config.require_divisible_shards and config.output_shards % config.world_size:
            raise ValueError("output_shards must be divisible by world_size")
    if any(index < 0 for index in config.drop_partitions):
        raise ValueError("drop partition indices must be nonnegative")
    if config.source_dir.resolve() == config.output_dir.resolve():
        raise ValueError("source and output directories must differ")


def resolve_output_shard_rows(config: CleanConfig) -> list[int] | None:
    """Return equalized formal shard sizes, or None for fixed-size streaming."""
    if config.output_shards is None:
        return None
    base, extra = divmod(config.expected_output_rows, config.output_shards)
    if base <= 0:
        raise ValueError("output_shards cannot exceed expected_output_rows")
    return [base + (index < extra) for index in range(config.output_shards)]


def _empty_removed() -> dict[str, int]:
    return {
        "bad_partition": 0,
        "precursor_mz_above_limit_outside_bad_partition": 0,
        "bad_partition_and_precursor_mz_above_limit": 0,
        "total": 0,
    }


def _add_removed(target: dict[str, int], addition: dict[str, int]) -> None:
    for key in target:
        target[key] += addition[key]


def _filter_segment(
    batch: pa.RecordBatch,
    global_start: int,
    config: CleanConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    rows = batch.num_rows
    global_ids = np.arange(global_start, global_start + rows, dtype=np.int64)
    partitions = global_ids // config.partition_size
    bad = np.isin(partitions, config.drop_partitions)
    precursor = batch.column(batch.schema.get_field_index(config.precursor_column))
    above = pc.fill_null(
        pc.greater(precursor, config.max_precursor_mz), False
    ).to_numpy(zero_copy_only=False)
    overlap = int(np.logical_and(bad, above).sum())
    bad_count = int(bad.sum())
    precursor_only = int(np.logical_and(~bad, above).sum())
    removed = {
        "bad_partition": bad_count,
        "precursor_mz_above_limit_outside_bad_partition": precursor_only,
        "bad_partition_and_precursor_mz_above_limit": overlap,
        "total": bad_count + precursor_only,
    }
    return np.logical_not(np.logical_or(bad, above)), removed


def _load_resume_state(
    output_dir: Path,
    audit_dir: Path,
    run_identity: str,
) -> tuple[
    list[dict[str, Any]], int, int, dict[str, Any], dict[str, Any], dict[str, int]
]:
    progress_paths = sorted((audit_dir / "output_progress").glob("shard_*.json"))
    records: list[dict[str, Any]] = []
    source_index = 0
    source_row_offset = 0
    input_stats = empty_stats()
    output_stats = empty_stats()
    removed = _empty_removed()
    for expected_index, path in enumerate(progress_paths):
        progress = load_json(path)
        if progress.get("run_identity") != run_identity:
            raise RuntimeError(f"output progress identity mismatch: {path}")
        if progress.get("output_index") != expected_index:
            raise RuntimeError("output progress is not a contiguous shard prefix")
        record = progress["output"]
        validate_completed_output(output_dir / record["name"], record)
        records.append(record)
        cursor = progress["next_source_cursor"]
        source_index = cursor["source_index"]
        source_row_offset = cursor["row_offset"]
        input_stats = progress["cumulative_input_stats"]
        output_stats = progress["cumulative_output_stats"]
        removed = progress["cumulative_removed"]
    return records, source_index, source_row_offset, input_stats, output_stats, removed


def _commit_output_shard(
    output_dir: Path,
    audit_dir: Path,
    run_identity: str,
    output_index: int,
    temporary: Path,
    shard_stats: dict[str, Any],
    source_index: int,
    source_row_offset: int,
    cumulative_input_stats: dict[str, Any],
    cumulative_output_stats: dict[str, Any],
    cumulative_removed: dict[str, int],
) -> dict[str, Any]:
    output = output_dir / f"clean_{output_index:04d}.parquet"
    digest = sha256_file(temporary)
    temporary_stat = temporary.stat()
    if output.exists():
        if (
            output.stat().st_size != temporary_stat.st_size
            or sha256_file(output) != digest
        ):
            temporary.unlink()
            raise RuntimeError(
                f"untracked output conflicts with deterministic replay: {output}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, output)
        fsync_directory(output.parent)
    stat = output.stat()
    record = {
        "name": output.name,
        "bytes": stat.st_size,
        "rows": shard_stats["rows"],
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "schema_fingerprint": schema_fingerprint(pq.ParquetFile(output).schema_arrow),
        "stats": shard_stats,
    }
    progress = {
        "manifest_version": MANIFEST_VERSION,
        "run_identity": run_identity,
        "output_index": output_index,
        "output": record,
        "next_source_cursor": {
            "source_index": source_index,
            "row_offset": source_row_offset,
        },
        "cumulative_input_stats": cumulative_input_stats,
        "cumulative_output_stats": cumulative_output_stats,
        "cumulative_removed": cumulative_removed,
    }
    atomic_write_json(
        audit_dir / "output_progress" / f"shard_{output_index:04d}.json",
        progress,
    )
    return record


def materialize_clean_dataset(
    config: CleanConfig,
    *,
    max_output_shards_this_run: int | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    source_dir = config.source_dir.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "_audit"
    (audit_dir / "output_progress").mkdir(parents=True, exist_ok=True)
    input_manifest = build_or_validate_input_manifest(
        source_dir,
        audit_dir,
        config.input_sha256_manifest,
        config.rehash_inputs,
    )
    if (
        config.precursor_column
        not in pq.ParquetFile(
            source_dir / input_manifest["files"][0]["name"]
        ).schema_arrow.names
    ):
        raise ValueError(f"missing required column {config.precursor_column!r}")

    stable_config = config.stable_dict()
    stable_config.pop("rehash_inputs", None)
    stable_config.pop("input_sha256_manifest", None)
    identity_config = dict(stable_config)
    identity_config.pop("source_dir", None)
    identity_config.pop("output_dir", None)
    run_identity = fingerprint(
        {"config": identity_config, "input": input_manifest["content_fingerprint"]}
    )
    identity_path = audit_dir / "run_identity.json"
    if (
        identity_path.exists()
        and load_json(identity_path).get("run_identity") != run_identity
    ):
        raise RuntimeError("configuration changed; use a new output directory")
    atomic_write_json(
        identity_path,
        {
            "run_identity": run_identity,
            "config": stable_config,
            "status": "in_progress",
        },
    )

    (
        output_records,
        source_index,
        source_row_offset,
        cumulative_input_stats,
        cumulative_output_stats,
        cumulative_removed,
    ) = _load_resume_state(output_dir, audit_dir, run_identity)
    output_index = len(output_records)
    formal_shard_rows = resolve_output_shard_rows(config)
    processed_now = 0
    writer: pq.ParquetWriter | None = None
    temporary: Path | None = None
    shard_stats = empty_stats()
    shard_rows = 0
    schema = pq.ParquetFile(
        source_dir / input_manifest["files"][0]["name"]
    ).schema_arrow

    def ensure_writer() -> pq.ParquetWriter:
        nonlocal writer, temporary
        if writer is None:
            temporary = (
                output_dir / f".clean_{output_index:04d}.parquet.partial.{os.getpid()}"
            )
            temporary.unlink(missing_ok=True)
            writer = pq.ParquetWriter(
                temporary,
                schema,
                compression=config.compression,
                use_dictionary=True,
                write_statistics=True,
            )
        return writer

    def finish_shard(next_source_index: int, next_row_offset: int) -> None:
        nonlocal writer, temporary, shard_stats, shard_rows, output_index, processed_now
        if writer is None or temporary is None or shard_rows <= 0:
            raise RuntimeError("attempted to finish an empty output shard")
        writer.close()
        writer = None
        record = _commit_output_shard(
            output_dir,
            audit_dir,
            run_identity,
            output_index,
            temporary,
            shard_stats,
            next_source_index,
            next_row_offset,
            cumulative_input_stats,
            cumulative_output_stats,
            cumulative_removed,
        )
        output_records.append(record)
        temporary = None
        shard_stats = empty_stats()
        shard_rows = 0
        output_index += 1
        processed_now += 1

    stopped = False
    try:
        while source_index < input_manifest["total_files"] and not stopped:
            item = input_manifest["files"][source_index]
            parquet = pq.ParquetFile(source_dir / item["name"])
            batch_file_start = 0
            for raw_batch in parquet.iter_batches(batch_size=config.batch_size):
                batch_file_end = batch_file_start + raw_batch.num_rows
                if batch_file_end <= source_row_offset:
                    batch_file_start = batch_file_end
                    continue
                slice_start = max(0, source_row_offset - batch_file_start)
                segment = raw_batch.slice(slice_start)
                segment_local_start = batch_file_start + slice_start
                while segment.num_rows:
                    global_start = item["global_start"] + segment_local_start
                    keep, _ = _filter_segment(segment, global_start, config)
                    kept_positions = np.flatnonzero(keep)
                    if len(kept_positions):
                        if formal_shard_rows is not None and output_index >= len(
                            formal_shard_rows
                        ):
                            raise RuntimeError(
                                "kept-row count exceeds expected_output_rows"
                            )
                        target_rows = (
                            formal_shard_rows[output_index]
                            if formal_shard_rows is not None
                            else config.output_rows_per_shard
                        )
                        remaining = target_rows - shard_rows
                        if len(kept_positions) > remaining:
                            consume_rows = int(kept_positions[remaining - 1]) + 1
                        else:
                            consume_rows = segment.num_rows
                    else:
                        consume_rows = segment.num_rows
                        target_rows = None
                    consumed = segment.slice(0, consume_rows)
                    consumed_keep, consumed_removed = _filter_segment(
                        consumed, global_start, config
                    )
                    add_stats(cumulative_input_stats, summarize_arrow(consumed))
                    _add_removed(cumulative_removed, consumed_removed)
                    if consumed_keep.any():
                        filtered = consumed.filter(pa.array(consumed_keep))
                        ensure_writer().write_batch(filtered)
                        filtered_stats = summarize_arrow(filtered)
                        add_stats(shard_stats, filtered_stats)
                        add_stats(cumulative_output_stats, filtered_stats)
                        shard_rows += filtered.num_rows
                    source_row_offset = segment_local_start + consume_rows
                    segment = segment.slice(consume_rows)
                    segment_local_start += consume_rows

                    if target_rows is not None and shard_rows == target_rows:
                        finish_shard(source_index, source_row_offset)
                        if (
                            max_output_shards_this_run is not None
                            and processed_now >= max_output_shards_this_run
                        ):
                            stopped = True
                            break
                if stopped:
                    break
                batch_file_start = batch_file_end
            if not stopped:
                if source_row_offset != item["rows"]:
                    raise RuntimeError(f"source cursor mismatch after {item['name']}")
                source_index += 1
                source_row_offset = 0

        if stopped:
            state = {
                "manifest_version": MANIFEST_VERSION,
                "run_identity": run_identity,
                "status": "in_progress",
                "completed_output_shards": len(output_records),
                "next_source_cursor": {
                    "source_index": source_index,
                    "row_offset": source_row_offset,
                },
            }
            atomic_write_json(audit_dir / "run_state.json", state)
            return state

        if shard_rows:
            finish_shard(source_index, source_row_offset)
        elif writer is not None:
            raise RuntimeError("writer exists without rows")
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()

    if source_index != input_manifest["total_files"] or source_row_offset != 0:
        raise RuntimeError("final source cursor did not reach end of dataset")
    if cumulative_input_stats["rows"] != input_manifest["total_rows"]:
        raise RuntimeError("not every source row was accounted for")
    if (
        cumulative_input_stats["rows"] - cumulative_removed["total"]
        != cumulative_output_stats["rows"]
    ):
        raise RuntimeError("global row accounting failed")
    if formal_shard_rows is not None:
        if cumulative_output_stats["rows"] != config.expected_output_rows:
            raise RuntimeError(
                "kept-row count does not match expected_output_rows: "
                f"{cumulative_output_stats['rows']} != {config.expected_output_rows}"
            )
        expected_files = config.output_shards
    else:
        expected_files = (
            cumulative_output_stats["rows"] + config.output_rows_per_shard - 1
        ) // config.output_rows_per_shard
    if len(output_records) != expected_files:
        raise RuntimeError("output shard count does not match kept-row count")
    if config.require_divisible_shards and len(output_records) % config.world_size:
        raise RuntimeError(
            f"output shard count {len(output_records)} is not divisible by world_size={config.world_size}"
        )
    if formal_shard_rows is not None:
        actual_rows = [record["rows"] for record in output_records]
        if actual_rows != formal_shard_rows:
            raise RuntimeError("formal output shards are not evenly balanced")
    else:
        for record in output_records[:-1]:
            if record["rows"] != config.output_rows_per_shard:
                raise RuntimeError("a non-final output shard is short")

    write_hash_manifest(output_dir / "files.sha256", output_records)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "dataset_name": config.dataset_name,
        "run_identity": run_identity,
        "config": stable_config,
        "input_manifest": "_audit/input_manifest.json",
        "input_content_fingerprint": input_manifest["content_fingerprint"],
        "schema": input_manifest["schema"],
        "schema_fingerprint": input_manifest["schema_fingerprint"],
        "input": {
            "files": input_manifest["total_files"],
            "bytes": input_manifest["total_bytes"],
            "stats": cumulative_input_stats,
        },
        "removed": cumulative_removed,
        "output": {
            "files": len(output_records),
            "rows_per_full_shard": config.output_rows_per_shard,
            "balanced_shard_rows": formal_shard_rows,
            "world_size": config.world_size,
            "files_divisible_by_world_size": len(output_records) % config.world_size
            == 0,
            "bytes": sum(record["bytes"] for record in output_records),
            "stats": cumulative_output_stats,
            "file_records": output_records,
            "sha256_manifest": "files.sha256",
        },
    }
    manifest["dataset_fingerprint"] = fingerprint(
        {
            "run_identity": run_identity,
            "files": [
                (record["name"], record["rows"], record["sha256"])
                for record in output_records
            ],
        }
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        audit_dir / "run_state.json",
        {
            "run_identity": run_identity,
            "status": "complete",
            "completed_output_shards": len(output_records),
            "next_source_cursor": {"source_index": source_index, "row_offset": 0},
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        },
    )
    identity = load_json(identity_path)
    identity["status"] = "complete"
    atomic_write_json(identity_path, identity)
    return manifest


def _parse_partition_list(value: str) -> tuple[int, ...]:
    return tuple(
        sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ae3_clean_reconstructed_v1")
    parser.add_argument("--partition-size", type=int, default=200_000)
    parser.add_argument(
        "--drop-partitions",
        type=_parse_partition_list,
        default=DEFAULT_BAD_PARTITIONS,
        help="comma-separated zero-based global partition indices",
    )
    parser.add_argument("--max-precursor-mz", type=float, default=1000.0)
    parser.add_argument("--output-rows-per-shard", type=int, default=200_000)
    parser.add_argument("--output-shards", type=int)
    parser.add_argument("--expected-output-rows", type=int)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--allow-nondivisible-shards", action="store_true")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--input-sha256-manifest", type=Path)
    parser.add_argument("--rehash-inputs", action="store_true")
    parser.add_argument(
        "--max-output-shards-this-run",
        type=int,
        help="operational canary: stop cleanly after this many newly committed output shards",
    )
    args = parser.parse_args()
    config = CleanConfig(
        source_dir=args.src,
        output_dir=args.dst,
        dataset_name=args.dataset_name,
        partition_size=args.partition_size,
        drop_partitions=args.drop_partitions,
        max_precursor_mz=args.max_precursor_mz,
        output_rows_per_shard=args.output_rows_per_shard,
        output_shards=args.output_shards,
        expected_output_rows=args.expected_output_rows,
        world_size=args.world_size,
        require_divisible_shards=not args.allow_nondivisible_shards,
        batch_size=args.batch_size,
        compression=args.compression,
        input_sha256_manifest=args.input_sha256_manifest,
        rehash_inputs=args.rehash_inputs,
    )
    result = materialize_clean_dataset(
        config, max_output_shards_this_run=args.max_output_shards_this_run
    )
    print(f"status={result['status']} output={config.output_dir}")
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
