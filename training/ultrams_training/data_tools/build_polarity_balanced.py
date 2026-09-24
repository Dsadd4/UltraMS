#!/usr/bin/env python3
"""Build an auditable, exactly balanced polarity dataset from UltraMSdata.

All ``polarity == 0`` rows are retained. Exactly the same number of
``polarity == 1`` rows are sampled uniformly without replacement using seed
42 by default. Sampling is streamed with batch-level hypergeometric draws, so
it never constructs one Python tuple per positive row. Selection sidecars and
output shards are independently atomic and resumable.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
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
        aggregate_stats,
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
        aggregate_stats,
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


@dataclass(frozen=True)
class PolarityConfig:
    source_dir: Path
    output_dir: Path
    dataset_name: str = "ae3_polarity_balanced_reconstructed_v1"
    seed: int = 42
    world_size: int = 6
    output_shards: int | None = None
    target_rows_per_shard: int = 500_000
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
        return result


def _validate_config(config: PolarityConfig) -> None:
    if (
        config.world_size <= 0
        or config.batch_size <= 0
        or config.target_rows_per_shard <= 0
    ):
        raise ValueError(
            "world_size, batch_size and target_rows_per_shard must be positive"
        )
    if config.output_shards is not None and config.output_shards <= 0:
        raise ValueError("output_shards must be positive")
    if config.source_dir.resolve() == config.output_dir.resolve():
        raise ValueError("source and output directories must differ")


def _arrow_positions(batch: pa.RecordBatch, value: int) -> np.ndarray:
    polarity = batch.column(batch.schema.get_field_index("polarity"))
    mask = pc.fill_null(pc.equal(polarity, value), False).to_numpy(zero_copy_only=False)
    return np.flatnonzero(mask)


def _count_polarity_file(path: Path, batch_size: int) -> dict[str, int]:
    counts = {"negative_0": 0, "positive_1": 0, "other_or_null": 0, "rows": 0}
    parquet = pq.ParquetFile(path)
    if "polarity" not in parquet.schema_arrow.names:
        raise ValueError(f"missing polarity column in {path}")
    for batch in parquet.iter_batches(columns=["polarity"], batch_size=batch_size):
        negative = len(_arrow_positions(batch, 0))
        positive = len(_arrow_positions(batch, 1))
        counts["rows"] += batch.num_rows
        counts["negative_0"] += negative
        counts["positive_1"] += positive
        counts["other_or_null"] += batch.num_rows - negative - positive
    return counts


def _build_counts(
    source_dir: Path,
    audit_dir: Path,
    input_manifest: dict[str, Any],
    run_identity: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    count_dir = audit_dir / "counts"
    count_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in input_manifest["files"]:
        path = count_dir / f"source_{item['index']:04d}.json"
        if path.exists():
            record = load_json(path)
            if (
                record.get("run_identity") != run_identity
                or record.get("source_sha256") != item["sha256"]
            ):
                raise RuntimeError(f"count checkpoint mismatch: {path}")
        else:
            counts = _count_polarity_file(source_dir / item["name"], batch_size)
            if counts["rows"] != item["rows"]:
                raise RuntimeError(f"row count changed while scanning {item['name']}")
            record = {
                "run_identity": run_identity,
                "source_index": item["index"],
                "source_name": item["name"],
                "source_sha256": item["sha256"],
                "counts": counts,
            }
            atomic_write_json(path, record)
        records.append(record)
    totals = {
        key: sum(record["counts"][key] for record in records)
        for key in ("rows", "negative_0", "positive_1", "other_or_null")
    }
    if totals["negative_0"] <= 0:
        raise ValueError("source dataset contains no polarity=0 rows")
    if totals["positive_1"] < totals["negative_0"]:
        raise ValueError("not enough positive rows to balance all negative rows")
    atomic_write_json(
        audit_dir / "polarity_counts.json",
        {"run_identity": run_identity, "files": records, "totals": totals},
    )
    return records, totals


def _rng_state_fingerprint(
    state: dict[str, Any], remaining_positive: int, remaining_needed: int
) -> str:
    return fingerprint(
        {
            "rng_state": state,
            "remaining_positive": remaining_positive,
            "remaining_needed": remaining_needed,
        }
    )


def _read_selection_sidecar(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        indices = archive["indices"].astype(np.int64, copy=False)
        metadata = json.loads(str(archive["metadata"].item()))
    return indices, metadata


def _write_selection_sidecar(
    path: Path, indices: np.ndarray, metadata: dict[str, Any]
) -> dict[str, Any]:
    metadata = dict(metadata)
    metadata["indices_sha256"] = hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices=indices.astype(np.int64, copy=False),
            metadata=np.array(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    if path.exists():
        if (
            path.stat().st_size != temporary.stat().st_size
            or sha256_file(path) != digest
        ):
            temporary.unlink()
            raise RuntimeError(
                f"selection sidecar conflicts with deterministic rebuild: {path}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "sha256": digest}


def _select_one_file(
    source: Path,
    source_item: dict[str, Any],
    rng: np.random.Generator,
    remaining_positive: int,
    remaining_needed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], int, int]:
    selected_parts: list[np.ndarray] = []
    local_offset = 0
    selected_negative = 0
    selected_positive = 0
    state_before = rng.bit_generator.state
    before_fingerprint = _rng_state_fingerprint(
        state_before, remaining_positive, remaining_needed
    )

    for batch in pq.ParquetFile(source).iter_batches(
        columns=["polarity"], batch_size=batch_size
    ):
        negative_positions = _arrow_positions(batch, 0)
        positive_positions = _arrow_positions(batch, 1)
        positive_count = len(positive_positions)
        if positive_count > remaining_positive:
            raise RuntimeError("positive population accounting underflow")
        if positive_count:
            draw_count = int(
                rng.hypergeometric(
                    remaining_needed,
                    remaining_positive - remaining_needed,
                    positive_count,
                )
            )
            if draw_count == positive_count:
                chosen_positive = positive_positions
            elif draw_count:
                chosen_ordinals = np.sort(
                    rng.choice(positive_count, size=draw_count, replace=False)
                )
                chosen_positive = positive_positions[chosen_ordinals]
            else:
                chosen_positive = np.empty(0, dtype=np.int64)
            remaining_positive -= positive_count
            remaining_needed -= draw_count
        else:
            chosen_positive = np.empty(0, dtype=np.int64)

        selected_negative += len(negative_positions)
        selected_positive += len(chosen_positive)
        selected = np.concatenate((negative_positions, chosen_positive)) + local_offset
        if len(selected):
            selected_parts.append(np.sort(selected))
        local_offset += batch.num_rows

    if local_offset != source_item["rows"]:
        raise RuntimeError(f"row count changed while selecting {source}")
    selected_indices = (
        np.concatenate(selected_parts)
        if selected_parts
        else np.empty(0, dtype=np.int64)
    )
    metadata = {
        "source_index": source_item["index"],
        "source_name": source_item["name"],
        "source_sha256": source_item["sha256"],
        "state_before_fingerprint": before_fingerprint,
        "rng_state_after": rng.bit_generator.state,
        "remaining_positive_after": remaining_positive,
        "remaining_needed_after": remaining_needed,
        "selected_rows": len(selected_indices),
        "selected_negative": selected_negative,
        "selected_positive": selected_positive,
        "indices_sha256": hashlib.sha256(
            selected_indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    return selected_indices, metadata, remaining_positive, remaining_needed


def _build_selection_sidecars(
    config: PolarityConfig,
    input_manifest: dict[str, Any],
    totals: dict[str, int],
    audit_dir: Path,
    run_identity: str,
    max_files_this_run: int | None,
) -> tuple[list[dict[str, Any]], bool]:
    selection_dir = audit_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    remaining_positive = totals["positive_1"]
    remaining_needed = totals["negative_0"]
    records: list[dict[str, Any]] = []
    processed_now = 0

    for item in input_manifest["files"]:
        sidecar = selection_dir / f"source_{item['index']:04d}.npz"
        before_fingerprint = _rng_state_fingerprint(
            rng.bit_generator.state, remaining_positive, remaining_needed
        )
        if sidecar.exists():
            indices, metadata = _read_selection_sidecar(sidecar)
            if metadata.get("state_before_fingerprint") != before_fingerprint:
                raise RuntimeError(f"selection resume state mismatch: {sidecar}")
            if metadata.get("source_sha256") != item["sha256"]:
                raise RuntimeError(f"selection source mismatch: {sidecar}")
            if len(indices) != metadata["selected_rows"]:
                raise RuntimeError(f"selection sidecar row mismatch: {sidecar}")
            rng.bit_generator.state = metadata["rng_state_after"]
            remaining_positive = metadata["remaining_positive_after"]
            remaining_needed = metadata["remaining_needed_after"]
            sidecar_record = {
                "name": sidecar.name,
                "bytes": sidecar.stat().st_size,
                "sha256": sha256_file(sidecar),
            }
        else:
            if max_files_this_run is not None and processed_now >= max_files_this_run:
                return records, False
            indices, metadata, remaining_positive, remaining_needed = _select_one_file(
                config.source_dir.resolve() / item["name"],
                item,
                rng,
                remaining_positive,
                remaining_needed,
                config.batch_size,
            )
            sidecar_record = _write_selection_sidecar(sidecar, indices, metadata)
            processed_now += 1
        records.append({**metadata, "sidecar": sidecar_record})
        atomic_write_json(
            audit_dir / "selection_state.json",
            {
                "run_identity": run_identity,
                "status": "in_progress",
                "completed_source_files": len(records),
                "total_source_files": input_manifest["total_files"],
                "remaining_positive": remaining_positive,
                "remaining_needed": remaining_needed,
            },
        )

    if remaining_needed != 0 or remaining_positive != 0:
        raise RuntimeError("selection did not consume the complete positive population")
    selected_negative = sum(record["selected_negative"] for record in records)
    selected_positive = sum(record["selected_positive"] for record in records)
    if (
        selected_negative != totals["negative_0"]
        or selected_positive != totals["negative_0"]
    ):
        raise RuntimeError("balanced selection accounting failed")
    selection_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_identity": run_identity,
        "algorithm": "streamed_batch_hypergeometric_uniform_without_replacement",
        "seed": config.seed,
        "selected_negative": selected_negative,
        "selected_positive": selected_positive,
        "selected_total": selected_negative + selected_positive,
        "files": records,
    }
    selection_manifest["selection_fingerprint"] = fingerprint(
        {
            "run_identity": run_identity,
            "files": [
                (record["source_name"], record["indices_sha256"]) for record in records
            ],
        }
    )
    atomic_write_json(audit_dir / "selection_manifest.json", selection_manifest)
    atomic_write_json(
        audit_dir / "selection_state.json",
        {
            "run_identity": run_identity,
            "status": "complete",
            "completed_source_files": len(records),
            "total_source_files": input_manifest["total_files"],
            "remaining_positive": 0,
            "remaining_needed": 0,
            "selection_fingerprint": selection_manifest["selection_fingerprint"],
        },
    )
    return records, True


def _resolve_output_shards(config: PolarityConfig, total_rows: int) -> int:
    if config.output_shards is not None:
        count = config.output_shards
    else:
        nominal = max(1, math.ceil(total_rows / config.target_rows_per_shard))
        count = math.ceil(nominal / config.world_size) * config.world_size
    if count > total_rows:
        raise ValueError("output_shards cannot exceed selected rows")
    return count


def _shard_ranges(total_rows: int, shards: int) -> list[tuple[int, int]]:
    base, extra = divmod(total_rows, shards)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        size = base + (1 if index < extra else 0)
        ranges.append((start, start + size))
        start += size
    assert start == total_rows
    return ranges


def _write_output_shard(
    output: Path,
    start: int,
    end: int,
    source_dir: Path,
    input_items: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
    selected_prefix: list[int],
    selection_dir: Path,
    compression: str,
) -> dict[str, Any]:
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    stats = empty_stats()
    source_slices: list[dict[str, int]] = []
    schema: pa.Schema | None = None
    position = start
    source_index = max(0, bisect.bisect_right(selected_prefix, start) - 1)
    try:
        while position < end:
            while (
                source_index + 1 < len(selected_prefix)
                and selected_prefix[source_index + 1] <= position
            ):
                source_index += 1
            source_start = selected_prefix[source_index]
            source_end = selected_prefix[source_index + 1]
            take_end = min(end, source_end)
            local_start = position - source_start
            local_end = take_end - source_start
            sidecar_path = (
                selection_dir / selection_records[source_index]["sidecar"]["name"]
            )
            selected_indices, _ = _read_selection_sidecar(sidecar_path)
            chosen = selected_indices[local_start:local_end]
            table = pq.read_table(source_dir / input_items[source_index]["name"])
            chunk = table.take(pa.array(chosen, type=pa.int64()))
            if schema is None:
                schema = chunk.schema
                writer = pq.ParquetWriter(
                    temporary,
                    schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif not schema.equals(chunk.schema, check_metadata=True):
                raise RuntimeError("schema changed while writing selected rows")
            assert writer is not None
            writer.write_table(chunk)
            add_stats(stats, summarize_arrow(chunk))
            source_slices.append(
                {
                    "source_index": source_index,
                    "selected_start": int(local_start),
                    "selected_end_exclusive": int(local_end),
                }
            )
            position = take_end
            source_index += 1
    finally:
        if writer is not None:
            writer.close()

    if stats["rows"] != end - start:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("output shard row accounting failed")
    digest = sha256_file(temporary)
    if output.exists():
        if (
            output.stat().st_size != temporary.stat().st_size
            or sha256_file(output) != digest
        ):
            temporary.unlink()
            raise RuntimeError(
                f"untracked output conflicts with deterministic rebuild: {output}"
            )
        temporary.unlink()
    else:
        os.replace(temporary, output)
        fsync_directory(output.parent)
    stat = output.stat()
    return {
        "name": output.name,
        "bytes": stat.st_size,
        "rows": stats["rows"],
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "schema_fingerprint": schema_fingerprint(pq.ParquetFile(output).schema_arrow),
        "selected_global_start": start,
        "selected_global_end_exclusive": end,
        "source_slices": source_slices,
        "stats": stats,
    }


def build_polarity_balanced_dataset(
    config: PolarityConfig,
    *,
    max_selection_files_this_run: int | None = None,
    max_output_shards_this_run: int | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    source_dir = config.source_dir.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = build_or_validate_input_manifest(
        source_dir,
        audit_dir,
        config.input_sha256_manifest,
        config.rehash_inputs,
    )
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

    _, totals = _build_counts(
        source_dir, audit_dir, input_manifest, run_identity, config.batch_size
    )
    selection_records, selection_complete = _build_selection_sidecars(
        config,
        input_manifest,
        totals,
        audit_dir,
        run_identity,
        max_selection_files_this_run,
    )
    if not selection_complete:
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_identity": run_identity,
            "status": "in_progress",
            "phase": "selection",
        }

    selected_counts = [record["selected_rows"] for record in selection_records]
    selected_prefix = [0]
    for count in selected_counts:
        selected_prefix.append(selected_prefix[-1] + count)
    total_selected = selected_prefix[-1]
    output_shards = _resolve_output_shards(config, total_selected)
    ranges = _shard_ranges(total_selected, output_shards)
    progress_dir = audit_dir / "output_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    processed_now = 0
    records: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(ranges):
        progress_path = progress_dir / f"shard_{index:04d}.json"
        output = output_dir / f"polarity_{index:04d}.parquet"
        if progress_path.exists():
            progress = load_json(progress_path)
            if progress.get("run_identity") != run_identity:
                raise RuntimeError(
                    f"output progress identity mismatch: {progress_path}"
                )
            validate_completed_output(output, progress["output"])
            records.append(progress["output"])
            continue
        if (
            max_output_shards_this_run is not None
            and processed_now >= max_output_shards_this_run
        ):
            break
        record = _write_output_shard(
            output,
            start,
            end,
            source_dir,
            input_manifest["files"],
            selection_records,
            selected_prefix,
            audit_dir / "selection",
            config.compression,
        )
        atomic_write_json(
            progress_path,
            {"run_identity": run_identity, "output": record},
        )
        records.append(record)
        processed_now += 1

    all_progress = sorted(progress_dir.glob("shard_*.json"))
    if len(all_progress) != output_shards:
        state = {
            "manifest_version": MANIFEST_VERSION,
            "run_identity": run_identity,
            "status": "in_progress",
            "phase": "output",
            "completed_output_shards": len(all_progress),
            "total_output_shards": output_shards,
        }
        atomic_write_json(audit_dir / "run_state.json", state)
        return state

    records = [load_json(path)["output"] for path in all_progress]
    output_stats = aggregate_stats(record["stats"] for record in records)
    expected_per_class = totals["negative_0"]
    if output_stats["rows"] != 2 * expected_per_class:
        raise RuntimeError("final output row accounting failed")
    if output_stats["polarity"]["negative_0"] != expected_per_class:
        raise RuntimeError("not every negative row reached the output")
    if output_stats["polarity"]["positive_1"] != expected_per_class:
        raise RuntimeError("positive output is not exactly balanced")
    if output_stats["polarity"]["other_or_null"] != 0:
        raise RuntimeError("unexpected non-binary polarity rows reached the output")

    write_hash_manifest(output_dir / "files.sha256", records)
    selection_manifest = load_json(audit_dir / "selection_manifest.json")
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
        "source_polarity_counts": totals,
        "selection": {
            "manifest": "_audit/selection_manifest.json",
            "algorithm": selection_manifest["algorithm"],
            "seed": config.seed,
            "fingerprint": selection_manifest["selection_fingerprint"],
        },
        "output": {
            "files": output_shards,
            "world_size": config.world_size,
            "files_divisible_by_world_size": output_shards % config.world_size == 0,
            "bytes": sum(record["bytes"] for record in records),
            "stats": output_stats,
            "file_records": records,
            "sha256_manifest": "files.sha256",
        },
    }
    manifest["dataset_fingerprint"] = fingerprint(
        {
            "run_identity": run_identity,
            "selection": selection_manifest["selection_fingerprint"],
            "files": [
                (record["name"], record["rows"], record["sha256"]) for record in records
            ],
        }
    )
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        audit_dir / "run_state.json",
        {
            "run_identity": run_identity,
            "status": "complete",
            "completed_output_shards": output_shards,
            "total_output_shards": output_shards,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        },
    )
    identity = load_json(identity_path)
    identity["status"] = "complete"
    atomic_write_json(identity_path, identity)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--dataset-name", default="ae3_polarity_balanced_reconstructed_v1"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--output-shards", type=int)
    parser.add_argument("--target-rows-per-shard", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--input-sha256-manifest", type=Path)
    parser.add_argument("--rehash-inputs", action="store_true")
    parser.add_argument("--max-selection-files-this-run", type=int)
    parser.add_argument("--max-output-shards-this-run", type=int)
    args = parser.parse_args()
    config = PolarityConfig(
        source_dir=args.src,
        output_dir=args.dst,
        dataset_name=args.dataset_name,
        seed=args.seed,
        world_size=args.world_size,
        output_shards=args.output_shards,
        target_rows_per_shard=args.target_rows_per_shard,
        batch_size=args.batch_size,
        compression=args.compression,
        input_sha256_manifest=args.input_sha256_manifest,
        rehash_inputs=args.rehash_inputs,
    )
    result = build_polarity_balanced_dataset(
        config,
        max_selection_files_this_run=args.max_selection_files_this_run,
        max_output_shards_this_run=args.max_output_shards_this_run,
    )
    print(f"status={result['status']} output={config.output_dir}")
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
