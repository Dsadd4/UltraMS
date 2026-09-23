from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MANIFEST_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def discover_parquet_files(source_dir: Path) -> list[Path]:
    # Lexical basename order is the audited reconstruction order. Do not change
    # this to natural sorting without creating a new dataset version.
    files = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix == ".parquet"
        ),
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError(f"no Parquet files found in {source_dir}")
    return files


def parse_sha256_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(
                    f"invalid SHA-256 manifest line {line_number}: {raw!r}"
                )
            name = parts[1].lstrip("* ")
            hashes[Path(name).name] = parts[0].lower()
    return hashes


def build_or_validate_input_manifest(
    source_dir: Path,
    audit_dir: Path,
    supplied_hash_manifest: Path | None = None,
    rehash_inputs: bool = False,
) -> dict[str, Any]:
    manifest_path = audit_dir / "input_manifest.json"
    files = discover_parquet_files(source_dir)
    supplied_hashes = parse_sha256_manifest(supplied_hash_manifest)
    cached = load_json(manifest_path) if manifest_path.exists() else None
    cached_by_name = (
        {item["name"]: item for item in cached.get("files", [])} if cached else {}
    )

    items: list[dict[str, Any]] = []
    common_schema: pa.Schema | None = None
    global_start = 0
    for index, path in enumerate(files):
        stat = path.stat()
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if common_schema is None:
            common_schema = schema
        elif not common_schema.equals(schema, check_metadata=True):
            raise ValueError(f"schema mismatch in {path}")
        rows = int(parquet.metadata.num_rows)
        cached_item = cached_by_name.get(path.name)
        unchanged = bool(
            cached_item
            and cached_item.get("bytes") == stat.st_size
            and cached_item.get("rows") == rows
            and cached_item.get("mtime_ns") == stat.st_mtime_ns
            and cached_item.get("device") == stat.st_dev
            and cached_item.get("inode") == stat.st_ino
            and cached_item.get("schema_fingerprint") == schema_fingerprint(schema)
        )
        if unchanged and not rehash_inputs:
            digest = cached_item["sha256"]
        elif path.name in supplied_hashes and not rehash_inputs:
            digest = supplied_hashes[path.name]
        else:
            digest = sha256_file(path)
        item = {
            "index": index,
            "name": path.name,
            "bytes": stat.st_size,
            "rows": rows,
            "global_start": global_start,
            "global_end_exclusive": global_start + rows,
            "sha256": digest,
            "schema_fingerprint": schema_fingerprint(schema),
            "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
        items.append(item)
        global_start += rows

    assert common_schema is not None
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "ordering": "lexical_basename",
        "source_dir": str(source_dir.resolve()),
        "schema": str(common_schema),
        "schema_fingerprint": schema_fingerprint(common_schema),
        "files": items,
        "total_files": len(items),
        "total_rows": global_start,
        "total_bytes": sum(item["bytes"] for item in items),
    }
    stable = {
        "manifest_version": MANIFEST_VERSION,
        "ordering": manifest["ordering"],
        "schema_fingerprint": manifest["schema_fingerprint"],
        "files": [
            {
                "index": item["index"],
                "name": item["name"],
                "bytes": item["bytes"],
                "rows": item["rows"],
                "global_start": item["global_start"],
                "global_end_exclusive": item["global_end_exclusive"],
                "sha256": item["sha256"],
                "schema_fingerprint": item["schema_fingerprint"],
            }
            for item in items
        ],
        "total_files": manifest["total_files"],
        "total_rows": manifest["total_rows"],
        "total_bytes": manifest["total_bytes"],
    }
    manifest["content_fingerprint"] = fingerprint(stable)
    if cached and cached.get("content_fingerprint") != manifest["content_fingerprint"]:
        raise RuntimeError(
            "source inventory changed after this output directory was initialized; "
            "use a new output directory"
        )
    atomic_write_json(manifest_path, manifest)
    return manifest


def empty_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "polarity": {"negative_0": 0, "positive_1": 0, "other_or_null": 0},
        "rt": {
            "positive": 0,
            "accepted_0_to_1500": 0,
            "above_1500": 0,
            "nonpositive_nan_or_null": 0,
        },
    }


def _count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    safe = pc.fill_null(mask, False)
    value = pc.sum(pc.cast(safe, pa.int64())).as_py()
    return int(value or 0)


def summarize_arrow(data: pa.RecordBatch | pa.Table) -> dict[str, Any]:
    result = empty_stats()
    rows = data.num_rows
    result["rows"] = rows

    if "polarity" in data.schema.names:
        polarity = data.column(data.schema.get_field_index("polarity"))
        negative = _count_true(pc.equal(polarity, 0))
        positive = _count_true(pc.equal(polarity, 1))
        result["polarity"] = {
            "negative_0": negative,
            "positive_1": positive,
            "other_or_null": rows - negative - positive,
        }

    rt_name = (
        "RT"
        if "RT" in data.schema.names
        else "rt"
        if "rt" in data.schema.names
        else None
    )
    if rt_name is not None:
        rt = data.column(data.schema.get_field_index(rt_name))
        positive_mask = pc.and_(pc.is_valid(rt), pc.greater(rt, 0))
        accepted_mask = pc.and_(positive_mask, pc.less_equal(rt, 1500))
        above_mask = pc.and_(pc.is_valid(rt), pc.greater(rt, 1500))
        rt_positive = _count_true(positive_mask)
        accepted = _count_true(accepted_mask)
        above = _count_true(above_mask)
        result["rt"] = {
            "positive": rt_positive,
            "accepted_0_to_1500": accepted,
            "above_1500": above,
            "nonpositive_nan_or_null": rows - rt_positive,
        }
    return result


def add_stats(target: dict[str, Any], addition: dict[str, Any]) -> None:
    target["rows"] += addition["rows"]
    for key in target["polarity"]:
        target["polarity"][key] += addition["polarity"][key]
    for key in target["rt"]:
        target["rt"][key] += addition["rt"][key]


def aggregate_stats(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = empty_stats()
    for value in values:
        add_stats(total, value)
    return total


def validate_completed_output(path: Path, record: dict[str, Any]) -> None:
    if not path.exists():
        raise RuntimeError(f"completed output is missing: {path}")
    stat = path.stat()
    if stat.st_size != record["bytes"]:
        raise RuntimeError(f"completed output size changed: {path}")
    if stat.st_mtime_ns != record.get("mtime_ns"):
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"completed output hash changed: {path}")
    parquet = pq.ParquetFile(path)
    if int(parquet.metadata.num_rows) != record["rows"]:
        raise RuntimeError(f"completed output row count changed: {path}")


def write_hash_manifest(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [
        f"{record['sha256']}  {record['name']}"
        for record in records
        if record.get("name")
    ]
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())
