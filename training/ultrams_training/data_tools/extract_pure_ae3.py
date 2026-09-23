"""Recover the proven Ae3 prefix of a frozen mixed bundle without changing rows.

All source files are verified before new outputs are written. The existing
resumable materializer only repartitions the prefix; its selection rules are
explicitly disabled. Originals are never rewritten or chmod'ed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

from ._audit_utils import atomic_write_json, fingerprint, load_json, sha256_file
from .materialize_clean_ae3 import CleanConfig, materialize_clean_dataset

DEFAULT_FINGERPRINT = "23ce892bdbdc0da19ad5b40367fde89c77c4ec6f9da83364b2101b9c8b2142ff"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _disjoint(paths: list[Path]) -> None:
    for i, left in enumerate(paths):
        for right in paths[i + 1 :]:
            require(
                left != right
                and left not in right.parents
                and right not in left.parents,
                f"source/output/work directories overlap: {left}, {right}",
            )


def verify_source(src: Path, contract: dict, expected_fingerprint: str) -> dict:
    manifest = load_json(src / "manifest.json")
    records = manifest["output"]["file_records"]
    require(manifest["status"] == "complete", "mixed manifest incomplete")
    identity = fingerprint(
        {
            "run_identity": manifest["run_identity"],
            "files": [(r["name"], r["rows"], r["sha256"]) for r in records],
        }
    )
    require(
        identity == manifest["dataset_fingerprint"] == expected_fingerprint,
        "mixed dataset fingerprint mismatch",
    )
    require(
        manifest["output"]["stats"]["rows"] == contract["final"]["rows"],
        "mixed row count mismatch",
    )
    require(len(records) == contract["final"]["files"], "mixed file count mismatch")
    require(
        sum(r["rows"] for r in records) == contract["final"]["rows"],
        "mixed record rows mismatch",
    )
    names = [r["name"] for r in records]
    require(
        names == sorted(names) and len(names) == len(set(names)),
        "mixed ordering invalid",
    )
    require(
        all(Path(n).name == n and n.endswith(".parquet") for n in names),
        "unsafe source filename",
    )
    require(
        sorted(p.name for p in src.glob("*.parquet")) == names,
        "mixed shard set differs",
    )
    config = manifest["config"]
    require(
        config["drop_partitions"] == []
        and config["max_precursor_mz"]
        == contract["historical_replay"]["max_precursor_mz"],
        "unexpected mixed selection rules",
    )
    inputs = load_json(src / manifest["input_manifest"])
    fields = (
        "index",
        "name",
        "bytes",
        "rows",
        "global_start",
        "global_end_exclusive",
        "sha256",
        "schema_fingerprint",
    )
    stable = {
        k: inputs[k]
        for k in (
            "manifest_version",
            "ordering",
            "schema_fingerprint",
            "total_files",
            "total_rows",
            "total_bytes",
        )
    }
    stable["files"] = [{k: r[k] for k in fields} for r in inputs["files"]]
    require(
        fingerprint(stable)
        == inputs["content_fingerprint"]
        == manifest["input_content_fingerprint"],
        "input manifest fingerprint mismatch",
    )
    require(inputs["ordering"] == "lexical_basename", "input source order unproven")
    source_names = [r["name"] for r in inputs["files"]]
    require(source_names == sorted(source_names), "input source order differs")
    h = contract["historical_replay"]
    expected_ae3 = [
        f"ae3_shard_{i:04d}.parquet"
        for i in range(h["balanced_files"])
        if f"shard_{i:04d}.parquet" not in h["excluded_balanced_shards"]
    ]
    public = contract["public_increment"]["files"]
    require(
        source_names
        == expected_ae3 + [f"public_{i:04d}.parquet" for i in range(len(public))],
        "input provenance inventory differs",
    )
    for item, (_, rows, digest) in zip(inputs["files"][len(expected_ae3) :], public):
        require(
            item["rows"] == rows and item["sha256"] == digest,
            "frozen public input mismatch",
        )
    require(
        sum(r[1] for r in public) == contract["public_increment"]["rows"],
        "public count mismatch",
    )
    pure_rows = h["clean_ae3_rows"]
    require(
        pure_rows + contract["public_increment"]["rows"] == contract["final"]["rows"],
        "source row closure failed",
    )
    start = 0
    boundary = None
    for index, record in enumerate(records):
        if start < pure_rows < start + record["rows"]:
            boundary = index
            break
        start += record["rows"]
    require(
        boundary is not None and boundary > 0,
        "expected interior source boundary missing",
    )
    for index in (boundary - 1, boundary):
        progress = load_json(
            src / "_audit" / "output_progress" / f"shard_{index:04d}.json"
        )
        require(
            progress["run_identity"] == manifest["run_identity"],
            "progress identity mismatch",
        )
        require(progress["output"] == records[index], "progress output record mismatch")
        require(
            progress["cumulative_output_stats"]["rows"]
            == sum(r["rows"] for r in records[: index + 1]),
            "progress row count mismatch",
        )
        cursor = progress["next_source_cursor"]
        if index < boundary:
            require(
                cursor["source_index"] == len(expected_ae3) - 1,
                "pre-boundary cursor is not in last Ae3 source",
            )
        else:
            require(
                cursor["source_index"] >= len(expected_ae3),
                "boundary cursor has not reached public source",
            )
        require(
            0 <= cursor["source_index"] < len(inputs["files"]), "invalid source cursor"
        )
        require(
            0
            <= cursor["row_offset"]
            <= inputs["files"][cursor["source_index"]]["rows"],
            "invalid source row cursor",
        )
    for index, record in enumerate(records):
        path = src / record["name"]
        require(
            path.stat().st_size == record["bytes"]
            and sha256_file(path) == record["sha256"],
            f"mixed hash mismatch: {path.name}",
        )
        require(
            pq.read_metadata(path).num_rows == record["rows"],
            f"mixed row mismatch: {path.name}",
        )
        if index % 32 == 0:
            print(
                f"verified mixed shard {index + 1}/{len(records)}",
                file=sys.stderr,
                flush=True,
            )
    return {
        "manifest": manifest,
        "boundary_index": boundary,
        "boundary_keep_rows": pure_rows - start,
        "pure_rows": pure_rows,
    }


def extract(
    src: Path,
    dst: Path,
    work_dir: Path,
    contract_path: Path,
    *,
    expected_fingerprint: str = DEFAULT_FINGERPRINT,
    dataset_name: str = "ae3_only_160641162",
    output_shards: int = 816,
    world_size: int = 4,
    max_output_shards_this_run: int | None = None,
) -> dict:
    src, dst, work_dir = [p.resolve() for p in (src, dst, work_dir)]
    _disjoint([src, dst, work_dir])
    contract = load_json(contract_path)
    proof = verify_source(src, contract, expected_fingerprint)
    ownership = {
        "source": str(src),
        "mixed_fingerprint": expected_fingerprint,
        "contract_sha256": sha256_file(contract_path),
        "destination": str(dst),
        "dataset_name": dataset_name,
        "output_shards": output_shards,
        "world_size": world_size,
        "boundary_index": proof["boundary_index"],
        "boundary_keep_rows": proof["boundary_keep_rows"],
    }
    for directory in (work_dir, dst):
        owner = directory / "pure_ae3_owner.json"
        if directory.exists() and any(directory.iterdir()):
            require(
                owner.exists() and load_json(owner) == ownership,
                f"refuse unrelated existing directory: {directory}",
            )
        directory.mkdir(parents=True, exist_ok=True)
        if not owner.exists():
            atomic_write_json(owner, ownership)
    prefix = work_dir / "prefix"
    prefix.mkdir(exist_ok=True)
    records = proof["manifest"]["output"]["file_records"]
    boundary = proof["boundary_index"]
    for index, record in enumerate(records[: boundary + 1]):
        target = prefix / record["name"]
        source = src / record["name"]
        if index < boundary:
            if target.is_symlink():
                require(target.resolve() == source.resolve(), "prefix link changed")
            else:
                require(not target.exists(), "prefix contains unexpected file")
                target.symlink_to(source)
        else:
            receipt = work_dir / "boundary.json"
            if target.exists():
                require(
                    receipt.exists()
                    and sha256_file(target) == load_json(receipt)["sha256"],
                    "boundary artifact changed",
                )
            else:
                temporary = target.with_suffix(".partial")
                remaining = proof["boundary_keep_rows"]
                parquet = pq.ParquetFile(source)
                with pq.ParquetWriter(
                    temporary, parquet.schema_arrow, compression="zstd"
                ) as writer:
                    for batch in parquet.iter_batches(batch_size=65536):
                        if remaining <= 0:
                            break
                        keep = min(remaining, batch.num_rows)
                        writer.write_batch(batch.slice(0, keep))
                        remaining -= keep
                require(remaining == 0, "boundary source too short")
                digest = sha256_file(temporary)
                # Receipt first: either an interrupted temporary or the verified final
                # artifact can be recovered without replacing an unknown output.
                atomic_write_json(
                    receipt, {"sha256": digest, "rows": proof["boundary_keep_rows"]}
                )
                os.replace(temporary, target)
    require(
        sorted(p.name for p in prefix.iterdir())
        == [r["name"] for r in records[: boundary + 1]],
        "unexpected prefix artifact",
    )
    manifest = materialize_clean_dataset(
        CleanConfig(
            source_dir=prefix,
            output_dir=dst,
            dataset_name=dataset_name,
            drop_partitions=(),
            max_precursor_mz=sys.float_info.max,
            output_shards=output_shards,
            expected_output_rows=proof["pure_rows"],
            world_size=world_size,
            rehash_inputs=True,
        ),
        max_output_shards_this_run=max_output_shards_this_run,
    )
    if manifest.get("status") == "complete":
        require(
            manifest["removed"]["total"] == 0, "repartition unexpectedly removed rows"
        )
        require(
            manifest["output"]["stats"]["polarity"]
            == contract["historical_replay"]["clean_ae3_polarity"],
            "pure Ae3 polarity closure failed",
        )
        require(
            manifest["output"]["stats"]["rows"] == proof["pure_rows"],
            "pure Ae3 row closure failed",
        )
        atomic_write_json(
            dst / "pure_ae3_provenance.json",
            {
                "status": "passed",
                **ownership,
                "pure_rows": proof["pure_rows"],
                "public_rows_removed": contract["public_increment"]["rows"],
                "all_source_hashes_verified": True,
                "selection": "proven global row prefix; all columns and order retained",
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("ae3_pretraining_contract_v2.json"),
    )
    parser.add_argument("--expected-mixed-fingerprint", default=DEFAULT_FINGERPRINT)
    parser.add_argument("--dataset-name", default="ae3_only_160641162")
    parser.add_argument("--output-shards", type=int, default=816)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    result = extract(
        args.src,
        args.dst,
        args.work_dir,
        args.contract,
        expected_fingerprint=args.expected_mixed_fingerprint,
        dataset_name=args.dataset_name,
        output_shards=args.output_shards,
        world_size=args.world_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_fingerprint": result.get("dataset_fingerprint"),
            }
        )
    )


if __name__ == "__main__":
    main()
