#!/usr/bin/env python3
"""Build UltraMSdata with resumable verification.

Only this module's output prefix is writable. Input objects are read only.
Recovery blobs are content addressed; a snapshot becomes eligible for recovery
only after every blob has been read back and its SHA-256 verified. Mutable
builder metadata is frozen while the child is paused and copied locally.
"""

from __future__ import annotations
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

MIXED_FINGERPRINT = "23ce892bdbdc0da19ad5b40367fde89c77c4ec6f9da83364b2101b9c8b2142ff"
PURE_ROWS = 160_641_162
NEGATIVE_ROWS = 10_793_129
PURE_POLARITY = {
    "positive_1": 149_767_344,
    "negative_0": NEGATIVE_ROWS,
    "other_or_null": 80_689,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def require_hash(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}")


def safe_relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("unsafe snapshot or manifest relative path")
    return path


def validate_dataset(root: Path, rows: int, files: int, polarity: dict) -> dict:
    """Rehash every output file and rescan all labels, independently of metadata."""
    import pyarrow.parquet as pq
    from ._audit_utils import empty_stats, add_stats, summarize_arrow

    manifest = load(root / "manifest.json")
    output = manifest["output"]
    if (
        manifest.get("status") != "complete"
        or output["files"] != files
        or output["stats"]["rows"] != rows
    ):
        raise RuntimeError("dataset completion or cardinality mismatch")
    records = output["file_records"]
    names = {item["name"] for item in records}
    if (
        len(records) != files
        or len(names) != files
        or names != {p.name for p in root.glob("*.parquet")}
    ):
        raise RuntimeError("output parquet inventory mismatch")
    total = empty_stats()
    for item in records:
        relative = safe_relative(item["name"])
        if len(relative.parts) != 1:
            raise RuntimeError("shards must have simple basenames")
        path = root / relative
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"output file size mismatch: {path.name}")
        require_hash(path, item["sha256"])
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != item["rows"]:
            raise RuntimeError("output shard row count mismatch")
        stats = empty_stats()
        for batch in parquet.iter_batches(
            batch_size=65536, columns=["precursor_mz", "polarity", "RT"]
        ):
            add_stats(stats, summarize_arrow(batch))
        if stats != item["stats"]:
            raise RuntimeError(f"output semantic statistics mismatch: {path.name}")
        add_stats(total, stats)
    if total != output["stats"] or total["polarity"] != polarity:
        raise RuntimeError("output aggregate semantic statistics mismatch")
    if total["rows"] != rows:
        raise RuntimeError("output aggregate row count mismatch")
    return {
        "manifest_sha256": sha(root / "manifest.json"),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "files": files,
        "stats": total,
        "all_file_hashes_recomputed": True,
        "all_rt_and_polarity_labels_rescanned": True,
    }


def portable_metadata(value):
    """Remove only host filesystem cache fields, never scientific identities."""
    if isinstance(value, dict):
        return {
            k: portable_metadata(v)
            for k, v in value.items()
            if k not in {"device", "inode", "mtime_ns"}
        }
    if isinstance(value, list):
        return [portable_metadata(v) for v in value]
    return value


def stage_publication(source: Path, target: Path) -> None:
    """Preserve builder files; expose deterministic metadata on the publish side."""
    if target.exists():
        shutil.rmtree(target)  # regenerable local hardlinks, not source or S3
    target.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(
            part.startswith(".") or ".partial" in part or ".tmp" in part
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise RuntimeError("published dataset must not contain symlinks")
        if not path.is_file():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            write(destination, portable_metadata(load(path)))
        else:
            os.link(path, destination)


class Job:
    def __init__(self):
        self.root = Path(os.environ["HOT_ROOT"]).resolve()
        if not str(self.root).startswith("/opt/ultrams-pure-ae3-"):
            raise ValueError("worker-local /opt root required")
        self.build = self.root / "build"
        self.mixed = self.root / "mixed"
        self.output = os.environ["S3_OUTPUT_PREFIX"].rstrip("/")
        self.recovery = self.output + "/_recovery/" + os.environ["RUN_ID"]
        self.rclone = [
            os.environ["RCLONE_BIN"],
            "--config",
            os.environ["RCLONE_CONFIG"],
        ]
        self.python = os.environ.get("PYTHON_BIN", sys.executable)
        self.child = None
        self.stopping = False
        self.identity = {
            key: os.environ[key]
            for key in (
                "RUN_ID",
                "S3_CODE_ARCHIVE",
                "S3_CODE_SHA256",
                "S3_MIXED_SOURCE",
                "MIXED_MANIFEST_SHA256",
                "MIXED_INPUT_MANIFEST_SHA256",
                "S3_SEMANTIC_AUDIT",
                "SEMANTIC_AUDIT_SHA256",
                "S3_OUTPUT_PREFIX",
            )
        }
        self.verified_blobs = set()
        self.hash_cache = {}

    def rc(self, *args, capture=False):
        return subprocess.run(
            [*self.rclone, *map(str, args)],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )

    def readback_hash(self, remote: str, expected: str) -> None:
        process = subprocess.Popen(
            [*self.rclone, "cat", remote], stdout=subprocess.PIPE
        )
        digest = hashlib.sha256()
        try:
            for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        finally:
            process.stdout.close()
            rc = process.wait()
        if rc or digest.hexdigest() != expected:
            raise RuntimeError(f"S3 readback failed: {remote}")

    def put_verified(self, local: Path, remote: str, digest=None):
        digest = digest or sha(local)
        self.rc("copyto", local, remote, "--immutable", "--checksum", "--retries", "8")
        self.readback_hash(remote, digest)

    def list_optional(self, remote: str) -> list[str]:
        result = subprocess.run(
            [*self.rclone, "lsf", remote, "--files-only"],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            if any(
                x in result.stderr.lower()
                for x in ("directory not found", "object not found", "nosuchkey")
            ):
                return []
            raise RuntimeError("S3 listing failed: " + result.stderr[-1000:])
        return result.stdout.splitlines()

    def restore(self):
        # Never overlay a partially resumed local tree. The builders validate it.
        if self.build.exists() and any(self.build.iterdir()):
            return
        ready = sorted(
            x
            for x in self.list_optional(self.recovery + "/ready")
            if x.endswith(".json")
        )
        if not ready:
            self.build.mkdir(parents=True, exist_ok=True)
            return
        local = self.root / "restore.json"
        self.rc("copyto", self.recovery + "/ready/" + ready[-1], local)
        snapshot = load(local)
        if not snapshot.get("all_blobs_readback_verified"):
            raise RuntimeError("recovery snapshot was not verified")
        restore_tree = self.root / "restore_build"
        restore_tree.mkdir(exist_ok=True)
        if snapshot["identity"] != self.identity:
            raise RuntimeError("recovery snapshot identity mismatch")
        for item in snapshot["files"]:
            path = restore_tree / safe_relative(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if item["kind"] == "symlink":
                target = Path(item["target"])
                if not target.is_absolute() or not target.is_relative_to(self.root):
                    raise RuntimeError("recovery symlink leaves worker root")
                if path.is_symlink():
                    if os.readlink(path) != str(target):
                        raise RuntimeError("existing recovery symlink conflicts")
                else:
                    path.symlink_to(target)
            else:
                remote = self.recovery + "/blobs/" + item["sha256"]
                self.rc("copyto", remote, path, "--retries", "8")
                require_hash(path, item["sha256"])
                self.verified_blobs.add(item["sha256"])
        if self.build.exists():
            self.build.rmdir()  # only an empty directory may be replaced
        restore_tree.rename(self.build)
        print(
            f"Recovered {len(snapshot['files'])} entries from {ready[-1]}", flush=True
        )

    def snapshot(self):
        if not self.build.exists():
            return
        stamp = (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        frozen = self.root / "snapshots" / stamp
        frozen.mkdir(parents=True)
        paused = self.child is not None and self.child.poll() is None
        records = []
        try:
            if paused:
                try:
                    os.killpg(self.child.pid, signal.SIGSTOP)
                    # A child may exit between poll and SIGSTOP; preserve its exit
                    # status rather than leaving Popen waiting for a reaped pid.
                    _, status = os.waitpid(self.child.pid, os.WUNTRACED)
                    if not os.WIFSTOPPED(status):
                        self.child.returncode = os.waitstatus_to_exitcode(status)
                        paused = False
                except ProcessLookupError:
                    paused = False
                    self.child.poll()
            for path in sorted(self.build.rglob("*")):
                relative = path.relative_to(self.build)
                if any(
                    part.startswith(".") or ".partial" in part or ".tmp" in part
                    for part in relative.parts
                ):
                    continue
                if path.is_symlink():
                    records.append(
                        {
                            "path": str(relative),
                            "kind": "symlink",
                            "target": os.readlink(path),
                        }
                    )
                elif path.is_file():
                    target = frozen / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix in {".parquet", ".npz"}:
                        os.link(
                            path, target
                        )  # committed builder data is never rewritten
                    else:
                        shutil.copyfile(path, target)
                    records.append(
                        {
                            "path": str(relative),
                            "kind": "file",
                            "bytes": target.stat().st_size,
                        }
                    )
        finally:
            if paused and self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGCONT)

        def upload(item):
            if item["kind"] != "file":
                return item
            path = frozen / item["path"]
            stat = path.stat()
            cache_key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            digest = self.hash_cache.get(cache_key)
            if digest is None:
                digest = sha(path)
                self.hash_cache[cache_key] = digest
            item["sha256"] = digest
            if digest not in self.verified_blobs:
                self.put_verified(path, self.recovery + "/blobs/" + digest, digest)
                self.verified_blobs.add(digest)
            return item

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(upload, records))
        manifest = self.root / "snapshots" / (stamp + ".json")
        write(
            manifest,
            {
                "identity": self.identity,
                "files": records,
                "all_blobs_readback_verified": True,
            },
        )
        self.put_verified(manifest, self.recovery + "/ready/" + manifest.name)
        shutil.rmtree(frozen)  # local hardlinks/copies only; never delete S3 objects
        print(
            f"Recovery snapshot verified: {stamp}, entries={len(records)}", flush=True
        )

    def run_child(self, *args):
        command = [self.python, "-u", "-m", *map(str, args)]
        print("Starting", " ".join(command), flush=True)
        self.child = subprocess.Popen(command, start_new_session=True)
        last = time.monotonic()
        try:
            while self.child.poll() is None:
                if self.stopping:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    self.child.wait(timeout=60)
                    raise RuntimeError(
                        "data job interrupted; saving verified recovery state"
                    )
                if time.monotonic() - last >= 900:
                    self.snapshot()
                    last = time.monotonic()
                time.sleep(2)
            if self.child.returncode:
                raise subprocess.CalledProcessError(self.child.returncode, command)
        finally:
            if self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)
                try:
                    self.child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.child = None
            self.snapshot()

    def publish(self, local: Path, relative: str):
        target = self.output + "/" + relative
        self.rc(
            "copy",
            local,
            target,
            "--immutable",
            "--checksum",
            "--transfers",
            "8",
            "--checkers",
            "16",
            "--retries",
            "8",
        )
        # Object API byte comparison reads every uploaded object back. A completion
        # marker is never published based on size, existence, or multipart ETag.
        self.rc("check", local, target, "--download", "--one-way", "--checkers", "8")

    def run(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stopping", True))
        self.root.mkdir(parents=True, exist_ok=True)
        # Disk includes retained source, pure output, balanced output, and recovery
        # hardlinks. Resumption counts existing local bytes toward the requirement.
        local_inodes = {}
        for path in self.root.rglob("*.parquet"):
            if not path.is_symlink():
                stat = path.stat()
                local_inodes[(stat.st_dev, stat.st_ino)] = stat.st_size
        if (
            shutil.disk_usage(self.root).free + sum(local_inodes.values())
            < 270 * 1024**3
        ):
            raise RuntimeError(
                "worker-local disk has less than 270 GiB usable build capacity"
            )
        owner = self.root / "owner.json"
        write(owner, self.identity)
        self.put_verified(owner, self.output + "/_owner.json")
        self.mixed.mkdir(parents=True, exist_ok=True)
        source = self.identity["S3_MIXED_SOURCE"].rstrip("/")
        # copy is restartable; immutable local destination exposes source drift.
        self.rc(
            "copy",
            source,
            self.mixed,
            "--immutable",
            "--checksum",
            "--transfers",
            "8",
            "--checkers",
            "16",
            "--retries",
            "12",
        )
        require_hash(
            self.mixed / "manifest.json", self.identity["MIXED_MANIFEST_SHA256"]
        )
        require_hash(
            self.mixed / "_audit/input_manifest.json",
            self.identity["MIXED_INPUT_MANIFEST_SHA256"],
        )
        audit = self.root / "source_semantic_audit.json"
        self.rc("copyto", self.identity["S3_SEMANTIC_AUDIT"], audit, "--retries", "8")
        require_hash(audit, self.identity["SEMANTIC_AUDIT_SHA256"])
        self.restore()
        pure = self.build / "pure"
        balanced = self.build / "polarity"
        self.run_child(
            "ultrams_training.data_tools.extract_ultramsdata",
            "--src",
            self.mixed,
            "--dst",
            pure,
            "--work-dir",
            self.build / "extract",
            "--contract",
            Path(os.environ["CODE_DIR"])
            / "ultrams_training/data_tools/ultramsdata_contract.json",
            "--expected-mixed-fingerprint",
            MIXED_FINGERPRINT,
            "--dataset-name",
            "ae3_only_160641162",
            "--output-shards",
            "816",
            "--world-size",
            "4",
        )
        self.run_child(
            "ultrams_training.data_tools.build_polarity_balanced",
            "--src",
            pure,
            "--dst",
            balanced,
            "--dataset-name",
            "ae3_only_polarity_balanced",
            "--seed",
            "42",
            "--world-size",
            "4",
            "--output-shards",
            "48",
            "--batch-size",
            "65536",
            "--compression",
            "zstd",
            "--input-sha256-manifest",
            pure / "files.sha256",
            "--rehash-inputs",
        )
        publication = self.root / "publication"
        publish_pure, publish_balanced = publication / "pure", publication / "polarity"
        stage_publication(pure, publish_pure)
        stage_publication(balanced, publish_balanced)
        pure_check = validate_dataset(publish_pure, PURE_ROWS, 816, PURE_POLARITY)
        balanced_check = validate_dataset(
            publish_balanced,
            2 * NEGATIVE_ROWS,
            48,
            {
                "positive_1": NEGATIVE_ROWS,
                "negative_0": NEGATIVE_ROWS,
                "other_or_null": 0,
            },
        )
        audit_dir = self.root / "final_audit"
        write(
            audit_dir / "semantic_audit.json",
            {
                "status": "complete",
                "identity": self.identity,
                "pure": pure_check,
                "polarity": balanced_check,
                "pure_source_only": True,
            },
        )
        shutil.copyfile(audit, audit_dir / "parent_semantic_audit.json")
        self.run_child(
            "ultrams_training.data_tools.validate_pretraining_data",
            "--clean-dir",
            publish_pure,
            "--polarity-dir",
            publish_balanced,
            "--expected-clean-name",
            "ae3_only_160641162",
            "--expected-polarity-name",
            "ae3_only_polarity_balanced",
            "--expected-clean-rows",
            str(PURE_ROWS),
            "--expected-polarity-rows",
            str(2 * NEGATIVE_ROWS),
            "--expected-schema-fingerprint",
            "182a8c7e2831a49d1b87515b2eac2c3be00c1e285fc04251091ee1d2b3520a2c",
            "--output",
            audit_dir / "final_semantic_audit.json",
        )
        if load(audit_dir / "final_semantic_audit.json")["status"] != "passed":
            raise RuntimeError("full-spectrum semantic audit failed")
        self.publish(publish_pure, "derived/ae3_only_160641162/shards")
        self.publish(publish_balanced, "derived/ae3_only_polarity_balanced/shards")
        self.publish(audit_dir, "audit")
        complete = self.root / "completion.json"
        write(
            complete,
            {
                "status": "complete",
                "identity": self.identity,
                "pure": pure_check,
                "polarity": balanced_check,
                "s3_full_readback_verified": True,
            },
        )
        self.put_verified(
            complete, self.output + "/_completion/" + os.environ["RUN_ID"] + ".json"
        )
        print(
            "UltraMSdata built, audited, and read back from S3.",
            flush=True,
        )


def main():
    Job().run()


if __name__ == "__main__":
    main()
