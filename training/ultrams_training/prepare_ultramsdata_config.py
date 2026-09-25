"""Prepare UltraMSdata training configuration from completed data audits.

MPR runs through peak_reconstruction. Later training uses its completed
epoch-5 checkpoint, SHA sidecar, and resolved configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from ultrams_training.data_tools._audit_utils import fingerprint, load_json, sha256_file
from ultrams_training.data_tools.extract_ultramsdata import DEFAULT_FINGERPRINT, require
from ultrams_training.data_tools.validate_pretraining_data import (
    _recompute_dataset_fingerprint,
    _source_inventory_fingerprint,
)

COUNTS = {
    "clean": (
        160641162,
        816,
        {"negative_0": 10793129, "positive_1": 149767344, "other_or_null": 80689},
    ),
    "polarity": (
        21586258,
        48,
        {"negative_0": 10793129, "positive_1": 10793129, "other_or_null": 0},
    ),
}


def read_verified_assets(
    clean_dir: Path, polarity_dir: Path, audit_path: Path
) -> tuple[dict, dict, dict]:
    audit = load_json(audit_path)
    require(
        audit.get("status") == "passed" and audit.get("errors") == [],
        "semantic audit did not pass",
    )
    manifests, assets, audits = {}, {}, {}
    for key, directory in (("clean", clean_dir), ("polarity", polarity_dir)):
        m = load_json(directory / "manifest.json")
        rows, files, polarity = COUNTS[key]
        stats, records = m["output"]["stats"], m["output"]["file_records"]
        require(m.get("status") == "complete", f"{key} incomplete")
        require(
            m["dataset_fingerprint"] == _recompute_dataset_fingerprint(m),
            f"{key} fingerprint invalid",
        )
        require(
            m["output"]["files"] == len(records) == files, f"{key} file count differs"
        )
        require(
            sum(r["rows"] for r in records) == stats["rows"] == rows
            and stats["polarity"] == polarity,
            f"{key} row/polarity contract differs",
        )
        names = [r["name"] for r in records]
        require(
            names == sorted(names) and len(set(names)) == len(names),
            f"{key} records unordered/duplicated",
        )
        require(all(Path(n).name == n for n in names), "unsafe shard name")
        report = audit["datasets"][key]
        require(
            report.get("status") == "passed" and report.get("errors") == [],
            f"{key} semantic audit failed",
        )
        require(
            report["manifest"]["dataset_fingerprint"] == m["dataset_fingerprint"],
            f"{key} semantic audit belongs to another dataset",
        )
        require(report["stats"] == stats, f"{key} audited statistics differ")
        require(
            [(r["name"], r["rows"], r["sha256"]) for r in report["files"]]
            == [(r["name"], r["rows"], r["sha256"]) for r in records],
            f"{key} audited file identity differs",
        )
        peaks = report["content_validation"]["peak_rows"]
        require(
            peaks["at_least_minimum"] == rows
            and peaks["below_minimum"] == peaks["unscannable"] == 0,
            f"{key} incomplete peak audit",
        )
        require(
            report["content_validation"]["rows"] == rows,
            f"{key} incomplete content scan",
        )
        asset = {
            k: m[k]
            for k in (
                "dataset_fingerprint",
                "input_content_fingerprint",
                "run_identity",
                "schema_fingerprint",
            )
        }
        asset.update(
            rows=rows,
            files=files,
            **polarity,
            accepted_0_to_1500=stats["rt"]["accepted_0_to_1500"],
            above_1500=stats["rt"]["above_1500"],
        )
        if key == "polarity":
            require(
                m["config"]["seed"] == m["selection"]["seed"] == 42,
                "polarity seed differs",
            )
            asset["seed"] = 42
        assets[key], manifests[key] = asset, m
        audits[key] = {
            "sha256": sha256_file(audit_path),
            "dataset_fingerprint": m["dataset_fingerprint"],
            "rows_at_least_three_valid_peaks": rows,
        }
    clean, pol = manifests["clean"], manifests["polarity"]
    require(
        pol["input_content_fingerprint"]
        == _source_inventory_fingerprint(
            clean["schema_fingerprint"], clean["output"]["file_records"]
        ),
        "polarity is not derived from this pure clean dataset",
    )
    require(
        pol["source_polarity_counts"]
        == {"rows": COUNTS["clean"][0], **COUNTS["clean"][2]},
        "polarity source counts differ",
    )
    provenance_path = find_provenance(clean_dir)
    proof = load_json(provenance_path)
    require(
        proof.get("status") == "passed"
        and proof.get("all_source_hashes_verified") is True,
        "UltraMSdata provenance not verified",
    )
    require(
        proof["mixed_fingerprint"] == DEFAULT_FINGERPRINT
        and proof["dataset_fingerprint"] == clean["dataset_fingerprint"],
        "UltraMSdata provenance identity differs",
    )
    require(
        proof["pure_rows"] == COUNTS["clean"][0]
        and proof["public_rows_removed"] == 775870,
        "UltraMSdata provenance row counts differ",
    )
    require(clean["removed"]["total"] == 0, "pure extraction changed row selection")
    return manifests, assets, audits


def find_provenance(clean_dir: Path) -> Path:
    candidates = sorted(clean_dir.glob("*provenance.json"))
    require(len(candidates) == 1, "expected one UltraMSdata provenance file")
    return candidates[0]


def phase1_identity(cfg: dict) -> str:
    payload = {k: cfg[k] for k in ("training_mode", "seed", "supported_world_sizes")}
    payload.update(
        distributed_profiles={
            w: {"phase1": p["phase1"]} for w, p in cfg["distributed_profiles"].items()
        },
        model={k: cfg["model"][k] for k in ("max_peaks", "mask_ratio")},
        optimization={
            k: cfg["optimization"][k]
            for k in ("weight_decay", "gradient_clip", "num_workers", "precision")
        },
        clean=cfg["expected_assets"]["clean"],
        stage=cfg["stages"][0],
    )
    return fingerprint(payload)


def freeze_config(
    *,
    mode: str,
    template: Path,
    clean_dir: Path,
    polarity_dir: Path,
    audit_path: Path,
    output: Path,
    run_name: str,
    clean_relative_path: str = "derived/ultramsdata_clean/shards",
    polarity_relative_path: str = "derived/ultramsdata_polarity/shards",
    phase1_config: Path | None = None,
    parent_checkpoint: Path | None = None,
    parent_sidecar: Path | None = None,
) -> dict:
    require(mode in ("phase1", "adaptation"), "unknown freeze mode")
    require(output.resolve() != template.resolve(), "refuse to overwrite template")
    require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", run_name)), "unsafe run name")
    for path in (clean_relative_path, polarity_relative_path):
        require(
            not Path(path).is_absolute() and ".." not in Path(path).parts,
            "data paths must be bundle-relative",
        )
    manifests, assets, audits = read_verified_assets(
        clean_dir, polarity_dir, audit_path
    )
    cfg = copy.deepcopy(load_json(template))
    cfg.update(
        run_name=run_name, world_size=4, expected_assets=assets, data_audits=audits,
        data_mode="ultramsdata_parquet",
    )
    cfg["paths"] = {
        "clean_shard_dir": clean_relative_path,
        "polarity_shard_dir": polarity_relative_path,
        "output_dir": f"runs/{run_name}",
    }
    evidence = {
        "mode": mode,
        "source_template_sha256": sha256_file(template),
        "semantic_audit_sha256": sha256_file(audit_path),
        "pure_provenance_sha256": sha256_file(find_provenance(clean_dir)),
        "mixed_source_fingerprint": DEFAULT_FINGERPRINT,
    }
    if mode == "phase1":
        require(
            cfg["stages"][0]["name"] == "peak_reconstruction"
            and cfg["stages"][0]["epochs"] == 5,
            "unexpected Phase-1 protocol",
        )
        evidence["required_start_stage"] = evidence["required_end_stage"] = (
            "peak_reconstruction"
        )
        require("parent_checkpoint" not in cfg, "Phase 1 must start from scratch")
    else:
        require(
            all(
                p is not None
                for p in (phase1_config, parent_checkpoint, parent_sidecar)
            ),
            "adaptation requires a new real Phase-1 checkpoint, sidecar and frozen config",
        )
        import torch
        from ultrams_training.train_pretraining import parquet_worker_parallel_layout

        cfg["paths"]["rt_shard_dir"] = clean_relative_path
        cfg["expected_assets"]["rt"] = copy.deepcopy(assets["clean"])
        parent_cfg = load_json(phase1_config)
        require(
            parent_cfg.get("ultramsdata_freeze", {}).get("mode") == "phase1",
            "parent config is not a completed UltraMSdata MPR stage",
        )
        require(
            phase1_identity(parent_cfg) == phase1_identity(cfg),
            "parent Phase-1 contract differs",
        )
        sidecar = load_json(parent_sidecar)
        require(
            parent_checkpoint.stat().st_size == sidecar["bytes"]
            and sha256_file(parent_checkpoint) == sidecar["sha256"],
            "parent checkpoint sidecar/hash mismatch",
        )
        checkpoint = torch.load(
            parent_checkpoint, map_location="cpu", weights_only=False
        )
        saved = checkpoint["config"]
        state = checkpoint["resume_state"]
        require(
            state == sidecar["resume_state"],
            "parent checkpoint progress differs from sidecar",
        )
        require(
            state["stage"] == "peak_reconstruction"
            and state["epoch"] == 6
            and state["batch_in_epoch"] == 0,
            "parent must be completed Phase-1 epoch 5",
        )
        require(
            saved["world_size"] == 4
            and saved["phase1_contract_sha256"] == phase1_identity(cfg),
            "parent trained a different Phase-1 contract",
        )
        require(
            saved["dataset_fingerprints"]["clean"]
            == assets["clean"]["dataset_fingerprint"],
            "parent weights saw another dataset",
        )
        codes = saved["code_fingerprints"]
        require(
            bool(codes)
            and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in codes.values()),
            "parent code fingerprints missing or invalid",
        )
        require(
            bool(checkpoint.get("model_state_dict")),
            "parent checkpoint has no model weights",
        )
        cfg["parent_checkpoint"] = {
            "sha256": sidecar["sha256"],
            "code_fingerprints": codes,
        }
        stage = cfg["stages"][1]
        profile = cfg["distributed_profiles"]["4"][stage["profile"]]
        layouts = {}
        for epoch in range(1, int(stage["epochs"]) + 1):
            workers, batches = parquet_worker_parallel_layout(
                manifests["clean"]["output"]["file_records"],
                4,
                profile["batch_size_per_rank"],
                cfg["optimization"]["adaptation_num_workers"],
                cfg["seed"],
                epoch,
                drop_last=True,
            )
            accumulation = profile["gradient_accumulation"]
            require(
                len(set(batches)) == 1
                and batches[0] > 0
                and batches[0] % accumulation == 0,
                "RT ranks cannot form synchronized optimizer batches",
            )
            layouts[str(epoch)] = {
                "optimizer_steps": batches[0] // accumulation,
                "microbatches_per_rank": batches,
                "worker_rows_by_rank": workers,
                "used_rows": sum(batches) * profile["batch_size_per_rank"],
            }
        stage["scheduler_total_steps_override"] = 3 * sum(
            x["optimizer_steps"] for x in layouts.values()
        )
        evidence.update(
            parent_sidecar_sha256=sha256_file(parent_sidecar),
            phase1_config_sha256=sha256_file(phase1_config),
            rt_epoch_layouts=layouts,
        )
        require(
            "UNRESOLVED" not in json.dumps(cfg), "adaptation config remains unresolved"
        )
    cfg["ultramsdata_freeze"] = evidence
    payload = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        require(
            output.read_text() == payload,
            f"refuse to overwrite unrelated config: {output}",
        )
    else:
        with output.open("x") as handle:
            handle.write(payload)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("phase1", "adaptation"), required=True)
    for name in ("template", "clean-dir", "polarity-dir", "audit-path", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--clean-relative-path", default="derived/ultramsdata_clean/shards"
    )
    parser.add_argument(
        "--polarity-relative-path", default="derived/ultramsdata_polarity/shards"
    )
    for name in ("phase1-config", "parent-checkpoint", "parent-sidecar"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    cfg = freeze_config(**vars(args))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "phase1_contract_sha256": phase1_identity(cfg),
                "mode": args.mode,
            }
        )
    )


if __name__ == "__main__":
    main()
