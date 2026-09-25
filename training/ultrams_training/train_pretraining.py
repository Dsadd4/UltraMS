#!/usr/bin/env python3
"""Run UltraMS pretraining stages with exact resume.

Production launches use two operational phases without changing the stage
implementations: five-epoch peak reconstruction first, then the historical
post-MLM curriculum from the terminal reconstruction checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from . import train_epoch3_to_stage_d11 as common
except ImportError:  # Direct execution from a staged code directory.
    import train_epoch3_to_stage_d11 as common


STAGE_ORDER = (
    "peak_reconstruction",
    "supervised_rt",
    "self_consistent_rt",
    "ion_mode",
)
LEGACY_OBJECTIVE = {
    "supervised_rt": "ae",
    "self_consistent_rt": "c",
    "ion_mode": "d",
}


def training_code_fingerprints(project_root: Path) -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    fixed_paths = {
        "package/train_pretraining.py": Path(__file__).resolve(),
        "package/train_epoch3_to_stage_d11.py": Path(common.__file__).resolve(),
    }
    project_files = sorted(project_root.rglob("*.py"))
    if not project_files:
        raise FileNotFoundError(
            f"project snapshot contains no Python files: {project_root}"
        )
    paths = {
        **fixed_paths,
        **{
            f"project_snapshot/{path.relative_to(project_root).as_posix()}": path
            for path in project_files
        },
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing training code: {missing}")
    if any(package_root not in path.parents for path in fixed_paths.values()):
        raise RuntimeError("training entry points are outside the release package")
    return {name: common.sha256_file(path) for name, path in paths.items()}


def expected_parent_code_fingerprints(
    cfg: Mapping[str, Any], current_code: Mapping[str, str]
) -> dict[str, str]:
    """Resolve the frozen producer identity for an initialization checkpoint.

    A later adaptation release is expected to have different orchestration and
    data-loader code from the Phase-1 release that produced its parent. Resume
    checkpoints still require exact current-code equality; only the immutable
    parent boundary may declare the historical producer fingerprints explicitly.
    """
    parent = cfg.get("parent_checkpoint")
    if not isinstance(parent, Mapping):
        raise RuntimeError("adaptation config lacks parent checkpoint identity")
    declared = parent.get("code_fingerprints", current_code)
    if not isinstance(declared, Mapping) or not declared:
        raise RuntimeError("parent checkpoint code fingerprints are invalid")
    fingerprints = {str(name): str(digest) for name, digest in declared.items()}
    if any(
        not name or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for name, digest in fingerprints.items()
    ):
        raise RuntimeError("parent checkpoint code fingerprints are invalid")
    return fingerprints


def verify_parent_code_identity(
    cfg: Mapping[str, Any],
    parent_cfg: Mapping[str, Any],
    current_code: Mapping[str, str],
) -> dict[str, str]:
    expected = expected_parent_code_fingerprints(cfg, current_code)
    if parent_cfg.get("code_fingerprints") != expected:
        raise RuntimeError("parent MLM training code fingerprints do not match")
    return expected


def load_config(path: Path, data_root: Path | None) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("training_mode") != "pretraining":
        raise ValueError("config is not an UltraMS pretraining configuration")
    root = data_root or path.parent.parent
    cfg["config_path"] = str(path.resolve())
    cfg["data_root"] = str(root.resolve())
    cfg["paths"] = {
        key: str((root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in cfg["paths"].items()
    }
    names = [stage.get("name") for stage in cfg.get("stages", [])]
    if names != list(STAGE_ORDER):
        raise ValueError(f"stages must be exactly {STAGE_ORDER}, got {names}")
    validate_profiles(cfg)
    contract = {
        key: cfg[key]
        for key in (
            "training_mode",
            "seed",
            "supported_world_sizes",
            "distributed_profiles",
            "model",
            "optimization",
            "expected_assets",
            "stages",
        )
    }
    cfg["training_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    phase1_contract = {
        "training_mode": cfg["training_mode"],
        "seed": cfg["seed"],
        "supported_world_sizes": cfg["supported_world_sizes"],
        "distributed_profiles": {
            world_size: {"phase1": profile["phase1"]}
            for world_size, profile in cfg["distributed_profiles"].items()
        },
        "model": {
            key: cfg["model"][key] for key in ("max_peaks", "mask_ratio")
        },
        "optimization": {
            key: cfg["optimization"][key]
            for key in ("weight_decay", "gradient_clip", "num_workers", "precision")
        },
        "clean": cfg["expected_assets"]["clean"],
        "stage": cfg["stages"][0],
    }
    cfg["phase1_contract_sha256"] = hashlib.sha256(
        json.dumps(phase1_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return cfg


def validate_profiles(cfg: Mapping[str, Any]) -> None:
    supported = [int(value) for value in cfg.get("supported_world_sizes", [])]
    if supported != [4, 6]:
        raise ValueError("the reviewed H200 launch supports exactly four or six GPUs")
    for world_size in supported:
        profile = cfg["distributed_profiles"].get(str(world_size), {})
        for name, expected in (("phase1", 3840), ("finetune", 1152)):
            item = profile.get(name, {})
            batch = int(item.get("batch_size_per_rank", 0))
            accumulation = int(item.get("gradient_accumulation", 0))
            declared = int(item.get("global_optimizer_batch_size", 0))
            if min(batch, accumulation, declared) <= 0:
                raise ValueError(f"incomplete {world_size}-GPU {name} profile")
            if world_size * batch * accumulation != expected or declared != expected:
                raise ValueError(
                    f"{world_size}-GPU {name} profile changes the reviewed global batch"
                )
    for stage in cfg["stages"]:
        if int(stage["mlm_frequency"]) != 1:
            raise ValueError("the reviewed curriculum computes MLM on every batch")
    for stage in cfg["stages"][1:]:
        four = profile_for_stage(cfg, stage, 4)
        six = profile_for_stage(cfg, stage, 6)
        if (
            4 * four["batch_size_per_rank"] != 6 * six["batch_size_per_rank"]
            or four["gradient_accumulation"] != six["gradient_accumulation"]
        ):
            raise ValueError(
                f"{stage['name']} must expose identical global microbatches "
                "and accumulation on four and six GPUs"
            )
    if cfg["optimization"]["precision"] != "bf16":
        raise ValueError("the reviewed pretraining path requires BF16")
    reconstruction, supervised_rt, self_consistent_rt, ion_mode = cfg["stages"]
    if reconstruction["trainable_scope"] != "all_base":
        raise ValueError("peak reconstruction must train the complete base model")
    if supervised_rt["trainable_scope"] != "last_layers_and_heads":
        raise ValueError("supervised RT trainable scope differs from the reviewed run")
    for stage in (self_consistent_rt, ion_mode):
        if stage["trainable_scope"] != "last_layers_and_heads":
            raise ValueError(
                f"{stage['name']} trainable scope differs from the reviewed run"
            )
    if supervised_rt["rt_input_unit"] != "normalized_600s":
        raise ValueError("supervised RT input unit differs from the historical loader")
    if any(
        stage["rt_input_unit"] != "seconds" for stage in (self_consistent_rt, ion_mode)
    ):
        raise ValueError("clean RT input units differ from the Parquet loader")
    if (
        self_consistent_rt["polarity_weight"] != 0.0
        or ion_mode["polarity_weight"] <= 0.0
    ):
        raise ValueError("polarity loss must be enabled only for ion-mode training")
    expected = {
        "peak_reconstruction": {
            "epochs": 5,
            "scheduler_planned_epochs": 5,
            "learning_rate": 3e-5,
            "scheduler": "warmup_cosine",
            "rt_weight": 0.0,
            "polarity_weight": 0.0,
        },
        "supervised_rt": {
            "epochs": 5,
            "learning_rate": 1e-5,
            "scheduler": "legacy_warmup_cosine",
            "gradient_accumulation_multiplier": 3,
            "rt_weight": 0.5,
            "polarity_weight": 0.0,
        },
        "self_consistent_rt": {
            "epochs": 1,
            "learning_rate": 5e-6,
            "scheduler": "constant",
            "rt_weight": 0.1,
            "polarity_weight": 0.0,
        },
        "ion_mode": {
            "epochs": 11,
            "learning_rate": 5e-6,
            "scheduler": "constant",
            "rt_weight": 0.1,
            "polarity_weight": 0.1,
        },
    }
    for objective in cfg["stages"]:
        for key, value in expected[objective["name"]].items():
            if objective.get(key) != value:
                raise ValueError(
                    f"{objective['name']} {key} differs from the pretraining contract"
                )


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pyarrow", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def runtime_resume_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: environment[key]
        for key in (
            "python",
            "packages",
            "cuda_runtime",
            "cudnn",
            "gpu_names",
            "deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
        )
    }


def initialize_unified_model(model_class, model_config: Mapping[str, Any]):
    base = model_class(model_config)
    rng_after_base = common.capture_rng_state()
    model = UnifiedUltraMS(base, model_config)
    common.restore_rng_state(rng_after_base)
    return model


def profile_for_stage(
    cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any], world_size: int
) -> dict[str, int]:
    profile_name = str(stage_cfg["profile"])
    raw = cfg["distributed_profiles"][str(world_size)][profile_name]
    accumulation = int(raw["gradient_accumulation"])
    accumulation *= int(stage_cfg.get("gradient_accumulation_multiplier", 1))
    return {
        "batch_size_per_rank": int(raw["batch_size_per_rank"]),
        "gradient_accumulation": accumulation,
        "global_optimizer_batch_size": int(raw["global_optimizer_batch_size"])
        * int(stage_cfg.get("gradient_accumulation_multiplier", 1)),
    }


def loader_worker_seed(seed: int, stage: str, epoch: int, rank: int) -> int:
    if stage not in STAGE_ORDER or epoch <= 0 or rank < 0:
        raise ValueError("invalid DataLoader seed coordinates")
    return int(seed) + STAGE_ORDER.index(stage) * 100_000 + epoch + rank * 1_000_003


class GlobalOptimizerBatchSampler:
    """Partition identical shuffled optimizer batches at any world size."""

    def __init__(
        self,
        dataset_size: int,
        rank: int,
        world_size: int,
        batch_size_per_rank: int,
        gradient_accumulation: int,
        *,
        seed: int = 0,
    ) -> None:
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size_per_rank = int(batch_size_per_rank)
        self.gradient_accumulation = int(gradient_accumulation)
        self.seed = int(seed)
        self.epoch = 0
        self.local_optimizer_batch = (
            self.batch_size_per_rank * self.gradient_accumulation
        )
        self.global_microbatch = self.batch_size_per_rank * self.world_size
        self.global_optimizer_batch = self.local_optimizer_batch * self.world_size
        self.optimizer_steps = self.dataset_size // self.global_optimizer_batch
        self.usable_size = self.optimizer_steps * self.global_optimizer_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.optimizer_steps * self.local_optimizer_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator)[
            : self.usable_size
        ]
        global_batches = order.reshape(
            self.optimizer_steps,
            self.gradient_accumulation,
            self.global_microbatch,
        )
        start = self.rank * self.batch_size_per_rank
        stop = start + self.batch_size_per_rank
        return iter(global_batches[:, :, start:stop].reshape(-1).tolist())


def _manifest_stats(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    output = manifest.get("output", {})
    return output.get("stats", manifest.get("stats", {}))


def validate_inputs(
    cfg: Mapping[str, Any], project_root: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    report: dict[str, Any] = {"files": {}, "directories": {}, "errors": []}
    launch_phase = cfg.get("launch_phase")
    hq_assets = () if launch_phase == "mlm" else (
        "massspecgym_csv",
        "msnlib_csv",
        "spectraverse_csv",
    )
    for name in hq_assets:
        path = Path(cfg["paths"][name])
        exists = path.is_file()
        report["files"][name] = {
            "path": str(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
        }
        if not exists:
            report["errors"].append(f"missing file: {path}")
            continue
        expected = cfg["expected_assets"][name]
        if path.stat().st_size != int(expected["bytes"]):
            report["errors"].append(f"size mismatch for {name}: {path}")
        if verify_hashes:
            digest = common.sha256_file(path)
            report["files"][name]["sha256"] = digest
            if digest != expected["sha256"]:
                report["errors"].append(f"SHA-256 mismatch for {name}: {path}")

    optimizer_steps: dict[str, dict[str, int]] = {}
    datasets = [("clean", "clean_shard_dir")]
    if launch_phase != "mlm":
        datasets.append(("polarity", "polarity_shard_dir"))
    for dataset_key, path_key in datasets:
        shard_dir = Path(cfg["paths"][path_key])
        parquet_files = (
            sorted(shard_dir.glob("*.parquet")) if shard_dir.is_dir() else []
        )
        manifest_path = shard_dir / "manifest.json"
        directory: dict[str, Any] = {
            "path": str(shard_dir),
            "parquet_files": len(parquet_files),
            "manifest": str(manifest_path),
            "parallel_layouts": {},
        }
        report["directories"][path_key] = directory
        if not parquet_files:
            report["errors"].append(f"no parquet files: {shard_dir}")
            continue
        if not manifest_path.is_file():
            report["errors"].append(f"missing dataset manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats = _manifest_stats(manifest)
        output = manifest.get("output", {})
        records = sorted(
            output.get("file_records", []), key=lambda record: str(record["name"])
        )
        directory.update(
            status=manifest.get("status"),
            dataset_name=manifest.get("dataset_name"),
            dataset_fingerprint=manifest.get("dataset_fingerprint"),
            rows=stats.get("rows"),
            stats=stats,
        )
        expected = cfg["expected_assets"][dataset_key]
        checks = {
            "dataset_name": manifest.get("dataset_name"),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "input_content_fingerprint": manifest.get("input_content_fingerprint"),
            "run_identity": manifest.get("run_identity"),
            "schema_fingerprint": manifest.get("schema_fingerprint"),
            "rows": stats.get("rows"),
            "negative_0": stats.get("polarity", {}).get("negative_0"),
            "positive_1": stats.get("polarity", {}).get("positive_1"),
            "other_or_null": stats.get("polarity", {}).get("other_or_null"),
            "accepted_0_to_1500": stats.get("rt", {}).get("accepted_0_to_1500"),
            "above_1500": stats.get("rt", {}).get("above_1500"),
            "files": len(parquet_files),
            "seed": manifest.get("config", {}).get("seed", manifest.get("seed")),
        }
        if manifest.get("status") != "complete":
            report["errors"].append(f"incomplete dataset manifest: {manifest_path}")
        for key, expected_value in expected.items():
            if checks.get(key) != expected_value:
                report["errors"].append(
                    f"{dataset_key} {key} mismatch: "
                    f"{checks.get(key)} != {expected_value}"
                )
        if len(records) != len(parquet_files):
            report["errors"].append(
                f"{dataset_key} manifest file_records are incomplete: "
                f"{len(records)} != {len(parquet_files)}"
            )
            continue
        record_names = {str(record["name"]) for record in records}
        actual_names = {path.name for path in parquet_files}
        if record_names != actual_names:
            report["errors"].append(
                f"{dataset_key} manifest paths differ from local Parquet files"
            )
            continue
        if verify_hashes:
            for record in records:
                path = shard_dir / str(record["name"])
                if path.stat().st_size != int(record["bytes"]):
                    report["errors"].append(f"size mismatch: {path}")
                    continue
                if common.sha256_file(path) != record["sha256"]:
                    report["errors"].append(f"SHA-256 mismatch: {path}")
        if launch_phase == "mlm":
            stage_names = ("peak_reconstruction",)
        else:
            stage_names = (
                ("peak_reconstruction", "self_consistent_rt")
                if dataset_key == "clean"
                else ("ion_mode",)
            )
        for stage_name in stage_names:
            stage_cfg = next(
                stage for stage in cfg["stages"] if stage["name"] == stage_name
            )
            stage_steps: dict[str, int] = {}
            optimizer_steps[f"{dataset_key}:{stage_name}"] = stage_steps
            for world_size in cfg["supported_world_sizes"]:
                profile = profile_for_stage(cfg, stage_cfg, int(world_size))
                worker_rows_by_epoch: dict[str, list[list[int]]] = {}
                microbatches_by_epoch: dict[str, list[int]] = {}
                optimizer_steps_by_epoch: dict[str, int] = {}
                for epoch in range(1, int(stage_cfg["epochs"]) + 1):
                    worker_rows, batches = parquet_worker_parallel_layout(
                        records,
                        int(world_size),
                        int(profile["batch_size_per_rank"]),
                        int(cfg["optimization"]["num_workers"]),
                        int(cfg["seed"]),
                        epoch,
                        drop_last=True,
                    )
                    if len(set(batches)) != 1:
                        report["errors"].append(
                            f"{dataset_key}/{stage_name} is not synchronized for "
                            f"{world_size} GPUs at epoch {epoch}: {batches}"
                        )
                    worker_rows_by_epoch[str(epoch)] = worker_rows
                    microbatches_by_epoch[str(epoch)] = batches
                    optimizer_steps_by_epoch[str(epoch)] = (
                        min(batches) // int(profile["gradient_accumulation"])
                    )
                steps = optimizer_steps_by_epoch["1"]
                stage_steps[str(world_size)] = steps
                directory["parallel_layouts"][f"{stage_name}:{world_size}"] = {
                    "batch_size_per_rank": profile["batch_size_per_rank"],
                    "gradient_accumulation": profile["gradient_accumulation"],
                    "worker_rows_by_rank_by_epoch": worker_rows_by_epoch,
                    "microbatches_per_rank_by_epoch": microbatches_by_epoch,
                    "optimizer_steps_per_epoch": steps,
                    "optimizer_steps_by_epoch": optimizer_steps_by_epoch,
                }

    report["optimizer_steps_per_epoch"] = optimizer_steps

    for path in (
        project_root / "train" / "train_ue_multiscale_v9_mlm.py",
        project_root / "pretrain" / "dataset_sharded.py",
        project_root / "train" / "phase2" / "dataset_hq.py",
    ):
        if not path.is_file():
            report["errors"].append(f"missing code dependency: {path}")
    return report


@dataclass
class UnifiedResumeState(common.ResumeState):
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UnifiedResumeState:
        return cls(
            stage=str(value["stage"]),
            epoch=int(value["epoch"]),
            batch_in_epoch=int(value["batch_in_epoch"]),
            optimizer_step=int(value["optimizer_step"]),
            stage_optimizer_step=int(value["stage_optimizer_step"]),
        )


class UnifiedUltraMS(common.Phase2Model):
    """Phase-1 compatible model for the reviewed four-stage curriculum."""

    def _freeze_structurally_unused_parameters(self) -> None:
        # BertModel is always called with inputs_embeds, so its vocabulary
        # embedding is intentionally outside the UltraMS computation graph.
        embeddings = getattr(self.base.encoder, "embeddings", None)
        word_embeddings = getattr(embeddings, "word_embeddings", None)
        if word_embeddings is not None:
            word_embeddings.requires_grad_(False)

    def configure_trainable(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if stage == "peak_reconstruction":
            for parameter in self.base.parameters():
                parameter.requires_grad = True
            self._freeze_structurally_unused_parameters()
            return
        # Historical Phase 2 kept both heads in every Ae/C/D optimizer; only
        # the stage loss weights changed.
        for parameter in self.base.parameters():
            parameter.requires_grad = True
        for parameter in self.rt_head.parameters():
            parameter.requires_grad = True
        for parameter in self.pol_head.parameters():
            parameter.requires_grad = True
        self._freeze_structurally_unused_parameters()
        self.freeze_backbone(int(self.config["freeze_layers_after_phase1"]))

    def forward(
        self,
        batch: Mapping[str, Any],
        device: torch.device,
        stage: str,
    ) -> dict[str, Any]:
        if stage == "peak_reconstruction":
            output = self.base(batch, device)
            result = {"loss": output["loss"], "mlm_loss": output["loss"].detach()}
            for key in (
                "mz_loss",
                "level0_loss",
                "level1_loss",
                "level2_loss",
                "int_loss",
                "mz_acc",
                "level0_acc",
                "level1_acc",
                "level2_acc",
                "int_acc",
                "mz_acc_1bin",
                "mz_acc_5bin",
                "high_mz_acc",
            ):
                if key in output:
                    result[key] = output[key]
            return result
        if stage in LEGACY_OBJECTIVE:
            return super().forward(batch, device, LEGACY_OBJECTIVE[stage])
        raise ValueError(f"unknown stage: {stage}")


class MetricWindow(common.MetricWindow):
    METRICS = common.MetricWindow.METRICS + (
        "mz_loss",
        "level0_loss",
        "level1_loss",
        "level2_loss",
        "int_loss",
        "level0_acc",
        "level1_acc",
        "level2_acc",
        "int_acc",
        "mz_acc_1bin",
        "mz_acc_5bin",
        "high_mz_acc",
    )

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.counts = {
            "rt_valid_samples": 0.0,
            "rt_selected_samples": 0.0,
            "rt_zero_selected_updates": 0.0,
            "polarity_samples": 0.0,
        }

    def update(self, result: Mapping[str, Any], batch_size: int) -> None:
        super().update(result, batch_size)
        self.counts["rt_valid_samples"] += float(result.get("rt_n", 0))
        self.counts["rt_selected_samples"] += float(result.get("rt_n_sc", 0))
        self.counts["polarity_samples"] += float(result.get("pol_n", 0))
        if "rt_n_sc" in result and int(result["rt_n_sc"]) < 2:
            self.counts["rt_zero_selected_updates"] += 1.0

    def reduce_and_reset(self) -> dict[str, float]:
        reduced = super().reduce_and_reset()
        names = tuple(self.counts)
        values = torch.tensor(
            [self.counts[name] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        reduced.update(
            {name: float(values[index].item()) for index, name in enumerate(names)}
        )
        valid = reduced["rt_valid_samples"]
        reduced["rt_selection_rate"] = (
            reduced["rt_selected_samples"] / valid if valid else 0.0
        )
        self.counts = {name: 0.0 for name in names}
        return reduced

    def state_dict(self) -> dict[str, Any]:
        return {
            "values": {name: list(value) for name, value in self.values.items()},
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        values = state.get("values", {})
        counts = state.get("counts", {})
        if set(values) != set(self.values) or set(counts) != set(self.counts):
            raise RuntimeError("checkpoint metric accumulator schema does not match")
        self.values = {
            name: [float(pair[0]), float(pair[1])] for name, pair in values.items()
        }
        self.counts = {name: float(value) for name, value in counts.items()}


def historical_phase1_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    no_decay = (
        "bias",
        "LayerNorm",
        "layer_norm",
        "Embedding",
        "embedding",
        "codebook",
        "alpha",
        "type_emb",
        "cls_emb",
        "precursor_type_emb",
    )
    decay: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay_parameters if any(key in name for key in no_decay) else decay
        target.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def make_loader(
    stage: str,
    cfg: Mapping[str, Any],
    profile: Mapping[str, int],
    rank: int,
    world_size: int,
    create_parquet,
    *,
    dataset_key: str | None = None,
):
    if stage == "supervised_rt":
        from phase2 import dataset_hq  # type: ignore
        from torch.utils.data import DataLoader

        dataset_hq.DATASETS["masspecgym"]["csv"] = cfg["paths"]["massspecgym_csv"]
        dataset_hq.DATASETS["msnlib"]["csv"] = cfg["paths"]["msnlib_csv"]
        dataset_hq.DATASETS["spectraverse"]["csv"] = cfg["paths"]["spectraverse_csv"]
        dataset = dataset_hq.HQMixedDataset(
            fold="train", max_peaks=int(cfg["model"]["max_peaks"])
        )
        wrapped = dataset_hq.PairSamplerWrapper(dataset)
        expected_hq = cfg["expected_assets"]["hq_rt"]
        if len(wrapped) != int(expected_hq["loader_eligible_rows"]):
            raise RuntimeError(
                "parsed HQ RT dataset size differs from the frozen contract: "
                f"{len(wrapped)} != {expected_hq['loader_eligible_rows']}"
            )
        sampler = GlobalOptimizerBatchSampler(
            len(wrapped),
            rank,
            world_size,
            int(profile["batch_size_per_rank"]),
            int(profile["gradient_accumulation"]),
            # DistributedSampler historically used seed=0.
            seed=0,
        )
        if sampler.optimizer_steps != int(expected_hq["optimizer_steps_per_epoch"]):
            raise RuntimeError("HQ RT optimizer-step count differs from the contract")
        workers = int(cfg["optimization"]["num_workers"])
        return DataLoader(
            wrapped,
            batch_size=int(profile["batch_size_per_rank"]),
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            collate_fn=dataset_hq.collate_fn_phase2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if workers else None,
        )
    selected_dataset = dataset_key or ("polarity" if stage == "ion_mode" else "clean")
    if selected_dataset not in ("clean", "polarity"):
        raise ValueError(f"unknown dataset key: {selected_dataset}")
    return create_parquet(
        shard_dir=cfg["paths"][f"{selected_dataset}_shard_dir"],
        batch_size=int(profile["batch_size_per_rank"]),
        max_peaks=int(cfg["model"]["max_peaks"]),
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        rank=rank,
        world_size=world_size,
        num_workers=int(cfg["optimization"]["num_workers"]),
        seed=int(cfg["seed"]),
        drop_last=True,
    )


def stage_optimizer_steps_per_epoch(
    cfg: Mapping[str, Any],
    dataset_key: str,
    profile: Mapping[str, int],
    world_size: int,
    *,
    epoch: int = 1,
    drop_last: bool,
) -> int:
    manifest = json.loads(
        (Path(cfg["paths"][f"{dataset_key}_shard_dir"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    records = sorted(
        manifest["output"]["file_records"], key=lambda record: str(record["name"])
    )
    _, batches = parquet_worker_parallel_layout(
        records,
        world_size,
        int(profile["batch_size_per_rank"]),
        int(cfg["optimization"]["num_workers"]),
        int(cfg["seed"]),
        epoch,
        drop_last=drop_last,
    )
    if len(set(batches)) != 1:
        raise RuntimeError(
            f"{dataset_key} exposes different microbatch counts across "
            f"world_size={world_size} ranks at epoch={epoch}: {batches}"
        )
    accumulation = int(profile["gradient_accumulation"])
    if drop_last:
        return batches[0] // accumulation
    return math.ceil(batches[0] / accumulation)


def parquet_worker_parallel_layout(
    file_records: list[Mapping[str, Any]],
    world_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
    *,
    drop_last: bool,
) -> tuple[list[list[int]], list[int]]:
    """Mirror the sharded iterable loader's rank and worker batch layout."""
    if min(world_size, batch_size) <= 0 or num_workers < 0 or epoch <= 0:
        raise ValueError("invalid rank/worker layout arguments")
    if len(file_records) % world_size:
        raise ValueError(
            f"{len(file_records)} shards are not divisible by world_size={world_size}"
        )
    loader_workers = min(num_workers, 2)
    logical_workers = max(1, loader_workers)
    per_rank = len(file_records) // world_size
    worker_rows_by_rank: list[list[int]] = []
    microbatches_by_rank: list[int] = []
    for rank in range(world_size):
        shard_ids = list(range(rank, per_rank * world_size, world_size))[:per_rank]
        rng = np.random.RandomState(seed + epoch * 100 + rank)
        rng.shuffle(shard_ids)
        worker_rows = [
            sum(
                int(file_records[shard_id]["rows"])
                for shard_id in shard_ids[worker_id::logical_workers]
            )
            for worker_id in range(logical_workers)
        ]
        if drop_last:
            worker_batches = [rows // batch_size for rows in worker_rows]
        else:
            worker_batches = [math.ceil(rows / batch_size) for rows in worker_rows]
        worker_rows_by_rank.append(worker_rows)
        microbatches_by_rank.append(sum(worker_batches))
    return worker_rows_by_rank, microbatches_by_rank


def scheduler_plan_from_checkpoint(
    checkpoint: Mapping[str, Any],
    world_size: int,
    fallback: Mapping[str, int],
) -> dict[str, int]:
    """Restore serialized scheduler geometry, or infer it for older checkpoints."""
    extra_by_rank = checkpoint.get("extra_state_by_rank", [])
    if not isinstance(extra_by_rank, list) or len(extra_by_rank) != world_size:
        raise RuntimeError("checkpoint lacks one progress state per rank")
    raw_plans = [
        extra.get("scheduler_plan") if isinstance(extra, Mapping) else None
        for extra in extra_by_rank
    ]
    if all(plan is None for plan in raw_plans):
        return normalize_scheduler_plan(fallback)
    if any(plan is None for plan in raw_plans):
        raise RuntimeError("checkpoint scheduler plan is missing on some ranks")
    plans = [normalize_scheduler_plan(plan) for plan in raw_plans]
    if any(plan != plans[0] for plan in plans[1:]):
        raise RuntimeError("checkpoint scheduler plans differ across ranks")
    return plans[0]


def normalize_scheduler_plan(plan: Mapping[str, Any]) -> dict[str, int]:
    total_steps = int(plan["total_steps"])
    warmup_steps = int(plan["warmup_steps"])
    if total_steps <= 0 or not 0 <= warmup_steps <= total_steps:
        raise RuntimeError(f"invalid scheduler plan: {dict(plan)}")
    return {"total_steps": total_steps, "warmup_steps": warmup_steps}


def checkpoint_progress_state(
    epoch_metrics: MetricWindow,
    micro_step: int,
    clean_batches: int,
    polarity_batches: int,
    scheduler_plan: Mapping[str, int] | None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "epoch_metrics": epoch_metrics.state_dict(),
        "micro_step": int(micro_step),
        "clean_batches": int(clean_batches),
        "polarity_batches": int(polarity_batches),
    }
    if scheduler_plan is not None:
        state["scheduler_plan"] = normalize_scheduler_plan(scheduler_plan)
    return state


def _select_stages(
    cfg: Mapping[str, Any], start: str, end: str
) -> list[dict[str, Any]]:
    start_index = STAGE_ORDER.index(start)
    end_index = STAGE_ORDER.index(end)
    if start_index > end_index:
        raise ValueError(f"start stage {start} is after end stage {end}")
    return [dict(stage) for stage in cfg["stages"][start_index : end_index + 1]]


def validate_stage_entry(
    start: str,
    end: str,
    resume_stage: str | None,
    initialized_from_mlm: bool = False,
) -> None:
    start_index = STAGE_ORDER.index(start)
    end_index = STAGE_ORDER.index(end)
    if start_index > end_index:
        raise ValueError(f"start stage {start} is after end stage {end}")
    if resume_stage is None and start != "peak_reconstruction" and not initialized_from_mlm:
        raise RuntimeError(
            "a from-scratch run without a checkpoint must start at peak reconstruction"
        )
    if initialized_from_mlm and start != "supervised_rt":
        raise RuntimeError("an MLM-initialized run must start at supervised RT")
    if resume_stage is not None:
        resume_index = STAGE_ORDER.index(resume_stage)
        if not start_index <= resume_index <= end_index:
            raise RuntimeError(
                f"resume stage {resume_stage} is outside the selected stage range"
            )


def train_worker(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {int(value) for value in cfg["supported_world_sizes"]}:
        raise ValueError(f"unsupported WORLD_SIZE={world_size}")
    cfg["world_size"] = world_size
    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    common.set_seed(int(cfg["seed"]), rank)

    project_root = args.project_root.resolve()
    cfg["code_fingerprints"] = training_code_fingerprints(project_root)
    v9_config, model_class, build_lr_scheduler, create_parquet = common.import_project(
        project_root
    )
    model_config = {
        **v9_config,
        **cfg["model"],
        "freeze_layers": cfg["model"]["freeze_layers_after_phase1"],
    }
    # The successful reconstruction run instantiated only the base MLM model.
    # Later heads must not advance its masking/dropout RNG trajectory.
    model = initialize_unified_model(model_class, model_config)

    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = str(args.output_dir.resolve())
    output_dir = Path(cfg["paths"]["output_dir"])
    if str(output_dir).startswith("/mnt/s3"):
        raise ValueError("live checkpoints must use worker-local storage")
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["runtime_environment"] = runtime_environment()
    if rank == 0:
        common.atomic_json(output_dir / "resolved_config.json", cfg)
        common.atomic_json(
            output_dir / "runtime_environment.json", cfg["runtime_environment"]
        )
        input_report = validate_inputs(cfg, project_root, verify_hashes=True)
        common.atomic_json(output_dir / "input_validation.json", input_report)
    else:
        input_report = None
    if dist.is_initialized():
        shared_report = [input_report]
        dist.broadcast_object_list(shared_report, src=0)
        input_report = shared_report[0]
    assert input_report is not None
    if input_report["errors"]:
        raise RuntimeError(
            "training input validation failed:\n" + "\n".join(input_report["errors"])
        )
    dataset_keys = ("clean",) if cfg.get("launch_phase") == "mlm" else (
        "clean",
        "polarity",
    )
    cfg["dataset_fingerprints"] = {
        key: input_report["directories"][f"{key}_shard_dir"].get("dataset_fingerprint")
        for key in dataset_keys
    }
    if rank == 0:
        # Persist the resolved data identities as part of the run contract.
        common.atomic_json(output_dir / "resolved_config.json", cfg)

    if args.resume is not None and args.initialize_from is not None:
        raise RuntimeError("--resume and --initialize-from are mutually exclusive")
    resume_path = common.find_resume_checkpoint(output_dir, args.resume)
    resume_checkpoint = None
    resume_state = None
    initialization_checkpoint = None
    initialization_rng_by_rank = None
    if resume_path:
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        saved_cfg = resume_checkpoint.get("config", {})
        if saved_cfg.get("training_contract_sha256") != cfg["training_contract_sha256"]:
            raise RuntimeError("resume checkpoint training contract does not match")
        if int(saved_cfg.get("world_size", -1)) != world_size:
            raise RuntimeError("resume checkpoint world size does not match")
        if saved_cfg.get("dataset_fingerprints") != cfg["dataset_fingerprints"]:
            raise RuntimeError("resume checkpoint dataset fingerprints do not match")
        if saved_cfg.get("code_fingerprints") != cfg["code_fingerprints"]:
            raise RuntimeError(
                "resume checkpoint training code fingerprints do not match"
            )
        saved_environment = saved_cfg.get("runtime_environment")
        if not isinstance(saved_environment, Mapping) or runtime_resume_identity(
            saved_environment
        ) != runtime_resume_identity(cfg["runtime_environment"]):
            raise RuntimeError("resume checkpoint runtime environment does not match")
        metrics_snapshot = resume_checkpoint.get("metrics_snapshot")
        if not isinstance(metrics_snapshot, Mapping):
            raise RuntimeError("resume checkpoint lacks its exact metrics snapshot")
        snapshot_text = metrics_snapshot.get("text")
        snapshot_sha = str(metrics_snapshot.get("sha256", ""))
        if (
            not isinstance(snapshot_text, str)
            or hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest() != snapshot_sha
        ):
            raise RuntimeError("resume checkpoint metrics snapshot is invalid")
        if rank == 0 and (
            not metrics_path.exists()
            or metrics_path.read_text(encoding="utf-8") != snapshot_text
        ):
            metrics_tmp = metrics_path.with_suffix(".jsonl.tmp")
            metrics_tmp.write_text(snapshot_text, encoding="utf-8")
            os.replace(metrics_tmp, metrics_path)
        if dist.is_initialized():
            dist.barrier()
        if metrics_path.read_text(encoding="utf-8") != snapshot_text:
            raise RuntimeError(
                "restored metrics history does not match resume checkpoint"
            )
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        resume_state = UnifiedResumeState.from_mapping(
            resume_checkpoint["resume_state"]
        )
    elif args.initialize_from is not None:
        initialization_checkpoint = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        parent_cfg = initialization_checkpoint.get("config", {})
        parent_state = UnifiedResumeState.from_mapping(
            initialization_checkpoint["resume_state"]
        )
        expected_parent = cfg.get("parent_checkpoint")
        if not isinstance(expected_parent, Mapping):
            raise RuntimeError("adaptation config lacks parent checkpoint identity")
        if common.sha256_file(args.initialize_from) != expected_parent.get("sha256"):
            raise RuntimeError("parent checkpoint SHA-256 differs from launch identity")
        if parent_cfg.get("phase1_contract_sha256") != cfg["phase1_contract_sha256"]:
            raise RuntimeError("parent MLM phase-1 contract does not match")
        if int(parent_cfg.get("world_size", -1)) != world_size:
            raise RuntimeError("parent MLM world size does not match")
        if parent_cfg.get("dataset_fingerprints", {}).get("clean") != cfg[
            "dataset_fingerprints"
        ].get("clean"):
            raise RuntimeError("parent MLM clean dataset fingerprint does not match")
        verify_parent_code_identity(cfg, parent_cfg, cfg["code_fingerprints"])
        if parent_state != UnifiedResumeState(
            stage="peak_reconstruction",
            epoch=6,
            batch_in_epoch=0,
            optimizer_step=parent_state.optimizer_step,
            stage_optimizer_step=parent_state.stage_optimizer_step,
        ):
            raise RuntimeError("parent checkpoint is not the terminal five-epoch MLM state")
        initialization_rng_by_rank = initialization_checkpoint.get(
            "rng_state_by_rank", []
        )
        if len(initialization_rng_by_rank) != world_size:
            raise RuntimeError("parent checkpoint lacks one RNG state per rank")
        model.load_state_dict(initialization_checkpoint["model_state_dict"], strict=True)

    validate_stage_entry(
        args.start_stage,
        args.end_stage,
        resume_state.stage if resume_state else None,
        initialization_checkpoint is not None,
    )

    if rank == 0:
        common.append_jsonl(
            metrics_path,
            {
                "event": "run_start",
                "run_name": cfg["run_name"],
                "training_mode": cfg["training_mode"],
                "launch_phase": cfg.get("launch_phase"),
                "world_size": world_size,
                "resume_checkpoint": str(resume_path) if resume_path else None,
                "initialize_from": (
                    str(args.initialize_from)
                    if initialization_checkpoint is not None
                    else None
                ),
                "time_unix": time.time(),
            },
        )

    model = model.to(device)
    global_optimizer_step = resume_state.optimizer_step if resume_state else 0
    stages = _select_stages(cfg, args.start_stage, args.end_stage)
    if resume_state:
        stages = [
            stage
            for stage in stages
            if STAGE_ORDER.index(stage["name"]) >= STAGE_ORDER.index(resume_state.stage)
        ]
        resumed_stage = next(
            stage for stage in cfg["stages"] if stage["name"] == resume_state.stage
        )
        if resume_state.batch_in_epoch == 0 and resume_state.epoch > int(
            resumed_stage["epochs"]
        ):
            assert resume_checkpoint is not None
            rng_by_rank = resume_checkpoint.get("rng_state_by_rank", [])
            if len(rng_by_rank) != world_size:
                raise RuntimeError("checkpoint lacks one RNG state per rank")
            common.restore_rng_state(rng_by_rank[rank])

    for stage_cfg in stages:
        stage = stage_cfg["name"]
        model.configure_trainable(stage)
        model.config.update(
            rt_weight=float(stage_cfg["rt_weight"]),
            polarity_weight=float(stage_cfg["polarity_weight"]),
        )
        profile = profile_for_stage(cfg, stage_cfg, world_size)
        loader = None
        loader = make_loader(stage, cfg, profile, rank, world_size, create_parquet)
        phase1_optimizer_steps_by_epoch: list[int] | None = None
        if stage == "supervised_rt":
            assert loader is not None
            steps_per_epoch = len(loader) // int(profile["gradient_accumulation"])
        else:
            dataset_key = "polarity" if stage == "ion_mode" else "clean"
            if stage == "peak_reconstruction":
                phase1_optimizer_steps_by_epoch = [
                    stage_optimizer_steps_per_epoch(
                        cfg,
                        dataset_key,
                        profile,
                        world_size,
                        epoch=epoch,
                        drop_last=True,
                    )
                    for epoch in range(1, int(stage_cfg["epochs"]) + 1)
                ]
                steps_per_epoch = phase1_optimizer_steps_by_epoch[0]
            else:
                steps_per_epoch = stage_optimizer_steps_per_epoch(
                    cfg,
                    dataset_key,
                    profile,
                    world_size,
                    drop_last=True,
                )
        wrapped: nn.Module = model
        if world_size > 1:
            wrapped = DDP(
                model,
                device_ids=[local_rank] if use_cuda else None,
                find_unused_parameters=False,
            )
        groups = (
            historical_phase1_parameter_groups(
                wrapped, float(cfg["optimization"]["weight_decay"])
            )
            if stage == "peak_reconstruction"
            else common.trainable_parameter_groups(
                wrapped, float(cfg["optimization"]["weight_decay"])
            )
        )
        optimizer = torch.optim.AdamW(
            groups, lr=float(stage_cfg["learning_rate"])
        )
        scheduler = None
        scheduler_plan: dict[str, int] | None = None
        if stage_cfg["scheduler"] == "warmup_cosine":
            planned_epochs = int(
                stage_cfg.get("scheduler_planned_epochs", stage_cfg["epochs"])
            )
            scheduler_total_steps = steps_per_epoch * planned_epochs
            if stage == "peak_reconstruction":
                assert phase1_optimizer_steps_by_epoch is not None
                recalibrated_resume = (
                    resume_state is not None
                    and resume_state.stage == stage
                    and resume_state.epoch
                    > int(stage_cfg["recalibrate_scheduler_after_epoch"])
                )
                if recalibrated_resume:
                    scheduler_total_steps = sum(phase1_optimizer_steps_by_epoch)
                else:
                    scheduler_total_steps = (
                        int(stage_cfg["initial_estimated_steps_per_epoch"])
                        * planned_epochs
                    )
            warmup = int(scheduler_total_steps * float(stage_cfg["warmup_ratio"]))
            scheduler_plan = {
                "total_steps": scheduler_total_steps,
                "warmup_steps": warmup,
            }
            if resume_state is not None and resume_state.stage == stage:
                assert resume_checkpoint is not None
                scheduler_plan = scheduler_plan_from_checkpoint(
                    resume_checkpoint,
                    world_size,
                    scheduler_plan,
                )
            scheduler = build_lr_scheduler(
                optimizer,
                scheduler_plan["warmup_steps"],
                scheduler_plan["total_steps"],
                float(stage_cfg["minimum_learning_rate"])
                / float(stage_cfg["learning_rate"]),
            )
        elif stage_cfg["scheduler"] == "legacy_warmup_cosine":
            assert loader is not None
            total_microbatches = int(
                stage_cfg.get(
                    "scheduler_total_steps_override",
                    len(loader) * int(stage_cfg["epochs"]),
                )
            )
            if total_microbatches <= 0:
                raise RuntimeError("legacy scheduler total steps must be positive")
            warmup = int(total_microbatches * float(stage_cfg["warmup_ratio"]))
            scheduler_plan = {
                "total_steps": total_microbatches,
                "warmup_steps": warmup,
            }
            if resume_state is not None and resume_state.stage == stage:
                assert resume_checkpoint is not None
                scheduler_plan = scheduler_plan_from_checkpoint(
                    resume_checkpoint,
                    world_size,
                    scheduler_plan,
                )
            scheduler = build_lr_scheduler(
                optimizer,
                scheduler_plan["warmup_steps"],
                scheduler_plan["total_steps"],
                float(stage_cfg["minimum_learning_rate"])
                / float(stage_cfg["learning_rate"]),
            )
        # Native BF16 has no dynamic loss scaling. Disabling the scaler also
        # prevents a skipped optimizer update from corrupting exact step counts.
        scaler = GradScaler("cuda", enabled=False)

        first_epoch = 1
        skip_batches = 0
        stage_optimizer_step = 0
        if resume_state and resume_state.stage == stage:
            assert resume_checkpoint is not None
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            if scheduler is not None and resume_checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            first_epoch = resume_state.epoch
            skip_batches = resume_state.batch_in_epoch
            stage_optimizer_step = resume_state.stage_optimizer_step

        if rank == 0:
            common.append_jsonl(
                metrics_path,
                {
                    "event": "stage_start",
                    "stage": stage,
                    "epochs": int(stage_cfg["epochs"]),
                    "batch_size_per_rank": profile["batch_size_per_rank"],
                    "gradient_accumulation": profile["gradient_accumulation"],
                    "optimizer_steps_per_epoch": steps_per_epoch,
                    "optimizer_steps_by_epoch": phase1_optimizer_steps_by_epoch,
                    "scheduler_plan": scheduler_plan,
                    "dataset": (
                        "polarity"
                        if stage == "ion_mode"
                        else "hq_rt"
                        if stage == "supervised_rt"
                        else "clean"
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                },
            )

        for epoch in range(first_epoch, int(stage_cfg["epochs"]) + 1):
            current_skip = skip_batches if epoch == first_epoch else 0
            clean_batches_in_epoch = 0
            polarity_batches_in_epoch = 0
            iterator = None
            assert loader is not None
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            loader.generator = torch.Generator().manual_seed(
                loader_worker_seed(int(cfg["seed"]), stage, epoch, rank)
            )
            iterator = iter(loader)
            for _ in range(current_skip):
                try:
                    next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        f"resume batch offset {current_skip} exceeds "
                        f"stage={stage} epoch={epoch}"
                    ) from error
            if resume_state and resume_state.stage == stage and epoch == first_epoch:
                assert resume_checkpoint is not None
                rng_by_rank = resume_checkpoint.get("rng_state_by_rank", [])
                if len(rng_by_rank) != world_size:
                    raise RuntimeError("checkpoint lacks one RNG state per rank")
                common.restore_rng_state(rng_by_rank[rank])
            elif initialization_rng_by_rank is not None:
                common.restore_rng_state(initialization_rng_by_rank[rank])
                initialization_rng_by_rank = None

            wrapped.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation = int(profile["gradient_accumulation"])
            micro_step = 0
            batch_in_epoch = current_skip
            metrics = MetricWindow(device)
            epoch_metrics = MetricWindow(device)
            if resume_state and resume_state.stage == stage and epoch == first_epoch:
                assert resume_checkpoint is not None
                extra_by_rank = resume_checkpoint.get("extra_state_by_rank", [])
                if len(extra_by_rank) != world_size:
                    raise RuntimeError(
                        "checkpoint lacks one metric accumulator state per rank"
                    )
                extra_state = extra_by_rank[rank]
                epoch_metrics.load_state_dict(extra_state["epoch_metrics"])
                micro_step = int(extra_state["micro_step"])
                clean_batches_in_epoch = int(extra_state["clean_batches"])
                polarity_batches_in_epoch = int(extra_state["polarity_batches"])
            while True:
                assert iterator is not None
                try:
                    batch = next(iterator)
                    has_data = torch.ones(1, dtype=torch.int32, device=device)
                except StopIteration:
                    batch = None
                    has_data = torch.zeros(1, dtype=torch.int32, device=device)
                if dist.is_initialized():
                    dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if not has_data.item():
                    break
                assert batch is not None
                batch_in_epoch += 1
                if stage == "ion_mode":
                    polarity_batches_in_epoch += 1
                elif stage in ("peak_reconstruction", "self_consistent_rt"):
                    clean_batches_in_epoch += 1
                micro_step += 1
                sync_step = micro_step % accumulation == 0
                sync_context = contextlib.nullcontext()
                if isinstance(wrapped, DDP) and not sync_step:
                    sync_context = wrapped.no_sync()
                amp_context = (
                    autocast("cuda", dtype=torch.bfloat16)
                    if use_cuda and cfg["optimization"]["precision"] == "bf16"
                    else contextlib.nullcontext()
                )
                mlm_frequency = int(stage_cfg.get("mlm_frequency", 1))
                with sync_context, amp_context:
                    result = wrapped(
                        batch,
                        device,
                        stage,
                    )
                    loss = result["loss"] / accumulation
                scaler.scale(loss).backward()
                batch_size = int(batch["orig_spectra"].shape[0])
                metrics.update(result, batch_size)
                epoch_metrics.update(result, batch_size)
                if not sync_step:
                    continue
                scaler.unscale_(optimizer)
                gradient_norm = nn.utils.clip_grad_norm_(
                    wrapped.parameters(), float(cfg["optimization"]["gradient_clip"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"non-finite gradient at stage={stage} epoch={epoch} "
                        f"batch={batch_in_epoch}"
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_optimizer_step += 1
                stage_optimizer_step += 1

                if (
                    stage_optimizer_step
                    % int(cfg["optimization"]["log_every_optimizer_steps"])
                    == 0
                ):
                    reduced = metrics.reduce_and_reset()
                    if rank == 0:
                        common.append_jsonl(
                            metrics_path,
                            {
                                "event": "train_window",
                                "stage": stage,
                                "epoch": epoch,
                                "batch_in_epoch": batch_in_epoch,
                                "optimizer_step": global_optimizer_step,
                                "stage_optimizer_step": stage_optimizer_step,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "mlm_frequency": mlm_frequency,
                                **reduced,
                            },
                        )

                if (
                    stage_optimizer_step
                    % int(
                        stage_cfg.get(
                            "save_every_optimizer_steps",
                            cfg["optimization"]["save_every_optimizer_steps"],
                        )
                    )
                    == 0
                ):
                    state = UnifiedResumeState(
                        stage=stage,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        optimizer_step=global_optimizer_step,
                        stage_optimizer_step=stage_optimizer_step,
                    )
                    common.save_checkpoint(
                        checkpoint_dir
                        / f"stage_{stage}_step_{stage_optimizer_step}.pt",
                        wrapped,
                        optimizer,
                        scaler,
                        scheduler,
                        state,
                        cfg,
                        rank,
                        world_size,
                        checkpoint_progress_state(
                            epoch_metrics,
                            micro_step,
                            clean_batches_in_epoch,
                            polarity_batches_in_epoch,
                            scheduler_plan,
                        ),
                        metrics_path,
                    )

            remainder = micro_step % accumulation
            if remainder:
                if stage in ("peak_reconstruction", "supervised_rt"):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    common.synchronize_incomplete_accumulation(
                        wrapped, world_size, accumulation, remainder
                    )
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        wrapped.parameters(),
                        float(cfg["optimization"]["gradient_clip"]),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_optimizer_step += 1
                    stage_optimizer_step += 1

            if stage == "peak_reconstruction":
                assert phase1_optimizer_steps_by_epoch is not None
                expected_stage_steps = sum(phase1_optimizer_steps_by_epoch[:epoch])
                if stage_optimizer_step != expected_stage_steps:
                    raise RuntimeError(
                        "peak-reconstruction optimizer-step count differs from "
                        f"the worker/drop_last layout at epoch {epoch}: "
                        f"{stage_optimizer_step} != {expected_stage_steps}"
                    )

            if stage == "peak_reconstruction" and epoch == int(
                stage_cfg["recalibrate_scheduler_after_epoch"]
            ):
                assert phase1_optimizer_steps_by_epoch is not None
                scheduler_plan = {
                    "total_steps": sum(phase1_optimizer_steps_by_epoch),
                    "warmup_steps": int(
                        sum(phase1_optimizer_steps_by_epoch)
                        * float(stage_cfg["warmup_ratio"])
                    ),
                }
                scheduler = build_lr_scheduler(
                    optimizer,
                    scheduler_plan["warmup_steps"],
                    scheduler_plan["total_steps"],
                    float(stage_cfg["minimum_learning_rate"])
                    / float(stage_cfg["learning_rate"]),
                )
                for _ in range(stage_optimizer_step):
                    scheduler.step()
                if rank == 0:
                    common.append_jsonl(
                        metrics_path,
                        {
                            "event": "scheduler_recalibrated",
                            "stage": stage,
                            "epoch": epoch,
                            "actual_optimizer_steps_per_epoch": (
                                phase1_optimizer_steps_by_epoch[epoch - 1]
                            ),
                            "total_scheduler_steps": scheduler_plan["total_steps"],
                            "warmup_steps": scheduler_plan["warmup_steps"],
                        },
                    )

            epoch_summary = epoch_metrics.reduce_and_reset()
            if rank == 0:
                common.append_jsonl(
                    metrics_path,
                    {
                        "event": "epoch_end",
                        "stage": stage,
                        "epoch": epoch,
                        "optimizer_step": global_optimizer_step,
                        "stage_optimizer_step": stage_optimizer_step,
                        "microbatches": micro_step,
                        "clean_batches": clean_batches_in_epoch,
                        "polarity_batches": polarity_batches_in_epoch,
                        **epoch_summary,
                    },
                )
            state = UnifiedResumeState(
                stage=stage,
                epoch=epoch + 1,
                batch_in_epoch=0,
                optimizer_step=global_optimizer_step,
                stage_optimizer_step=stage_optimizer_step,
            )
            common.save_checkpoint(
                checkpoint_dir / f"stage_{stage}_epoch_{epoch}.pt",
                wrapped,
                optimizer,
                scaler,
                scheduler,
                state,
                cfg,
                rank,
                world_size,
                checkpoint_progress_state(
                    epoch_metrics,
                    0,
                    0,
                    0,
                    scheduler_plan,
                ),
                metrics_path,
            )
            skip_batches = 0

        if rank == 0:
            common.append_jsonl(
                metrics_path,
                {
                    "event": "stage_end",
                    "stage": stage,
                    "optimizer_step": global_optimizer_step,
                    "stage_optimizer_step": stage_optimizer_step,
                },
            )
        if isinstance(wrapped, DDP):
            model = wrapped.module
            del wrapped
        resume_state = None
        resume_checkpoint = None

    if dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        common.append_jsonl(
            metrics_path,
            {
                "event": "run_end",
                "run_name": cfg["run_name"],
                "optimizer_step": global_optimizer_step,
                "time_unix": time.time(),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--start-stage", choices=STAGE_ORDER, default="peak_reconstruction"
    )
    parser.add_argument("--end-stage", choices=STAGE_ORDER, default="ion_mode")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-hash-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.data_root)
    if args.validate_only:
        report = validate_inputs(
            cfg,
            args.project_root.resolve(),
            verify_hashes=not args.skip_hash_validation,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["errors"] else 0
    train_worker(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
