#!/usr/bin/env python3
"""Run the reviewed UltraMS epoch-3 -> Stage D epoch-11 training chain.

The numerical objectives and stage boundaries follow the archived 2026 Phase-2
trainer. This implementation makes the lineage explicit, emits machine-readable
metrics, and stores enough state for deterministic same-stage recovery.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

STAGE_ORDER = ("ae", "c", "d")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(value)
    record.setdefault("time_unix", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path, data_root: Path | None) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    root = data_root or path.parent.parent
    cfg["config_path"] = str(path.resolve())
    cfg["data_root"] = str(root.resolve())
    cfg["paths"] = {
        key: str((root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in cfg["paths"].items()
    }
    stages = cfg.get("stages", [])
    names = [stage.get("name") for stage in stages]
    if names != list(STAGE_ORDER):
        raise ValueError(f"stages must be exactly {STAGE_ORDER}, got {names}")
    return cfg


def set_seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"])


class RTHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(cls_embedding).squeeze(-1)


class PolarityHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(cls_embedding).squeeze(-1)


class Phase2Model(nn.Module):
    """Checkpoint-compatible Phase-2 wrapper around UltraExplorerMLM."""

    def __init__(self, base_model: nn.Module, config: Mapping[str, Any]):
        super().__init__()
        self.base = base_model
        self.config = dict(config)
        d_model = int(config["d_model"])
        self.rt_head = RTHead(d_model)
        self.pol_head = PolarityHead(d_model)
        self.freeze_backbone(int(config["freeze_layers"]))

    def freeze_backbone(self, freeze_layers: int) -> None:
        for parameter in self.base.peak_encoder.parameters():
            parameter.requires_grad = False
        layers = self.base.encoder.encoder.layer
        for layer in layers[: min(freeze_layers, len(layers))]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def encode(
        self,
        spectra: torch.Tensor,
        attention_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spectra.shape[0]
        peak_embeddings = self.base.peak_encoder(spectra)
        cls_embedding = self.base.cls_emb.expand(batch_size, 1, -1)
        precursor_input = torch.stack(
            [precursor_mz, torch.full_like(precursor_mz, 1.1)], dim=-1
        ).unsqueeze(1)
        precursor_embedding = (
            self.base.peak_encoder(precursor_input) + self.base.precursor_type_emb
        )
        embeddings = torch.cat(
            [cls_embedding, precursor_embedding, peak_embeddings], dim=1
        )
        positions = torch.arange(embeddings.shape[1], device=device).unsqueeze(0)
        embeddings = self.base.dropout(embeddings + self.base.pos_emb(positions))
        prefix_attention = torch.ones(
            batch_size, 2, dtype=attention_mask.dtype, device=device
        )
        full_attention = torch.cat([prefix_attention, attention_mask], dim=1)
        hidden = self.base.encoder(
            inputs_embeds=embeddings, attention_mask=full_attention
        ).last_hidden_state
        return hidden, hidden[:, 0, :]

    def forward(
        self, batch: Mapping[str, Any], device: torch.device, stage: str
    ) -> dict[str, Any]:
        cfg = self.config
        mlm_result = self.base(batch, device)
        result: dict[str, Any] = {"mlm_loss": mlm_result["loss"].detach()}
        for key in ("mz_acc", "level0_acc", "level1_acc", "level2_acc", "int_acc"):
            if key in mlm_result:
                result[key] = mlm_result[key]
        total_loss = float(cfg["mlm_weight"]) * mlm_result["loss"]

        attention_mask = batch["attn_mask"].to(device, non_blocking=True)
        precursor_mz = batch["precursor_mz"].to(device, non_blocking=True)
        spectra = batch["orig_spectra"].to(device, non_blocking=True)
        _, cls_embedding = self.encode(spectra, attention_mask, precursor_mz, device)
        predicted_rt = self.rt_head(cls_embedding)
        predicted_polarity = self.pol_head(cls_embedding)
        total_loss = (
            total_loss + (predicted_rt * 0).sum() + (predicted_polarity * 0).sum()
        )

        rt_target = batch.get("rt")
        if rt_target is not None:
            rt_target = torch.as_tensor(rt_target, dtype=torch.float32, device=device)
            rt_scale = float(cfg["rt_norm_scale"])
            if stage in ("c", "d"):
                rt_target = rt_target / rt_scale
                valid_upper = float(cfg["rt_max_seconds"]) / rt_scale
            else:
                # Stage Ae's archived loader already normalized RT by 600. The
                # historical trainer nevertheless compared that value with
                # rt_max_seconds directly; retain the training behavior here.
                valid_upper = float(cfg["rt_max_seconds"])
            valid_rt = (rt_target > 0) & (rt_target <= valid_upper)
            if valid_rt.sum() >= 2:
                rt_loss = F.huber_loss(
                    predicted_rt[valid_rt],
                    rt_target[valid_rt],
                    delta=float(cfg["rt_huber_delta"]),
                )
                rt_error_seconds = (
                    predicted_rt[valid_rt] - rt_target[valid_rt]
                ).abs() * rt_scale
                result["rt_mae"] = rt_error_seconds.mean().detach()
                result["rt_n"] = int(valid_rt.sum())
                if stage in ("c", "d"):
                    with torch.no_grad():
                        self_consistent = (
                            predicted_rt.detach() - rt_target
                        ).abs() < float(cfg["rt_self_consistency_tau"])
                    selected = valid_rt & self_consistent
                    if selected.sum() >= 2:
                        rt_loss = F.huber_loss(
                            predicted_rt[selected],
                            rt_target[selected],
                            delta=float(cfg["rt_huber_delta"]),
                        )
                        result["rt_mae_sc"] = (
                            (predicted_rt[selected] - rt_target[selected]).abs().mean()
                            * rt_scale
                        ).detach()
                    else:
                        rt_loss = predicted_rt.sum() * 0
                    result["rt_n_sc"] = int(selected.sum())
                result["rt_loss"] = rt_loss.detach()
                total_loss = total_loss + float(cfg["rt_weight"]) * rt_loss

        if stage == "d":
            polarity_target = batch.get("polarity")
            if polarity_target is not None:
                polarity_target = torch.as_tensor(
                    polarity_target, dtype=torch.long, device=device
                )
                valid_polarity = polarity_target >= 0
                if valid_polarity.sum() >= 2:
                    polarity_loss = F.binary_cross_entropy_with_logits(
                        predicted_polarity[valid_polarity],
                        polarity_target[valid_polarity].float(),
                    )
                    prediction = (predicted_polarity[valid_polarity] > 0).long()
                    result["pol_loss"] = polarity_loss.detach()
                    result["pol_acc"] = (
                        (prediction == polarity_target[valid_polarity])
                        .float()
                        .mean()
                        .detach()
                    )
                    result["pol_n"] = int(valid_polarity.sum())
                    total_loss = (
                        total_loss + float(cfg["polarity_weight"]) * polarity_loss
                    )

        result["loss"] = total_loss
        return result


@dataclass
class ResumeState:
    stage: str
    epoch: int
    batch_in_epoch: int
    optimizer_step: int
    stage_optimizer_step: int


class MetricWindow:
    METRICS = (
        "loss",
        "mlm_loss",
        "rt_loss",
        "rt_mae",
        "rt_mae_sc",
        "pol_loss",
        "pol_acc",
        "mz_acc",
    )

    def __init__(self, device: torch.device):
        self.device = device
        self.values = {name: [0.0, 0.0] for name in self.METRICS}

    def update(self, result: Mapping[str, Any], batch_size: int) -> None:
        for name in self.METRICS:
            if name not in result:
                continue
            weight = batch_size
            if name == "rt_loss":
                weight = int(result.get("rt_n_sc", result.get("rt_n", 0)))
            elif name == "rt_mae":
                weight = int(result.get("rt_n", 0))
            elif name == "rt_mae_sc":
                weight = int(result.get("rt_n_sc", 0))
            elif name in ("pol_loss", "pol_acc"):
                weight = int(result.get("pol_n", 0))
            if weight <= 0:
                continue
            value = result[name]
            scalar = float(value.detach().item() if torch.is_tensor(value) else value)
            self.values[name][0] += scalar * weight
            self.values[name][1] += weight

    def reduce_and_reset(self) -> dict[str, float]:
        flat: list[float] = []
        for name in self.METRICS:
            flat.extend(self.values[name])
        tensor = torch.tensor(flat, dtype=torch.float64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced: dict[str, float] = {}
        for index, name in enumerate(self.METRICS):
            total = float(tensor[2 * index].item())
            count = float(tensor[2 * index + 1].item())
            if count:
                reduced[name] = total / count
        self.values = {name: [0.0, 0.0] for name in self.METRICS}
        return reduced


def trainable_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "LayerNorm" in name or "bias" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def import_project(project_root: Path):
    train_root = project_root / "train"
    for path in (project_root, train_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from pretrain.dataset_sharded import create_parquet_dataloader_mlm  # type: ignore
    from train_ue_multiscale_v9_mlm import (  # type: ignore
        CONFIG as v9_config,
    )
    from train_ue_multiscale_v9_mlm import (
        UltraExplorerMLM,
        build_lr_scheduler,
    )

    return (
        v9_config,
        UltraExplorerMLM,
        build_lr_scheduler,
        create_parquet_dataloader_mlm,
    )


def make_loader(
    stage: str,
    cfg: Mapping[str, Any],
    rank: int,
    world_size: int,
    project_root: Path,
    create_parquet_dataloader_mlm,
):
    optimization = cfg["optimization"]
    model_cfg = cfg["model"]
    paths = cfg["paths"]
    if stage == "ae":
        from phase2 import dataset_hq  # type: ignore
        from torch.utils.data import DataLoader, DistributedSampler

        dataset_hq.DATASETS["masspecgym"]["csv"] = paths["massspecgym_csv"]
        dataset_hq.DATASETS["msnlib"]["csv"] = paths["msnlib_csv"]
        dataset_hq.DATASETS["spectraverse"]["csv"] = paths["spectraverse_csv"]
        dataset = dataset_hq.HQMixedDataset(
            fold="train", max_peaks=int(model_cfg["max_peaks"])
        )
        wrapped = dataset_hq.PairSamplerWrapper(dataset)
        sampler = (
            DistributedSampler(
                wrapped, num_replicas=world_size, rank=rank, shuffle=True
            )
            if world_size > 1
            else None
        )
        workers = int(optimization["num_workers"])
        return DataLoader(
            wrapped,
            batch_size=int(optimization["batch_size_per_rank"]),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=workers,
            collate_fn=dataset_hq.collate_fn_phase2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if workers else None,
        )
    shard_dir = (
        paths["clean_shard_dir"] if stage == "c" else paths["polarity_shard_dir"]
    )
    parquet_files = sorted(Path(shard_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet shards found in {shard_dir}")
    if len(parquet_files) % world_size:
        raise ValueError(
            f"{stage} has {len(parquet_files)} shards, not divisible by world_size={world_size}; "
            "the legacy loader would silently drop shards"
        )
    return create_parquet_dataloader_mlm(
        shard_dir=shard_dir,
        batch_size=int(optimization["batch_size_per_rank"]),
        max_peaks=int(model_cfg["max_peaks"]),
        mask_ratio=float(model_cfg["mask_ratio"]),
        rank=rank,
        world_size=world_size,
        num_workers=int(optimization["num_workers"]),
        seed=int(cfg["seed"]),
        drop_last=False,
    )


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    state: ResumeState,
    cfg: Mapping[str, Any],
    rng_states: list[dict[str, Any]],
    extra_state_by_rank: list[dict[str, Any]] | None = None,
    metrics_snapshot: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DDP) else model
    completed_epoch = state.epoch - 1 if state.batch_in_epoch == 0 else state.epoch
    return {
        "format_version": 5,
        "model_state_dict": raw_model.state_dict(),
        "base_state_dict": raw_model.base.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "resume_state": state.__dict__,
        "epoch": completed_epoch,
        "global_step": state.optimizer_step,
        "stage_optimizer_step": state.stage_optimizer_step,
        "rng_state_by_rank": rng_states,
        "extra_state_by_rank": extra_state_by_rank or [],
        "metrics_snapshot": metrics_snapshot,
        "config": dict(cfg),
        "saved_at_unix": time.time(),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    state: ResumeState,
    cfg: Mapping[str, Any],
    rank: int,
    world_size: int,
    local_extra_state: dict[str, Any] | None = None,
    metrics_path: Path | None = None,
) -> None:
    local_rng = capture_rng_state()
    if dist.is_initialized():
        gathered: list[dict[str, Any]] | None = (
            [None] * world_size if rank == 0 else None
        )  # type: ignore[list-item]
        dist.gather_object(local_rng, gathered, dst=0)
    else:
        gathered = [local_rng]
    if dist.is_initialized():
        gathered_extra: list[dict[str, Any]] | None = (
            [None] * world_size if rank == 0 else None
        )  # type: ignore[list-item]
        dist.gather_object(local_extra_state or {}, gathered_extra, dst=0)
    else:
        gathered_extra = [local_extra_state or {}]
    if rank == 0:
        metrics_snapshot = None
        if metrics_path is not None:
            metrics_text = metrics_path.read_text(encoding="utf-8")
            metrics_snapshot = {
                "sha256": hashlib.sha256(metrics_text.encode("utf-8")).hexdigest(),
                "text": metrics_text,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                scaler,
                scheduler,
                state,
                cfg,
                gathered or [],
                gathered_extra or [],
                metrics_snapshot,
            ),
            tmp,
        )
        torch.load(tmp, map_location="cpu", weights_only=False)
        os.replace(tmp, path)
        atomic_json(
            path.with_suffix(path.suffix + ".json"),
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "resume_state": state.__dict__,
            },
        )
        latest = path.parent / "latest.pt"
        latest_tmp = path.parent / "latest.pt.tmp"
        latest_tmp.unlink(missing_ok=True)
        os.link(path, latest_tmp)
        os.replace(latest_tmp, latest)
    if dist.is_initialized():
        dist.barrier()


def find_resume_checkpoint(output_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    latest = output_dir / "checkpoints" / "latest.pt"
    return latest if latest.exists() else None


def validate_inputs(
    cfg: Mapping[str, Any], project_root: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    required_files = (
        "phase1_epoch3_checkpoint",
        "reference_stage_d11_checkpoint",
        "massspecgym_csv",
        "msnlib_csv",
        "spectraverse_csv",
    )
    report: dict[str, Any] = {"files": {}, "directories": {}, "errors": []}
    for name in required_files:
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
        expected = cfg.get("expected_assets", {}).get(name, {})
        if expected.get("bytes") != path.stat().st_size:
            report["errors"].append(
                f"size mismatch for {name}: {path.stat().st_size} != {expected.get('bytes')}"
            )
        if verify_hashes and expected.get("sha256"):
            digest = sha256_file(path)
            report["files"][name]["sha256"] = digest
            if digest != expected["sha256"]:
                report["errors"].append(f"SHA-256 mismatch for {name}: {path}")
    for name in ("clean_shard_dir", "polarity_shard_dir"):
        path = Path(cfg["paths"][name])
        files = sorted(path.glob("*.parquet")) if path.is_dir() else []
        dataset_key = "clean" if name == "clean_shard_dir" else "polarity"
        expected = cfg.get("expected_assets", {}).get(dataset_key, {})
        manifest_path = path / "manifest.json"
        entry: dict[str, Any] = {
            "path": str(path),
            "parquet_files": len(files),
            "manifest": str(manifest_path),
        }
        report["directories"][name] = entry
        if not files:
            report["errors"].append(f"no parquet files: {path}")
            continue
        supported_world_sizes = [
            int(value) for value in cfg.get("supported_world_sizes", [])
        ]
        if not supported_world_sizes:
            report["errors"].append("supported_world_sizes must not be empty")
        if not manifest_path.is_file():
            report["errors"].append(f"missing dataset manifest: {manifest_path}")
            continue
        manifest = json.loads(manifest_path.read_text())
        entry["status"] = manifest.get("status")
        entry["dataset_name"] = manifest.get("dataset_name")
        output = manifest.get("output", {})
        stats = output.get("stats", manifest.get("stats", {}))
        entry["rows"] = stats.get("rows")
        entry["stats"] = stats
        checks = {
            "dataset_name": manifest.get("dataset_name"),
            "rows": stats.get("rows"),
            "negative_0": stats.get("polarity", {}).get("negative_0"),
            "positive_1": stats.get("polarity", {}).get("positive_1"),
            "other_or_null": stats.get("polarity", {}).get("other_or_null"),
            "files": len(files),
        }
        if dataset_key == "clean":
            checks.update(
                {
                    "accepted_0_to_1500": stats.get("rt", {}).get("accepted_0_to_1500"),
                    "above_1500": stats.get("rt", {}).get("above_1500"),
                }
            )
        if manifest.get("status") != "complete":
            report["errors"].append(f"incomplete dataset manifest: {manifest_path}")
        file_records = output.get("file_records", [])
        if len(file_records) == len(files):
            entry["parallel_layouts"] = {}
            optimizer_steps_by_world_size: dict[str, int] = {}
            for candidate_world_size in supported_world_sizes:
                if len(files) % candidate_world_size:
                    report["errors"].append(
                        f"{name} shard count {len(files)} is not divisible by "
                        f"world_size={candidate_world_size}"
                    )
                    continue
                profile = cfg["distributed_profiles"].get(str(candidate_world_size), {})
                batch_size = int(profile.get("batch_size_per_rank", 0))
                accumulation = int(profile.get("gradient_accumulation_multiplier", 0))
                target_batch = int(
                    cfg["optimization"]["target_global_optimizer_batch_size"]
                )
                if (
                    batch_size <= 0
                    or accumulation <= 0
                    or candidate_world_size * batch_size * accumulation != target_batch
                ):
                    report["errors"].append(
                        f"invalid distributed profile for world_size="
                        f"{candidate_world_size}: batch={batch_size}, "
                        f"accumulation={accumulation}, target={target_batch}"
                    )
                    continue
                per_rank_rows, per_rank_batches = dataset_parallel_layout(
                    file_records, candidate_world_size, batch_size
                )
                optimizer_steps = math.ceil(per_rank_batches[0] / accumulation)
                optimizer_steps_by_world_size[str(candidate_world_size)] = (
                    optimizer_steps
                )
                entry["parallel_layouts"][str(candidate_world_size)] = {
                    "batch_size_per_rank": batch_size,
                    "gradient_accumulation": accumulation,
                    "rows_per_rank": per_rank_rows,
                    "microbatches_per_rank": per_rank_batches,
                    "optimizer_steps_per_epoch": optimizer_steps,
                }
                if max(per_rank_rows) - min(per_rank_rows) > 1:
                    report["errors"].append(
                        f"{dataset_key} rows differ by more than one across "
                        f"{candidate_world_size} ranks: {per_rank_rows}"
                    )
                if len(set(per_rank_batches)) != 1:
                    report["errors"].append(
                        f"{dataset_key} batch counts are imbalanced across "
                        f"{candidate_world_size} ranks: {per_rank_batches}"
                    )
            if len(set(optimizer_steps_by_world_size.values())) > 1:
                report["errors"].append(
                    f"{dataset_key} optimizer-step counts differ across supported "
                    f"world sizes: {optimizer_steps_by_world_size}"
                )
        else:
            report["errors"].append(
                f"{dataset_key} manifest file_records are incomplete: "
                f"{len(file_records)} != {len(files)}"
            )
        for key, expected_value in expected.items():
            if key == "seed":
                actual = manifest.get("config", {}).get("seed", manifest.get("seed"))
            else:
                actual = checks.get(key)
            if actual != expected_value:
                report["errors"].append(
                    f"{dataset_key} {key} mismatch: {actual} != {expected_value}"
                )
    for path in (
        project_root / "train" / "train_ue_multiscale_v9_mlm.py",
        project_root / "pretrain" / "dataset_sharded.py",
        project_root / "train" / "phase2" / "dataset_hq.py",
    ):
        if not path.is_file():
            report["errors"].append(f"missing code dependency: {path}")
    return report


def dataset_parallel_layout(
    file_records: list[Mapping[str, Any]], world_size: int, batch_size: int
) -> tuple[list[int], list[int]]:
    if world_size <= 0 or batch_size <= 0:
        raise ValueError("world_size and batch_size must be positive")
    rows = [
        sum(int(record["rows"]) for record in file_records[rank::world_size])
        for rank in range(world_size)
    ]
    return rows, [math.ceil(value / batch_size) for value in rows]


def synchronize_incomplete_accumulation(
    model: nn.Module, world_size: int, accumulation: int, remainder: int
) -> None:
    if not 0 < remainder < accumulation:
        raise ValueError("remainder must be between zero and accumulation")
    if isinstance(model, DDP):
        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)
    correction = accumulation / remainder
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(correction)


def select_stage_range(
    cfg: Mapping[str, Any], start: str, end: str
) -> list[dict[str, Any]]:
    start_index = STAGE_ORDER.index(start)
    end_index = STAGE_ORDER.index(end)
    if start_index > end_index:
        raise ValueError(f"start stage {start} is after end stage {end}")
    return [dict(stage) for stage in cfg["stages"][start_index : end_index + 1]]


def train_worker(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    supported_world_sizes = {
        int(value) for value in cfg.get("supported_world_sizes", [])
    }
    if world_size not in supported_world_sizes:
        raise ValueError(
            f"launcher WORLD_SIZE={world_size} is unsupported; expected one of "
            f"{sorted(supported_world_sizes)}"
        )
    cfg["world_size"] = world_size
    profile = cfg["distributed_profiles"].get(str(world_size), {})
    batch_size = int(profile.get("batch_size_per_rank", 0))
    accumulation_multiplier = int(profile.get("gradient_accumulation_multiplier", 0))
    target_batch = int(cfg["optimization"]["target_global_optimizer_batch_size"])
    if (
        batch_size <= 0
        or accumulation_multiplier <= 0
        or world_size * batch_size * accumulation_multiplier != target_batch
    ):
        raise ValueError(
            f"invalid distributed profile for WORLD_SIZE={world_size}: "
            f"batch={batch_size}, accumulation={accumulation_multiplier}, "
            f"target={target_batch}"
        )
    cfg["optimization"]["batch_size_per_rank"] = batch_size
    cfg["optimization"]["gradient_accumulation_multiplier"] = accumulation_multiplier
    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    set_seed(int(cfg["seed"]), rank)

    project_root = args.project_root.resolve()
    v9_config, model_class, build_lr_scheduler, create_parquet = import_project(
        project_root
    )
    model_config = {
        **v9_config,
        **cfg["model"],
        "freeze_layers": cfg["model"]["freeze_layers"],
    }
    base_model = model_class(v9_config)
    model = Phase2Model(base_model, model_config)

    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = str(args.output_dir.resolve())
    output_dir = Path(cfg["paths"]["output_dir"])
    if str(output_dir).startswith("/mnt/s3"):
        raise ValueError(
            "live checkpoints must use worker-local storage, not the S3 FUSE mount; "
            "pass --output-dir /tmp/<run> and archive verified artifacts separately"
        )
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(output_dir / "resolved_config.json", cfg)
        input_report = validate_inputs(cfg, project_root, verify_hashes=True)
        atomic_json(output_dir / "input_validation.json", input_report)
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

    resume_path = find_resume_checkpoint(output_dir, args.resume)
    resume_checkpoint = None
    resume_state = None
    if resume_path:
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        resume_state = ResumeState(**resume_checkpoint["resume_state"])
    else:
        phase1 = torch.load(
            cfg["paths"]["phase1_epoch3_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        missing, unexpected = model.base.load_state_dict(
            phase1["model_state_dict"], strict=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"phase1 load mismatch: missing={missing}, unexpected={unexpected}"
            )
        del phase1

    if rank == 0:
        append_jsonl(
            metrics_path,
            {
                "event": "run_start",
                "run_name": cfg["run_name"],
                "world_size": world_size,
                "batch_size_per_rank": cfg["optimization"]["batch_size_per_rank"],
                "resume_checkpoint": str(resume_path) if resume_path else None,
                "time_unix": time.time(),
            },
        )

    model = model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=False,
        )

    global_optimizer_step = resume_state.optimizer_step if resume_state else 0
    stages = select_stage_range(cfg, args.start_stage, args.end_stage)
    if resume_state:
        stages = [
            stage
            for stage in stages
            if STAGE_ORDER.index(stage["name"]) >= STAGE_ORDER.index(resume_state.stage)
        ]

    for stage_cfg in stages:
        stage = stage_cfg["name"]
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.config.update(
            {
                "rt_weight": float(stage_cfg["rt_weight"]),
                "polarity_weight": float(stage_cfg["polarity_weight"]),
            }
        )
        loader = make_loader(stage, cfg, rank, world_size, project_root, create_parquet)
        optimizer = torch.optim.AdamW(
            trainable_parameter_groups(
                model, float(cfg["optimization"]["weight_decay"])
            ),
            lr=float(cfg["optimization"]["learning_rate"])
            * float(stage_cfg["learning_rate_multiplier"]),
        )
        scheduler = None
        if stage_cfg["scheduler"] == "legacy_warmup_cosine":
            total_microbatches = len(loader) * int(stage_cfg["epochs"])
            warmup = int(
                total_microbatches * float(cfg["optimization"]["warmup_ratio"])
            )
            scheduler = build_lr_scheduler(
                optimizer,
                warmup,
                total_microbatches,
                float(cfg["optimization"]["minimum_learning_rate"])
                / float(cfg["optimization"]["learning_rate"]),
            )
        scaler = GradScaler("cuda", enabled=use_cuda)

        first_epoch = 1
        skip_batches = 0
        stage_optimizer_step = 0
        if resume_state and resume_state.stage == stage:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            if scheduler is not None and resume_checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            first_epoch = resume_state.epoch
            skip_batches = resume_state.batch_in_epoch
            stage_optimizer_step = resume_state.stage_optimizer_step

        for epoch in range(first_epoch, int(stage_cfg["epochs"]) + 1):
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            loader.generator = torch.Generator().manual_seed(
                int(cfg["seed"]) + STAGE_ORDER.index(stage) * 100_000 + epoch
            )
            iterator = iter(loader)
            current_skip = skip_batches if epoch == first_epoch else 0
            for _ in range(current_skip):
                try:
                    next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        f"resume batch offset {current_skip} exceeds stage={stage} epoch={epoch}"
                    ) from error
            if resume_state and resume_state.stage == stage and epoch == first_epoch:
                rng_by_rank = resume_checkpoint.get("rng_state_by_rank", [])
                if len(rng_by_rank) != world_size:
                    raise RuntimeError(
                        "checkpoint does not contain one RNG state per rank"
                    )
                restore_rng_state(rng_by_rank[rank])

            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation = int(stage_cfg["gradient_accumulation"]) * int(
                cfg["optimization"]["gradient_accumulation_multiplier"]
            )
            micro_step = 0
            batch_in_epoch = current_skip
            metrics = MetricWindow(device)
            epoch_metrics = MetricWindow(device)
            while True:
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
                batch_in_epoch += 1
                micro_step += 1
                sync_step = micro_step % accumulation == 0
                sync_context = contextlib.nullcontext()
                if isinstance(model, DDP) and not sync_step:
                    sync_context = model.no_sync()
                amp_context = (
                    autocast("cuda", dtype=torch.bfloat16)
                    if use_cuda and cfg["optimization"]["precision"] == "bf16"
                    else contextlib.nullcontext()
                )
                with sync_context, amp_context:
                    result = model(batch, device, stage)
                    loss = result["loss"] / accumulation
                scaler.scale(loss).backward()
                metrics.update(result, int(batch["orig_spectra"].shape[0]))
                epoch_metrics.update(result, int(batch["orig_spectra"].shape[0]))
                if not sync_step:
                    continue
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["optimization"]["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_optimizer_step += 1
                stage_optimizer_step += 1

                should_log = (
                    stage_optimizer_step
                    % int(cfg["optimization"]["log_every_optimizer_steps"])
                    == 0
                )
                if should_log:
                    reduced = metrics.reduce_and_reset()
                    if rank == 0:
                        append_jsonl(
                            metrics_path,
                            {
                                "event": "train_window",
                                "stage": stage,
                                "epoch": epoch,
                                "batch_in_epoch": batch_in_epoch,
                                "optimizer_step": global_optimizer_step,
                                "stage_optimizer_step": stage_optimizer_step,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                **reduced,
                            },
                        )

                should_save = (
                    stage_optimizer_step
                    % int(cfg["optimization"]["save_every_optimizer_steps"])
                    == 0
                )
                if should_save:
                    state = ResumeState(
                        stage=stage,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        optimizer_step=global_optimizer_step,
                        stage_optimizer_step=stage_optimizer_step,
                    )
                    save_checkpoint(
                        checkpoint_dir
                        / f"stage_{stage}_step_{stage_optimizer_step}.pt",
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                        state,
                        cfg,
                        rank,
                        world_size,
                    )
                    if rank == 0:
                        step_files = sorted(
                            checkpoint_dir.glob(f"stage_{stage}_step_*.pt"),
                            key=lambda item: item.stat().st_mtime,
                        )
                        keep = int(cfg["optimization"]["keep_last_step_checkpoints"])
                        obsolete = step_files[:-keep] if keep else step_files
                        for old_path in obsolete:
                            old_path.unlink(missing_ok=True)
                            old_path.with_suffix(old_path.suffix + ".json").unlink(
                                missing_ok=True
                            )

            remainder = micro_step % accumulation
            if remainder:
                if stage == "ae":
                    # Match the archived Ae behavior: only complete accumulation
                    # windows contribute an optimizer update.
                    optimizer.zero_grad(set_to_none=True)
                else:
                    synchronize_incomplete_accumulation(
                        model, world_size, accumulation, remainder
                    )
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(cfg["optimization"]["gradient_clip"]),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_optimizer_step += 1
                    stage_optimizer_step += 1
            epoch_summary = epoch_metrics.reduce_and_reset()
            if rank == 0:
                append_jsonl(
                    metrics_path,
                    {
                        "event": "epoch_end",
                        "stage": stage,
                        "epoch": epoch,
                        "optimizer_step": global_optimizer_step,
                        "stage_optimizer_step": stage_optimizer_step,
                        "microbatches": micro_step,
                        **epoch_summary,
                    },
                )
            next_epoch = epoch + 1
            state = ResumeState(
                stage=stage,
                epoch=next_epoch,
                batch_in_epoch=0,
                optimizer_step=global_optimizer_step,
                stage_optimizer_step=stage_optimizer_step,
            )
            save_checkpoint(
                checkpoint_dir / f"stage_{stage}_epoch_{epoch}.pt",
                model,
                optimizer,
                scaler,
                scheduler,
                state,
                cfg,
                rank,
                world_size,
            )
            skip_batches = 0

        resume_state = None
        resume_checkpoint = None
        if rank == 0:
            append_jsonl(
                metrics_path,
                {
                    "event": "stage_end",
                    "stage": stage,
                    "optimizer_step": global_optimizer_step,
                },
            )

    if dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        append_jsonl(
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
    parser.add_argument("--start-stage", choices=STAGE_ORDER, default="ae")
    parser.add_argument("--end-stage", choices=STAGE_ORDER, default="d")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve(), args.data_root)
    if args.validate_only:
        report = validate_inputs(cfg, args.project_root.resolve(), verify_hashes=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["errors"]:
            raise SystemExit(2)
        return
    train_worker(args, cfg)


if __name__ == "__main__":
    main()
