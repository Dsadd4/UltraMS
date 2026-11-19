"""Shared model objectives, checkpoint state and distributed training utilities."""
from __future__ import annotations

import hashlib
import json
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
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


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


class SpectrumObjectiveModel(nn.Module):
    """Masked-spectrum, retention-time and ion-mode objectives."""

    def __init__(self, base_model: nn.Module, config: Mapping[str, Any]):
        super().__init__()
        self.base = base_model
        self.config = dict(config)
        d_model = int(config["d_model"])
        self.rt_head = RTHead(d_model)
        self.pol_head = PolarityHead(d_model)

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
            if stage == "ion_mode":
                rt_target = rt_target / rt_scale
                valid_upper = float(cfg["rt_max_seconds"]) / rt_scale
            else:
                # The RT loader supplies targets normalized by 600 seconds.
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
                if stage == "ion_mode":
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

        if stage == "ion_mode":
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
    from ultrams_pretraining_model import (  # type: ignore
        CONFIG as ultrams_config,
    )
    from ultrams_pretraining_model import (
        UltraMSPretrainingModel,
        build_lr_scheduler,
    )

    return (
        ultrams_config,
        UltraMSPretrainingModel,
        build_lr_scheduler,
        create_parquet_dataloader_mlm,
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


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Read a complete training checkpoint for resuming or the MPR-to-RT transition."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model_state_dict", "config", "resume_state", "optimizer_state_dict", "rng_state_by_rank")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            f"{path} is not a complete training checkpoint (missing {', '.join(missing)}). "
            "Use the original training run's checkpoints/latest.pt for --resume, "
            "or its completed MPR checkpoint for --initialize-from. "
            "Hugging Face inference weights are for inference and downstream fine-tuning."
        )
    return checkpoint


def find_resume_checkpoint(
    output_dir: Path, explicit: Path | None, initialize_from: Path | None = None
) -> Path | None:
    latest = output_dir / "checkpoints" / "latest.pt"
    resume = explicit if explicit is not None else (latest if latest.exists() else None)
    if resume is not None and initialize_from is not None:
        raise RuntimeError(
            "A training checkpoint was selected for resume. "
            "Resume that run without --initialize-from, or use a new --output-dir."
        )
    return resume


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

