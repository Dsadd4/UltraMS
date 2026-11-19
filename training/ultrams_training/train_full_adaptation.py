#!/usr/bin/env python3
"""Train UltraMS retention time and ion mode on UltraMSdata.

Start from the completed five-epoch masked-peak-reconstruction checkpoint.
Train all effective parameters for five retention-time epochs, followed by
11 joint retention-time and ion-mode epochs on the polarity-balanced subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from . import train_pretraining as pretraining
except ImportError:  # Direct execution from a staged release package.
    import train_pretraining as pretraining


STAGE_ORDER = ("peak_reconstruction", "supervised_rt", "ion_mode")
ADAPTATION_STAGES = ("supervised_rt", "ion_mode")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase1_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "training_mode": cfg["training_mode"],
        "seed": cfg["seed"],
        "supported_world_sizes": cfg["supported_world_sizes"],
        "distributed_profiles": {
            world_size: {"phase1": profile["phase1"]}
            for world_size, profile in cfg["distributed_profiles"].items()
        },
        "model": {key: cfg["model"][key] for key in ("max_peaks", "mask_ratio")},
        "optimization": {
            key: cfg["optimization"][key]
            for key in (
                "weight_decay",
                "gradient_clip",
                "num_workers",
                "precision",
            )
        },
        "clean": cfg["expected_assets"]["clean"],
        "stage": cfg["stages"][0],
    }


def profile_for_stage(
    cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any], world_size: int
) -> dict[str, int]:
    raw = cfg["distributed_profiles"][str(world_size)][str(stage_cfg["profile"])]
    return {
        "batch_size_per_rank": int(raw["batch_size_per_rank"]),
        "gradient_accumulation": int(raw["gradient_accumulation"]),
        "global_optimizer_batch_size": int(raw["global_optimizer_batch_size"]),
    }


def validate_profiles(cfg: Mapping[str, Any]) -> None:
    if [int(value) for value in cfg["supported_world_sizes"]] != [4, 6]:
        raise ValueError("training supports four or six GPUs")
    names = [stage.get("name") for stage in cfg.get("stages", [])]
    if names != list(STAGE_ORDER):
        raise ValueError(f"stages must be exactly {STAGE_ORDER}, got {names}")
    if cfg["optimization"]["precision"] != "bf16":
        raise ValueError("full adaptation requires BF16")
    if int(cfg["optimization"]["adaptation_num_workers"]) < 1:
        raise ValueError("adaptation_num_workers must be positive")

    expected_batches = {
        "phase1": 3840,
        "supervised_rt_full": 3456,
        "ion_mode_full": 1152,
    }
    for world_size in (4, 6):
        profiles = cfg["distributed_profiles"][str(world_size)]
        for name, expected in expected_batches.items():
            profile = profiles[name]
            actual = (
                world_size
                * int(profile["batch_size_per_rank"])
                * int(profile["gradient_accumulation"])
            )
            if actual != expected:
                raise ValueError(
                    f"{world_size}-GPU {name} global batch is {actual}, expected {expected}"
                )
            if int(profile["global_optimizer_batch_size"]) != expected:
                raise ValueError(f"{world_size}-GPU {name} declares the wrong batch")

    reconstruction, supervised_rt, ion_mode = cfg["stages"]
    if reconstruction["trainable_scope"] != "all_base":
        raise ValueError("the parent Phase-1 contract changed")
    for stage in (supervised_rt, ion_mode):
        if stage["trainable_scope"] != "all_effective_parameters":
            raise ValueError(f"{stage['name']} must train all effective parameters")
        if int(stage["mlm_frequency"]) != 1:
            raise ValueError("MLM must be computed on every adaptation batch")
        rates = stage.get("parameter_group_learning_rates", {})
        if set(rates) != {"task_heads", "shared_late", "shared_early"}:
            raise ValueError(f"{stage['name']} has an incomplete layer-wise LR policy")
        if any(float(value) <= 0 for value in rates.values()):
            raise ValueError(f"{stage['name']} learning rates must be positive")
        if float(stage["learning_rate"]) != float(rates["task_heads"]):
            raise ValueError(f"{stage['name']} base LR must equal the task-head LR")
    if supervised_rt["rt_input_unit"] != "normalized_600s":
        raise ValueError("supervised RT input units differ from the successful loader")
    if ion_mode["rt_input_unit"] != "seconds":
        raise ValueError("ion-mode RT input units differ from the polarity corpus")
    if supervised_rt["polarity_weight"] != 0.0 or ion_mode["polarity_weight"] <= 0.0:
        raise ValueError("polarity loss must be enabled only in ion mode")


def load_config(path: Path, data_root: Path | None) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("training_mode") != "pretraining":
        raise ValueError("config is not an UltraMS pretraining configuration")
    if cfg.get("protocol") != "full_adaptation_v1":
        raise ValueError("config is not the full-adaptation protocol")
    root = data_root or path.parent.parent
    cfg["config_path"] = str(path.resolve())
    cfg["data_root"] = str(root.resolve())
    cfg["paths"] = {
        key: str((root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in cfg["paths"].items()
    }
    validate_profiles(cfg)
    validate_ultramsdata_contract(cfg)

    wrapper_sha = sha256_file(Path(__file__).resolve())
    protocol_contract = {
        key: cfg[key]
        for key in (
            "protocol",
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
    protocol_contract["data_mode"] = cfg["data_mode"]
    protocol_contract["rt_data_contract"] = cfg["rt_data_contract"]
    protocol_contract["entrypoint_sha256"] = wrapper_sha
    cfg["adaptation_code_fingerprints"] = {
        "package/train_full_adaptation.py": wrapper_sha
    }
    cfg["training_contract_sha256"] = hashlib.sha256(
        json.dumps(protocol_contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cfg["phase1_contract_sha256"] = hashlib.sha256(
        json.dumps(
            _phase1_contract(cfg), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return cfg


def validate_ultramsdata_contract(cfg: Mapping[str, Any]) -> None:
    """Validate the UltraMSdata inputs and the complete training schedule."""
    if cfg.get("data_mode") != "ultramsdata_parquet":
        raise ValueError("UltraMS training requires the prepared UltraMSdata Parquet corpus")
    required = (
        "expected_assets",
        "data_audits",
        "parent_checkpoint",
        "rt_data_contract",
    )
    for key in required:
        value = cfg.get(key)
        if not isinstance(value, Mapping) or "UNRESOLVED" in json.dumps(value):
            raise ValueError(f"UltraMSdata {key} is unresolved")
    if cfg.get("world_size") not in cfg["supported_world_sizes"]:
        raise ValueError(
            "UltraMSdata runtime must explicitly select a supported world_size"
        )
    contract = cfg["rt_data_contract"]
    if contract != {
        "source": "ultramsdata",
        "source_unit": "seconds",
        "loader_unit": "normalized_600s",
        "selection": "all_clean_rows",
        "scheduler_optimizer_step_factor": 3,
    }:
        raise ValueError("unexpected UltraMSdata RT data contract")
    if float(cfg["model"]["rt_norm_scale"]) != 600.0:
        raise ValueError("UltraMSdata RT loader requires the 600-second scale")
    if [int(stage["epochs"]) for stage in cfg["stages"]] != [5, 5, 11]:
        raise ValueError("UltraMSdata requires the complete 5/5/11 epoch protocol")
    if cfg["paths"]["rt_shard_dir"] != cfg["paths"]["clean_shard_dir"]:
        raise ValueError("RT must consume all rows of UltraMSdata")
    if cfg["expected_assets"]["rt"] != cfg["expected_assets"]["clean"]:
        raise ValueError("RT and UltraMSdata identities must match exactly")
    override = cfg["stages"][1].get("scheduler_total_steps_override")
    if not isinstance(override, int) or isinstance(override, bool) or override <= 0:
        raise ValueError("UltraMSdata RT scheduler steps are unresolved")


def pure_epoch_layouts(cfg, records, stage_name):
    """Count exact batches for every epoch of the explicitly selected GPU layout."""
    stage = next(item for item in cfg["stages"] if item["name"] == stage_name)
    result = {}
    world_size = cfg.get("world_size")
    if (
        not isinstance(world_size, int)
        or world_size not in cfg["supported_world_sizes"]
    ):
        raise ValueError(
            "UltraMSdata runtime must explicitly select a supported world_size"
        )
    for world_size in (world_size,):
        profile = profile_for_stage(cfg, stage, int(world_size))
        accumulation = int(profile["gradient_accumulation"])
        epochs = {}
        for epoch in range(1, int(stage["epochs"]) + 1):
            worker_rows, batches = pretraining.parquet_worker_parallel_layout(
                records,
                int(world_size),
                profile["batch_size_per_rank"],
                int(cfg["optimization"]["adaptation_num_workers"]),
                int(cfg["seed"]),
                epoch,
                drop_last=True,
            )
            if not batches or min(batches) <= 0 or len(set(batches)) != 1:
                raise ValueError(
                    f"{stage_name} epoch {epoch}: unequal/empty rank batches {batches}"
                )
            if batches[0] % accumulation:
                raise ValueError(
                    f"{stage_name} epoch {epoch}: incomplete optimizer batch"
                )
            epochs[str(epoch)] = {
                "microbatches_per_rank": batches,
                "optimizer_steps": batches[0] // accumulation,
                "worker_rows_by_rank": worker_rows,
                "used_rows": sum(batches) * profile["batch_size_per_rank"],
            }
        if stage_name == "supervised_rt":
            expected_total = 3 * sum(
                item["optimizer_steps"] for item in epochs.values()
            )
            if stage.get("scheduler_total_steps_override") != expected_total:
                raise ValueError(
                    f"RT scheduler must be {expected_total} for {world_size} GPUs "
                    "to preserve the 3:1 scheduler geometry"
                )
        result[str(world_size)] = epochs
    return result


def validate_pure_parquet_dataset(cfg, dataset_key, stage_name, *, verify_hashes):
    shard_dir = Path(cfg["paths"][f"{dataset_key}_shard_dir"])
    manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("incomplete Parquet manifest")
    records = sorted(manifest["output"]["file_records"], key=lambda item: item["name"])
    names = [str(item["name"]) for item in records]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("unsafe or duplicate manifest shard names")
    if set(names) != {path.name for path in shard_dir.glob("*.parquet")}:
        raise ValueError("manifest does not enumerate exactly the Parquet files")
    stats = _manifest_stats(manifest)
    expected = cfg["expected_assets"][dataset_key]
    actual = {
        key: manifest.get(key)
        for key in (
            "dataset_name",
            "dataset_fingerprint",
            "input_content_fingerprint",
            "run_identity",
            "schema_fingerprint",
        )
    }
    actual.update(
        rows=stats.get("rows"),
        files=len(records),
        **{
            key: stats.get("polarity", {}).get(key)
            for key in ("negative_0", "positive_1", "other_or_null")
        },
        **{
            key: stats.get("rt", {}).get(key)
            for key in ("accepted_0_to_1500", "above_1500")
        },
        seed=manifest.get("config", {}).get("seed", manifest.get("seed")),
    )
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(
                f"{dataset_key} manifest {key} mismatch: {actual.get(key)} != {value}"
            )
    if sum(int(item["rows"]) for item in records) != int(stats["rows"]):
        raise ValueError("manifest file rows do not sum to corpus rows")
    for item in records:
        path = shard_dir / item["name"]
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"size mismatch: {path}")
        if verify_hashes and sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
    return {
        "path": str(shard_dir),
        "manifest": str(shard_dir / "manifest.json"),
        "rows": stats["rows"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "parallel_layouts": pure_epoch_layouts(cfg, records, stage_name),
    }


class NormalizedRTLoader:
    """Normalize retention times in Parquet batches to the 600-second training scale."""

    def __init__(self, loader, epoch_layouts, scale):
        self.loader = loader
        self.epoch_layouts = epoch_layouts
        self.scale = scale

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def generator(self):
        return self.loader.generator

    @generator.setter
    def generator(self, value):
        self.loader.generator = value

    def __len__(self):
        epoch = max(1, int(self.dataset.epoch))
        return self.epoch_layouts[str(epoch)]["microbatches_per_rank"][0]

    def __iter__(self):
        expected = len(self)
        count = 0
        for batch in self.loader:
            count += 1
            if count > expected:
                raise RuntimeError("RT loader exceeded its audited batch count")
            yield {**batch, "rt": batch["rt"] / self.scale}
        if count != expected:
            raise RuntimeError(
                f"RT loader produced {count} batches, expected {expected}"
            )


def _manifest_stats(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    output = manifest.get("output", {})
    return output.get("stats", manifest.get("stats", {}))


def validate_inputs(
    cfg: Mapping[str, Any], project_root: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    """Validate UltraMSdata and its polarity-balanced subset."""
    validate_ultramsdata_contract(cfg)
    report: dict[str, Any] = {"files": {}, "directories": {}, "errors": []}
    for dataset_key, stage_name in (
        ("rt", "supervised_rt"),
        ("polarity", "ion_mode"),
    ):
        try:
            detail = validate_pure_parquet_dataset(
                cfg, dataset_key, stage_name, verify_hashes=verify_hashes
            )
            report["directories"][f"{dataset_key}_shard_dir"] = detail
            if dataset_key == "rt":
                report["directories"]["clean_shard_dir"] = detail
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            report["errors"].append(f"UltraMSdata {dataset_key}: {error}")

    for path in (
        project_root / "train" / "ultrams_pretraining_model.py",
        project_root / "pretrain" / "dataset_sharded.py",
    ):
        if not path.is_file():
            report["errors"].append(f"missing code dependency: {path}")
    return report


class FullAdaptationUltraMS(pretraining.UnifiedUltraMS):
    """Checkpoint-compatible model with every effective parameter trainable."""

    def configure_trainable(self, stage: str) -> None:
        if stage == "peak_reconstruction":
            super().configure_trainable(stage)
            return
        if stage not in ADAPTATION_STAGES:
            raise ValueError(f"unknown full-adaptation stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad = True
        self._freeze_structurally_unused_parameters()
        self._full_adaptation_stage = stage

    def forward(
        self, batch: Mapping[str, Any], device: torch.device, stage: str
    ) -> dict[str, Any]:
        if stage == "peak_reconstruction":
            return super().forward(batch, device, stage)
        return pretraining.common.SpectrumObjectiveModel.forward(
            self, batch, device, stage
        )


def _strip_ddp(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def _group_category(name: str) -> str:
    name = _strip_ddp(name)
    if name.startswith(("rt_head.", "pol_head.")):
        return "task_heads"
    match = re.match(r"base\.encoder\.encoder\.layer\.(\d+)\.", name)
    if match and int(match.group(1)) >= 12:
        return "shared_late"
    if name.startswith(
        (
            "base.mz_level0_head.",
            "base.mz_level1_head.",
            "base.mz_level2_head.",
            "base.int_predictor.",
        )
    ):
        return "shared_late"
    return "shared_early"


def _uses_weight_decay(name: str) -> bool:
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
    return not any(token in name for token in no_decay)


def full_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    raw_model = model.module if hasattr(model, "module") else model
    stage = getattr(raw_model, "_full_adaptation_stage", None)
    if stage not in ADAPTATION_STAGES:
        raise RuntimeError("layer-wise groups require an active adaptation stage")
    stage_cfg = next(
        item for item in raw_model._protocol_stages if item["name"] == stage
    )
    rates = stage_cfg["parameter_group_learning_rates"]
    buckets: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {
        (category, decay): []
        for category in ("task_heads", "shared_late", "shared_early")
        for decay in (True, False)
    }
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            buckets[(_group_category(name), _uses_weight_decay(name))].append(
                (name, parameter)
            )

    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for category in ("task_heads", "shared_late", "shared_early"):
        for decay in (True, False):
            entries = buckets[(category, decay)]
            if not entries:
                raise RuntimeError(f"empty optimizer group: {category}/decay={decay}")
            names = [name for name, _ in entries]
            parameters = [parameter for _, parameter in entries]
            overlap = seen.intersection(map(id, parameters))
            if overlap:
                raise RuntimeError(
                    "a parameter appears in more than one optimizer group"
                )
            seen.update(map(id, parameters))
            groups.append(
                {
                    "params": parameters,
                    "lr": float(rates[category]),
                    "weight_decay": float(weight_decay) if decay else 0.0,
                    "group_name": f"{category}_{'decay' if decay else 'no_decay'}",
                    "parameter_names_sha256": hashlib.sha256(
                        "\n".join(names).encode("utf-8")
                    ).hexdigest(),
                    "parameter_count": sum(
                        parameter.numel() for parameter in parameters
                    ),
                    "protocol_stage": stage,
                }
            )
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if seen != trainable:
        raise RuntimeError("optimizer groups do not cover all trainable parameters")
    return groups


class ContinuousAdamWFactory:
    """Create AdamW once for RT and retain moments across the RT-to-ion boundary."""

    def __init__(self, adamw_type: type[torch.optim.AdamW], *, fused: bool) -> None:
        self.adamw_type = adamw_type
        self.fused = bool(fused)
        self.optimizer: torch.optim.AdamW | None = None
        self.signatures: tuple[tuple[str, str], ...] | None = None

    @staticmethod
    def _signature(groups: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(group["group_name"]), str(group["parameter_names_sha256"]))
            for group in groups
        )

    def __call__(self, groups, *args, **kwargs):
        groups = list(groups)
        signature = self._signature(groups)
        if self.optimizer is None:
            use_fused = self.fused and torch.cuda.is_available()
            self.optimizer = self.adamw_type(groups, *args, fused=use_fused, **kwargs)
            self.signatures = signature
            return self.optimizer
        if signature != self.signatures:
            raise RuntimeError("RT-to-ion optimizer parameter groups changed")
        for current, target in zip(self.optimizer.param_groups, groups, strict=True):
            current["lr"] = float(target["lr"])
            current["weight_decay"] = float(target["weight_decay"])
            current["protocol_stage"] = target["protocol_stage"]
        return self.optimizer


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
        detail = validate_pure_parquet_dataset(cfg, "rt", stage, verify_hashes=False)
        loader = create_parquet(
            shard_dir=cfg["paths"]["rt_shard_dir"],
            batch_size=int(profile["batch_size_per_rank"]),
            max_peaks=int(cfg["model"]["max_peaks"]),
            mask_ratio=float(cfg["model"]["mask_ratio"]),
            rank=rank,
            world_size=world_size,
            num_workers=int(cfg["optimization"]["adaptation_num_workers"]),
            seed=int(cfg["seed"]),
            drop_last=True,
        )
        return NormalizedRTLoader(
            loader,
            detail["parallel_layouts"][str(world_size)],
            float(cfg["model"]["rt_norm_scale"]),
        )
    if stage != "ion_mode":
        raise ValueError(f"full adaptation cannot load stage: {stage}")
    return create_parquet(
        shard_dir=cfg["paths"]["polarity_shard_dir"],
        batch_size=int(profile["batch_size_per_rank"]),
        max_peaks=int(cfg["model"]["max_peaks"]),
        mask_ratio=float(cfg["model"]["mask_ratio"]),
        rank=rank,
        world_size=world_size,
        num_workers=int(cfg["optimization"]["adaptation_num_workers"]),
        seed=int(cfg["seed"]),
        drop_last=True,
    )


def install_protocol(cfg: Mapping[str, Any]) -> ContinuousAdamWFactory:
    pretraining.STAGE_ORDER = STAGE_ORDER
    pretraining.UnifiedUltraMS = FullAdaptationUltraMS
    pretraining.profile_for_stage = profile_for_stage
    pretraining.validate_inputs = validate_inputs
    pretraining.make_loader = make_loader
    pretraining.common.trainable_parameter_groups = full_parameter_groups
    factory = ContinuousAdamWFactory(
        torch.optim.AdamW, fused=bool(cfg["optimization"].get("fused_adamw", False))
    )
    pretraining.torch.optim.AdamW = factory
    return factory


def verify_parent_checkpoint(
    cfg: Mapping[str, Any], project_root: Path, checkpoint_path: Path
) -> dict[str, Any]:
    """Verify the complete Phase-1 boundary without entering training."""
    checkpoint = pretraining.common.load_training_checkpoint(checkpoint_path)
    parent_cfg = checkpoint.get("config", {})
    expected = cfg.get("parent_checkpoint")
    if not isinstance(expected, Mapping):
        raise RuntimeError("adaptation config lacks parent checkpoint identity")
    digest = sha256_file(checkpoint_path)
    if digest != expected.get("sha256"):
        raise RuntimeError("parent checkpoint SHA-256 differs from launch identity")
    if parent_cfg.get("phase1_contract_sha256") != cfg["phase1_contract_sha256"]:
        raise RuntimeError("parent MLM phase-1 contract does not match")
    if int(parent_cfg.get("world_size", -1)) != int(cfg["world_size"]):
        raise RuntimeError("parent MLM world size does not match")
    clean_fingerprint = cfg["expected_assets"]["clean"]["dataset_fingerprint"]
    if parent_cfg.get("dataset_fingerprints", {}).get("clean") != clean_fingerprint:
        raise RuntimeError("parent MPR UltraMSdata fingerprint does not match")
    current_code = pretraining.training_code_fingerprints(project_root)
    expected_parent_code = pretraining.verify_parent_code_identity(
        cfg, parent_cfg, current_code
    )
    state = pretraining.UnifiedResumeState.from_mapping(checkpoint["resume_state"])
    if (
        state.stage != "peak_reconstruction"
        or state.epoch != 6
        or state.batch_in_epoch != 0
    ):
        raise RuntimeError("parent checkpoint is not the terminal five-epoch MLM state")
    rng_by_rank = checkpoint.get("rng_state_by_rank", [])
    if len(rng_by_rank) != int(cfg["world_size"]):
        raise RuntimeError("parent checkpoint lacks one RNG state per rank")

    ultrams_config, model_class, _, _ = pretraining.common.import_project(project_root)
    model_config = {
        **ultrams_config,
        **cfg["model"],
        "freeze_layers": cfg["model"]["freeze_layers_after_phase1"],
    }
    base = model_class(model_config)
    rng_after_base = pretraining.common.capture_rng_state()
    model = FullAdaptationUltraMS(base, model_config)
    pretraining.common.restore_rng_state(rng_after_base)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("parent model tensors do not strictly match")
    return {
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "sha256": digest,
        "world_size": int(cfg["world_size"]),
        "phase1_contract_sha256": cfg["phase1_contract_sha256"],
        "clean_dataset_fingerprint": clean_fingerprint,
        "model_tensors": len(checkpoint["model_state_dict"]),
        "rng_states": len(rng_by_rank),
        "resume_state": dict(checkpoint["resume_state"]),
        "parent_code_fingerprints": expected_parent_code,
        "adaptation_code_fingerprints": current_code,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--start-stage", choices=ADAPTATION_STAGES, default="supervised_rt"
    )
    parser.add_argument("--end-stage", choices=ADAPTATION_STAGES, default="ion_mode")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verify-parent-only", action="store_true")
    parser.add_argument("--verification-output", type=Path)
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
    if args.verify_parent_only:
        if args.initialize_from is None or args.resume is not None:
            raise RuntimeError(
                "--verify-parent-only requires --initialize-from and forbids --resume"
            )
        report = verify_parent_checkpoint(
            cfg, args.project_root.resolve(), args.initialize_from.resolve()
        )
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.verification_output is not None:
            args.verification_output.parent.mkdir(parents=True, exist_ok=True)
            args.verification_output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    install_protocol(cfg)
    # The optimizer grouping function reads the immutable stage policy from
    # the model, so it survives DDP wrapping without global mutable stage data.
    original_initialize = pretraining.initialize_unified_model

    def initialize(model_class, model_config):
        model = original_initialize(model_class, model_config)
        model._protocol_stages = cfg["stages"]
        return model

    pretraining.initialize_unified_model = initialize
    pretraining.train_worker(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
