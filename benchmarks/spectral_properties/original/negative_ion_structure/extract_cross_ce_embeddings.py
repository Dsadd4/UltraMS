"""Extract frozen embeddings for the cross-energy isomer benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from pathlib import PosixPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(benchmark_dir: Path, split: str, input_peaks: int) -> tuple[pd.DataFrame, list[dict]]:
    metadata = pd.read_csv(benchmark_dir / f"{split}_metadata.csv")
    with np.load(benchmark_dir / f"{split}_top{input_peaks}_peaks.npz") as peaks:
        mzs = peaks["mzs"]
        intensities = peaks["intensities"]
        lengths = peaks["lengths"]
    samples = []
    for i, row in metadata.iterrows():
        length = int(lengths[i])
        spectrum = np.stack(
            [mzs[i, :length], intensities[i, :length]], axis=1
        ).astype(np.float32)
        samples.append({"spectrum": spectrum, "precursor_mz": float(row["precursor_mz"])})
    return metadata, samples


@torch.no_grad()
def extract_ultra_ssl(
    checkpoint: Path,
    samples: list[dict],
    device: torch.device,
    batch_size: int,
    max_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    from appliedmodel.common import load_ultra_backbone, ultra_forward_tokens, ultra_prepare

    model = load_ultra_backbone(device, checkpoint_path=str(checkpoint), state_key="base_state_dict")
    outputs, fusion_outputs = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        peaks, attention, precursor = ultra_prepare(batch, device, max_peaks=max_peaks)
        cls, peak_tokens = ultra_forward_tokens(model, peaks, attention, precursor)
        weights = peaks[:, :, 1] * attention
        peak_iw = (peak_tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        outputs.append(cls.detach().cpu().numpy().astype(np.float32))
        fusion = F.normalize(torch.cat([F.normalize(cls, dim=1), F.normalize(peak_iw, dim=1)], dim=1), dim=1)
        fusion_outputs.append(fusion.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"UltraMS SSL: {start}/{len(samples)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(outputs), np.concatenate(fusion_outputs)


def build_projection(in_dim: int, state: dict, kind: str) -> nn.Module:
    if kind == "linear" or (kind == "auto" and set(state) == {"weight", "bias"}):
        module = nn.Linear(in_dim, in_dim)
    else:
        hidden = int(state["1.weight"].shape[0])
        module = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden, in_dim),
        )
    module.load_state_dict(state, strict=False)
    return module


@torch.no_grad()
def extract_ultra_supervised(checkpoint: Path, samples: list[dict], device, batch_size: int):
    from appliedmodel.common import (
        get_backbone_hidden_dim, load_ultra_backbone, ultra_forward_tokens, ultra_prepare,
    )

    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    config = checkpoint_data.get("config", {})
    model = load_ultra_backbone(
        device, checkpoint_path=os.environ.get("ULTRAMS_STAGE_D_CKPT") or config.get("ultra_ckpt") or None,
        state_key=config.get("state_key", "base_state_dict"),
    )
    model.load_state_dict(checkpoint_data["encoder_state_dict"], strict=False)
    model.eval()
    projection = build_projection(
        get_backbone_hidden_dim(model), checkpoint_data["projection_state_dict"],
        config.get("projection", "auto"),
    ).to(device).eval()
    outputs, fusion_outputs = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        peaks, attention, precursor = ultra_prepare(
            batch, device, max_peaks=int(config.get("max_peaks", 100))
        )
        cls, peak_tokens = ultra_forward_tokens(model, peaks, attention, precursor)
        projected = projection(cls)
        weights = peaks[:, :, 1] * attention
        peak_iw = (peak_tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fusion = F.normalize(
            torch.cat([F.normalize(projected, dim=1), F.normalize(peak_iw, dim=1)], dim=1), dim=1
        )
        outputs.append(projected.detach().cpu().numpy().astype(np.float32))
        fusion_outputs.append(fusion.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"UltraMS supervised: {start}/{len(samples)}", flush=True)
    del model, projection, checkpoint_data
    torch.cuda.empty_cache()
    return np.concatenate(outputs), np.concatenate(fusion_outputs)


def patch_dreams_aliases() -> None:
    from comparison import dreams_loader

    dreams_loader._patch_msml()
    msml = sys.modules.get("msml")
    if msml is not None and not hasattr(msml, "__path__"):
        msml.__path__ = []
    aliases = {
        "msml.models": "dreams.models",
        "msml.models.heads": "dreams.models.heads",
        "msml.models.heads.heads": "dreams.models.heads.heads",
        "msml.models.dreams": "dreams.models.dreams",
        "msml.models.dreams.dreams": "dreams.models.dreams.dreams",
    }
    for legacy, current in aliases.items():
        sys.modules[legacy] = importlib.import_module(current)
    sys.modules["msml"].models = sys.modules["msml.models"]
    sys.modules["msml.models"].heads = sys.modules["msml.models.heads"]
    sys.modules["msml.models"].dreams = sys.modules["msml.models.dreams"]


@torch.no_grad()
def extract_dreams_ssl(root: Path, checkpoint: Path, samples: list[dict], device, batch_size: int):
    sys.path.insert(0, os.environ.get("ULTRAMS_DREAMS_ROOT", str(root / "train/comparison/resources/dreams/DreaMS")))
    from comparison import dreams_loader

    model = dreams_loader.load_dreams_encoder(device=device, ckpt_path=str(checkpoint))
    outputs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        peaks = dreams_loader.build_dreams_batch(model, batch, device)
        hidden = model(peaks, charge=None)
        outputs.append(hidden[:, 0, :].detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"DreaMS SSL precursor token: {start}/{len(samples)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(outputs)


@torch.no_grad()
def extract_dreams_supervised(root: Path, checkpoint: Path, samples: list[dict], device, batch_size: int):
    sys.path.insert(0, os.environ.get("ULTRAMS_DREAMS_ROOT", str(root / "train/comparison/resources/dreams/DreaMS")))
    from comparison import dreams_loader

    patch_dreams_aliases()
    from dreams.models.heads.heads import ContrastiveHead

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath, argparse.Namespace])
    original_torch_load = torch.load

    def trusted_checkpoint_load(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_checkpoint_load
    try:
        model = ContrastiveHead.load_from_checkpoint(
            checkpoint,
            backbone_pth=Path(os.environ.get("ULTRAMS_DREAMS_CKPT", str(root / "train/comparison/resources/DreaMS_Check/ssl_model.ckpt"))),
            map_location=device,
        ).to(device).eval()
    finally:
        torch.load = original_torch_load
    defaults = {
        "prec_intens": 1.1, "n_highest_peaks": 60, "spec_entropy_cleaning": False,
        "normalize_mzs": False, "to_relative_intensities": True, "precision": 32,
        "mz_shift_aug_p": 0, "mz_shift_aug_max": 0,
    }
    for key, value in defaults.items():
        if not hasattr(model.backbone.spec_preproc, key):
            setattr(model.backbone.spec_preproc, key, value)
    model.backbone.ff_out = None
    model.backbone.ro_out = None
    outputs, fusion_outputs = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        peaks = dreams_loader.build_dreams_batch(model.backbone, batch, device)
        hidden = model.backbone(peaks, charge=None)
        projected = model.head(hidden[:, 0, :])
        peak_tokens = hidden[:, 1:, :]
        peak_inputs = peaks[:, 1 : 1 + peak_tokens.shape[1], :]
        mask = (peak_inputs.abs().sum(-1) > 0).to(peak_tokens.dtype)
        weights = peak_inputs[:, :, 1].clamp_min(0) * mask
        peak_iw = (peak_tokens * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fusion = F.normalize(
            torch.cat([F.normalize(projected, dim=1), F.normalize(peak_iw, dim=1)], dim=1), dim=1
        )
        outputs.append(projected.detach().cpu().numpy().astype(np.float32))
        fusion_outputs.append(fusion.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"DreaMS supervised: {start}/{len(samples)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(outputs), np.concatenate(fusion_outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ultra-ssl", type=Path, required=True)
    parser.add_argument("--ultra-supervised", type=Path, required=True)
    parser.add_argument("--dreams-supervised", type=Path, required=True)
    parser.add_argument("--dreams-ssl", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-peaks", type=int, default=60)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root))
    sys.path.insert(0, str(args.root / "train"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dreams_ssl = args.dreams_ssl or (
        args.root / "train/comparison/resources/DreaMS_Check/ssl_model.ckpt"
    )
    print(f"device={device}", flush=True)

    manifest = {
        "input_policy": (
            f"frozen top {args.input_peaks} peaks; DreaMS applies its native top-60 preprocessor, "
            "UltraMS supervised applies checkpoint-native top-100, and UltraMS SSL uses up to top-150"
        ),
        "checkpoints": {
            "ultrams_ssl": {"path": str(args.ultra_ssl), "sha256": sha256(args.ultra_ssl)},
            "ultrams_supervised": {
                "path": str(args.ultra_supervised), "sha256": sha256(args.ultra_supervised)
            },
            "dreams_supervised": {
                "path": str(args.dreams_supervised), "sha256": sha256(args.dreams_supervised)
            },
            "dreams_ssl": {"path": str(dreams_ssl), "sha256": sha256(dreams_ssl)},
        },
        "splits": {},
    }
    for split in ("val", "test"):
        metadata, samples = load_samples(args.benchmark_dir, split, args.input_peaks)
        ultra_ssl, ultra_ssl_fusion = extract_ultra_ssl(
            args.ultra_ssl, samples, device, args.batch_size, min(args.input_peaks, 150)
        )
        ultra_supervised, ultra_supervised_fusion = extract_ultra_supervised(
            args.ultra_supervised, samples, device, args.batch_size
        )
        dreams_supervised, dreams_supervised_fusion = extract_dreams_supervised(
            args.root, args.dreams_supervised, samples, device, args.batch_size
        )
        dreams_ssl_values = extract_dreams_ssl(
            args.root, dreams_ssl, samples, device, args.batch_size
        )
        arrays = {
            "ultrams_ssl": ultra_ssl,
            "ultrams_ssl_fusion": ultra_ssl_fusion,
            "ultrams_supervised": ultra_supervised,
            "ultrams_supervised_fusion": ultra_supervised_fusion,
            "dreams_supervised": dreams_supervised,
            "dreams_supervised_fusion": dreams_supervised_fusion,
            "dreams_ssl": dreams_ssl_values,
        }
        manifest["splits"][split] = {"n_spectra": len(metadata), "models": {}}
        for name, values in arrays.items():
            if values.shape[0] != len(metadata) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid {name} embeddings for {split}: {values.shape}")
            path = args.out_dir / f"{split}_{name}.npy"
            np.save(path, values)
            manifest["splits"][split]["models"][name] = {
                "shape": list(values.shape), "sha256": sha256(path)
            }
            print(f"saved {path} {values.shape}", flush=True)
    (args.out_dir / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
