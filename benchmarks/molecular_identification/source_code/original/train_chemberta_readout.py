#!/usr/bin/env python3
"""Train matched ChemBERTa-readout baselines for MSnLib Figure 3.

The four spectrum encoders are kept deliberately simple.  The only conceptual
change from the archived fingerprint controls is that their final output is a
768-dimensional vector trained against the frozen ChemBERTa molecule space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_common import atomic_write_json, sha256_file


MODEL_NAMES = (
    "linear",
    "binned_ffn",
    "deepsets",
    "deepsets_fourier",
    "shallow_deepsets",
    "fourier_projection",
    "ultrams_codebook",
)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class DenseMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = in_dim
        for _ in range(layers - 1):
            modules.extend((nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)))
            current = hidden_dim
        modules.append(nn.Linear(current, out_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class LinearReadout(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1005, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class BinnedFFNReadout(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.network = DenseMLP(1005, 128, out_dim, layers=4, dropout=0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DreamsFourierFeatures(nn.Module):
    """Fixed-frequency m/z construction used by the MassSpecGym DeepSets baseline."""

    def __init__(self, x_min: float = 1e-4, x_max: float = 1000.0) -> None:
        super().__init__()
        frequencies = [1.0 / (x_min * i) for i in range(2, math.ceil(1.0 / x_min), 2)]
        frequencies += [1.0 / i for i in range(2, math.ceil(x_max), 1)]
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32).view(1, 1, -1))

    @property
    def num_features(self) -> int:
        return int(self.frequencies.numel() * 2)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * torch.pi * mz * self.frequencies
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class DeepSetsReadout(nn.Module):
    def __init__(self, out_dim: int, fourier: bool) -> None:
        super().__init__()
        self.fourier = bool(fourier)
        hidden = 128
        if self.fourier:
            self.ff = DreamsFourierFeatures()
            mz_channels = int(0.8 * hidden)
            self.ff_proj_mz = nn.Linear(self.ff.num_features, mz_channels)
            self.ff_proj_i = nn.Linear(1, hidden - mz_channels)
            phi_in = hidden
        else:
            phi_in = 2
        self.phi = DenseMLP(phi_in, hidden, hidden, layers=4, dropout=0.1)
        self.rho = DenseMLP(hidden, hidden, out_dim, layers=4, dropout=0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.fourier:
            mz = self.ff_proj_mz(self.ff(values[:, :, 0:1]))
            intensity = self.ff_proj_i(values[:, :, 1:2])
            values = torch.cat((mz, intensity), dim=-1)
        values = self.phi(values)
        values = values.sum(dim=-2)
        return self.rho(values)


class ShallowDeepSetsReadout(nn.Module):
    """Low-capacity set encoder fixed before validation/test evaluation.

    This keeps the same 61-peak input and sum aggregation as the full DeepSets
    control, but uses one shared peak projection and one linear molecule-space
    readout.  It is therefore a capacity ablation, not a test-tuned weak model.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.peak_projection = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        self.readout = nn.Linear(64, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.readout(self.peak_projection(values).sum(dim=-2))


class FourierProjectionReadout(nn.Module):
    """Fixed Fourier spectrum pooling followed by one learned projection."""

    def __init__(self, out_dim: int, n_frequencies: int = 64) -> None:
        super().__init__()
        wavelengths = torch.logspace(math.log10(0.25), math.log10(1000.0), n_frequencies)
        self.register_buffer("wavelengths", wavelengths.view(1, 1, -1))
        self.projection = nn.Linear(2 * n_frequencies, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Row 0 is the precursor token in the shared cache.  This control is a
        # fragment-spectrum Fourier baseline, so it excludes that row.
        peaks = values[:, 1:, :]
        mz = peaks[:, :, 0:1]
        intensity = peaks[:, :, 1:2].clamp_min(0.0)
        valid = (mz > 0).to(intensity.dtype)
        weights = intensity * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        phase = 2.0 * torch.pi * mz / self.wavelengths
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        pooled = (encoded * weights).sum(dim=1)
        return self.projection(pooled)


class UltraMSCodebookReadout(nn.Module):
    """Linear readout from a frozen, transformer-free UltraMS codebook pool."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1024, out_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


def make_model(name: str, out_dim: int = 768) -> nn.Module:
    if name == "linear":
        return LinearReadout(out_dim)
    if name == "binned_ffn":
        return BinnedFFNReadout(out_dim)
    if name == "deepsets":
        return DeepSetsReadout(out_dim, fourier=False)
    if name == "deepsets_fourier":
        return DeepSetsReadout(out_dim, fourier=True)
    if name == "shallow_deepsets":
        return ShallowDeepSetsReadout(out_dim)
    if name == "fourier_projection":
        return FourierProjectionReadout(out_dim)
    if name == "ultrams_codebook":
        return UltraMSCodebookReadout(out_dim)
    raise ValueError(name)


def iter_batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def model_inputs(
    name: str,
    bins: np.ndarray,
    peaks: np.ndarray,
    indices: np.ndarray,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    if name == "ultrams_codebook":
        if codebook is None:
            raise RuntimeError("ultrams_codebook requires --codebook-cache-dir")
        source = codebook
    else:
        source = bins if name in {"linear", "binned_ffn"} else peaks
    return np.asarray(source[indices], dtype=np.float32)


def molecule_targets(
    target_embeddings: np.ndarray,
    row_smiles_index: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = np.asarray(target_embeddings[np.asarray(row_smiles_index[indices])], dtype=np.float32)
    return F.normalize(torch.from_numpy(values).to(device), dim=1)


@torch.no_grad()
def validation_paired_cosine_loss(
    model: nn.Module,
    name: str,
    bins: np.ndarray,
    peaks: np.ndarray,
    codebook: np.ndarray | None,
    row_smiles_index: np.ndarray,
    target_embeddings: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    for batch in iter_batches(indices, batch_size):
        x = torch.from_numpy(model_inputs(name, bins, peaks, batch, codebook)).to(device)
        pred = F.normalize(model(x).float(), dim=1)
        target = molecule_targets(target_embeddings, row_smiles_index, batch, device)
        total += float((1.0 - F.cosine_similarity(pred, target, dim=1)).sum())
    return total / len(indices)


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    if args.seed != 0:
        raise ValueError("the preregistered main comparison uses seed 0 only")
    set_determinism(args.seed)
    cache = Path(args.ffn_cache_dir)
    peak_cache = Path(args.peak_cache_dir)
    targets_dir = Path(args.targets_dir)
    out = Path(args.output_dir) / args.model / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists() and not args.force:
        print((out / "DONE.json").read_text())
        return

    bins = np.load(cache / "ffn_bins_float16.npy", mmap_mode="r")
    peaks = np.load(peak_cache / "peaksets_float32.npy", mmap_mode="r")
    codebook = None
    if args.model == "ultrams_codebook":
        if not args.codebook_cache_dir:
            raise RuntimeError("--codebook-cache-dir is required for ultrams_codebook")
        codebook_dir = Path(args.codebook_cache_dir)
        codebook = np.load(codebook_dir / "ultrams_codebook_pool_float32.npy", mmap_mode="r")
        if codebook.shape != (len(bins), 1024) or codebook.dtype != np.float32:
            raise RuntimeError(f"invalid UltraMS codebook cache: {codebook.shape} {codebook.dtype}")
    folds = np.load(cache / "fold_codes_uint8.npy", mmap_mode="r")
    row_smiles_index = np.load(cache / "row_smiles_index_int32.npy", mmap_mode="r")
    targets = np.load(targets_dir / "target_chemberta_float32.npy", mmap_mode="r")
    target_valid = np.load(targets_dir / "target_chemberta_valid.npy", mmap_mode="r")
    target_smiles = json.loads((cache / "unique_target_smiles.json").read_text())
    if len(target_smiles) != len(targets) or targets.shape[1] != 768:
        raise RuntimeError("ChemBERTa target cache does not align with the frozen target-SMILES list")
    if not np.all(target_valid[np.unique(row_smiles_index[folds <= 2])]):
        raise RuntimeError("an MSnLib target lacks a ChemBERTa embedding")

    train_indices = np.where(folds == 0)[0]
    validation_indices = np.where(folds == 1)[0]
    if len(train_indices) != 446_667 or len(validation_indices) != 55_980:
        raise RuntimeError((len(train_indices), len(validation_indices)))
    if args.max_train_samples:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_validation_samples:
        validation_indices = validation_indices[: args.max_validation_samples]

    architecture = {
        "linear": "Linear(1005,768)",
        "binned_ffn": "MLP 1005->128->128->128->768 with ReLU/dropout(0.1)",
        "deepsets": "phi MLP(2->128,4 layers)+sum+rho MLP(128->768,4 layers)",
        "deepsets_fourier": "fixed Fourier m/z features; 102/26 m/z/intensity projection; phi/rho as DeepSets",
        "shallow_deepsets": "shared Linear(2,64)+ReLU per peak; sum pooling; Linear(64,768)",
        "fourier_projection": "64 fixed log-spaced Fourier wavelengths; intensity-weighted pooling; Linear(128,768)",
        "ultrams_codebook": "frozen UltraMS peak-codebook encoder; intensity-weighted pooling; Linear(1024,768)",
    }[args.model]
    config = {
        "experiment": "MSnLib matched ChemBERTa-readout baselines",
        "method": args.model,
        "display_name": {
            "linear": "Linear",
            "binned_ffn": "Binned FFN",
            "deepsets": "DeepSets",
            "deepsets_fourier": "DeepSets + Fourier",
            "shallow_deepsets": "Shallow DeepSets",
            "fourier_projection": "Fourier projection",
            "ultrams_codebook": "UltraMS codebook",
        }[args.model],
        "seed": args.seed,
        "architecture": architecture,
        "molecule_target": "frozen 768-dimensional ChemBERTa embedding used by UltraMS/DreaMS retrieval",
        "training_objective": "in-batch InfoNCE on normalized spectrum and ChemBERTa vectors",
        "temperature": args.temperature,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "epochs": args.epochs,
        "checkpoint_selection": "lowest full-validation mean paired cosine loss",
        "amp": False,
        "deterministic_algorithms": True,
        "main_seed_policy": "seed 0 only; split and query identities remain frozen",
        "canary_limits": {
            "max_train_samples": args.max_train_samples,
            "max_validation_samples": args.max_validation_samples,
        },
        "input_cache_summary_sha256": sha256_file(
            Path(args.codebook_cache_dir) / "DONE.json"
            if args.model == "ultrams_codebook"
            else cache / "prepare_summary.json"
            if args.model in {"linear", "binned_ffn"}
            else peak_cache / "DONE.json"
        ),
        "target_cache_summary_sha256": sha256_file(targets_dir / "projection_target_summary.json"),
        "target_smiles_sha256": sha256_file(cache / "unique_target_smiles.json"),
    }
    atomic_write_json(out / "config.json", config)

    device = torch.device(args.device)
    model = make_model(args.model, out_dim=targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    generator = np.random.default_rng(args.seed)
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    overall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        permutation = generator.permutation(train_indices)
        model.train()
        train_total = 0.0
        for step, batch in enumerate(iter_batches(permutation, args.batch_size), start=1):
            x = torch.from_numpy(model_inputs(args.model, bins, peaks, batch, codebook)).to(device)
            target = molecule_targets(targets, row_smiles_index, batch, device)
            pred = F.normalize(model(x).float(), dim=1)
            logits = pred @ target.T / args.temperature
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(batch)
            if step == 1 or step % args.log_every == 0:
                seen = min(step * args.batch_size, len(permutation))
                print(
                    f"model={args.model} epoch={epoch} step={step:,} train_loss={train_total / seen:.6f}",
                    flush=True,
                )
        scheduler.step()
        val_loss = validation_paired_cosine_loss(
            model,
            args.model,
            bins,
            peaks,
            codebook,
            row_smiles_index,
            targets,
            validation_indices,
            device,
            args.validation_batch_size,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "config": config,
                },
                out / "best.pt",
            )
        row = {
            "epoch": epoch,
            "train_infonce_loss": train_total / len(train_indices),
            "validation_paired_cosine_loss": val_loss,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "improved": int(improved),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_start,
            "elapsed_seconds": time.time() - overall_start,
        }
        history.append(row)
        write_history(out / "history.csv", history)
        atomic_write_json(out / "history.json", history)
        print(json.dumps(row), flush=True)

    checkpoint = out / "best.pt"
    done = {
        "status": "complete",
        "method": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history_sha256": sha256_file(out / "history.csv"),
        "runtime_seconds": time.time() - overall_start,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    atomic_write_json(out / "DONE.json", done)
    print(json.dumps(done, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--ffn-cache-dir", required=True)
    parser.add_argument("--peak-cache-dir", required=True)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--codebook-cache-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=0, help="canary only")
    parser.add_argument("--max-validation-samples", type=int, default=0, help="canary only")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
