#!/usr/bin/env python3
"""Train one Figure 3h split-specific ChemBERTa readout baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, sha256_file
from train_chemberta_readout import make_model, set_determinism


METHODS = ("linear", "deepsets", "fourier_projection", "ultrams_codebook")


def arrays_for(cache: Path, mode: str, method: str):
    root = cache / mode
    if method == "linear":
        return np.load(root / "bins_float16.npy", mmap_mode="r")
    if method in {"deepsets", "fourier_projection"}:
        return np.load(root / "peaksets_float32.npy", mmap_mode="r")
    return np.load(root / "codebook_float32.npy", mmap_mode="r")


def iter_batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def validation_loss(model, inputs, targets, row_labels, rows, device, batch_size):
    model.eval(); total = 0.0
    for batch in iter_batches(rows, batch_size):
        x = torch.from_numpy(np.asarray(inputs[batch], dtype=np.float32)).to(device)
        target = torch.from_numpy(np.asarray(targets[row_labels[batch]], dtype=np.float32)).to(device)
        prediction = F.normalize(model(x).float(), dim=1)
        total += float((1.0 - F.cosine_similarity(prediction, target, dim=1)).sum())
    return total / len(rows)


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("the frozen main comparison uses seed 0 only")
    set_determinism(args.seed)
    manifest = Path(args.manifest_dir)
    cache = Path(args.cache_dir)
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE.json").exists() and not args.force:
        print((output / "DONE.json").read_text()); return
    protocol = np.load(manifest / f"protocol_{args.mode}.npz")
    train_rows = np.asarray(protocol["train_rows"], dtype=np.int64)
    validation_rows = np.asarray(protocol["val_query_rows"], dtype=np.int64)
    inputs = arrays_for(cache, args.mode, args.model)
    targets = np.load(cache / args.mode / "target_chemberta_float32.npy", mmap_mode="r")
    with h5py.File(manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        row_labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
    if not np.isfinite(np.asarray(targets)).all():
        raise RuntimeError("non-finite molecule target")
    config = {
        "status": "configured", "experiment": "Figure 3h split-specific readout reuse",
        "model": args.model, "mode": args.mode, "seed": args.seed,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "training_objective": "in-batch InfoNCE to the frozen ChemBERTa molecule space",
        "checkpoint_selection": "lowest paired cosine loss on frozen validation queries",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "leakage_control": "trained only on the mode-specific Figure 3h train compounds",
        "manifest_summary_sha256": sha256_file(manifest / "manifest_summary.json"),
    }
    atomic_write_json(output / "config.json", config)
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    rng = np.random.default_rng(args.seed)
    history = []; best_loss = float("inf"); best_epoch = 0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(train_rows)
        model.train(); total = 0.0; epoch_started = time.time()
        for batch in iter_batches(permutation, args.batch_size):
            x = torch.from_numpy(np.asarray(inputs[batch], dtype=np.float32)).to(device)
            target = torch.from_numpy(np.asarray(targets[row_labels[batch]], dtype=np.float32)).to(device)
            prediction = F.normalize(model(x).float(), dim=1)
            logits = prediction @ target.T / args.temperature
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch)
        scheduler.step()
        val_loss = validation_loss(
            model, inputs, targets, row_labels, validation_rows, device, args.validation_batch_size
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_loss": val_loss, "config": config}, output / "best.pt")
        row = {
            "epoch": epoch, "train_infonce_loss": total / len(train_rows),
            "validation_paired_cosine_loss": val_loss, "best_validation_loss": best_loss,
            "best_epoch": best_epoch, "improved": int(improved),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row); write_history(output / "history.csv", history)
        atomic_write_json(output / "history.json", history)
        print(json.dumps({"model": args.model, "mode": args.mode, **row}), flush=True)
    atomic_write_json(output / "DONE.json", {
        "status": "complete", "model": args.model, "mode": args.mode, "seed": args.seed,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "epochs_completed": args.epochs, "runtime_seconds": time.time() - started,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "history_sha256": sha256_file(output / "history.csv"),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
    })


if __name__ == "__main__":
    main()
