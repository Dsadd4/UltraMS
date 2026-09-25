#!/usr/bin/env python3
"""Train one cross-CE split-safe ChemBERTa readout at a fixed epoch."""

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


def inputs_for(cache: Path, mode: str, method: str):
    root = cache / mode
    if method == "linear":
        return np.load(root / "bins_float16.npy", mmap_mode="r")
    if method in {"deepsets", "fourier_projection"}:
        return np.load(root / "peaksets_float32.npy", mmap_mode="r")
    return np.load(root / "codebook_float32.npy", mmap_mode="r")


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest-dir", required=True)
    parser.add_argument("--crossce-manifest-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("This comparison follows the original single-seed protocol (seed 0).")
    set_determinism(args.seed)
    source_manifest = Path(args.source_manifest_dir)
    crossce_manifest = Path(args.crossce_manifest_dir)
    cache = Path(args.cache_dir)
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE.json").exists() and not args.force:
        print((output / "DONE.json").read_text())
        return
    protocol = np.load(crossce_manifest / f"protocol_{args.mode}.npz")
    train_rows = np.asarray(protocol["train_rows"], dtype=np.int64)
    inputs = inputs_for(cache, args.mode, args.model)
    targets = np.load(cache / args.mode / "target_chemberta_float32.npy", mmap_mode="r")
    with h5py.File(source_manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        row_labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
    config = {
        "status": "configured",
        "experiment": "cross-CE split-safe ChemBERTa readout",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "train_rows": len(train_rows),
        "training_objective": "in-batch InfoNCE to frozen ChemBERTa molecule embeddings",
        "checkpoint_selection": "fixed final epoch; no validation/test selection",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "leakage_control": "only the 70% cross-CE training compounds are used for optimization",
        "crossce_manifest_sha256": sha256_file(crossce_manifest / "manifest_summary.json"),
    }
    atomic_write_json(output / "config.json", config)
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    rng = np.random.default_rng(args.seed)
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(train_rows)
        model.train()
        total = 0.0
        epoch_started = time.time()
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start:start + args.batch_size]
            x = torch.from_numpy(np.asarray(inputs[batch], dtype=np.float32)).to(device)
            target = torch.from_numpy(np.asarray(targets[row_labels[batch]], dtype=np.float32)).to(device)
            prediction = F.normalize(model(x).float(), dim=1)
            target = F.normalize(target, dim=1)
            logits = prediction @ target.T / args.temperature
            labels = torch.arange(len(batch), device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_infonce_loss": total / len(train_rows),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        write_history(output / "history.csv", history)
        atomic_write_json(output / "history.json", history)
        print(json.dumps({"model": args.model, "mode": args.mode, **row}), flush=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "config": config,
    }, output / "final.pt")
    atomic_write_json(output / "DONE.json", {
        "status": "complete",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "epochs_completed": args.epochs,
        "runtime_seconds": time.time() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_sha256": sha256_file(output / "final.pt"),
        "history_sha256": sha256_file(output / "history.csv"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    })


if __name__ == "__main__":
    main()
