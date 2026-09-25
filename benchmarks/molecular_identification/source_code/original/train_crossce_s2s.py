#!/usr/bin/env python3
"""Train simple Figure 3c controls with low/high-CE spectrum-pair InfoNCE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from baseline_common import atomic_write_json, sha256_file
from train_chemberta_readout import make_model, set_determinism
from train_fig3h_s2s_readout import arrays_for, encode, pair_loss, validation_metrics


METHODS = ("linear", "deepsets", "fourier_projection", "ultrams_codebook")


def groups_by_label(rows: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    output: dict[int, list[int]] = {}
    for row in np.asarray(rows, dtype=np.int64):
        output.setdefault(int(labels[row]), []).append(int(row))
    return {label: np.asarray(values, dtype=np.int64) for label, values in output.items()}


def sample_pairs(
    low: dict[int, np.ndarray], high: dict[int, np.ndarray], labels: list[int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels, dtype=np.int64)[rng.permutation(len(labels))]
    left = np.asarray([rng.choice(low[int(label)]) for label in shuffled], dtype=np.int64)
    right = np.asarray([rng.choice(high[int(label)]) for label in shuffled], dtype=np.int64)
    return left, right


def write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Figure 3 comparison uses the frozen seed-0 policy")
    set_determinism(args.seed)
    output = args.output_dir / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE.json").exists():
        print((output / "DONE.json").read_text()); return

    protocol = np.load(args.manifest_dir / f"protocol_{args.mode}.npz")
    inputs = arrays_for(args.cache_dir, args.mode, args.model)
    with h5py.File(args.manifest_dir / "msnlib_fig3h_spectra.h5", "r") as h5:
        labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
        source_rows = np.asarray(h5[args.mode]["source_row"], dtype=np.int64)
    low = groups_by_label(protocol["train_low_rows"], labels)
    high = groups_by_label(protocol["train_high_rows"], labels)
    train_labels = sorted(set(low) & set(high))
    val_left = np.asarray(protocol["val_query_rows"], dtype=np.int64)
    val_right = np.asarray(protocol["val_library_rows"], dtype=np.int64)
    if len(train_labels) < 100 or len(val_left) < 100:
        raise RuntimeError("insufficient cross-CE train/validation pairs")
    np.savez_compressed(
        output / "validation_pairs.npz",
        query_rows=val_left,
        positive_rows=val_right,
        query_source_rows=source_rows[val_left],
        positive_source_rows=source_rows[val_right],
    )
    config = {
        "status": "configured",
        "experiment": "Figure 3c controls with UltraMS-matched cross-CE spectrum-pair supervision",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "embedding_dimension": 256,
        "training_objective": "symmetric in-batch InfoNCE between low-CE and high-CE spectra of the same molecule",
        "molecule_embedding_supervision": "none",
        "chemberta_used": False,
        "checkpoint_selection": "minimum symmetric InfoNCE on frozen validation low/high-CE pairs",
        "n_train_pairable_molecules": len(train_labels),
        "n_validation_pairs": len(val_left),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "warmup_ratio": args.warmup_ratio,
        "gradient_clip_norm": 1.0,
        "manifest_summary_sha256": sha256_file(args.manifest_dir / "manifest_summary.json"),
    }
    atomic_write_json(output / "config.json", config)
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_labels) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        left, right = sample_pairs(low, high, train_labels, args.seed * 1_000_003 + epoch)
        model.train(); loss_sum = 0.0; acc_sum = 0.0; count_sum = 0
        epoch_started = time.time()
        for start in range(0, len(left), args.batch_size):
            stop = min(start + args.batch_size, len(left))
            z_a = encode(model, inputs, left[start:stop], device)
            z_b = encode(model, inputs, right[start:stop], device)
            loss, accuracy = pair_loss(z_a, z_b, args.temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            count = stop - start
            loss_sum += float(loss.detach()) * count
            acc_sum += float(accuracy.detach()) * count
            count_sum += count
        val_loss, val_acc, val_cos = validation_metrics(
            model, inputs, val_left, val_right, device, args.batch_size, args.temperature
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss; best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch,
                 "validation_infonce_loss": val_loss, "config": config},
                output / "best.pt",
            )
        row = {
            "epoch": epoch,
            "train_infonce_loss": loss_sum / count_sum,
            "train_pair_accuracy": acc_sum / count_sum,
            "validation_infonce_loss": val_loss,
            "validation_pair_accuracy": val_acc,
            "validation_positive_cosine": val_cos,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "improved": int(improved),
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row); write_history(output / "history.csv", history)
        atomic_write_json(output / "history.json", history)
        print(json.dumps({"model": args.model, "mode": args.mode, **row}), flush=True)
    done = {
        "status": "complete",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_infonce_loss": best_loss,
        "epochs_completed": args.epochs,
        "runtime_seconds": time.time() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "history_sha256": sha256_file(output / "history.csv"),
        "config_sha256": sha256_file(output / "config.json"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    atomic_write_json(output / "DONE.json", done)


if __name__ == "__main__":
    main()
