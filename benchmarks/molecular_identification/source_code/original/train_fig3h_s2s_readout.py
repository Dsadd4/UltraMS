#!/usr/bin/env python3
"""Train Figure 3h controls with the same spectrum-pair InfoNCE objective as UltraMS.

One training item is one molecule and two distinct spectra of that molecule.
ChemBERTa and every other molecule embedding are deliberately absent.
"""

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


def grouped_rows(rows: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(labels[rows], kind="stable")
    ordered = rows[order]
    ordered_labels = labels[ordered]
    boundaries = np.flatnonzero(np.diff(ordered_labels)) + 1
    return [group for group in np.split(ordered, boundaries) if len(group) >= 2]


def fixed_validation_pairs(
    query_rows: np.ndarray,
    library_rows: np.ndarray,
    labels: np.ndarray,
    source_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_rows = np.asarray(query_rows, dtype=np.int64)
    library_rows = np.asarray(library_rows, dtype=np.int64)
    library_order = np.lexsort((source_rows[library_rows], labels[library_rows]))
    ordered_library = library_rows[library_order]
    ordered_labels = labels[ordered_library]
    unique_labels, starts = np.unique(ordered_labels, return_index=True)
    lookup = {int(label): int(ordered_library[start]) for label, start in zip(unique_labels, starts)}
    keep_query, positives = [], []
    seen = set()
    for row in query_rows:
        label = int(labels[row])
        if label in seen:
            raise RuntimeError("validation query labels are not molecule-unique")
        seen.add(label)
        if label in lookup:
            keep_query.append(int(row)); positives.append(lookup[label])
    return np.asarray(keep_query, np.int64), np.asarray(positives, np.int64)


def sample_epoch_pairs(groups: list[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    left = np.empty(len(groups), dtype=np.int64)
    right = np.empty(len(groups), dtype=np.int64)
    for output_index, group_index in enumerate(order):
        group = groups[int(group_index)]
        chosen = rng.choice(len(group), size=2, replace=False)
        left[output_index] = group[int(chosen[0])]
        right[output_index] = group[int(chosen[1])]
    return left, right


def pair_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float):
    logits_ab = z_a @ z_b.T / temperature
    logits_ba = z_b @ z_a.T / temperature
    labels = torch.arange(len(z_a), device=z_a.device)
    loss = (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels)) / 2
    accuracy = ((logits_ab.argmax(1) == labels).float().mean() +
                (logits_ba.argmax(1) == labels).float().mean()) / 2
    return loss, accuracy


def encode(model, inputs, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(inputs[rows], dtype=np.float32)).to(device)
    return F.normalize(model(values).float(), dim=1)


@torch.no_grad()
def validation_metrics(model, inputs, left, right, device, batch_size, temperature):
    model.eval(); total_loss = 0.0; total_acc = 0.0; total_n = 0; cosine_sum = 0.0
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        z_a = encode(model, inputs, left[start:stop], device)
        z_b = encode(model, inputs, right[start:stop], device)
        loss, accuracy = pair_loss(z_a, z_b, temperature)
        count = stop - start
        total_loss += float(loss) * count
        total_acc += float(accuracy) * count
        cosine_sum += float(F.cosine_similarity(z_a, z_b, dim=1).sum())
        total_n += count
    return total_loss / total_n, total_acc / total_n, cosine_sum / total_n


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Figure 3 uses the same single frozen seed-0 protocol as the existing experiment")
    set_determinism(args.seed)
    manifest = Path(args.manifest_dir); cache = Path(args.cache_dir)
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE.json").exists() and not args.force:
        print((output / "DONE.json").read_text()); return

    protocol = np.load(manifest / f"protocol_{args.mode}.npz")
    inputs = arrays_for(cache, args.mode, args.model)
    with h5py.File(manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
        source_rows = np.asarray(h5[args.mode]["source_row"], dtype=np.int64)
    if len(inputs) != len(labels):
        raise RuntimeError("row-aligned input cache and manifest differ")
    groups = grouped_rows(protocol["train_rows"], labels)
    val_left, val_right = fixed_validation_pairs(
        protocol["val_query_rows"], protocol["val_library_rows"], labels, source_rows
    )
    if len(groups) < 100 or len(val_left) < 100:
        raise RuntimeError("insufficient same-molecule spectrum pairs")
    np.savez_compressed(
        output / "validation_pairs.npz", query_rows=val_left, positive_rows=val_right,
        query_source_rows=source_rows[val_left], positive_source_rows=source_rows[val_right],
    )

    config = {
        "status": "configured",
        "experiment": "Figure 3h architecture controls with UltraMS-matched spectrum-pair supervision",
        "model": args.model, "mode": args.mode, "seed": args.seed,
        "embedding_dimension": 256,
        "training_objective": "symmetric in-batch InfoNCE between two distinct spectra of the same molecule",
        "molecule_embedding_supervision": "none",
        "chemberta_used": False,
        "training_sampling": "one same-molecule spectrum pair per eligible train molecule per epoch",
        "checkpoint_selection": "minimum symmetric InfoNCE on frozen molecule-unique validation spectrum pairs",
        "n_train_pairable_molecules": len(groups),
        "n_validation_pairs": len(val_left),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "temperature": args.temperature, "warmup_ratio": args.warmup_ratio,
        "gradient_clip_norm": 1.0,
        "manifest_summary_sha256": sha256_file(manifest / "manifest_summary.json"),
        "validation_pairs_sha256": sha256_file(output / "validation_pairs.npz"),
    }
    atomic_write_json(output / "config.json", config)
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(groups) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []; best_loss = float("inf"); best_epoch = 0; started = time.time()
    for epoch in range(1, args.epochs + 1):
        left, right = sample_epoch_pairs(groups, args.seed * 1_000_003 + epoch)
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
        val_loss, val_acc, val_cosine = validation_metrics(
            model, inputs, val_left, val_right, device,
            args.validation_batch_size, args.temperature,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss; best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "validation_infonce_loss": val_loss, "config": config}, output / "best.pt")
        row = {
            "epoch": epoch, "train_infonce_loss": loss_sum / count_sum,
            "train_pair_accuracy": acc_sum / count_sum,
            "validation_infonce_loss": val_loss,
            "validation_pair_accuracy": val_acc,
            "validation_positive_cosine": val_cosine,
            "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "improved": int(improved), "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row); write_history(output / "history.csv", history)
        atomic_write_json(output / "history.json", history)
        print(json.dumps({"model": args.model, "mode": args.mode, **row}), flush=True)

    done = {
        "status": "complete", "model": args.model, "mode": args.mode, "seed": args.seed,
        "best_epoch": best_epoch, "best_validation_infonce_loss": best_loss,
        "epochs_completed": args.epochs, "runtime_seconds": time.time() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "history_sha256": sha256_file(output / "history.csv"),
        "config_sha256": sha256_file(output / "config.json"),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
    }
    atomic_write_json(output / "DONE.json", done)


if __name__ == "__main__":
    main()
