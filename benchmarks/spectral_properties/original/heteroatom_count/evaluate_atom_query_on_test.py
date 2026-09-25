"""Evaluate epoch-15 UltraMS and DreaMS atom-query probes on test spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import os

_TRAIN_ROOT = Path(os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', Path(__file__).resolve().parents[1]))
_PROJECT_ROOT = _TRAIN_ROOT.parent
sys.path.insert(0, str(_TRAIN_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "supplement" / "pretrain_lighting"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

from showcase import data as showcase_data
from showcase.data import load_fold
from showcase.models_atom_query import FORMULA_IDX, load_probe_from_ckpt


ELEMENTS = ["S", "Cl", "F", "Br"]


class SpectrumDataset(Dataset):
    def __init__(self, samples: list[dict], labels: np.ndarray):
        self.samples = samples
        self.labels = labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict, np.ndarray]:
        return self.samples[index], self.labels[index]


def collate(batch: list[tuple[dict, np.ndarray]]) -> tuple[list[dict], torch.Tensor]:
    return [row[0] for row in batch], torch.from_numpy(np.stack([row[1] for row in batch]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    checkpoint: Path,
    backbone_checkpoint: Path | None,
    samples: list[dict],
    targets: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    model, atom_names, norm_mean, norm_std = load_probe_from_ckpt(
        str(checkpoint),
        device,
        ultra_ckpt=str(backbone_checkpoint) if backbone_checkpoint else None,
    )
    if atom_names != ELEMENTS:
        raise ValueError(f"Expected {ELEMENTS}, received {atom_names}")

    loader = DataLoader(
        SpectrumDataset(samples, targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_index, (batch, _) in enumerate(loader, start=1):
            logits = model(batch, device).detach().cpu().float().numpy()
            predictions.append(logits * norm_std + norm_mean)
            if batch_index % 20 == 0:
                print(f"{checkpoint.name}: batches={batch_index}/{len(loader)}", flush=True)
    prediction = np.concatenate(predictions)
    rows = []
    for index, element in enumerate(ELEMENTS):
        rows.append(
            {
                "element": element,
                "r2": float(r2_score(targets[:, index], prediction[:, index])),
                "test_n": int(len(targets)),
            }
        )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(checkpoint_data["epoch"]),
        "checkpoint_validation_mean_r2": float(checkpoint_data["val_mean_r2"]),
    }
    return pd.DataFrame(rows), prediction, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ultra-probe", type=Path, required=True)
    parser.add_argument("--dreams-probe", type=Path, required=True)
    parser.add_argument("--ultra-backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output-tag", default="full")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    showcase_data.TASKS["formula_count"]["parquet"] = str(args.parquet.resolve())
    samples, all_targets, _ = load_fold(str(args.parquet), "test", "formula_count")
    indices = [FORMULA_IDX[element] for element in ELEMENTS]
    targets = all_targets[:, indices].astype(np.float32)
    full_test_n = len(samples)
    start = max(0, args.start)
    end = full_test_n if args.end is None else min(full_test_n, args.end)
    if start >= end:
        raise ValueError(f"Invalid test slice: [{start}, {end}) of {full_test_n}")
    samples = samples[start:end]
    targets = targets[start:end]
    print(
        f"Loaded MassSpecGym test slice [{start}, {end}): n={len(samples):,}/{full_test_n:,}; "
        f"device={device}",
        flush=True,
    )

    all_metrics = []
    prediction_data: dict[str, np.ndarray] = {
        "test_index": np.arange(start, end, dtype=np.int64)
    }
    prediction_data.update(
        {
            f"true_{element}": targets[:, index]
            for index, element in enumerate(ELEMENTS)
        }
    )
    metadata = {}
    for model_name, probe, backbone in (
        ("UltraMS", args.ultra_probe, args.ultra_backbone),
        ("DreaMS", args.dreams_probe, None),
    ):
        metrics, prediction, model_metadata = evaluate(
            probe,
            backbone,
            samples,
            targets,
            device,
            args.batch_size,
        )
        metrics.insert(1, "model", model_name)
        all_metrics.append(metrics)
        metadata[model_name] = model_metadata
        for index, element in enumerate(ELEMENTS):
            prediction_data[f"pred_{model_name.lower()}_{element}"] = prediction[:, index]

    metric_frame = pd.concat(all_metrics, ignore_index=True)
    stem = f"atom_count_test_{args.output_tag}"
    metric_frame.to_csv(args.output_dir / f"{stem}_metrics.csv", index=False)
    pd.DataFrame(prediction_data).to_csv(
        args.output_dir / f"{stem}_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    config = {
        "task": "element count regression",
        "evaluation_fold": "test",
        "full_test_n": int(full_test_n),
        "slice_start": int(start),
        "slice_end": int(end),
        "slice_n": int(len(samples)),
        "elements": ELEMENTS,
        "checkpoint_policy": "explicit probe checkpoint paths; actual epoch recorded for each model",
        "device": str(device),
        "batch_size": args.batch_size,
        "models": metadata,
        "metrics": metric_frame.to_dict(orient="records"),
    }
    with (args.output_dir / f"{stem}_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    print(metric_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
