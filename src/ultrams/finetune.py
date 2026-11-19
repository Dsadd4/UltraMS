"""Small supervised fine-tuning interface for UltraMS embeddings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .inference import UltraMS


class UltraMSPredictor(nn.Module):
    """An UltraMS encoder and a supervised task head."""

    def __init__(
        self, encoder: UltraMS, task: str, classes: list[str | int] | None = None
    ) -> None:
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if task == "classification" and (classes is None or len(classes) < 2):
            raise ValueError("classification needs at least two classes")
        self.encoder = encoder
        self.task = task
        self.classes = classes
        self.head = nn.Linear(encoder.embedding_dim, len(classes) if classes else 1)
        self.history: list[dict[str, float | int]] = []
        self.output_dir: Path | None = None

    def forward(
        self,
        peaks: torch.Tensor,
        attention_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.encoder(peaks, attention_mask, precursor_mz))

    @torch.inference_mode()
    def predict(self, mz: Any, intensity: Any, *, precursor_mz: float) -> float | str | int:
        self.eval()
        embedding = self.encoder.encode_tensor(mz, intensity, precursor_mz=precursor_mz)
        output = self.head(embedding)
        if self.task == "regression":
            return float(output.squeeze())
        assert self.classes is not None
        return self.classes[int(output.argmax(dim=-1).item())]

    def save_pretrained(self, output_dir: str | Path) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        encoder = self.encoder
        config = dict(encoder.config)
        config["max_peaks"] = encoder.input_max_peaks
        if encoder.model_family == "mona":
            payload = {
                "encoder_state_dict": encoder.model.state_dict(),
                "projection_state_dict": encoder.projection.state_dict(),
                "config": config,
            }
        elif encoder.model_family == "atlas":
            payload = {
                "encoder_state_dict": encoder.model.state_dict(),
                "proj_state_dict": {
                    "net." + key: value
                    for key, value in encoder.projection.state_dict().items()
                },
                "config": config,
            }
        else:
            payload = {"model_state_dict": encoder.model.state_dict(), "config": config}
        torch.save(payload, path / "encoder.pt")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "config.json").write_text(
            json.dumps({"task": self.task, "classes": self.classes}, indent=2) + "\n"
        )
        (path / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
        self.output_dir = path
        return path

    @classmethod
    def from_pretrained(
        cls, output_dir: str | Path, *, device: str | torch.device = "cpu"
    ) -> "UltraMSPredictor":
        path = Path(output_dir)
        config = json.loads((path / "config.json").read_text())
        encoder = UltraMS.from_checkpoint(path / "encoder.pt", device=device)
        predictor = cls(encoder, config["task"], config["classes"]).to(device)
        predictor.head.load_state_dict(
            torch.load(path / "head.pt", map_location=device, weights_only=True)
        )
        history_path = path / "history.json"
        if history_path.exists():
            predictor.history = json.loads(history_path.read_text())
        predictor.output_dir = path
        return predictor.eval()


def _prepare_records(
    records: Sequence[Mapping[str, Any]], task: str, classes: list[str | int] | None
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        label = record["label"]
        if task == "regression":
            target = float(label)
            if not np.isfinite(target):
                raise ValueError("regression labels must be finite")
        else:
            assert classes is not None
            if label not in classes:
                raise ValueError("validation label was not present in training data")
            target = classes.index(label)
        prepared.append(
            {
                "mz": record["mz"],
                "intensity": record["intensity"],
                "precursor_mz": record["precursor_mz"],
                "target": target,
            }
        )
    return prepared


def _loss(prediction: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    if task == "regression":
        return nn.functional.mse_loss(prediction.squeeze(-1), target)
    return nn.functional.cross_entropy(prediction, target.long())


def finetune(
    model: UltraMS,
    records: Sequence[Mapping[str, Any]],
    *,
    task: str = "regression",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-5,
    validation_records: Sequence[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    seed: int = 42,
) -> UltraMSPredictor:
    """Train on records with ``mz``, ``intensity``, ``precursor_mz``, and ``label``."""
    if not records:
        raise ValueError("records must contain labelled spectra")
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    classes = None
    if task == "classification":
        labels = [record["label"] for record in records]
        if not all(isinstance(label, (str, int)) and not isinstance(label, bool) for label in labels):
            raise ValueError("classification labels must be strings or integers")
        classes = list(dict.fromkeys(labels))
    train_records = _prepare_records(records, task, classes)
    valid_records = (
        _prepare_records(validation_records, task, classes)
        if validation_records is not None
        else None
    )
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        predictor = UltraMSPredictor(model, task, classes).to(device)
    loader = DataLoader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=model.batch_converter(),
        generator=torch.Generator().manual_seed(seed),
    )
    valid_loader = (
        DataLoader(valid_records, batch_size=batch_size, collate_fn=model.batch_converter())
        if valid_records
        else None
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        predictor.train()
        total_loss = 0.0
        for batch in loader:
            peaks = batch["peaks"].to(device)
            attention = batch["attention_mask"].to(device)
            precursor = batch["precursor_mz"].to(device)
            target = batch["target"].to(device)
            loss = _loss(predictor(peaks, attention, precursor), target, task)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_records),
        }
        if valid_loader is not None:
            predictor.eval()
            valid_loss = 0.0
            with torch.inference_mode():
                for batch in valid_loader:
                    prediction = predictor(
                        batch["peaks"].to(device),
                        batch["attention_mask"].to(device),
                        batch["precursor_mz"].to(device),
                    )
                    target = batch["target"].to(device)
                    valid_loss += float(_loss(prediction, target, task)) * len(target)
            row["validation_loss"] = valid_loss / len(valid_records)
        predictor.history.append(row)
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = Path("ultrams_finetune") / timestamp
    run_dir = predictor.save_pretrained(output_dir)
    (run_dir / "training.json").write_text(
        json.dumps(
            {
                "task": task,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "seed": seed,
                "training_spectra": len(records),
                "validation_spectra": len(validation_records) if validation_records else 0,
            },
            indent=2,
        )
        + "\n"
    )
    return predictor.eval()
