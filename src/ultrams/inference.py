"""Load an UltraMS checkpoint and encode one MS/MS spectrum."""

from __future__ import annotations

from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ._architecture import UltraMSBackbone, UltraMSWithHeads, model_config


PRETRAINED_MODELS = {
    "unsupervised": "dsadd4/UltraMS-Unsupervised",
    "base": "dsadd4/UltraMS-Unsupervised",
    "mona": "dsadd4/UltraMS-MoNA-Contrastive",
    "search": "dsadd4/UltraMS-Search",
}


@dataclass(frozen=True)
class SpectrumEmbedding:
    cls: np.ndarray
    peak_intensity_weighted: np.ndarray
    fusion: np.ndarray
    projection: np.ndarray | None = None
    rt_seconds: float | None = None
    peak_embeddings: np.ndarray | None = None
    peak_mz: np.ndarray | None = None
    peak_intensity: np.ndarray | None = None

    @property
    def embedding(self) -> np.ndarray:
        """The model-specific representation used for similarity search."""
        return self.projection if self.projection is not None else self.cls


def _prepare_spectrum(
    mz: Any, intensity: Any, max_peaks: int
) -> tuple[np.ndarray, np.ndarray]:
    mz_array = np.asarray(mz, dtype=np.float32)
    intensity_array = np.asarray(intensity, dtype=np.float32)
    if mz_array.ndim != 1 or intensity_array.ndim != 1 or len(mz_array) != len(intensity_array):
        raise ValueError("mz and intensity must be one-dimensional arrays of equal length")
    if not np.isfinite(mz_array).all() or not np.isfinite(intensity_array).all():
        raise ValueError("mz and intensity must contain finite values")
    valid = mz_array > 0
    mz_array = mz_array[valid]
    intensity_array = intensity_array[valid]
    if len(mz_array) < 3:
        raise ValueError("a spectrum needs at least three peaks with positive m/z")
    maximum = intensity_array.max()
    if maximum > 0:
        intensity_array = intensity_array / maximum
    intensity_array = np.clip(intensity_array, 0, 1)
    if len(mz_array) > max_peaks:
        selected = np.argsort(intensity_array)[-max_peaks:]
        selected = np.sort(selected)
        mz_array = mz_array[selected]
        intensity_array = intensity_array[selected]
    order = np.argsort(mz_array)
    peaks = np.stack((mz_array[order], intensity_array[order]), axis=-1).astype(np.float32)
    mask = np.ones(len(peaks), dtype=np.int64)
    return peaks, mask


@dataclass(frozen=True)
class SpectrumCollator:
    """Convert variable-length spectra into a padded PyTorch batch."""

    max_peaks: int

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        precursor_values = np.asarray([row["precursor_mz"] for row in examples], dtype=np.float32)
        if not np.isfinite(precursor_values).all() or np.any(precursor_values <= 0):
            raise ValueError("precursor_mz must be a positive finite number")
        prepared = [
            _prepare_spectrum(row["mz"], row["intensity"], self.max_peaks)[0]
            for row in examples
        ]
        length = max(len(spectrum) for spectrum in prepared)
        peaks = torch.zeros((len(prepared), length, 2), dtype=torch.float32)
        attention_mask = torch.zeros((len(prepared), length), dtype=torch.long)
        for index, spectrum in enumerate(prepared):
            count = len(spectrum)
            peaks[index, :count] = torch.from_numpy(spectrum)
            attention_mask[index, :count] = 1
        batch = {
            "peaks": peaks,
            "attention_mask": attention_mask,
            "precursor_mz": torch.tensor(
                precursor_values, dtype=torch.float32
            ),
        }
        if all("target" in row for row in examples):
            batch["target"] = torch.tensor(
                [float(row["target"]) for row in examples], dtype=torch.float32
            )
        return batch


def _checkpoint_state(
    payload: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], int, str, torch.nn.Module | None]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a mapping")
    model_family = "base"
    projection = None
    if "encoder_state_dict" in payload:
        state = payload["encoder_state_dict"]
        if "proj_state_dict" in payload:
            model_family = "atlas"
            projection = _atlas_projection(payload["proj_state_dict"])
        elif "projection_state_dict" in payload:
            model_family = "mona"
            projection = _mona_projection(payload["projection_state_dict"])
        else:
            raise ValueError("encoder checkpoint has no projection state")
    else:
        state = payload.get("model_state_dict")
        if state is None:
            state = payload.get("base_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no supported model state")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("checkpoint state must contain named tensors")
    saved_config = payload.get("config")
    config = model_config(saved_config)
    first_key = "base.pos_emb.weight" if any(key.startswith("base.") for key in state) else "pos_emb.weight"
    if first_key in state:
        config["max_peaks"] = int(state[first_key].shape[0]) - 2
    input_max_peaks = config["max_peaks"]
    if model_family == "mona" and isinstance(saved_config, dict):
        input_max_peaks = int(saved_config.get("max_peaks", 100))
    return state, config, input_max_peaks, model_family, projection


def _mona_projection(state: Any) -> torch.nn.Module:
    if not isinstance(state, dict):
        raise ValueError("MoNA checkpoint has no projection weights")
    if set(state) == {"weight", "bias"}:
        size = int(state["weight"].shape[1])
        projection: torch.nn.Module = torch.nn.Linear(size, size)
    elif "1.weight" in state and "4.weight" in state:
        size = int(state["1.weight"].shape[1])
        hidden = int(state["1.weight"].shape[0])
        projection = torch.nn.Sequential(
            torch.nn.LayerNorm(size),
            torch.nn.Linear(size, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden, size),
        )
    else:
        raise ValueError("unrecognized MoNA projection layout")
    projection.load_state_dict(state, strict=True)
    return projection.eval()


def _atlas_projection(state: Any) -> torch.nn.Module:
    if not isinstance(state, dict) or "net.3.weight" not in state:
        raise ValueError("unrecognized UltraAtlas projection layout")
    hidden = int(state["net.1.weight"].shape[0])
    output = int(state["net.3.weight"].shape[0])
    projection = torch.nn.Sequential(
        torch.nn.LayerNorm(hidden),
        torch.nn.Linear(hidden, hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, output),
    )
    named_state = {key.removeprefix("net."): value for key, value in state.items()}
    projection.load_state_dict(named_state, strict=True)
    return projection.eval()


class UltraMS(torch.nn.Module):
    """Pretrained MS/MS encoder that works as a standard PyTorch module."""

    def __init__(
        self,
        model: UltraMSBackbone | UltraMSWithHeads,
        config: dict[str, Any],
        input_max_peaks: int,
        model_family: str,
        projection: torch.nn.Module | None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.input_max_peaks = input_max_peaks
        self.model_family = model_family
        self.projection = projection
        self.eval()

    @property
    def embedding_dim(self) -> int:
        if self.projection is not None:
            for layer in reversed(list(self.projection.modules())):
                if isinstance(layer, torch.nn.Linear):
                    return layer.out_features
        return int(self.config["d_model"])

    def forward(
        self,
        peaks: torch.Tensor,
        attention_mask: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> torch.Tensor:
        """Return a ``[batch, embedding_dim]`` embedding with gradients."""
        if peaks.ndim != 3 or peaks.shape[-1] != 2:
            raise ValueError("peaks must have shape [batch, peaks, 2]")
        if attention_mask.shape != peaks.shape[:2] or precursor_mz.shape != peaks.shape[:1]:
            raise ValueError("attention_mask or precursor_mz has the wrong shape")
        if peaks.shape[1] > self.input_max_peaks:
            raise ValueError("batch contains more peaks than this model supports")
        _, cls = self.model.encode(
            peaks.float(), attention_mask.long(), precursor_mz.float()
        )
        embedding = self.projection(cls) if self.projection is not None else cls
        if self.model_family == "mona":
            return embedding.float()
        return F.normalize(embedding.float(), dim=-1)

    def batch_converter(self) -> SpectrumCollator:
        """Return a lightweight ``DataLoader(collate_fn=...)`` converter."""
        return SpectrumCollator(self.input_max_peaks)

    def collate(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """Convert a batch of spectra for PyTorch."""
        return self.batch_converter()(examples)

    def encode_tensor(
        self, mz: Any, intensity: Any, *, precursor_mz: float
    ) -> torch.Tensor:
        """Differentiable model embedding of one spectrum, shape ``(1, embedding_dim)``."""
        peaks, attention = _prepare_spectrum(mz, intensity, self.input_max_peaks)
        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            raise ValueError("precursor_mz must be a positive finite number")
        device = next(self.parameters()).device
        return self.forward(
            torch.as_tensor(peaks, device=device).unsqueeze(0),
            torch.as_tensor(attention, device=device).unsqueeze(0),
            torch.tensor([precursor_mz], dtype=torch.float32, device=device),
        )

    def encode_batch(
        self, spectra: Iterable[Mapping[str, Any]], *, batch_size: int = 32
    ) -> np.ndarray:
        """Encode spectra in input order as a ``[N, embedding_dim]`` NumPy array."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        iterator = iter(spectra)
        converter = self.batch_converter()
        device = next(self.parameters()).device
        outputs: list[np.ndarray] = []
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                while rows := list(islice(iterator, batch_size)):
                    batch = converter(rows)
                    embeddings = self.forward(
                        batch["peaks"].to(device),
                        batch["attention_mask"].to(device),
                        batch["precursor_mz"].to(device),
                    )
                    outputs.append(embeddings.cpu().numpy())
        finally:
            self.train(was_training)
        if not outputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, device: str | torch.device = "cpu"
    ) -> "UltraMS":
        path = Path(checkpoint).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state, config, input_max_peaks, model_family, projection = _checkpoint_state(payload)
        unified = any(key.startswith("base.") for key in state)
        model = UltraMSWithHeads(config) if unified else UltraMSBackbone(config)
        model.load_state_dict(state, strict=True)
        return cls(model, config, input_max_peaks, model_family, projection).to(device)

    @classmethod
    def from_pretrained(
        cls, name: str = "unsupervised", *, device: str | torch.device = "cpu"
    ) -> "UltraMS":
        """Load ``unsupervised``, ``mona``, or ``search`` from Hugging Face."""
        try:
            repo_id = PRETRAINED_MODELS[name.lower()]
        except KeyError as exc:
            raise ValueError("name must be 'unsupervised', 'mona', or 'search'") from exc
        return cls.from_hub(repo_id, device=device)

    def finetune(self, records: Sequence[Mapping[str, Any]], **kwargs: Any):
        """Fine-tune on labelled spectra and return a task predictor."""
        from .finetune import finetune

        return finetune(self, records, **kwargs)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        filename: str = "model.pt",
        revision: str | None = None,
        device: str | torch.device = "cpu",
    ) -> "UltraMS":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Hugging Face downloads require huggingface-hub") from exc
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return cls.from_checkpoint(path, device=device)

    @torch.inference_mode()
    def encode(
        self, mz: Any, intensity: Any, *, precursor_mz: float, return_peaks: bool = False
    ) -> SpectrumEmbedding:
        """Encode one MS/MS spectrum; optionally return aligned peak embeddings."""
        was_training = self.training
        self.eval()
        try:
            return self._encode_one(mz, intensity, precursor_mz, return_peaks)
        finally:
            self.train(was_training)

    def _encode_one(
        self, mz: Any, intensity: Any, precursor_mz: float, return_peaks: bool
    ) -> SpectrumEmbedding:
        peaks, attention = _prepare_spectrum(mz, intensity, self.input_max_peaks)
        if not np.isfinite(precursor_mz) or precursor_mz <= 0:
            raise ValueError("precursor_mz must be a positive finite number")
        device = next(self.model.parameters()).device
        spectra = torch.as_tensor(peaks, device=device).unsqueeze(0)
        mask = torch.as_tensor(attention, device=device).unsqueeze(0)
        precursor = torch.tensor([precursor_mz], dtype=torch.float32, device=device)
        hidden, raw_cls = self.model.encode(spectra, mask, precursor)
        peak_hidden = hidden[:, 2 : 2 + spectra.shape[1]]
        weights = spectra[:, :, 1] * mask.float()
        weighted = (peak_hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(
            1, keepdim=True
        ).clamp_min(1e-8)
        cls = F.normalize(raw_cls.float(), dim=-1)
        weighted = F.normalize(weighted.float(), dim=-1)
        projected = None
        if self.projection is not None:
            projected = self.projection(raw_cls)
            if self.model_family == "atlas":
                projected = F.normalize(projected, dim=-1)
        fusion_cls = F.normalize(projected.float(), dim=-1) if self.model_family == "mona" else cls
        fusion = F.normalize(torch.cat((fusion_cls, weighted), dim=-1), dim=-1)
        rt_seconds = None
        if isinstance(self.model, UltraMSWithHeads):
            rt_seconds = float(self.model.rt_head(hidden[:, 0])[0] * 600.0)
        return SpectrumEmbedding(
            cls=cls[0].cpu().numpy(),
            peak_intensity_weighted=weighted[0].cpu().numpy(),
            fusion=fusion[0].cpu().numpy(),
            projection=projected[0].cpu().numpy() if projected is not None else None,
            rt_seconds=rt_seconds,
            peak_embeddings=peak_hidden[0].float().cpu().numpy() if return_peaks else None,
            peak_mz=peaks[:, 0].copy() if return_peaks else None,
            peak_intensity=peaks[:, 1].copy() if return_peaks else None,
        )
