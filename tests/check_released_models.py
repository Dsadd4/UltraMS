"""Verify the public checkpoints against the package before a release."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ultrams import UltraMS, read_spectra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.export_inference_checkpoint import CONFIG_KEYS, STATE_KEYS


MODELS = (
    (
        "unsupervised",
        "dsadd4/UltraMS-Unsupervised",
        1024,
        150,
    ),
    (
        "mona",
        "dsadd4/UltraMS-MoNA-Contrastive",
        1024,
        100,
    ),
    (
        "search",
        "dsadd4/UltraMS-Search",
        512,
        150,
    ),
)


def main() -> None:
    example = Path(__file__).resolve().parents[1] / "examples/data/example_5_spectra.mgf"
    rows = list(read_spectra(example))
    assert len(rows) == 5
    summary = []
    for name, repo, dimension, max_peaks in MODELS:
        checkpoint = hf_hub_download(repo, "model.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert set(payload) <= set(STATE_KEYS) | {"config"}
        assert set(payload["config"]) == set(CONFIG_KEYS)
        for key, state in payload.items():
            if key != "config":
                assert state and all(isinstance(value, torch.Tensor) for value in state.values())
        del payload
        model = UltraMS.from_checkpoint(checkpoint)
        assert model.embedding_dim == dimension
        assert model.input_max_peaks == max_peaks
        batched = model.encode_batch(rows[:2], batch_size=2)
        singles = np.stack(
            [
                model.encode(row["mz"], row["intensity"], precursor_mz=row["precursor_mz"]).embedding
                for row in rows[:2]
            ]
        )
        np.testing.assert_allclose(batched, singles, rtol=1e-4, atol=1e-4)
        assert np.isfinite(batched).all()
        peak_result = model.encode(
            rows[0]["mz"], rows[0]["intensity"], precursor_mz=rows[0]["precursor_mz"],
            return_peaks=True,
        )
        assert peak_result.peak_embeddings.shape == (len(peak_result.peak_mz), 1024)
        assert np.all(np.diff(peak_result.peak_mz) >= 0)
        summary.append({"model": name, "embeddings": list(batched.shape), "inference_checkpoint": True})
        del model
        gc.collect()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
