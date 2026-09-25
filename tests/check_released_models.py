"""Verify the public checkpoints against the package before a release."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from ultrams import UltraMS, read_spectra


MODELS = (
    (
        "unsupervised",
        "dsadd4/UltraMS-Unsupervised",
        "6a4c6660999848c409303119f6caa54fbae9444d0b75c7fcd8bafd303cde9830",
        1024,
        150,
    ),
    (
        "mona",
        "dsadd4/UltraMS-MoNA-Contrastive",
        "3b8ae5bd85ff8f6991f8a78b6d70531aab74f32914878730e7296aaddb79afb0",
        1024,
        100,
    ),
    (
        "search",
        "dsadd4/UltraMS-Search",
        "447e41e99e5f6ee4155f5dd9938bd7b5fe3f09239b6d5eb0f4973b44c963e383",
        512,
        150,
    ),
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    example = Path(__file__).resolve().parents[1] / "examples/data/example_5_spectra.mgf"
    rows = list(read_spectra(example))
    assert len(rows) == 5
    summary = []
    for name, repo, expected_hash, dimension, max_peaks in MODELS:
        checkpoint = hf_hub_download(repo, "model.pt")
        actual_hash = _sha256(checkpoint)
        assert actual_hash == expected_hash, f"{name} checkpoint hash changed"
        model = UltraMS.from_pretrained(name)
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
        summary.append({"model": name, "embeddings": list(batched.shape), "sha256": actual_hash})
        del model
        gc.collect()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
