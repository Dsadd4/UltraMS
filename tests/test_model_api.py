"""Check the public batch API against single-spectrum inference."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from ultrams import UltraMS
from ultrams._architecture import UltraMSBackbone, model_config
from ultrams.cli import main as embed_file


class ModelAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(11)
        config = model_config(
            {
                "d_model": 16,
                "num_heads": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "max_peaks": 8,
                "mz_codebook_dim": 8,
                "int_codebook_dim": 8,
                "coarse_num": 20,
            }
        )
        cls.model = UltraMS(UltraMSBackbone(config), config, 8, "base", None)

    def test_batch_matches_single_spectrum_and_restores_mode(self) -> None:
        spectra = [
            {"mz": [122, 100, 150], "intensity": [4, 10, 2], "precursor_mz": 301.2},
            {"mz": [99, 100, 102, 155, 180], "intensity": [2, 8, 3, 7, 1], "precursor_mz": 315.3},
            {"mz": [80, 120, 140, 190], "intensity": [4, 10, 2, 6], "precursor_mz": 250.1},
        ]
        expected = np.stack(
            [
                self.model.encode(row["mz"], row["intensity"], precursor_mz=row["precursor_mz"]).embedding
                for row in spectra
            ]
        )
        self.model.train()
        actual = self.model.encode_batch(iter(spectra), batch_size=2)
        self.assertTrue(self.model.training)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_empty_batch_has_embedding_width(self) -> None:
        actual = self.model.encode_batch([], batch_size=2)
        self.assertEqual(actual.shape, (0, self.model.embedding_dim))

    def test_peak_embeddings_align_with_selected_sorted_peaks(self) -> None:
        mz = [180, 100, 150, 110, 140, 120, 170, 130, 190, 160]
        intensity = [8, 1, 5, 2, 4, 3, 7, 6, 10, 9]
        result = self.model.encode(mz, intensity, precursor_mz=301.2, return_peaks=True)
        self.assertEqual(result.peak_embeddings.shape, (8, 16))
        np.testing.assert_array_equal(
            result.peak_mz,
            np.asarray([120, 130, 140, 150, 160, 170, 180, 190], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.peak_intensity,
            np.asarray([3, 6, 4, 5, 9, 7, 8, 10], dtype=np.float32) / 10,
        )
        self.assertIsNone(self.model.encode(mz, intensity, precursor_mz=301.2).peak_embeddings)

    def test_real_mgf_to_npz_with_local_checkpoint(self) -> None:
        source = Path(__file__).resolve().parents[1] / "examples/data/example_5_spectra.mgf"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            output = Path(directory) / "embeddings.npz"
            torch.save(
                {"model_state_dict": self.model.model.state_dict(), "config": self.model.config},
                checkpoint,
            )
            embed_file([str(source), str(output), "--checkpoint", str(checkpoint), "--batch-size", "2"])
            with np.load(output) as values:
                self.assertEqual(len(values["ids"]), 5)
                self.assertEqual(len(set(values["ids"].tolist())), 5)
                self.assertEqual(values["embeddings"].shape, (5, 16))
                self.assertTrue(np.isfinite(values["embeddings"]).all())


if __name__ == "__main__":
    unittest.main()
