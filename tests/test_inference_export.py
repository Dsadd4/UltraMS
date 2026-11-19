"""Check inference exports across the three released state-dict layouts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ultrams import UltraMS
from ultrams._architecture import UltraMSBackbone, UltraMSWithHeads, model_config
from tools.export_inference_checkpoint import CONFIG_KEYS, export_checkpoint


class InferenceExportTests(unittest.TestCase):
    def test_library_search_training_export_keeps_projection(self):
        config = model_config({
            "d_model": 16, "num_heads": 4, "num_layers": 1,
            "dim_feedforward": 32, "max_peaks": 8,
            "mz_codebook_dim": 8, "int_codebook_dim": 8, "coarse_num": 20,
        })
        encoder = UltraMSBackbone(config).state_dict()
        projection = torch.nn.Sequential(
            torch.nn.LayerNorm(16), torch.nn.Linear(16, 16),
            torch.nn.GELU(), torch.nn.Linear(16, 4),
        ).state_dict()
        projection = {"net." + key: value for key, value in projection.items()}
        with tempfile.TemporaryDirectory() as directory:
            source, output, reference = [Path(directory) / name for name in ("training.pt", "inference.pt", "reference.pt")]
            torch.save({"model_state_dict": encoder, "proj_state_dict": projection, "config": config}, source)
            torch.save({"encoder_state_dict": encoder, "proj_state_dict": projection, "config": config}, reference)
            export_checkpoint(source, output)
            model, expected = UltraMS.from_checkpoint(output), UltraMS.from_checkpoint(reference)
            self.assertEqual(model.embedding_dim, 4)
            inputs = dict(mz=[100, 110, 140], intensity=[1, 3, 2], precursor_mz=301.2)
            np.testing.assert_array_equal(model.encode(**inputs).embedding, expected.encode(**inputs).embedding)

    def test_training_checkpoint_with_numpy_rng_and_nested_config(self):
        config = model_config({
            "d_model": 16, "num_heads": 4, "num_layers": 1,
            "dim_feedforward": 32, "max_peaks": 8,
            "mz_codebook_dim": 8, "int_codebook_dim": 8, "coarse_num": 20,
        })
        model = UltraMSWithHeads(config)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "training.pt", Path(directory) / "model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_state_dict": model.base.state_dict(),
                "config": {"model": config, "runtime_environment": {"torch": torch.__version__}},
                "rng_state_by_rank": [{"numpy": np.random.get_state()}],
                "optimizer_state_dict": {},
                "resume_state": {"stage": "peak_reconstruction", "epoch": 6},
            }, source)
            export_checkpoint(source, output)
            before, after = UltraMS.from_checkpoint(source), UltraMS.from_checkpoint(output)
            inputs = dict(mz=[100, 110, 140], intensity=[1, 3, 2], precursor_mz=301.2)
            np.testing.assert_array_equal(before.encode(**inputs).embedding, after.encode(**inputs).embedding)
            self.assertEqual(after.config["d_model"], 16)
            self.assertEqual(after.input_max_peaks, 8)
            self.assertIn("rng_state_by_rank", torch.load(source, weights_only=False))
            self.assertNotIn("rng_state_by_rank", torch.load(output, weights_only=True))

    def test_all_model_families_keep_outputs_and_remove_training_metadata(self):
        config = model_config({
            "d_model": 16, "num_heads": 4, "num_layers": 1,
            "dim_feedforward": 32, "max_peaks": 8,
            "mz_codebook_dim": 8, "int_codebook_dim": 8, "coarse_num": 20,
        })
        unified = UltraMSWithHeads(config)
        encoder = UltraMSBackbone(config).state_dict()
        projection = torch.nn.Linear(16, 16).state_dict()
        search = torch.nn.Sequential(
            torch.nn.LayerNorm(16), torch.nn.Linear(16, 16),
            torch.nn.GELU(), torch.nn.Linear(16, 4),
        ).state_dict()
        layouts = {
            "base": {"model_state_dict": unified.state_dict(), "base_state_dict": unified.base.state_dict()},
            "mona": {"encoder_state_dict": encoder, "projection_state_dict": projection},
            "search": {"encoder_state_dict": encoder, "proj_state_dict": {"net." + k: v for k, v in search.items()}},
        }
        with tempfile.TemporaryDirectory() as directory:
            for family, weights in layouts.items():
                with self.subTest(family=family):
                    source, output = Path(directory) / "training.pt", Path(directory) / "inference.pt"
                    saved_config = {**config, "lr": 1e-5, "output_dir": "/training/run"}
                    if family == "mona":
                        saved_config["max_peaks"] = 5
                    torch.save({**weights, "config": saved_config, "epoch": 11,
                                "history": [1, 2], "optimizer_state_dict": {}}, source)
                    export_checkpoint(source, output)
                    clean = torch.load(output, weights_only=True)
                    self.assertEqual(set(clean), set(weights) | {"config"})
                    self.assertEqual(set(clean["config"]), set(CONFIG_KEYS))
                    before, after = UltraMS.from_checkpoint(source), UltraMS.from_checkpoint(output)
                    self.assertEqual(before.input_max_peaks, after.input_max_peaks)
                    for model in (before, after):
                        model.eval()
                    inputs = dict(mz=[100, 110, 120, 130, 140, 150, 160, 170, 180],
                                  intensity=[1, 2, 3, 4, 5, 6, 7, 8, 9], precursor_mz=301.2,
                                  return_peaks=True)
                    a, b = before.encode(**inputs), after.encode(**inputs)
                    for key in vars(a):
                        if getattr(a, key) is None:
                            self.assertIsNone(getattr(b, key))
                        else:
                            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


if __name__ == "__main__":
    unittest.main()
