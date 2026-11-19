"""Exercise native MPR checkpoint save/reload and optimizer/RNG continuity."""

import tempfile
import unittest
from pathlib import Path

import torch

from ultrams_training import train_pretraining as training
from ultrams_training import training_runtime as runtime


class CheckpointRoundtripTests(unittest.TestCase):
    def test_initialization_does_not_silently_select_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "completed_mpr.pt"
            self.assertIsNone(runtime.find_resume_checkpoint(root, None, parent))
            latest = root / "checkpoints" / "latest.pt"
            latest.parent.mkdir()
            latest.touch()
            self.assertEqual(runtime.find_resume_checkpoint(root, None), latest)
            with self.assertRaisesRegex(RuntimeError, "new --output-dir"):
                runtime.find_resume_checkpoint(root, None, parent)
            with self.assertRaisesRegex(RuntimeError, "new --output-dir"):
                runtime.find_resume_checkpoint(root, latest, parent)

    def test_next_mpr_step_matches_after_checkpoint_resume(self):
        root = Path(training.__file__).parent / "model"
        defaults, model_class, _, _ = runtime.import_project(root)
        config = dict(defaults, d_model=16, num_heads=4, num_layers=1,
                      dim_feedforward=32, max_peaks=8, mz_codebook_dim=8,
                      int_codebook_dim=8, coarse_num=20)
        runtime.set_seed(27, 0)

        def components():
            model = training.initialize_unified_model(model_class, config)
            model.configure_trainable("peak_reconstruction")
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            return model, optimizer, scheduler, scaler

        original = torch.tensor([[[100.1, .5], [110.2, 1.], [140.3, .7]],
                                 [[99.1, .8], [120.2, .4], [150.3, 1.]]])
        flags = torch.tensor([[False, True, False], [True, False, False]])
        masked = original.clone()
        masked[flags] = 0
        batch = dict(orig_spectra=original, masked_spectra=masked, mask_flags=flags,
                     attn_mask=torch.ones(2, 3, dtype=torch.long),
                     precursor_mz=torch.tensor([301.2, 315.3]))

        def step(model, optimizer, scheduler):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch, torch.device("cpu"), "peak_reconstruction")["loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.detach().clone()

        model, optimizer, scheduler, scaler = components()
        step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints" / "step1.pt"
            state = training.UnifiedResumeState("peak_reconstruction", 1, 1, 1, 1)
            runtime.save_checkpoint(path, model, optimizer, scaler, scheduler, state,
                                    {"model": config}, rank=0, world_size=1)
            self.assertTrue((path.parent / "latest.pt").samefile(path))
            expected_loss = step(model, optimizer, scheduler)
            checkpoint = runtime.load_training_checkpoint(path)
            resumed, resumed_optimizer, resumed_scheduler, resumed_scaler = components()
            resumed.load_state_dict(checkpoint["model_state_dict"], strict=True)
            resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            resumed_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            resumed_scaler.load_state_dict(checkpoint["scaler_state_dict"])
            runtime.restore_rng_state(checkpoint["rng_state_by_rank"][0])
            actual_loss = step(resumed, resumed_optimizer, resumed_scheduler)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], tensor, rtol=0, atol=0)
            expected_state = optimizer.state_dict()
            actual_state = resumed_optimizer.state_dict()
            self.assertEqual(actual_state["param_groups"], expected_state["param_groups"])
            for key, values in expected_state["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(actual_state["state"][key][name], value, rtol=0, atol=0)
            self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())

    def test_inference_weights_are_not_mistaken_for_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state_dict": {"weight": torch.ones(2)}, "config": {}}, path)
            with self.assertRaisesRegex(ValueError, "original training run"):
                runtime.load_training_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
