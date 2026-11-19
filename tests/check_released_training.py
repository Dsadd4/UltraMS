"""Exercise fine-tuning and save/reload with all three public checkpoints."""
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import torch
from ultrams import UltraMS, UltraMSPredictor, read_spectra


def main():
    torch.set_num_threads(4)
    rows = list(read_spectra(Path(__file__).resolve().parents[1] / "examples/data/example_5_spectra.mgf"))[:2]
    # An engineering test target from the measured precursor masses.
    targets = torch.tensor([row["precursor_mz"] / 1000 for row in rows])
    report = []
    for name in ("unsupervised", "mona", "search"):
        torch.manual_seed(42)
        encoder = UltraMS.from_pretrained(name)
        predictor = UltraMSPredictor(encoder, task="regression").train()
        batch = encoder.collate(rows)
        inputs = (batch["peaks"], batch["attention_mask"], batch["precursor_mz"])
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-5)
        tracked = next(parameter for key, parameter in encoder.named_parameters() if key.endswith("cls_emb"))
        before = tracked.detach().clone()
        loss = torch.nn.functional.mse_loss(predictor(*inputs).squeeze(-1), targets)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {key: parameter.grad for key, parameter in encoder.named_parameters() if parameter.grad is not None}
        assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients.values())
        assert any(bool(value.abs().max() > 0) for value in gradients.values())
        optimizer.step()
        assert not torch.equal(tracked.detach(), before)
        predictor.eval()
        with torch.no_grad():
            expected = predictor(*inputs).detach().clone()
        gradient_count = len(gradients)
        del gradients, optimizer
        predictor.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory(prefix="ultrams-checkpoint-training-") as directory:
            predictor.save_pretrained(directory)
            del predictor, encoder, tracked, loss
            gc.collect()
            restored = UltraMSPredictor.from_pretrained(directory)
            with torch.no_grad():
                actual = restored(*inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            restored.train()
            next_loss = torch.nn.functional.mse_loss(restored(*inputs).squeeze(-1), targets)
            next_loss.backward()
            assert any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in restored.encoder.parameters())
            del restored, next_loss
        report.append({"model": name, "real_spectra": len(rows), "finite_encoder_gradient_tensors": gradient_count,
                       "optimizer_updated_encoder": True, "save_reload_exact": True,
                       "backward_after_reload": True})
        print(json.dumps(report[-1]), flush=True)
        gc.collect()
    print("All released checkpoint training checks passed.", flush=True)


if __name__ == "__main__":
    main()
