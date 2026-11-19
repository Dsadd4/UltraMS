"""Reload selected ion-mode MLP heads and evaluate the saved full val/test splits."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from train_test_ion_mode import make_mlp, _mlp_forward_logits, seed_probe


def evaluate_head(checkpoint, embeddings, splits, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    seed_probe(config["seed"])
    model = make_mlp(config["input_dim"], config["hidden1"],
                     config["hidden2"], config["dropout"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    threshold = config["threshold_from_val"]
    metrics, predictions = {}, {}
    for name in ("val", "test"):
        indices = splits[f"{name}_idx"]
        labels = splits["labels"][indices]
        logits = _mlp_forward_logits(model, embeddings[indices], device,
                                     bs=config["forward_batch_size"])
        predicted = (logits > threshold).astype(np.int64)
        predictions.update({f"{name}_indices": indices, f"{name}_labels": labels,
                            f"{name}_logits": logits, f"{name}_predictions": predicted})
        metrics[f"mlp_{name}"] = {
            "accuracy": float(accuracy_score(labels, predicted)),
            "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "f1_neg": float(f1_score(labels, predicted, pos_label=0, zero_division=0)),
            "f1_pos": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
            "n": len(indices),
        }
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["dreams", "ultrams"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with np.load(args.artifact_dir / "splits.npz") as splits:
        for name in args.models:
            directory = args.artifact_dir / name
            embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
            metrics, predictions = evaluate_head(directory / "mlp.pt", embeddings,
                                                  splits, torch.device(args.device))
            (directory / "mlp_reloaded_metrics.json").write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(directory / "mlp_reloaded_predictions.npz", **predictions)
            print(name, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
