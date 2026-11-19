"""Fine-tune UltraMS with a native PyTorch loop on example spectra."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ultrams import UltraMS, UltraMSPredictor, read_spectra


EXAMPLE_MGF = Path(__file__).parent / "data" / "example_5_spectra.mgf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLE_MGF, help="MGF input file")
    parser.add_argument("--output-dir", type=Path, help="Directory for training history")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    spectra = list(read_spectra(args.input))
    if len(spectra) < 2:
        raise ValueError("this example needs at least two spectra")
    # Demonstration targets, not measurements or scientific labels.
    dataset = [
        {**spectrum, "target": index / (len(spectra) - 1)}
        for index, spectrum in enumerate(spectra)
    ]

    model = UltraMS.from_pretrained("unsupervised", device=args.device).train()
    head = torch.nn.Linear(model.embedding_dim, 1).to(args.device)
    loader = DataLoader(dataset, batch_size=2, collate_fn=model.batch_converter())
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-5)
    history = []

    for batch in loader:
        batch = {name: value.to(args.device) for name, value in batch.items()}
        prediction = head(model(batch["peaks"], batch["attention_mask"], batch["precursor_mz"]))
        loss = torch.nn.functional.mse_loss(prediction.squeeze(-1), batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="ultrams-finetune-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = UltraMSPredictor(model, task="regression")
    predictor.head = head
    predictor.history = [{"epoch": 1, "train_loss": sum(history) / len(history)}]
    predictor.save_pretrained(output_dir)
    record = {
        "model": "unsupervised",
        "target": "demonstration labels from input order, scaled to [0, 1]",
        "batch_size": 2,
        "learning_rate": 1e-5,
        "loss_per_batch": history,
    }
    (output_dir / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Trained on {len(spectra)} spectra with demonstration targets; final loss: {history[-1]:.4f}")
    print(output_dir / "encoder.pt")
    print(output_dir / "head.pt")
    print(output_dir / "training.json")


if __name__ == "__main__":
    main()
