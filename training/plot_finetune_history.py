"""Plot a fine-tuning run from its saved history.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    history = json.loads((args.run_dir / "history.json").read_text())
    if not history:
        raise ValueError("history.json has no epochs")
    epochs = [row["epoch"] for row in history]
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
    figure, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.plot(epochs, [row["train_loss"] for row in history], color="#3965BB", lw=1.0, marker="o", ms=2.5, label="Training")
    if all("validation_loss" in row for row in history):
        ax.plot(epochs, [row["validation_loss"] for row in history], color="#E9A0AA", lw=1.0, marker="o", ms=2.5, label="Validation")
        ax.legend(frameon=False)
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=0.4)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(args.run_dir / f"loss.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
