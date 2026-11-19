"""Export UltraMS weights and inference configuration from a training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultrams._architecture import model_config


STATE_KEYS = (
    "model_state_dict", "base_state_dict", "encoder_state_dict",
    "projection_state_dict", "proj_state_dict",
)
CONFIG_KEYS = (
    "d_model", "num_heads", "num_layers", "dim_feedforward", "dropout",
    "max_peaks", "mz_range", "mz_bin_size", "int_bin_size",
    "mz_codebook_dim", "int_codebook_dim", "coarse_num",
)


def inference_payload(payload: dict) -> dict:
    """Keep the existing state-dict layout and the input settings used by UltraMS."""
    weights = {key: payload[key] for key in STATE_KEYS if key in payload}
    # The library-search trainer saves its encoder under model_state_dict.
    # Use the public encoder/projection layout so the projection is loaded too.
    if "proj_state_dict" in weights and "encoder_state_dict" not in weights and "model_state_dict" in weights:
        weights["encoder_state_dict"] = weights.pop("model_state_dict")
    state = weights.get("encoder_state_dict", weights.get("model_state_dict", weights.get("base_state_dict")))
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no UltraMS model weights")
    config = model_config(payload.get("config"))
    position_key = "base.pos_emb.weight" if "base.pos_emb.weight" in state else "pos_emb.weight"
    max_peaks = int(state[position_key].shape[0]) - 2
    if "projection_state_dict" in weights:
        # MoNA uses 100 input peaks with a backbone that has 150 peak positions.
        max_peaks = int(payload.get("config", {}).get("max_peaks", 100))
    config["max_peaks"] = max_peaks
    return {**weights, "config": {key: config[key] for key in CONFIG_KEYS}}


def export_checkpoint(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("use a separate output file to preserve the training checkpoint")
    # Training checkpoints include NumPy RNG state and runtime version objects.
    # Read the locally produced training file, then retain only tensors/config.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    clean = inference_payload(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
