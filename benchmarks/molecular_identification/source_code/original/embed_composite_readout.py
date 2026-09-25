#!/usr/bin/env python3
"""Embed the frozen composite-polarity manifest with one trained readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baseline_common import atomic_write_json, sha256_file
from embed_search_fig3h_readout import embed_all, inputs_for
from train_chemberta_readout import make_model


METHODS = ("linear", "deepsets", "fourier_projection", "ultrams_codebook")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    trained = Path(args.model_dir) / args.model / args.mode / f"seed_{args.seed}"
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = trained / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=768).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    inputs = inputs_for(Path(args.cache_dir), args.mode, args.model)
    vector_path = output / "all_vectors_float32.npy"
    audit_path = output / "DONE.json"
    if vector_path.exists() and audit_path.exists() and not args.force:
        print(audit_path.read_text())
        return
    embedding = embed_all(model, inputs, vector_path, device, args.batch_size)
    atomic_write_json(audit_path, {
        "status": "complete",
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "embedding": embedding,
    })
    print(json.dumps(json.loads(audit_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
