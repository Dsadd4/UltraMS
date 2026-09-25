#!/usr/bin/env python3
"""Encode the molecules used by additional-adduct readout training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.output.exists():
        existing = torch.load(args.output, map_location="cpu", weights_only=True)
        labels = [
            smiles
            for mode in ("pos", "neg")
            for smiles in json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text())
        ]
        if all(smiles in existing for smiles in labels):
            print(f"ready {args.output}: {len(existing):,} molecules", flush=True)
            return

    source = Path(__file__).resolve().parent / "original"
    sys.path.insert(0, str(source))
    os.environ.setdefault("LIGHT_ULTRA_ROOT", str(args.output.parents[4]))
    from evaluate_molecule_identification import MolEncoder

    labels = list(dict.fromkeys(
        smiles
        for mode in ("pos", "neg")
        for smiles in json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text())
    ))
    device = torch.device(args.device)
    model = MolEncoder(str(args.model_dir)).to(device).eval()
    vectors = {}
    for start in range(0, len(labels), args.batch_size):
        batch = labels[start:start + args.batch_size]
        encoded = model.encode_batch(batch, device).cpu()
        vectors.update(zip(batch, encoded))
        if start % (args.batch_size * 50) == 0:
            print(f"molecules={start:,}/{len(labels):,}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(vectors, temporary)
    temporary.replace(args.output)
    print(f"saved {len(vectors):,} molecules to {args.output}", flush=True)


if __name__ == "__main__":
    main()
