"""Extract only the two supervised CLS-derived embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from extract_cross_ce_embeddings import (
    extract_dreams_supervised,
    extract_ultra_supervised,
    load_samples,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ultra", type=Path, required=True)
    parser.add_argument("--dreams", type=Path, required=True)
    parser.add_argument("--input-peaks", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root)); sys.path.insert(0, str(args.root / "train"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "representation": {"UltraMS": "projection(CLS)", "DreaMS": "head(precursor token)"},
        "excluded": ["peak pooling", "peak matching", "fusion"],
        "checkpoints": {
            "UltraMS": {"path": str(args.ultra), "sha256": sha256(args.ultra)},
            "DreaMS": {"path": str(args.dreams), "sha256": sha256(args.dreams)},
        }, "splits": {},
    }
    for split in ("val", "test"):
        metadata, samples = load_samples(args.benchmark_dir, split, args.input_peaks)
        ultra, _ = extract_ultra_supervised(args.ultra, samples, device, args.batch_size)
        dreams, _ = extract_dreams_supervised(args.root, args.dreams, samples, device, args.batch_size)
        for model, values in (("ultrams_supervised", ultra), ("dreams_supervised", dreams)):
            if values.shape[0] != len(metadata) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid {model} {split}: {values.shape}")
            path = args.out_dir / f"{split}_{model}.npy"
            np.save(path, values.astype(np.float32))
        manifest["splits"][split] = {"n_spectra": len(metadata), "dimension": int(ultra.shape[1])}
    (args.out_dir / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
