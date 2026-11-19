"""Command-line embedding extraction for MGF and mzML files."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .inference import UltraMS
from .spectra import read_spectra


def _batches(records: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    while batch := list(islice(records, size)):
        yield batch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ultrams-embed", description="Extract UltraMS embeddings from MGF or mzML."
    )
    parser.add_argument("input", type=Path, help="input .mgf, .mgf.gz, or .mzML file")
    parser.add_argument("output", type=Path, help="output .npz file containing ids and embeddings")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model", choices=("unsupervised", "mona", "search"), default="unsupervised"
    )
    source.add_argument("--checkpoint", type=Path, help="local UltraMS checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("output path must end in .npz")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    model = (
        UltraMS.from_checkpoint(args.checkpoint, device=device)
        if args.checkpoint is not None
        else UltraMS.from_pretrained(args.model, device=device)
    )
    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for batch in _batches(read_spectra(args.input), args.batch_size):
        ids.extend(str(record["id"]) for record in batch)
        embeddings.append(model.encode_batch(batch, batch_size=args.batch_size))
    if not embeddings:
        parser.error("input file contains no MS/MS spectra")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids, dtype=str),
        embeddings=np.concatenate(embeddings, axis=0),
    )
    print(f"Saved {len(ids)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
