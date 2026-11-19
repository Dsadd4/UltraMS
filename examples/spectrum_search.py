"""Rank example MS/MS spectra with the UltraMS Search representation."""

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from ultrams import UltraMS, read_spectra


EXAMPLE_MGF = Path(__file__).parent / "data" / "example_5_spectra.mgf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLE_MGF, help="MGF input file")
    parser.add_argument("--query-index", type=int, default=0, help="Zero-based query spectrum index")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    spectra = list(read_spectra(args.input))
    if not 0 <= args.query_index < len(spectra):
        raise ValueError("query-index is outside the input spectra")
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    embeddings = UltraMS.from_pretrained("search", device=args.device).encode_batch(spectra)
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    scores = unit @ unit[args.query_index]
    ranked = [i for i in np.argsort(-scores) if i != args.query_index][: args.top_k]

    output = args.output or Path(tempfile.mkdtemp(prefix="ultrams-search-")) / "matches.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_id", "rank", "match_id", "cosine_similarity"))
        for rank, index in enumerate(ranked, start=1):
            writer.writerow((spectra[args.query_index]["id"], rank, spectra[index]["id"], float(scores[index])))
    print(f"Query: {spectra[args.query_index]['id']}; ranked {len(ranked)} spectra")
    print(output)


if __name__ == "__main__":
    main()
