"""Read an MGF file and save one UltraMS embedding per spectrum."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from ultrams import UltraMS, read_spectra


EXAMPLE_MGF = Path(__file__).parent / "data" / "example_5_spectra.mgf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLE_MGF, help="MGF input file")
    parser.add_argument("--model", choices=("unsupervised", "mona", "search"), default="unsupervised")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Output .npz path")
    args = parser.parse_args()

    spectra = list(read_spectra(args.input))
    model = UltraMS.from_pretrained(args.model, device=args.device)
    embeddings = model.encode_batch(spectra, batch_size=args.batch_size)
    output = args.output or Path(tempfile.mkdtemp(prefix="ultrams-embeddings-")) / "embeddings.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray([spectrum["id"] for spectrum in spectra]), embeddings=embeddings)
    print(f"Encoded {len(spectra)} spectra into {embeddings.shape} embeddings")
    print(output)


if __name__ == "__main__":
    main()
