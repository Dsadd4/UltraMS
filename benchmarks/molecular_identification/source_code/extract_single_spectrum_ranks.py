"""Extract Figure 3a test ranks from the completed six-method JSONL runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


SOURCES = {
    "UltraMS": "train/output/comparison/10_per_spectrum_msnlib_mass_rt_only_d11_test.jsonl",
    "DreaMS": "train/output/comparison/10_per_spectrum_msnlib_mass_dreams_test.jsonl",
    "Linear": "train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/linear/seed_0/evaluation/per_spectrum_msnlib_mass_linear_chemberta_test.jsonl",
    "DeepSets": "train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/deepsets/seed_0/evaluation/per_spectrum_msnlib_mass_deepsets_chemberta_test.jsonl",
    "Fourier projection": "train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/fourier_projection/seed_0/evaluation/per_spectrum_msnlib_mass_fourier_projection_chemberta_test.jsonl",
    "UltraMS codebook": "train/output/comparison/fig3_simple_baselines_20260821/chemberta_readout_v1/formal_runs/ultrams_codebook/seed_0/evaluation/per_spectrum_msnlib_mass_ultrams_codebook_chemberta_test.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-directory", required=True, type=Path,
                        help="Directory containing the completed train/output/comparison runs")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference_ids = None
    with gzip.open(args.output, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "sample_idx", "rank"))
        for method, relative_path in SOURCES.items():
            seen = set()
            with (args.experiment_directory / relative_path).open() as source:
                for line in source:
                    record = json.loads(line)
                    sample_idx = int(record["sample_idx"])
                    rank = int(record["rank"])
                    if sample_idx in seen or rank < 1:
                        raise ValueError(f"{method}: repeated sample or invalid rank")
                    seen.add(sample_idx)
                    writer.writerow((method, sample_idx, rank))
            if len(seen) != 57_437:
                raise ValueError(f"{method}: expected 57,437 test spectra, found {len(seen)}")
            if reference_ids is None:
                reference_ids = seen
            elif seen != reference_ids:
                raise ValueError(f"{method}: test spectra differ from first method")
            print(f"{method}: {len(seen)} ranks")


if __name__ == "__main__":
    main()
