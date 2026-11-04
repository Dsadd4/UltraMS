"""Verify the released library-spectrum tables used by the benchmark programs.

Both tables are distributed in first-party UltraMS benchmark releases and are
downloaded by ``download_public_inputs.py``:

* **MSnLib** -- ``MSnLib.csv`` of the UltraMS MSnLib benchmark release
  (Hugging Face ``dsadd4/UltraMS-MSnLib-Benchmark``, DOI 10.5281/zenodo.23963489),
  derived from the public MSnLib library (Brungs et al., Zenodo 11163380,
  CC BY 4.0). Its ``fold`` column is the UltraMS formula-level 80/10/10 split.
* **SpectraVerse** -- ``Spectraverse.csv`` of the UltraMS SpectraVerse dataset
  (Hugging Face ``dsadd4/UltraMS-Spectraverse-Dataset``,
  DOI 10.5281/zenodo.23964105), derived from SpectraVerse v1.0.1
  (Zenodo 17870921, CC BY 4.0). Its ``fold`` column is the UltraMS molecule-level
  (exact-SMILES) 8:1:1 split.

Each mode verifies the published byte size and SHA-256, the header, the released
number of data rows and the fold counts, then places the table at the requested
output path. No spectra are reparsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

RELEASED_TABLES = {
    "msnlib": {
        "size": 752_707_651,
        "sha256": "be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07",
        "rows": 560_084,
        "header": "mzs,intensities,smiles,precursor_mz,adduct,collision_energy,fold",
        "folds": {"train": 446_667, "val": 55_980, "test": 57_437},
        "release": "UltraMS MSnLib benchmark release (dsadd4/UltraMS-MSnLib-Benchmark; DOI 10.5281/zenodo.23963489)",
    },
    "spectraverse": {
        "size": 707_447_509,
        "sha256": "06e084501e91f92cc642c3d60de2863266ff3d0db958354a011e32851bef3976",
        "rows": 464_346,
        "header": "mzs,intensities,smiles,precursor_mz,adduct,collision_energy,fold",
        "folds": {"train": 370_362, "val": 47_825, "test": 46_159},
        "release": "UltraMS SpectraVerse dataset (dsadd4/UltraMS-Spectraverse-Dataset; DOI 10.5281/zenodo.23964105)",
    },
}


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_table(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run: python benchmarks/spectral_properties/"
            "download_public_inputs.py --output-root INPUTS --asset "
            f"{'msnlib_spectra' if spec is RELEASED_TABLES['msnlib'] else 'spectraverse_spectra'}"
        )
    size = path.stat().st_size
    if size != spec["size"]:
        raise ValueError(f"{path}: {size} bytes, expected {spec['size']}")
    actual = checksum(path)
    if actual != spec["sha256"]:
        raise ValueError(f"{path}: sha256 {actual}, expected {spec['sha256']}")


def verify_folds(path: Path, spec: dict) -> dict:
    counts: dict = {}
    rows = 0
    with path.open(newline="") as stream:
        header = stream.readline().rstrip("\r\n")
        if header != spec["header"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in stream:
            rows += 1
            fold = line.rsplit(",", 1)[-1].strip().strip('"').lower()
            counts[fold] = counts.get(fold, 0) + 1
    if rows != spec["rows"]:
        raise ValueError(f"{path}: {rows} data rows, expected {spec['rows']}")
    if counts != spec["folds"]:
        raise ValueError(f"{path}: fold counts {counts}, expected {spec['folds']}")
    return counts


def prepare(dataset: str, source: Path, output: Path) -> dict:
    spec = RELEASED_TABLES[dataset]
    verify_table(source, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copyfile(source, output)
    if checksum(output) != spec["sha256"]:
        raise ValueError(f"{output}: copied table does not match the released SHA-256")
    folds = verify_folds(output, spec)
    return {
        "dataset": dataset,
        "source": str(source),
        "source_sha256": spec["sha256"],
        "output": str(output),
        "output_sha256": spec["sha256"],
        "rows": spec["rows"],
        "folds": folds,
        "source_release": spec["release"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(RELEASED_TABLES))
    parser.add_argument("--source", type=Path, required=True, help="released benchmark CSV")
    parser.add_argument("--output", type=Path, required=True, help="benchmark CSV")
    args = parser.parse_args()
    report = prepare(args.dataset, args.source.resolve(), args.output.resolve())
    args.output.resolve().with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
