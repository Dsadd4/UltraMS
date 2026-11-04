"""Verify the released UltraMS MSnLib tables used by the benchmark programs.

The UltraMS MSnLib benchmark release (Hugging Face dataset
``dsadd4/UltraMS-MSnLib-Benchmark``; DOI pending) hosts two converted tables that
the spectrum-to-molecule programs consume:

* ``MSnLib.csv`` -- one row per MSnLib spectrum with the peak arrays, SMILES,
  precursor m/z, adduct, collision energy and the ``fold`` column of the UltraMS
  formula-level 80/10/10 train/validation/test split;
* ``MSnLib_candidates.json`` -- the per-query candidate lists of the
  spectrum-to-molecule task.

``fetch_public_inputs.py --dataset msnlib`` downloads both into
``ROOT/datasets/MSnLib/``. This script verifies that the local copies match the
published byte sizes and SHA-256 values, that ``MSnLib.csv`` has the released
number of data rows and fold counts, and then leaves the tables in place; it does
not rebuild or reparse spectra. The tables are derived from the public MSnLib
release (Brungs et al., Zenodo 11163380, CC BY 4.0); the dataset README documents
the rebuild route from the original records.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

MSNLIB_TABLE = {
    "name": "MSnLib.csv",
    "size": 752_707_651,
    "sha256": "be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07",
    "header": "mzs,intensities,smiles,precursor_mz,adduct,collision_energy,fold",
    "rows": 560_084,
}
MSNLIB_CANDIDATES = {
    "name": "MSnLib_candidates.json",
    "size": 1_368_264_026,
    "sha256": "125a1cdcc57a3fcef8cc76d5322ac0c434cd09b0bfb9e0c9d28bddf052f45fcc",
}
# UltraMS MSnLib formula-level 80/10/10 split carried by the fold column.
EXPECTED_FOLDS = {"train": 446_667, "val": 55_980, "test": 57_437}

REBUILD_NOTE = """\
The released tables are the authoritative artifacts for this benchmark and are
not rebuilt by this repository. The documented rebuild route is in the dataset
README (section "Where to get it" / "Rebuilding split A"):

  - rebuild MSnLib.csv from the public MSnLib original release (Brungs et al.,
    Zenodo 11163380, CC BY 4.0), which provides 560,084 MS2 records, writing one
    row per record in the frozen conversion order with the columns of the released
    table;
  - take the row -> formula -> fold assignment from the shipped
    split_A_full_msnlib/row_formula_fold.csv mapping;
  - verify the result against the published size and SHA-256 below.

That rebuild route is documented but not executed or verified by this repository.
"""


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str, spec: dict) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing {path}; run: python benchmarks/molecular_identification/"
            "source_code/fetch_public_inputs.py --root ROOT --dataset msnlib"
        )
    size = os.path.getsize(path)
    if size != spec["size"]:
        raise ValueError(f"{path}: {size} bytes, expected {spec['size']}")
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"{path}: sha256 {actual}, expected {spec['sha256']}")
    print(f"ok {os.path.basename(path)}: {size:,} bytes, sha256 {actual}")


def verify_folds(path: str) -> None:
    """Check the row count, the header and the fold counts of the released CSV."""
    counts = {}
    rows = 0
    with open(path, newline="") as stream:
        header = stream.readline().rstrip("\r\n")
        if header != MSNLIB_TABLE["header"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in stream:
            rows += 1
            fold = line.rsplit(",", 1)[-1].strip().strip('"').lower()
            counts[fold] = counts.get(fold, 0) + 1
    if rows != MSNLIB_TABLE["rows"]:
        raise ValueError(f"{path}: {rows} data rows, expected {MSNLIB_TABLE['rows']}")
    if counts != EXPECTED_FOLDS:
        raise ValueError(f"{path}: fold counts {counts}, expected {EXPECTED_FOLDS}")
    print(
        "ok fold column: "
        + ", ".join(f"{name} {count:,}" for name, count in EXPECTED_FOLDS.items())
        + f" ({rows:,} spectra; UltraMS formula-level 80/10/10 split)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True, help="Benchmark workspace root")
    parser.add_argument(
        "--spectra-only",
        action="store_true",
        help="Verify MSnLib.csv only, without requiring MSnLib_candidates.json",
    )
    parser.add_argument(
        "--rebuild-from-original",
        action="store_true",
        help="Print the documented rebuild route from the MSnLib original release and exit",
    )
    args = parser.parse_args()

    if args.rebuild_from_original:
        print(REBUILD_NOTE, end="")
        return

    source_dir = os.path.join(args.root, "datasets", "MSnLib")
    csv_path = os.path.join(source_dir, MSNLIB_TABLE["name"])
    try:
        verify_file(csv_path, MSNLIB_TABLE)
        verify_folds(csv_path)
        if not args.spectra_only:
            verify_file(os.path.join(source_dir, MSNLIB_CANDIDATES["name"]), MSNLIB_CANDIDATES)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(f"ready: {source_dir}")


if __name__ == "__main__":
    main()
