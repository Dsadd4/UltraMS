"""Convert the pinned public MassSpecGym table to the benchmark CSV layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "0c9cc50450def3f0d4fe2dc09dea1105fc15e635db8c6656bc3e3be37a3bcd95"
EXPECTED_ROWS = 231104
REQUIRED_COLUMNS = {
    "identifier", "mzs", "intensities", "smiles", "inchikey", "formula",
    "precursor_mz", "adduct", "fold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="official MassSpecGym.tsv")
    parser.add_argument("--output", type=Path, required=True, help="benchmark MassSpecGym.csv")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    source_sha = sha256(source)
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"Expected the pinned public MassSpecGym.tsv; got SHA-256 {source_sha}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with source.open(newline="") as source_handle, temporary.open("w", newline="") as output_handle:
        reader = csv.reader(source_handle, delimiter="\t")
        header = next(reader)
        if not REQUIRED_COLUMNS.issubset(header):
            raise ValueError(f"Missing required columns: {sorted(REQUIRED_COLUMNS.difference(header))}")
        writer = csv.writer(output_handle, lineterminator="\n")
        writer.writerow(["", *header])
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f"Row {count} has {len(row)} columns; expected {len(header)}")
            writer.writerow([count - 1, *row])
    if count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} spectra; got {count}")
    temporary.replace(output)
    report = {
        "source": str(source),
        "source_sha256": source_sha,
        "output": str(output),
        "output_sha256": sha256(output),
        "rows": count,
        "columns": header,
    }
    output.with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
