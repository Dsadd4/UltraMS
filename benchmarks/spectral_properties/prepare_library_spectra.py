"""Build or verify the library-spectrum CSV used by the benchmark programs.

For **MSnLib**, the CSV is the released ``MSnLib.csv`` table of the UltraMS MSnLib
benchmark release (Hugging Face dataset ``dsadd4/UltraMS-MSnLib-Benchmark``; DOI
pending), downloaded by ``download_public_inputs.py --asset msnlib_spectra``. The
table is derived from the public MSnLib library (Brungs et al., Zenodo 11163380,
CC BY 4.0) and carries the UltraMS formula-level 80/10/10 split in its ``fold``
column. This script verifies the published size and SHA-256, places the table at
the requested output path, and checks the released row count and fold counts; it
does not reparse spectra.

For **SpectraVerse**, the input is the legacy SpectraVerse-derived MGF record
listed in ``INPUTS.md``; this script converts it to the same CSV layout (peak
arrays, SMILES, precursor m/z, adduct, collision energy and partition label) and
verifies the source checksum. A first-party SpectraVerse table is pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Released UltraMS MSnLib table (downloaded by download_public_inputs.py).
MSNLIB_TABLE = {
    "size": 752_707_651,
    "sha256": "be54b33039cf18675751ea3c6354b592177c899a0cd5df303ae616330133ec07",
    "rows": 560_084,
    "header": "mzs,intensities,smiles,precursor_mz,adduct,collision_energy,fold",
}
# UltraMS MSnLib formula-level 80/10/10 split carried by the fold column.
MSNLIB_FOLDS = {"train": 446_667, "val": 55_980, "test": 57_437}

# Legacy SpectraVerse-derived MGF record (still converted from the MGF layout).
SPECTRAVERSE_MGF = {"md5": "6af0588372e00781ed23a018841938aa", "rows": 464_346}


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_msnlib_table(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run: python benchmarks/spectral_properties/"
            "download_public_inputs.py --output-root INPUTS --asset msnlib_spectra"
        )
    size = path.stat().st_size
    if size != MSNLIB_TABLE["size"]:
        raise ValueError(f"{path}: {size} bytes, expected {MSNLIB_TABLE['size']}")
    actual = checksum(path, "sha256")
    if actual != MSNLIB_TABLE["sha256"]:
        raise ValueError(f"{path}: sha256 {actual}, expected {MSNLIB_TABLE['sha256']}")


def verify_msnlib_folds(path: Path) -> dict:
    counts: dict = {}
    rows = 0
    with path.open(newline="") as stream:
        header = stream.readline().rstrip("\r\n")
        if header != MSNLIB_TABLE["header"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in stream:
            rows += 1
            fold = line.rsplit(",", 1)[-1].strip().strip('"').lower()
            counts[fold] = counts.get(fold, 0) + 1
    if rows != MSNLIB_TABLE["rows"]:
        raise ValueError(f"{path}: {rows} data rows, expected {MSNLIB_TABLE['rows']}")
    if counts != MSNLIB_FOLDS:
        raise ValueError(f"{path}: fold counts {counts}, expected {MSNLIB_FOLDS}")
    return counts


def parse_mgf_file(path: Path) -> list[dict]:
    import numpy as np
    from pyteomics import mgf

    data = []
    with mgf.MGF(str(path)) as reader:
        for spectrum in reader:
            params = spectrum.get("params", {})
            mzs = spectrum.get("m/z array", np.array([]))
            intensities = spectrum.get("intensity array", np.array([]))
            if len(mzs) == 0:
                continue
            smiles = params.get("SMILES") or params.get("smiles", "")
            if not smiles:
                continue
            pepmass = params.get("PEPMASS") or params.get("pepmass")
            if pepmass is None:
                precursor_mz = 0.0
            elif isinstance(pepmass, (list, tuple)):
                precursor_mz = float(pepmass[0])
            else:
                precursor_mz = float(pepmass)
            adduct = params.get("ADDUCT") or params.get("adduct", "")
            if adduct:
                adduct = str(adduct).strip()
            collision_energy = params.get("COLLISION_ENERGY") or params.get("collision_energy")
            if collision_energy is not None:
                if isinstance(collision_energy, (list, tuple)):
                    collision_energy = collision_energy[0] if collision_energy else None
                if isinstance(collision_energy, str):
                    collision_energy = collision_energy.strip("[]").split(",")[0].strip()
                try:
                    collision_energy = (
                        float(collision_energy)
                        if collision_energy and collision_energy != "nan" else None
                    )
                except (ValueError, TypeError):
                    collision_energy = None
            fold = params.get("FOLD") or params.get("fold", "unknown")
            fold = str(fold).strip().lower()
            if fold in {"valid", "validation"}:
                fold = "val"
            data.append({
                "mzs": mzs.astype(np.float32),
                "intensities": intensities.astype(np.float32),
                "smiles": smiles,
                "precursor_mz": precursor_mz,
                "adduct": adduct,
                "collision_energy": collision_energy,
                "fold": fold,
            })
    return data


def convert_to_csv(data: list[dict], output: Path):
    import pandas as pd

    rows = []
    for item in data:
        rows.append({
            "mzs": ",".join(map(str, item["mzs"].tolist())),
            "intensities": ",".join(map(str, item["intensities"].tolist())),
            "smiles": item["smiles"],
            "precursor_mz": item["precursor_mz"],
            "adduct": item["adduct"] if item["adduct"] else "",
            "collision_energy": (
                item["collision_energy"] if item["collision_energy"] is not None else ""
            ),
            "fold": item["fold"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def prepare_msnlib(source: Path, output: Path) -> dict:
    verify_msnlib_table(source)
    source_sha256 = checksum(source, "sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != output.resolve():
        shutil.copyfile(source, output)
    if checksum(output, "sha256") != MSNLIB_TABLE["sha256"]:
        raise ValueError(f"{output}: copied table does not match the released SHA-256")
    folds = verify_msnlib_folds(output)
    return {
        "dataset": "msnlib",
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "output_sha256": MSNLIB_TABLE["sha256"],
        "rows": MSNLIB_TABLE["rows"],
        "folds": folds,
        "source_release": "UltraMS MSnLib benchmark release (dsadd4/UltraMS-MSnLib-Benchmark; DOI 10.5281/zenodo.23963489)",
    }


def prepare_spectraverse(source: Path, output: Path) -> dict:
    source_md5 = checksum(source, "md5")
    if source_md5 != SPECTRAVERSE_MGF["md5"]:
        raise ValueError(f"Source MGF mismatch: md5={source_md5}")
    data = parse_mgf_file(source)
    if len(data) != SPECTRAVERSE_MGF["rows"]:
        raise ValueError(f"Expected {SPECTRAVERSE_MGF['rows']} spectra; got {len(data)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame = convert_to_csv(data, temporary)
    temporary.replace(output)
    return {
        "dataset": "spectraverse",
        "source": str(source),
        "source_md5": source_md5,
        "output": str(output),
        "output_sha256": checksum(output, "sha256"),
        "rows": len(frame),
        "folds": {str(k): int(v) for k, v in frame.fold.value_counts().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["msnlib", "spectraverse"])
    parser.add_argument(
        "--source", type=Path, required=True,
        help="MSnLib: released MSnLib.csv; SpectraVerse: legacy source MGF record",
    )
    parser.add_argument("--output", type=Path, required=True, help="benchmark CSV")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.dataset == "msnlib":
        report = prepare_msnlib(source, output)
    else:
        report = prepare_spectraverse(source, output)
    output.with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
