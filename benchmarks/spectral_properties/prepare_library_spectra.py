"""Build the MSnLib or SpectraVerse benchmark CSV from the source MGF.

This is the CSV-producing path of the original ``convert_specbridge.py``. The
original Arrow-shard and molecular-candidate exports are separate experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyteomics import mgf


SOURCES = {
    "msnlib": ("e7c648b89841d10759f6b796aa7e3e50", 560084),
    "spectraverse": ("6af0588372e00781ed23a018841938aa", 464346),
}


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mgf_file(path: Path) -> list[dict]:
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


def convert_to_csv(data: list[dict], output: Path) -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    parser.add_argument("--source", type=Path, required=True, help="public SpecBridge MGF")
    parser.add_argument("--output", type=Path, required=True, help="benchmark CSV")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    expected_md5, expected_rows = SOURCES[args.dataset]
    source_md5 = checksum(source, "md5")
    if source_md5 != expected_md5:
        raise ValueError(f"Source MGF mismatch: md5={source_md5}")
    data = parse_mgf_file(source)
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} spectra; got {len(data)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame = convert_to_csv(data, temporary)
    temporary.replace(output)
    report = {
        "dataset": args.dataset,
        "source": str(source),
        "source_md5": source_md5,
        "output": str(output),
        "output_sha256": checksum(output, "sha256"),
        "rows": len(frame),
        "folds": {str(k): int(v) for k, v in frame.fold.value_counts().items()},
    }
    output.with_suffix(".source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
