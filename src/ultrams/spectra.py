"""Read MS/MS spectra from common mass spectrometry files."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any, Iterator


def _precursor(value: Any, spectrum_id: str) -> float:
    """Return a positive precursor m/z with a useful spectrum-specific error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"spectrum {spectrum_id!r} has no valid precursor m/z") from exc
    if not 0 < result < float("inf"):
        raise ValueError(f"spectrum {spectrum_id!r} has no valid precursor m/z")
    return result


def read_mgf(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield spectra from an MGF file without loading the full file into memory.

    Each record has ``id``, ``mz``, ``intensity``, and ``precursor_mz`` keys.
    ``TITLE`` or ``NAME`` is used as the ID, followed by a valid ``SCANS``
    value or the spectrum number.
    Peak lines may contain additional columns after m/z and intensity.
    """
    source = Path(path).expanduser()
    opener = gzip.open if source.name.lower().endswith(".gz") else open
    with opener(source, "rt", encoding="utf-8-sig") as handle:
        inside = False
        metadata: dict[str, str] = {}
        mz: list[float] = []
        intensity: list[float] = []
        number = 0
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith(("#", ";", "!")):
                continue
            keyword = line.upper()
            if keyword == "BEGIN IONS":
                if inside:
                    raise ValueError(f"nested BEGIN IONS at line {line_number} in {source}")
                inside = True
                metadata, mz, intensity = {}, [], []
                number += 1
                continue
            if keyword == "END IONS":
                if not inside:
                    raise ValueError(f"END IONS without BEGIN IONS at line {line_number} in {source}")
                scans = metadata.get("SCANS", "")
                spectrum_id = (
                    metadata.get("TITLE")
                    or metadata.get("NAME")
                    or (scans if scans not in {"", "-1"} else None)
                    or f"spectrum_{number}"
                )
                precursor_field = re.split(r"[,\s]+", metadata.get("PEPMASS", "").strip())
                precursor_mz = _precursor(precursor_field[0] if precursor_field else None, spectrum_id)
                if not mz:
                    raise ValueError(f"spectrum {spectrum_id!r} has no peaks")
                yield {
                    "id": spectrum_id,
                    "mz": mz,
                    "intensity": intensity,
                    "precursor_mz": precursor_mz,
                }
                inside = False
                continue
            if not inside:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key.strip().upper()] = value.strip()
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"invalid peak at line {line_number} in {source}")
            try:
                mz.append(float(fields[0]))
                intensity.append(float(fields[1]))
            except ValueError as exc:
                raise ValueError(f"invalid peak at line {line_number} in {source}") from exc
        if inside:
            raise ValueError(f"unfinished BEGIN IONS block in {source}")


def _mzml_precursor(spectrum: dict[str, Any], spectrum_id: str) -> float:
    precursor_list = spectrum.get("precursorList", {}).get("precursor", [])
    for precursor in precursor_list:
        selected_ions = precursor.get("selectedIonList", {}).get("selectedIon", [])
        for ion in selected_ions:
            value = ion.get("selected ion m/z")
            if value is not None:
                return _precursor(value, spectrum_id)
    raise ValueError(f"spectrum {spectrum_id!r} has no valid precursor m/z")


def read_mzml(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield MS2 spectra from mzML; requires ``pip install ultrams[io]``."""
    try:
        from pyteomics import mzml
    except ImportError as exc:
        raise ImportError("mzML reading requires: python -m pip install 'ultrams[io]'") from exc

    source = Path(path).expanduser()
    with mzml.read(str(source)) as reader:
        for number, spectrum in enumerate(reader, 1):
            if int(spectrum.get("ms level", 0)) != 2:
                continue
            spectrum_id = str(spectrum.get("id") or f"spectrum_{number}")
            mz = spectrum.get("m/z array")
            intensity = spectrum.get("intensity array")
            if mz is None or intensity is None or len(mz) == 0:
                raise ValueError(f"spectrum {spectrum_id!r} has no peaks")
            yield {
                "id": spectrum_id,
                "mz": mz,
                "intensity": intensity,
                "precursor_mz": _mzml_precursor(spectrum, spectrum_id),
            }


def read_spectra(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a ``.mgf``, ``.mgf.gz``, or ``.mzML`` file."""
    source = Path(path).expanduser()
    name = source.name.lower()
    if name.endswith((".mgf", ".mgf.gz")):
        return read_mgf(source)
    if name.endswith((".mzml", ".mzml.gz")):
        return read_mzml(source)
    raise ValueError(f"unsupported spectrum file {source}; use .mgf or .mzML")
