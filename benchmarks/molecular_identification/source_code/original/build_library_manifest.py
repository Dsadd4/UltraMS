#!/usr/bin/env python3
"""Replay and freeze the exact Figure 3h query/library construction with source rows."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem

from baseline_common import (
    atomic_write_json,
    read_csv_rows,
    sha256_file,
    sha256_lines,
    split_compounds,
)


MODE_ADDUCT = {"pos": "[M+H]+", "neg": "[M-H]-"}


def producer_preprocess(mzs_text: object, intensities_text: object, max_peaks: int = 150):
    mz = np.fromstring(str(mzs_text), dtype=np.float32, sep=",")
    intensity = np.fromstring(str(intensities_text), dtype=np.float32, sep=",")
    n = min(len(mz), len(intensity))
    mz, intensity = mz[:n], intensity[:n]
    valid = mz > 0
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) < 3:
        return None
    maximum = float(intensity.max())
    if maximum <= 0:
        return None
    intensity = np.clip(intensity / maximum, 0, 1)
    if len(mz) > max_peaks:
        chosen = np.sort(np.argsort(intensity)[-max_peaks:])
        mz, intensity = mz[chosen], intensity[chosen]
    order = np.argsort(mz)
    return np.stack([mz[order], intensity[order]], axis=-1).astype(np.float32)


class ModeWriter:
    def __init__(self, group: h5py.Group, mode: str, buffer_size: int = 5000) -> None:
        self.mode = mode
        self.buffer_size = buffer_size
        self.spectra = group.create_dataset(
            "spectra",
            shape=(0, 150, 2),
            maxshape=(None, 150, 2),
            dtype="f4",
            chunks=(256, 150, 2),
            compression="lzf",
        )
        self.n_peaks = group.create_dataset(
            "n_peaks", shape=(0,), maxshape=(None,), dtype="u2", chunks=True, compression="lzf"
        )
        self.source_row = group.create_dataset(
            "source_row", shape=(0,), maxshape=(None,), dtype="i8", chunks=True, compression="lzf"
        )
        self.precursor_mz = group.create_dataset(
            "precursor_mz", shape=(0,), maxshape=(None,), dtype="f4", chunks=True, compression="lzf"
        )
        self.smiles_index = group.create_dataset(
            "smiles_index", shape=(0,), maxshape=(None,), dtype="i4", chunks=True, compression="lzf"
        )
        self.adduct_index = group.create_dataset(
            "adduct_index", shape=(0,), maxshape=(None,), dtype="i2", chunks=True, compression="lzf"
        )
        self.smiles: list[str] = []
        self.smiles_to_idx: dict[str, int] = {}
        self.adducts: list[str] = []
        self.adduct_to_idx: dict[str, int] = {}
        self.row_ids_by_smiles: defaultdict[str, list[int]] = defaultdict(list)
        self.buffer: list[tuple[np.ndarray, int, int, float, int, int]] = []
        self.count = 0

    def append(self, spectrum: np.ndarray, source_row: int, precursor_mz: float, smiles: str, adduct: str) -> None:
        if smiles not in self.smiles_to_idx:
            self.smiles_to_idx[smiles] = len(self.smiles)
            self.smiles.append(smiles)
        if adduct not in self.adduct_to_idx:
            self.adduct_to_idx[adduct] = len(self.adducts)
            self.adducts.append(adduct)
        local_row = self.count + len(self.buffer)
        self.row_ids_by_smiles[smiles].append(local_row)
        padded = np.zeros((150, 2), dtype=np.float32)
        padded[: len(spectrum)] = spectrum
        self.buffer.append(
            (
                padded,
                len(spectrum),
                source_row,
                precursor_mz,
                self.smiles_to_idx[smiles],
                self.adduct_to_idx[adduct],
            )
        )
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        start, end = self.count, self.count + len(self.buffer)
        for dataset in (
            self.spectra,
            self.n_peaks,
            self.source_row,
            self.precursor_mz,
            self.smiles_index,
            self.adduct_index,
        ):
            dataset.resize((end,) + dataset.shape[1:])
        self.spectra[start:end] = np.stack([row[0] for row in self.buffer])
        self.n_peaks[start:end] = [row[1] for row in self.buffer]
        self.source_row[start:end] = [row[2] for row in self.buffer]
        self.precursor_mz[start:end] = [row[3] for row in self.buffer]
        self.smiles_index[start:end] = [row[4] for row in self.buffer]
        self.adduct_index[start:end] = [row[5] for row in self.buffer]
        self.count = end
        self.buffer.clear()


def choose_queries(row_ids_by_smiles: dict[str, list[int]], compounds: list[str], seed: int) -> tuple[list[int], list[str]]:
    rng = random.Random(seed)
    rows, smiles = [], []
    for smi in sorted(compounds):
        rows.append(rng.choice(row_ids_by_smiles[smi]))
        smiles.append(smi)
    return rows, smiles


def build_library(
    row_ids_by_smiles: dict[str, list[int]],
    excluded: list[int],
    allowed: set[str] | None = None,
) -> tuple[list[int], list[str]]:
    excluded_set = set(excluded)
    rows, smiles = [], []
    # Historical producer retained canonical-compound insertion order from the
    # source CSV.  This order is part of the frozen embedding/library identity.
    for smi in row_ids_by_smiles:
        if allowed is not None and smi not in allowed:
            continue
        for row_id in row_ids_by_smiles[smi]:
            if row_id not in excluded_set:
                rows.append(row_id)
                smiles.append(smi)
    return rows, smiles


def build(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / "msnlib_fig3h_spectra.h5"
    if h5_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {h5_path}; use --force only for a fresh versioned run")
    if h5_path.exists():
        h5_path.unlink()
    started = time.time()
    raw_counts = {"pos": 0, "neg": 0}
    invalid_counts = {"pos": 0, "neg": 0}
    # The frozen Figure 3h producer used the two strict adduct subsets below.
    # A newer fine-tuning script in the snapshot broadened these to composite
    # ion-mode sets, but those sets do not reproduce the frozen embeddings.
    mode_sets = {mode: {adduct} for mode, adduct in MODE_ADDUCT.items()}
    with h5py.File(h5_path, "w") as h5:
        writers = {mode: ModeWriter(h5.create_group(mode), mode) for mode in ("pos", "neg")}
        for source_row, row in enumerate(read_csv_rows(args.csv)):
            adduct = str(row.get("adduct", ""))
            mode = "pos" if adduct == MODE_ADDUCT["pos"] else "neg" if adduct == MODE_ADDUCT["neg"] else None
            if mode is None:
                continue
            raw_counts[mode] += 1
            spectrum = producer_preprocess(row["mzs"], row["intensities"], max_peaks=150)
            raw_smiles = str(row.get("smiles", "")).strip()
            molecule = None if raw_smiles in ("", "nan") else Chem.MolFromSmiles(raw_smiles)
            smiles = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""
            if spectrum is None or not smiles:
                invalid_counts[mode] += 1
                continue
            writers[mode].append(
                spectrum,
                source_row,
                float(row["precursor_mz"]),
                smiles,
                adduct,
            )
            total = raw_counts["pos"] + raw_counts["neg"]
            if total % 50_000 == 0:
                print(f"eligible_rows_seen={total:,}", flush=True)
        for writer in writers.values():
            writer.flush()

        summary: dict[str, object] = {
            "status": "complete",
            "csv": str(Path(args.csv).resolve()),
            "csv_sha256": sha256_file(args.csv),
            "split_seed": args.split_seed,
            "query_seed": args.query_seed,
            "mode_labels": {"pos": "[M+H]+", "neg": "[M-H]-"},
            "compound_identity": "RDKit canonical isomeric SMILES",
            "adduct_sets": {mode: sorted(values) for mode, values in mode_sets.items()},
            "modes": {},
        }
        for mode, writer in writers.items():
            # The historical producer split every valid canonical compound,
            # including singletons.  A test singleton has no remaining library
            # positive after its only spectrum becomes the query (1 positive,
            # 33 negative in the frozen benchmark).
            valid_rows = dict(writer.row_ids_by_smiles)
            singleton_spectra = sum(len(rows) for rows in valid_rows.values() if len(rows) == 1)
            mode_offset = 0 if mode == "pos" else 1
            mode_split_seed = args.split_seed + mode_offset
            mode_query_seed = args.query_seed
            splits = split_compounds(list(valid_rows), seed=mode_split_seed)
            test_query_rows, test_query_smis = choose_queries(valid_rows, splits["test"], mode_query_seed)
            test_library_rows, test_library_smis = build_library(valid_rows, test_query_rows)
            val_query_rows, val_query_smis = choose_queries(valid_rows, splits["val"], mode_query_seed)
            val_allowed = set(splits["train"]) | set(splits["val"])
            val_library_rows, val_library_smis = build_library(
                valid_rows, val_query_rows, allowed=val_allowed
            )
            train_rows = [row for smi in sorted(splits["train"]) for row in valid_rows[smi]]

            if not args.skip_reference_regression:
                if not args.reference_embedding_dir:
                    raise ValueError("--reference-embedding-dir is required for reference regression")
                reference_dir = Path(args.reference_embedding_dir)
                reference_query = json.loads(
                    (reference_dir / f"emb_msnlib_{mode}_query_smis.json").read_text()
                )
                reference_library = json.loads(
                    (reference_dir / f"emb_msnlib_{mode}_lib_smis.json").read_text()
                )
                if reference_query != test_query_smis:
                    raise RuntimeError(f"{mode} replayed query SMILES differ from frozen reference")
                if reference_library != test_library_smis:
                    raise RuntimeError(f"{mode} replayed library SMILES differ from frozen reference")

            smiles_path = out / f"smiles_{mode}.json"
            adduct_path = out / f"adducts_{mode}.json"
            atomic_write_json(smiles_path, writer.smiles)
            atomic_write_json(adduct_path, writer.adducts)
            protocol_path = out / f"protocol_{mode}.npz"
            np.savez_compressed(
                protocol_path,
                train_rows=np.asarray(train_rows, dtype=np.int64),
                val_query_rows=np.asarray(val_query_rows, dtype=np.int64),
                val_library_rows=np.asarray(val_library_rows, dtype=np.int64),
                test_query_rows=np.asarray(test_query_rows, dtype=np.int64),
                test_library_rows=np.asarray(test_library_rows, dtype=np.int64),
            )
            library_counts = defaultdict(int)
            for smi in test_library_smis:
                library_counts[smi] += 1
            n_eligible = sum(library_counts[smi] > 0 for smi in test_query_smis)
            source_rows_array = np.asarray(writer.source_row)
            mode_summary = {
                "raw_adduct_rows": raw_counts[mode],
                "invalid_rows": invalid_counts[mode],
                "valid_preprocessed_rows_before_singleton_filter": writer.count,
                "singleton_compounds_retained": singleton_spectra,
                "valid_compounds": len(valid_rows),
                "split_seed": mode_split_seed,
                "query_seed": mode_query_seed,
                "split_compound_counts": {key: len(value) for key, value in splits.items()},
                "train_spectra": len(train_rows),
                "validation": {
                    "n_query": len(val_query_rows),
                    "n_library": len(val_library_rows),
                    "ordered_query_smiles_sha256": sha256_lines(val_query_smis),
                    "ordered_library_smiles_sha256": sha256_lines(val_library_smis),
                    "library_scope": "train and validation compounds only; selected validation query excluded",
                },
                "test": {
                    "n_query": len(test_query_rows),
                    "n_library": len(test_library_rows),
                    "n_queries_with_library_positive": n_eligible,
                    "ordered_query_smiles_sha256": sha256_lines(test_query_smis),
                    "ordered_library_smiles_sha256": sha256_lines(test_library_smis),
                    "query_source_row_sha256": sha256_lines(
                        map(str, source_rows_array[np.asarray(test_query_rows)].tolist())
                    ),
                    "library_source_row_sha256": sha256_lines(
                        map(str, source_rows_array[np.asarray(test_library_rows)].tolist())
                    ),
                },
                "vocabulary": {
                    "n_smiles": len(writer.smiles),
                    "n_adducts": len(writer.adducts),
                    "adducts": writer.adducts,
                },
                "artifacts": {
                    protocol_path.name: {"sha256": sha256_file(protocol_path)},
                    smiles_path.name: {"sha256": sha256_file(smiles_path)},
                    adduct_path.name: {"sha256": sha256_file(adduct_path)},
                },
            }
            summary["modes"][mode] = mode_summary

    summary["h5"] = {
        "path": str(h5_path),
        "sha256": sha256_file(h5_path),
        "bytes": h5_path.stat().st_size,
    }
    summary["runtime_seconds"] = time.time() - started
    atomic_write_json(out / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--reference-embedding-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--query-seed", type=int, default=119)
    parser.add_argument("--skip-reference-regression", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
