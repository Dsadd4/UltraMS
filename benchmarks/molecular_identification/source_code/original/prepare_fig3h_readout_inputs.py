#!/usr/bin/env python3
"""Prepare leakage-safe Figure 3h inputs for the four ChemBERTa readouts.

The spectra and compound split come exclusively from the frozen strict-adduct
Figure 3h manifest.  This deliberately does not reuse the Figure 3a checkpoints,
because that independent split would expose some Figure 3h test compounds during
training.  It recreates the same four encoder families on the Figure 3h train
split and then reuses their spectrum embeddings for spectrum-to-spectrum search.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, fingerprint_ffn_bins, sha256_file


def import_retrieval_module(path: Path):
    spec = importlib.util.spec_from_file_location("fig3h_readout_retrieval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def peakset(spectrum: np.ndarray, precursor_mz: float) -> np.ndarray:
    mz = spectrum[:, 0]
    intensity = spectrum[:, 1]
    keep = (mz >= 10.0) & (mz <= 1005.0) & (intensity >= 0)
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) > 60:
        chosen = np.argpartition(intensity, -60)[-60:]
        chosen = chosen[np.argsort(mz[chosen], kind="stable")]
        mz, intensity = mz[chosen], intensity[chosen]
    maximum = float(intensity.max(initial=0.0))
    if maximum > 0:
        intensity = intensity / maximum
    result = np.zeros((61, 2), dtype=np.float32)
    result[0] = (float(precursor_mz), 1.1)
    count = min(60, len(mz))
    if count:
        result[1 : count + 1, 0] = mz[:count]
        result[1 : count + 1, 1] = intensity[:count]
    return result


def prepare_spectrum_caches(manifest: Path, output: Path, mode: str, chunk_size: int) -> None:
    mode_out = output / mode
    mode_out.mkdir(parents=True, exist_ok=True)
    bins_path = mode_out / "bins_float16.npy"
    peaks_path = mode_out / "peaksets_float32.npy"
    state_path = mode_out / "spectrum_cache_state.json"
    h5_path = manifest / "msnlib_fig3h_spectra.h5"
    with h5py.File(h5_path, "r") as h5:
        group = h5[mode]
        n_rows = len(group["n_peaks"])
        if bins_path.exists() and peaks_path.exists() and state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("status") == "complete" and state.get("n_rows") == n_rows:
                return
        bins = np.lib.format.open_memmap(bins_path, "w+", dtype=np.float16, shape=(n_rows, 1005))
        peaks = np.lib.format.open_memmap(peaks_path, "w+", dtype=np.float32, shape=(n_rows, 61, 2))
        started = time.time()
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            spectra = np.asarray(group["spectra"][start:end], dtype=np.float32)
            counts = np.asarray(group["n_peaks"][start:end], dtype=np.int64)
            precursors = np.asarray(group["precursor_mz"][start:end], dtype=np.float32)
            for local, count in enumerate(counts):
                spectrum = spectra[local, : int(count)]
                bins[start + local] = fingerprint_ffn_bins(
                    spectrum[:, 0], spectrum[:, 1], max_mz=1005, bin_width=1.0
                ).astype(np.float16)
                peaks[start + local] = peakset(spectrum, float(precursors[local]))
            bins.flush(); peaks.flush()
            atomic_write_json(state_path, {
                "status": "building", "mode": mode, "n_rows": n_rows,
                "next_row": end, "elapsed_seconds": time.time() - started,
            })
            print(f"cache mode={mode} rows={end:,}/{n_rows:,}", flush=True)
    atomic_write_json(state_path, {
        "status": "complete", "mode": mode, "n_rows": n_rows,
        "bins_shape": [n_rows, 1005], "peaksets_shape": [n_rows, 61, 2],
        "bins_sha256": sha256_file(bins_path), "peaksets_sha256": sha256_file(peaks_path),
        "protocol": {
            "bins": "1-Da sum bins, m/z <1005, per-spectrum max normalization, no precursor token",
            "peaksets": "precursor token plus strongest 60 fragments in m/z 10..1005; full float32 m/z",
        },
    })


def prepare_targets(manifest: Path, output: Path, mol_cache_path: Path) -> None:
    needed = []
    for mode in ("pos", "neg"):
        path = output / mode / "target_chemberta_float32.npy"
        if not path.exists():
            needed.append(mode)
    if not needed:
        return
    manifest_summary = json.loads((manifest / "manifest_summary.json").read_text())
    csv_path = Path(manifest_summary["csv"])
    raw_smiles = pd.read_csv(csv_path, usecols=["smiles"]).smiles.astype(str).to_numpy()
    print(f"loading molecule embedding cache: {mol_cache_path}", flush=True)
    molecule_cache = torch.load(mol_cache_path, map_location="cpu")
    for mode in needed:
        vocab = json.loads((manifest / f"smiles_{mode}.json").read_text())
        with h5py.File(manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
            group = h5[mode]
            labels = np.asarray(group["smiles_index"], dtype=np.int64)
            source_rows = np.asarray(group["source_row"], dtype=np.int64)
        representative_source = np.full(len(vocab), -1, dtype=np.int64)
        for local_row, label in enumerate(labels):
            if representative_source[label] < 0:
                representative_source[label] = source_rows[local_row]
        if np.any(representative_source < 0):
            raise RuntimeError(f"{mode}: a vocabulary entry has no source row")
        representative_raw = [raw_smiles[row].strip() for row in representative_source]
        cache_keys = [
            canonical if canonical in molecule_cache else raw
            for canonical, raw in zip(vocab, representative_raw)
        ]
        missing = [key for key in cache_keys if key not in molecule_cache]
        if missing:
            raise RuntimeError(
                f"{mode}: {len(missing)} manifest molecules lack both canonical and source-row raw ChemBERTa keys"
            )
        target = torch.stack([
            torch.as_tensor(molecule_cache[key], dtype=torch.float32).reshape(-1)
            for key in cache_keys
        ])
        if target.shape[1] != 768:
            raise RuntimeError(target.shape)
        target = F.normalize(target, dim=1).numpy()
        path = output / mode / "target_chemberta_float32.npy"
        np.save(path, target)
        key_path = output / mode / "target_cache_keys.json"
        atomic_write_json(key_path, cache_keys)
        atomic_write_json(output / mode / "target_summary.json", {
            "status": "complete", "mode": mode, "n_smiles": len(vocab),
            "shape": list(target.shape), "molecule_cache_sha256": sha256_file(mol_cache_path),
            "target_sha256": sha256_file(path),
            "target_cache_keys_sha256": sha256_file(key_path),
            "n_canonical_cache_keys": int(sum(key == canonical for key, canonical in zip(cache_keys, vocab))),
            "n_source_row_raw_cache_keys": int(sum(key != canonical for key, canonical in zip(cache_keys, vocab))),
            "source_csv_sha256": manifest_summary["csv_sha256"],
        })
    del molecule_cache
    gc.collect()


@torch.no_grad()
def prepare_codebook(
    manifest: Path,
    output: Path,
    retrieval_script: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    missing_modes = []
    for mode in ("pos", "neg"):
        path = output / mode / "codebook_float32.npy"
        summary = output / mode / "codebook_summary.json"
        if not path.exists() or not summary.exists() or json.loads(summary.read_text()).get("status") != "complete":
            missing_modes.append(mode)
    if not missing_modes:
        return
    retrieval = import_retrieval_module(retrieval_script)
    model, dimension = retrieval.load_model("rt_only_d11", device)
    if dimension != 1024:
        raise RuntimeError(dimension)
    workspace = os.environ.get("LIGHT_ULTRA_ROOT")
    checkpoint = (
        Path(workspace) / "train" / "output" / "phase2_rt_only" / "stage_d_epoch_11.pt"
        if workspace else retrieval_script.parents[1] / "output" / "phase2_rt_only" / "stage_d_epoch_11.pt"
    )
    h5_path = manifest / "msnlib_fig3h_spectra.h5"
    with h5py.File(h5_path, "r") as h5:
        for mode in missing_modes:
            group = h5[mode]
            n_rows = len(group["n_peaks"])
            path = output / mode / "codebook_float32.npy"
            result = np.lib.format.open_memmap(path, "w+", dtype=np.float32, shape=(n_rows, 1024))
            started = time.time()
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                values = torch.from_numpy(np.asarray(group["spectra"][start:end], dtype=np.float32)).to(device)
                counts = torch.from_numpy(np.asarray(group["n_peaks"][start:end], dtype=np.int64)).to(device)
                valid = torch.arange(values.shape[1], device=device)[None, :] < counts[:, None]
                encoded = model.peak_encoder(values)
                weights = values[:, :, 1].clamp_min(0.0) * valid.to(values.dtype)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                result[start:end] = (encoded * weights.unsqueeze(-1)).sum(dim=1).float().cpu().numpy()
                if end % 20_000 < batch_size or end == n_rows:
                    result.flush()
                    print(f"codebook mode={mode} rows={end:,}/{n_rows:,}", flush=True)
            result.flush()
            atomic_write_json(output / mode / "codebook_summary.json", {
                "status": "complete", "mode": mode, "shape": [n_rows, 1024],
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "cache_sha256": sha256_file(path), "runtime_seconds": time.time() - started,
                "excluded_components": ["CLS token", "precursor token", "transformer encoder", "Phase-2 head"],
            })
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--molecule-cache")
    parser.add_argument("--skip-molecule-targets", action="store_true")
    parser.add_argument("--retrieval-script", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--codebook-batch-size", type=int, default=256)
    args = parser.parse_args()
    manifest = Path(args.manifest_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for mode in ("pos", "neg"):
        prepare_spectrum_caches(manifest, output, mode, args.chunk_size)
    if not args.skip_molecule_targets:
        if not args.molecule_cache:
            parser.error("--molecule-cache is required unless --skip-molecule-targets is set")
        prepare_targets(manifest, output, Path(args.molecule_cache))
    prepare_codebook(
        manifest, output, Path(args.retrieval_script), torch.device(args.device), args.codebook_batch_size
    )
    atomic_write_json(output / "DONE.json", {
        "status": "complete",
        "manifest_summary_sha256": sha256_file(manifest / "manifest_summary.json"),
        "modes": ["pos", "neg"],
        "molecule_targets": not args.skip_molecule_targets,
    })


if __name__ == "__main__":
    main()
