#!/usr/bin/env python3
"""Prepare the molecule targets used by low-to-high collision-energy readouts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    source = Path(__file__).resolve().parent
    sys.path.insert(0, str(source / "original"))
    from baseline_common import atomic_write_json, sha256_file
    from evaluate_molecule_identification import MolEncoder

    key_files = {
        mode: source.parent / "data/benchmark_inputs" / f"collision_energy_target_keys_{mode}.json"
        for mode in ("pos", "neg")
    }
    keys = {mode: json.loads(path.read_text()) for mode, path in key_files.items()}
    raw_smiles = pd.read_csv(args.csv, usecols=["smiles"]).smiles.astype(str).to_numpy()
    with h5py.File(args.manifest_dir / "msnlib_fig3h_spectra.h5", "r") as h5:
        for mode in ("pos", "neg"):
            vocab = json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text())
            if len(keys[mode]) != len(vocab):
                raise ValueError(f"{mode}: molecule target count differs from the manifest")
            group = h5[mode]
            labels = np.asarray(group["smiles_index"], dtype=np.int64)
            source_rows = np.asarray(group["source_row"], dtype=np.int64)
            first_source = np.full(len(vocab), -1, dtype=np.int64)
            for label, row in zip(labels, source_rows):
                if first_source[label] < 0:
                    first_source[label] = row
            if np.any(first_source < 0):
                raise ValueError(f"{mode}: a manifest molecule has no source spectrum")
            for index, (key, canonical) in enumerate(zip(keys[mode], vocab)):
                if key != canonical and key != raw_smiles[first_source[index]].strip():
                    raise ValueError(f"{mode}: target key differs from its molecule at index {index}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = all(
        (args.output_dir / mode / "target_summary.json").is_file()
        and json.loads((args.output_dir / mode / "target_summary.json").read_text()).get("status") == "complete"
        and (args.output_dir / mode / "target_chemberta_float32.npy").is_file()
        and (args.output_dir / mode / "target_cache_keys.json").is_file()
        and sha256_file(args.output_dir / mode / "target_cache_keys.json") == sha256_file(key_files[mode])
        for mode in ("pos", "neg")
    )
    if complete:
        print(f"ready {args.output_dir}", flush=True)
        return

    unique_keys = list(dict.fromkeys(key for mode in ("pos", "neg") for key in keys[mode]))
    device = torch.device(args.device)
    encoder = MolEncoder(str(args.model_dir)).to(device).eval()
    encoded = {}
    for start in range(0, len(unique_keys), args.batch_size):
        batch = unique_keys[start:start + args.batch_size]
        values = encoder.encode_batch(batch, device).cpu().float().numpy()
        encoded.update(zip(batch, values))
        if start % (args.batch_size * 50) == 0:
            print(f"molecule targets: {start:,}/{len(unique_keys):,}", flush=True)
    del encoder

    for mode in ("pos", "neg"):
        directory = args.output_dir / mode
        directory.mkdir(parents=True, exist_ok=True)
        values = torch.from_numpy(np.stack([encoded[key] for key in keys[mode]], axis=0))
        target = F.normalize(values, dim=1).numpy()
        path = directory / "target_chemberta_float32.npy"
        temporary = path.with_suffix(".part.npy")
        np.save(temporary, target)
        os.replace(temporary, path)
        shutil.copyfile(key_files[mode], directory / "target_cache_keys.json")
        atomic_write_json(directory / "target_summary.json", {
            "status": "complete", "mode": mode, "shape": list(target.shape),
            "target_sha256": sha256_file(path),
            "target_cache_keys_sha256": sha256_file(directory / "target_cache_keys.json"),
        })
        print(f"saved {mode}: {target.shape}", flush=True)


if __name__ == "__main__":
    main()
