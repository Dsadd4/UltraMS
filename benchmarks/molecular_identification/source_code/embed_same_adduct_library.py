#!/usr/bin/env python3
"""Embed the complete strict-adduct test library with a trained spectrum-pair encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", choices=("ultra", "dreams"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    public = Path(__file__).resolve().parent
    source_model = public.parents[2] / "training/ultrams_training/model"
    sys.path.insert(0, str(source_model / "train"))
    sys.path.insert(0, str(source_model))
    import train_library_search as producer

    root = args.workspace.resolve()
    producer._PROJECT_ROOT = root
    producer._TRAIN_DIR = str(source_model / "train")
    producer._COMP_DIR = str(root / "train/comparison")
    producer.ULTRA_CKPT = str(root / "train/output/phase2_rt_only/stage_d_epoch_11.pt")
    producer.DREAMS_LOADER = str(public / "original/dreams_loader.py")
    device = torch.device(args.device)
    if args.backbone == "ultra":
        model, dimension = producer.load_ultra_backbone(device, n_unfreeze=2)
    else:
        model, dimension = producer.load_dreams_backbone(device, n_unfreeze=2)
    projection = producer.UL2Proj(dimension).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    projection.load_state_dict(checkpoint["proj_state_dict"], strict=True)
    model.eval()
    projection.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.manifest_dir / "msnlib_fig3h_spectra.h5", "r") as h5:
        for mode in ("pos", "neg"):
            group = h5[mode]
            protocol = np.load(args.manifest_dir / f"protocol_{mode}.npz")
            vocabulary = json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text())
            spectra = np.asarray(group["spectra"], dtype=np.float32)
            counts = np.asarray(group["n_peaks"], dtype=np.int64)
            precursors = np.asarray(group["precursor_mz"], dtype=np.float32)
            labels = np.asarray(group["smiles_index"], dtype=np.int64)
            for role in ("query", "lib"):
                indices = protocol[f"test_{'query' if role == 'query' else 'library'}_rows"]
                out = args.output_dir / f"emb_msnlib_{mode}_{role}_{args.backbone}.npy"
                temporary = out.with_suffix(".part.npy")
                vectors = np.lib.format.open_memmap(temporary, "w+", dtype=np.float32, shape=(len(indices), 256))
                for start in range(0, len(indices), args.batch_size):
                    chunk = indices[start:start + args.batch_size]
                    samples = [
                        {"spectrum": spectra[row, :counts[row]], "precursor_mz": float(precursors[row])}
                        for row in chunk
                    ]
                    vectors[start:start + len(chunk)] = producer.extract_embeddings(
                        model, projection, samples, device, args.backbone, bs=args.batch_size
                    )
                    if start % (args.batch_size * 100) == 0:
                        print(f"{args.backbone} {mode} {role}: {start:,}/{len(indices):,}", flush=True)
                vectors.flush()
                del vectors
                os.replace(temporary, out)
                identities = [vocabulary[labels[row]] for row in indices]
                identity_path = args.output_dir / f"emb_msnlib_{mode}_{role}_smis.json"
                identity_temp = identity_path.with_suffix(".part.json")
                identity_temp.write_text(json.dumps(identities))
                os.replace(identity_temp, identity_path)
                print(f"saved {out}: {len(indices)} × 256", flush=True)


if __name__ == "__main__":
    main()
