#!/usr/bin/env python3
"""Embed and exact-search Figure 3h spectrum-pair-supervised controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, sha256_file
from embed_search_fig3h_readout import DISPLAY, METHODS, exact_search, inputs_for
from train_chemberta_readout import make_model


EMBEDDING_DIMENSION = 256


@torch.no_grad()
def embed_all(model, inputs, path: Path, device: torch.device, batch_size: int) -> dict:
    vectors = np.lib.format.open_memmap(
        path, "w+", dtype=np.float32, shape=(len(inputs), EMBEDDING_DIMENSION)
    )
    started = time.time(); model.eval()
    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        values = torch.from_numpy(np.asarray(inputs[start:end], dtype=np.float32)).to(device)
        vectors[start:end] = F.normalize(model(values).float(), dim=1).cpu().numpy()
        if end % 20_000 < batch_size or end == len(inputs):
            vectors.flush(); print(f"embed rows={end:,}/{len(inputs):,}", flush=True)
    vectors.flush()
    norms = np.linalg.norm(np.asarray(vectors), axis=1)
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 2e-5):
        raise RuntimeError("invalid normalized embeddings")
    return {
        "n_rows": len(inputs),
        "shape": [len(inputs), EMBEDDING_DIMENSION],
        "dtype": "float32",
        "sha256": sha256_file(path),
        "runtime_seconds": time.time() - started,
        "training_supervision": "same-molecule spectrum-pair InfoNCE; no ChemBERTa",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--model", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embed-batch-size", type=int, default=1024)
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--library-block", type=int, default=20000)
    parser.add_argument("--tie-seed", type=int, default=20260824)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shared-polarity-model", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest_dir); cache = Path(args.cache_dir)
    trained = Path(args.model_dir) / args.model
    if not args.shared_polarity_model:
        trained = trained / args.mode
    trained = trained / f"seed_{args.seed}"
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(trained / "best.pt", map_location="cpu")
    config = checkpoint.get("config", {})
    if config.get("chemberta_used") is not False or config.get("embedding_dimension") != 256:
        raise RuntimeError("checkpoint is not a Figure 3h pure spectrum-pair model")
    if args.shared_polarity_model and config.get("modes") != ["pos", "neg"]:
        raise RuntimeError("checkpoint is not shared across positive and negative modes")
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=EMBEDDING_DIMENSION).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    inputs = inputs_for(cache, args.mode, args.model)
    vector_path = output / "all_vectors_float32.npy"
    if not vector_path.exists() or args.force:
        embedding_audit = embed_all(model, inputs, vector_path, device, args.embed_batch_size)
        atomic_write_json(output / "embedding_audit.json", embedding_audit)
    else:
        embedding_audit = json.loads((output / "embedding_audit.json").read_text())
    vectors = np.load(vector_path, mmap_mode="r")
    protocol = np.load(manifest / f"protocol_{args.mode}.npz")
    with h5py.File(manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
        source_rows = np.asarray(h5[args.mode]["source_row"], dtype=np.int64)
    split_summaries = {}
    for split in ("val", "test"):
        split_out = output / split; split_out.mkdir(exist_ok=True)
        top1_path = split_out / "top1.npz"
        if top1_path.exists() and not args.force:
            split_summaries[split] = json.loads((split_out / "DONE.json").read_text())
            continue
        query_rows = np.asarray(protocol[f"{split}_query_rows"], dtype=np.int64)
        library_rows = np.asarray(protocol[f"{split}_library_rows"], dtype=np.int64)
        arrays = exact_search(
            vectors, query_rows, library_rows, labels, source_rows, device,
            args.query_block, args.library_block, args.tie_seed,
        )
        np.savez_compressed(top1_path, query_rows=query_rows, library_rows=library_rows, **arrays)
        summary = {
            "status": "complete", "method": DISPLAY[args.model], "model_key": args.model,
            "mode": args.mode, "split": split, "seed": args.seed,
            "n_query": len(query_rows), "n_library": len(library_rows),
            "n_eligible": int(arrays["has_library_positive"].sum()),
            "hit_at_1": float(arrays["top1_correct"].mean()),
            "top1_sha256": sha256_file(top1_path),
            "checkpoint_sha256": sha256_file(trained / "best.pt"),
            "embedding_sha256": embedding_audit["sha256"],
            "embedding_dimension": EMBEDDING_DIMENSION,
            "training_supervision": "same-molecule spectrum-pair InfoNCE; no ChemBERTa",
            "shared_polarity_model": bool(args.shared_polarity_model),
            "tie_rule": f"minimum BLAKE2b-64(seed={args.tie_seed}, source row) among exact-score ties",
        }
        atomic_write_json(split_out / "DONE.json", summary); split_summaries[split] = summary
    atomic_write_json(output / "DONE.json", {
        "status": "complete", "model": args.model, "mode": args.mode,
        "embedding": embedding_audit, "splits": split_summaries,
    })


if __name__ == "__main__":
    main()
