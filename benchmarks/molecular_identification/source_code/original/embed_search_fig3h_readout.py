#!/usr/bin/env python3
"""Embed every strict-adduct spectrum and run exact Figure 3h library search."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, sha256_file
from train_chemberta_readout import make_model


METHODS = ("linear", "deepsets", "fourier_projection", "ultrams_codebook")
DISPLAY = {
    "linear": "Linear", "deepsets": "DeepSets",
    "fourier_projection": "Fourier", "ultrams_codebook": "Codebook",
}


def tie_keys(source_rows: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.blake2b(f"{seed}\0{int(row)}".encode("ascii"), digest_size=8).digest(), "big")
        for row in source_rows
    ], dtype=np.uint64)


def inputs_for(cache: Path, mode: str, method: str):
    root = cache / mode
    if method == "linear":
        return np.load(root / "bins_float16.npy", mmap_mode="r")
    if method in {"deepsets", "fourier_projection"}:
        return np.load(root / "peaksets_float32.npy", mmap_mode="r")
    return np.load(root / "codebook_float32.npy", mmap_mode="r")


@torch.no_grad()
def embed_all(model, inputs, path: Path, device: torch.device, batch_size: int) -> dict:
    vectors = np.lib.format.open_memmap(path, "w+", dtype=np.float32, shape=(len(inputs), 768))
    started = time.time()
    model.eval()
    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        x = torch.from_numpy(np.asarray(inputs[start:end], dtype=np.float32)).to(device)
        vectors[start:end] = F.normalize(model(x).float(), dim=1).cpu().numpy()
        if end % 20_000 < batch_size or end == len(inputs):
            vectors.flush(); print(f"embed rows={end:,}/{len(inputs):,}", flush=True)
    vectors.flush()
    probe_norm = np.linalg.norm(np.asarray(vectors), axis=1)
    if not np.isfinite(probe_norm).all() or np.any(probe_norm < 0.999):
        raise RuntimeError("invalid normalized embeddings")
    return {"n_rows": len(inputs), "shape": [len(inputs), 768],
            "sha256": sha256_file(path), "runtime_seconds": time.time() - started}


@torch.no_grad()
def exact_search(vectors, query_rows, library_rows, labels, source_rows, device, query_block, library_block, seed):
    query_labels = labels[query_rows]; library_labels = labels[library_rows]
    keys = tie_keys(source_rows[library_rows], seed)
    top_score = np.full(len(query_rows), -np.inf, np.float32)
    top_position = np.full(len(query_rows), -1, np.int64)
    top_key = np.full(len(query_rows), np.iinfo(np.uint64).max, np.uint64)
    n_ties = np.zeros(len(query_rows), np.int64)
    n_ties_correct = np.zeros(len(query_rows), np.int64)
    started = time.time()
    for qs in range(0, len(query_rows), query_block):
        qe = min(qs + query_block, len(query_rows))
        query = torch.from_numpy(np.array(vectors[query_rows[qs:qe]], copy=True)).to(device)
        for ls in range(0, len(library_rows), library_block):
            le = min(ls + library_block, len(library_rows))
            library = torch.from_numpy(np.array(vectors[library_rows[ls:le]], copy=True)).to(device)
            scores = (query @ library.T).cpu().numpy()
            maxima = scores.max(axis=1)
            for local in range(qe - qs):
                qi = qs + local
                positions = np.flatnonzero(scores[local] == maxima[local]) + ls
                selected = int(positions[int(np.argmin(keys[positions]))])
                correct_ties = int(np.sum(library_labels[positions] == query_labels[qi]))
                if maxima[local] > top_score[qi]:
                    top_score[qi] = maxima[local]; top_position[qi] = selected
                    top_key[qi] = keys[selected]; n_ties[qi] = len(positions)
                    n_ties_correct[qi] = correct_ties
                elif maxima[local] == top_score[qi]:
                    n_ties[qi] += len(positions); n_ties_correct[qi] += correct_ties
                    if keys[selected] < top_key[qi]:
                        top_position[qi] = selected; top_key[qi] = keys[selected]
        print(f"search queries={qe:,}/{len(query_rows):,} elapsed={time.time()-started:.1f}s", flush=True)
    counts = np.bincount(library_labels, minlength=int(max(labels.max(), query_labels.max())) + 1)
    return {
        "top1_score": top_score, "top1_library_position": top_position,
        "top1_correct": library_labels[top_position] == query_labels,
        "has_library_positive": counts[query_labels] > 0,
        "n_top_score_ties": n_ties, "n_top_ties_correct": n_ties_correct,
        "n_top_ties_mixed_label": (n_ties_correct > 0) & (n_ties_correct < n_ties),
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
    parser.add_argument("--tie-seed", type=int, default=20260821)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest_dir); cache = Path(args.cache_dir)
    trained = Path(args.model_dir) / args.model / args.mode / f"seed_{args.seed}"
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(trained / "best.pt", map_location="cpu")
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=768).to(device)
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
            split_summaries[split] = json.loads((split_out / "DONE.json").read_text()); continue
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
            "tie_rule": f"minimum BLAKE2b-64(seed={args.tie_seed}, source row) among exact-score ties",
        }
        atomic_write_json(split_out / "DONE.json", summary); split_summaries[split] = summary
    atomic_write_json(output / "DONE.json", {
        "status": "complete", "model": args.model, "mode": args.mode,
        "embedding": embedding_audit, "splits": split_summaries,
    })


if __name__ == "__main__":
    main()
