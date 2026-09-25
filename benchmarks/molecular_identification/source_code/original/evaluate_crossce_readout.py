#!/usr/bin/env python3
"""Embed frozen cross-CE pairs and run exact one-to-one library retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, sha256_file
from train_chemberta_readout import make_model


METHODS = ("linear", "deepsets", "fourier_projection", "ultrams_codebook")
DISPLAY = {
    "linear": "Linear",
    "deepsets": "DeepSets",
    "fourier_projection": "Fourier",
    "ultrams_codebook": "Codebook",
}


def inputs_for(cache: Path, mode: str, method: str):
    root = cache / mode
    if method == "linear":
        return np.load(root / "bins_float16.npy", mmap_mode="r")
    if method in {"deepsets", "fourier_projection"}:
        return np.load(root / "peaksets_float32.npy", mmap_mode="r")
    return np.load(root / "codebook_float32.npy", mmap_mode="r")


def tie_keys(source_rows: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.blake2b(f"{seed}\0{int(row)}".encode("ascii"), digest_size=8).digest(), "big")
        for row in source_rows
    ], dtype=np.uint64)


@torch.no_grad()
def embed(model, inputs, rows: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    result = np.empty((len(rows), 768), dtype=np.float32)
    model.eval()
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        x = torch.from_numpy(np.asarray(inputs[rows[start:end]], dtype=np.float32)).to(device)
        result[start:end] = F.normalize(model(x).float(), dim=1).cpu().numpy()
    return result


@torch.no_grad()
def exact_search(query: np.ndarray, library: np.ndarray, keys: np.ndarray, device: torch.device,
                 query_block: int, library_block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(query)
    top_position = np.full(n, -1, dtype=np.int64)
    top_score = np.full(n, -np.inf, dtype=np.float32)
    ranks = np.ones(n, dtype=np.int64)
    for qs in range(0, n, query_block):
        qe = min(qs + query_block, n)
        q = torch.from_numpy(query[qs:qe]).to(device)
        # Keep one query block's exact GPU matmul scores.  Reusing these same
        # float32 values for top-1 and ground-truth ranks avoids numerical
        # disagreement between an einsum GT score and blockwise GEMM scores.
        block_scores = np.empty((qe - qs, n), dtype=np.float32)
        for ls in range(0, n, library_block):
            le = min(ls + library_block, n)
            scores = (q @ torch.from_numpy(library[ls:le]).to(device).T).cpu().numpy()
            block_scores[:, ls:le] = scores
        for local in range(qe - qs):
            global_q = qs + local
            row = block_scores[local]
            maximum = row.max()
            tied_top = np.flatnonzero(row == maximum)
            selected = int(tied_top[np.argmin(keys[tied_top])])
            gt_score = row[global_q]
            ranks[global_q] = 1 + int(np.sum(row > gt_score)) + int(
                np.sum((row == gt_score) & (keys < keys[global_q]))
            )
            top_position[global_q] = selected
            top_score[global_q] = maximum
        print(f"search queries={qe:,}/{n:,}", flush=True)
    return ranks, top_position, top_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest-dir", required=True)
    parser.add_argument("--crossce-manifest-dir", required=True)
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
    args = parser.parse_args()
    source_manifest = Path(args.source_manifest_dir)
    crossce_manifest = Path(args.crossce_manifest_dir)
    cache = Path(args.cache_dir)
    trained = Path(args.model_dir) / args.model / args.mode / f"seed_{args.seed}"
    output = Path(args.output_dir) / args.model / args.mode / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(trained / "final.pt", map_location="cpu")
    device = torch.device(args.device)
    model = make_model(args.model, out_dim=768).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    inputs = inputs_for(cache, args.mode, args.model)
    protocol = np.load(crossce_manifest / f"protocol_{args.mode}.npz")
    query_rows = np.asarray(protocol["query_rows"], dtype=np.int64)
    library_rows = np.asarray(protocol["library_rows"], dtype=np.int64)
    query_source_rows = np.asarray(protocol["query_source_rows"], dtype=np.int64)
    library_source_rows = np.asarray(protocol["library_source_rows"], dtype=np.int64)
    with h5py.File(source_manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        group = h5[args.mode]
        row_labels = np.asarray(group["smiles_index"], dtype=np.int64)
    smiles = json.loads((source_manifest / f"smiles_{args.mode}.json").read_text())
    query_labels = row_labels[query_rows]
    library_labels = row_labels[library_rows]
    if not np.array_equal(query_labels, library_labels):
        raise RuntimeError("query and library identities are not aligned")
    query_vectors = embed(model, inputs, query_rows, device, args.embed_batch_size)
    library_vectors = embed(model, inputs, library_rows, device, args.embed_batch_size)
    keys = tie_keys(library_source_rows, args.tie_seed)
    ranks, top_positions, top_scores = exact_search(
        query_vectors, library_vectors, keys, device, args.query_block, args.library_block
    )
    top_labels = library_labels[top_positions]
    correct = top_labels == query_labels
    np.savez_compressed(
        output / "retrieval.npz",
        query_rows=query_rows,
        library_rows=library_rows,
        query_source_rows=query_source_rows,
        library_source_rows=library_source_rows,
        query_labels=query_labels,
        library_labels=library_labels,
        ranks=ranks,
        top_positions=top_positions,
        top_scores=top_scores,
        top1_correct=correct,
    )
    records = [{
        "smiles": smiles[int(label)],
        "query_source_row": int(query_source_rows[index]),
        "library_source_row": int(library_source_rows[index]),
        "rank": int(ranks[index]),
        "top1_correct": bool(correct[index]),
        "top1_smiles": smiles[int(top_labels[index])],
        "top1_score": float(top_scores[index]),
    } for index, label in enumerate(query_labels)]
    atomic_write_json(output / "per_sample.json", records)
    summary = {
        "status": "complete",
        "model": DISPLAY[args.model],
        "model_key": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "n_query": len(query_rows),
        "hit_at_1": float(correct.mean()),
        "hit_at_5": float((ranks <= 5).mean()),
        "hit_at_10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "checkpoint_sha256": sha256_file(trained / "final.pt"),
        "retrieval_sha256": sha256_file(output / "retrieval.npz"),
        "per_sample_sha256": sha256_file(output / "per_sample.json"),
        "tie_rule": f"minimum BLAKE2b-64(seed={args.tie_seed}, library source row)",
    }
    atomic_write_json(output / "DONE.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
