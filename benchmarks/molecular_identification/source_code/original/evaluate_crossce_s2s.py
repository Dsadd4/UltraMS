#!/usr/bin/env python3
"""Evaluate frozen Figure 3h learned spectrum encoders on cross-CE retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


METHODS = {
    "deepsets": "DeepSets",
    "ultrams_codebook": "Codebook",
    "linear": "Linear",
    "fourier_projection": "Fourier",
    "dreams": "DreaMS",
    "ultra": "UltraMS",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tie_keys(source_rows: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.blake2b(f"{seed}\0{int(row)}".encode("ascii"), digest_size=8).digest(),
                "big",
            )
            for row in source_rows
        ],
        dtype=np.uint64,
    )


@torch.no_grad()
def exact_search(
    vectors: np.ndarray,
    query_rows: np.ndarray,
    library_rows: np.ndarray,
    labels: np.ndarray,
    library_keys: np.ndarray,
    device: torch.device,
    block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(vectors[query_rows], dtype=np.float32)
    library = np.asarray(vectors[library_rows], dtype=np.float32)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    library /= np.maximum(np.linalg.norm(library, axis=1, keepdims=True), 1e-12)
    n = len(query)
    top_position = np.empty(n, dtype=np.int64)
    top_score = np.empty(n, dtype=np.float32)
    ranks = np.empty(n, dtype=np.int64)
    lib_tensor = torch.from_numpy(library).to(device)
    for start in range(0, n, block):
        end = min(start + block, n)
        scores = (torch.from_numpy(query[start:end]).to(device) @ lib_tensor.T).cpu().numpy()
        for local, row in enumerate(scores):
            global_index = start + local
            maximum = row.max()
            tied = np.flatnonzero(row == maximum)
            selected = int(tied[np.argmin(library_keys[tied])])
            gt_score = row[global_index]
            ranks[global_index] = 1 + int(np.sum(row > gt_score)) + int(
                np.sum((row == gt_score) & (library_keys < library_keys[global_index]))
            )
            top_position[global_index] = selected
            top_score[global_index] = maximum
        print(f"search {end:,}/{n:,}", flush=True)
    return ranks, top_position, top_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig3h-manifest", type=Path, required=True)
    parser.add_argument("--crossce-manifest", type=Path, required=True)
    parser.add_argument("--s2s-result", type=Path, required=True)
    parser.add_argument("--backbone-result", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pos", "neg"), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--tie-seed", type=int, default=20260824)
    args = parser.parse_args()

    protocol = np.load(args.crossce_manifest / f"protocol_{args.mode}.npz")
    query_rows = np.asarray(protocol["query_rows"], dtype=np.int64)
    library_rows = np.asarray(protocol["library_rows"], dtype=np.int64)
    query_source_rows = np.asarray(protocol["query_source_rows"], dtype=np.int64)
    library_source_rows = np.asarray(protocol["library_source_rows"], dtype=np.int64)
    with h5py.File(args.fig3h_manifest / "msnlib_fig3h_spectra.h5", "r") as h5:
        labels = np.asarray(h5[args.mode]["smiles_index"], dtype=np.int64)
    smiles = json.loads((args.fig3h_manifest / f"smiles_{args.mode}.json").read_text())

    if args.method in {"ultra", "dreams"}:
        vector_path = (
            args.backbone_result / args.method / "seed_0" /
            f"all_vectors_{args.mode}_float32.npy"
        )
    else:
        vector_path = (
            args.s2s_result / "search" / args.method / args.mode / "seed_0" /
            "all_vectors_float32.npy"
        )
    vectors = np.load(vector_path, mmap_mode="r")
    if args.method in {"ultra", "dreams"}:
        # Existing backbone embeddings are row-aligned to the extended-adduct
        # manifest, not to the strict-adduct H5.  Resolve exact rows by the
        # immutable MSnLib source-row identifier before taking any vectors.
        with h5py.File(args.fig3h_manifest / "msnlib_fig3h_spectra.h5", "r") as strict_h5:
            strict_source = np.asarray(strict_h5[args.mode]["source_row"], dtype=np.int64)
        with h5py.File(args.backbone_manifest / "msnlib_fig3h_spectra.h5", "r") as extended_h5:
            extended_source = np.asarray(extended_h5[args.mode]["source_row"], dtype=np.int64)
        position = {int(source_row): index for index, source_row in enumerate(extended_source)}
        try:
            strict_to_extended = np.asarray(
                [position[int(source_row)] for source_row in strict_source], dtype=np.int64
            )
        except KeyError as exc:
            raise RuntimeError(f"strict source row absent from backbone manifest: {exc}") from exc
        vectors = np.asarray(vectors[strict_to_extended], dtype=np.float32)
    else:
        if len(vectors) != len(labels):
            raise RuntimeError(f"simple-control vector rows {len(vectors)} != {len(labels)}")
    keys = tie_keys(library_source_rows, args.tie_seed)
    ranks, top_positions, top_scores = exact_search(
        vectors, query_rows, library_rows, labels, keys, torch.device(args.device), args.block
    )
    top_labels = labels[library_rows[top_positions]]
    correct = top_labels == labels[query_rows]
    output = args.output_dir / args.method / args.mode / "seed_0"
    output.mkdir(parents=True, exist_ok=True)
    retrieval_path = output / "retrieval.npz"
    np.savez_compressed(
        retrieval_path,
        query_rows=query_rows,
        library_rows=library_rows,
        query_source_rows=query_source_rows,
        library_source_rows=library_source_rows,
        ranks=ranks,
        top_positions=top_positions,
        top_scores=top_scores,
        top1_correct=correct,
    )
    records = [
        {
            "smiles": smiles[int(labels[query_rows[index]])],
            "query_source_row": int(query_source_rows[index]),
            "library_source_row": int(library_source_rows[index]),
            "rank": int(ranks[index]),
            "top1_correct": bool(correct[index]),
            "top1_smiles": smiles[int(top_labels[index])],
            "top1_score": float(top_scores[index]),
        }
        for index in range(len(query_rows))
    ]
    per_sample_path = output / "per_sample.json"
    per_sample_path.write_text(json.dumps(records) + "\n")
    summary = {
        "status": "complete",
        "method": METHODS[args.method],
        "method_key": args.method,
        "mode": args.mode,
        "n_query": len(query_rows),
        "n_library": len(library_rows),
        "hit_at_1": float(correct.mean()),
        "hit_at_5": float((ranks <= 5).mean()),
        "hit_at_10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "training_supervision": (
            "frozen Figure 3h spectrum encoder; simple controls use same-molecule "
            "spectrum-pair InfoNCE only; no ChemBERTa"
        ),
        "vector_sha256": sha256_file(vector_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "per_sample_sha256": sha256_file(per_sample_path),
    }
    (output / "DONE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
