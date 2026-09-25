#!/usr/bin/env python3
"""Score strict-adduct spectrum-library queries from trained encoder embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-block", type=int, default=128)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "original"))
    from baseline_common import atomic_write_json, sha256_lines

    manifest = json.loads((args.manifest_dir / "manifest_summary.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "modes": {}}
    for mode in ("pos", "neg"):
        query_smiles = json.loads((args.embedding_dir / f"emb_msnlib_{mode}_query_smis.json").read_text())
        library_smiles = json.loads((args.embedding_dir / f"emb_msnlib_{mode}_lib_smis.json").read_text())
        expected = manifest["modes"][mode]["test"]
        if sha256_lines(query_smiles) != expected["ordered_query_smiles_sha256"]:
            raise RuntimeError(f"{mode}: query identities differ from the strict-adduct manifest")
        if sha256_lines(library_smiles) != expected["ordered_library_smiles_sha256"]:
            raise RuntimeError(f"{mode}: library identities differ from the strict-adduct manifest")
        counts = Counter(library_smiles)
        has_positive = np.asarray([counts[smiles] > 0 for smiles in query_smiles], dtype=bool)
        summary["modes"][mode] = {}
        library_labels = np.asarray(library_smiles)
        query_labels = np.asarray(query_smiles)
        for backbone, display in (("ultra", "UltraMS"), ("dreams", "DreaMS")):
            query = np.load(args.embedding_dir / f"emb_msnlib_{mode}_query_{backbone}.npy")
            library = np.load(args.embedding_dir / f"emb_msnlib_{mode}_lib_{backbone}.npy")
            if len(query) != len(query_smiles) or len(library) != len(library_smiles):
                raise RuntimeError(f"{display} {mode}: embedding and identity lengths differ")
            query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            library = library / (np.linalg.norm(library, axis=1, keepdims=True) + 1e-8)
            top1_score = np.empty(len(query), dtype=np.float32)
            top1_correct = np.zeros(len(query), dtype=bool)
            for start in range(0, len(query), args.query_block):
                end = min(start + args.query_block, len(query))
                similarity = query[start:end] @ library.T
                best = np.argmax(similarity, axis=1)
                top1_score[start:end] = similarity[np.arange(end - start), best]
                top1_correct[start:end] = library_labels[best] == query_labels[start:end]
                print(f"{display} {mode}: {end:,}/{len(query):,}", flush=True)
            output = args.output_dir / f"top1_{backbone}_{mode}_test.npz"
            temporary = output.with_suffix(".part.npz")
            np.savez_compressed(temporary, top1_score=top1_score,
                                top1_correct=top1_correct, has_library_positive=has_positive)
            os.replace(temporary, output)
            summary["modes"][mode][display] = {
                "n_query": len(query), "n_library": len(library),
                "n_queries_with_library_positive": int(has_positive.sum()),
                "unthresholded_recall": float(top1_correct.sum() / has_positive.sum()),
                "scores": str(output),
            }
    atomic_write_json(args.output_dir / "search_summary.json", summary)


if __name__ == "__main__":
    main()
