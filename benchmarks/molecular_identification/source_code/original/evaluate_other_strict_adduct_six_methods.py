#!/usr/bin/env python3
"""Evaluate six frozen encoders on five exact additional-adduct tasks.

UltraMS and DreaMS use their frozen composite-polarity library-search heads.
Linear, DeepSets, Fourier and Codebook use readouts trained on the exact same
composite-polarity train compounds, which includes the corresponding additional
adduct spectra.  Every exact-adduct test task reuses the frozen composite test
compound identities and performs exact full-library cosine top-1 search.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from evaluate_other_adduct_low_fdr import (
    MIN_ACCEPTED,
    NEGATIVE_ADDUCTS,
    POSITIVE_ADDUCTS,
    PRIMARY_ADDUCTS,
    QUERY_SEED,
    SEED,
    TARGET_FDR,
    exact_top1,
    frontier,
    mean_low_fdr_recall,
    sha256,
    working_point,
)
from evaluate_other_strict_adduct_low_fdr import replay_composite_metadata


READOUTS = {
    "Linear": "linear",
    "DeepSets": "deepsets",
    "Fourier": "fourier_projection",
    "Codebook": "ultrams_codebook",
}


def build_task(
    source_rows: np.ndarray,
    smiles: np.ndarray,
    adducts: np.ndarray,
    included_rows: np.ndarray,
    frozen_test_smiles: set[str],
    adduct: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    local_rows = included_rows[adducts[included_rows] == adduct]
    table = pd.DataFrame({
        "local_row": local_rows,
        "source_row": source_rows[local_rows],
        "smiles": smiles[local_rows],
        "adduct": adduct,
    })
    groups = {
        smi: group.sort_values("source_row", kind="stable")
        for smi, group in table.groupby("smiles", sort=True)
    }
    test_smiles = sorted(
        smi for smi in frozen_test_smiles
        if smi in groups and len(groups[smi]) >= 2
    )
    rng = random.Random(QUERY_SEED)
    query_local_rows = np.asarray([
        rng.choice(groups[smi].local_row.tolist()) for smi in test_smiles
    ], dtype=np.int64)
    chosen = set(query_local_rows.tolist())
    query = (
        table.set_index("local_row").loc[query_local_rows].reset_index()
        if len(query_local_rows) else table.iloc[:0].copy()
    )
    library = table[~table.local_row.isin(chosen)].copy()
    library = library.sort_values(["smiles", "source_row"], kind="stable").reset_index(drop=True)
    return (
        query,
        library,
        query.local_row.to_numpy(dtype=np.int64),
        library.local_row.to_numpy(dtype=np.int64),
    )


def load_backbone_vectors(csv_path: Path, embedding_dir: Path, mode: str, token: str):
    cached_query, cached_library = replay_composite_metadata(csv_path, mode)
    saved_query_smiles = json.loads((embedding_dir / f"emb_msnlib_{mode}_query_smis.json").read_text())
    saved_library_smiles = json.loads((embedding_dir / f"emb_msnlib_{mode}_lib_smis.json").read_text())
    if cached_query.smiles.tolist() != saved_query_smiles:
        raise RuntimeError(f"{mode}: frozen query identity mismatch")
    if cached_library.smiles.tolist() != saved_library_smiles:
        raise RuntimeError(f"{mode}: frozen library identity mismatch")
    q = np.load(embedding_dir / f"emb_msnlib_{mode}_query_{token}.npy", mmap_mode="r")
    l = np.load(embedding_dir / f"emb_msnlib_{mode}_lib_{token}.npy", mmap_mode="r")
    rows = np.r_[cached_query.source_row.to_numpy(np.int64), cached_library.source_row.to_numpy(np.int64)]
    vectors = np.concatenate([np.asarray(q), np.asarray(l)], axis=0)
    return {int(row): idx for idx, row in enumerate(rows)}, vectors


def assert_task_regression(task_rows: pd.DataFrame, reference_path: Path) -> dict[str, object]:
    reference = pd.read_csv(reference_path)
    keys = ["mode", "adduct", "role", "source_row", "smiles"]
    left = task_rows[keys].sort_values(keys[:-1], kind="stable").reset_index(drop=True)
    right = reference[keys].sort_values(keys[:-1], kind="stable").reset_index(drop=True)
    if not left.equals(right):
        merged = left.merge(right, on=keys, how="outer", indicator=True)
        raise RuntimeError(f"frozen task-row regression failed: {merged._merge.value_counts().to_dict()}")
    return {"status": "exact_match", "reference": str(reference_path), "reference_sha256": sha256(reference_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--readout-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-task-rows", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-block", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    all_per_query: list[pd.DataFrame] = []
    all_frontiers: list[pd.DataFrame] = []
    all_task_rows: list[pd.DataFrame] = []
    audit: dict[str, object] = {
        "status": "complete",
        "protocol": "matched composite-polarity training; frozen exact-adduct held-out test full-library search",
        "split_seed": SEED,
        "query_seed": QUERY_SEED,
        "target_fdr": TARGET_FDR,
        "minimum_accepted": MIN_ACCEPTED,
        "csv_sha256": sha256(args.csv),
        "methods": ["DeepSets", "Codebook", "Fourier", "Linear", "DreaMS", "UltraMS"],
        "modes": {},
    }

    with h5py.File(args.manifest_dir / "msnlib_fig3h_spectra.h5", "r") as h5:
        for mode in ("pos", "neg"):
            protocol = np.load(args.manifest_dir / f"protocol_{mode}.npz")
            group = h5[mode]
            source_rows = np.asarray(group["source_row"], dtype=np.int64)
            label_ids = np.asarray(group["smiles_index"], dtype=np.int64)
            adduct_ids = np.asarray(group["adduct_index"], dtype=np.int64)
            vocab = np.asarray(json.loads((args.manifest_dir / f"smiles_{mode}.json").read_text()), dtype=object)
            adduct_vocab = np.asarray(json.loads((args.manifest_dir / f"adducts_{mode}.json").read_text()), dtype=object)
            smiles = vocab[label_ids]
            adducts = adduct_vocab[adduct_ids]
            test_query_rows = np.asarray(protocol["test_query_rows"], dtype=np.int64)
            test_library_rows = np.asarray(protocol["test_library_rows"], dtype=np.int64)
            train_rows = np.asarray(protocol["train_rows"], dtype=np.int64)
            included_rows = np.unique(np.r_[test_query_rows, test_library_rows])
            frozen_test_smiles = set(smiles[test_query_rows].tolist())
            training_counts = {
                adduct: {
                    "n_train_spectra": int(np.sum(adducts[train_rows] == adduct)),
                    "n_train_compounds": int(len(set(smiles[train_rows][adducts[train_rows] == adduct].tolist()))),
                }
                for adduct in sorted(set(adducts[train_rows].tolist()))
            }
            mode_audit = {
                "n_train_spectra": int(len(train_rows)),
                "n_train_compounds": int(len(set(smiles[train_rows].tolist()))),
                "corresponding_adduct_training_counts": training_counts,
                "tasks": {},
            }

            ud = {
                "UltraMS": load_backbone_vectors(args.csv, args.embedding_dir, mode, "ultra"),
                "DreaMS": load_backbone_vectors(args.csv, args.embedding_dir, mode, "dreams"),
            }
            readout_vectors = {
                display: np.load(
                    args.readout_embedding_dir / key / mode / "seed_0" / "all_vectors_float32.npy",
                    mmap_mode="r",
                )
                for display, key in READOUTS.items()
            }
            for display, vectors in readout_vectors.items():
                if len(vectors) != len(source_rows):
                    raise RuntimeError(f"{display}/{mode}: vector count mismatch")

            target_adducts = sorted((POSITIVE_ADDUCTS if mode == "pos" else NEGATIVE_ADDUCTS) - PRIMARY_ADDUCTS)
            for adduct in target_adducts:
                query, library, query_local, library_local = build_task(
                    source_rows, smiles, adducts, included_rows, frozen_test_smiles, adduct
                )
                if len(query) < MIN_ACCEPTED or library.empty:
                    continue
                all_task_rows.extend([
                    pd.DataFrame({"mode": mode, "adduct": adduct, "role": "query", "source_row": query.source_row, "smiles": query.smiles}),
                    pd.DataFrame({"mode": mode, "adduct": adduct, "role": "library", "source_row": library.source_row, "smiles": library.smiles}),
                ])
                library_counts = Counter(library.smiles)
                has_positive = query.smiles.map(lambda smi: library_counts[smi] > 0).to_numpy()
                method_arrays = {}
                for method, (row_map, vectors) in ud.items():
                    qidx = np.asarray([row_map[int(row)] for row in query.source_row], dtype=np.int64)
                    lidx = np.asarray([row_map[int(row)] for row in library.source_row], dtype=np.int64)
                    method_arrays[method] = (vectors[qidx], vectors[lidx])
                for method, vectors in readout_vectors.items():
                    method_arrays[method] = (vectors[query_local], vectors[library_local])

                for method, (query_vectors, library_vectors) in method_arrays.items():
                    scores, top1_indices, correct = exact_top1(
                        query_vectors, library_vectors, query.smiles.tolist(), library.smiles.tolist(),
                        args.device, args.query_block,
                    )
                    per_query = query.copy()
                    per_query["mode"] = mode
                    per_query["method"] = method
                    per_query["has_library_positive"] = has_positive
                    per_query["top1_score"] = scores
                    per_query["top1_library_source_row"] = library.source_row.to_numpy()[top1_indices]
                    per_query["top1_correct"] = correct
                    all_per_query.append(per_query)
                    n_positive = int(has_positive.sum())
                    local_frontier = frontier(scores, correct, n_positive)
                    local_frontier.insert(0, "method", method)
                    local_frontier.insert(1, "mode", mode)
                    local_frontier.insert(2, "adduct", adduct)
                    all_frontiers.append(local_frontier)
                    all_summaries.append({
                        "method": method,
                        "mode": mode,
                        "adduct": adduct,
                        "n_query": len(query),
                        "n_library": len(library),
                        "n_library_positive": n_positive,
                        "n_corresponding_train_spectra": training_counts.get(adduct, {}).get("n_train_spectra", 0),
                        "n_corresponding_train_compounds": training_counts.get(adduct, {}).get("n_train_compounds", 0),
                        "full_library_top1_accuracy": float(correct.mean()),
                        "mean_identification_recall_fdr_0_5pct": mean_low_fdr_recall(local_frontier, TARGET_FDR),
                        **working_point(local_frontier, TARGET_FDR),
                    })
                mode_audit["tasks"][adduct] = {
                    "n_query": len(query),
                    "n_library": len(library),
                    "n_library_positive": int(has_positive.sum()),
                    **training_counts.get(adduct, {}),
                }
            audit["modes"][mode] = mode_audit

    task_rows = pd.concat(all_task_rows, ignore_index=True).drop_duplicates(
        ["mode", "adduct", "role", "source_row"], keep="first"
    )
    audit["task_identity_regression"] = assert_task_regression(task_rows, args.reference_task_rows)
    outputs = {
        "other_strict_adduct_per_query.csv.gz": pd.concat(all_per_query, ignore_index=True),
        "other_strict_adduct_exact_fdr_frontiers.csv.gz": pd.concat(all_frontiers, ignore_index=True),
        "other_strict_adduct_low_fdr_summary.csv": pd.DataFrame(all_summaries).sort_values(
            ["mode", "adduct", "method"], kind="stable"
        ),
        "other_strict_adduct_task_rows.csv.gz": task_rows,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)
    audit["outputs"] = {
        name: {"sha256": sha256(args.output_dir / name), "rows": len(table)}
        for name, table in outputs.items()
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(outputs["other_strict_adduct_low_fdr_summary.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
