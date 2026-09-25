#!/usr/bin/env python3
"""Evaluate cached UltraMS/DreaMS library-search embeddings by query adduct.

This replays the exact composite-polarity query construction in
``train/comparison/08_libsearch.py`` while retaining the query adduct label.
It then performs exact full-library cosine top-1 search and constructs
tie-aware empirical-FDR frontiers separately for the non-primary adducts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SEED = 42
QUERY_SEED = SEED + 77
TARGET_FDR = 0.05
MIN_ACCEPTED = 10
PRIMARY_ADDUCTS = {"[M+H]+", "[M-H]-"}
POSITIVE_ADDUCTS = {"[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+"}
NEGATIVE_ADDUCTS = {"[M-H]-", "[M+Cl]-", "[M+FA]-", "[M+FA-H]-", "[M+CH3COO]-"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def valid_spectrum(mzs_value: object, intensities_value: object) -> bool:
    try:
        mz = np.fromstring(str(mzs_value), dtype=np.float32, sep=",")
        intensity = np.fromstring(str(intensities_value), dtype=np.float32, sep=",")
    except Exception:
        return False
    valid = mz > 0
    mz, intensity = mz[valid], intensity[valid]
    return len(mz) >= 3 and len(intensity) == len(mz) and bool(intensity.max() > 0)


def replay_query_metadata(csv_path: Path, mode: str) -> tuple[pd.DataFrame, list[str]]:
    keep = POSITIVE_ADDUCTS if mode == "pos" else NEGATIVE_ADDUCTS
    frame = pd.read_csv(
        csv_path,
        usecols=["mzs", "intensities", "smiles", "precursor_mz", "adduct"],
    )
    frame = frame[frame.adduct.isin(keep)].reset_index().rename(columns={"index": "source_row"})
    by_smiles: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        smiles = str(row.smiles).strip()
        if smiles in {"", "nan"} or not valid_spectrum(row.mzs, row.intensities):
            continue
        by_smiles[smiles].append(
            {
                "source_row": int(row.source_row),
                "smiles": smiles,
                "adduct": str(row.adduct),
                "precursor_mz": float(row.precursor_mz),
            }
        )
    by_smiles = {smiles: rows for smiles, rows in by_smiles.items() if len(rows) >= 2}
    smiles = sorted(by_smiles)
    rng_split = random.Random(SEED)
    rng_split.shuffle(smiles)
    n_test = int(len(smiles) * 0.15)
    test_smiles = smiles[:n_test]
    rng_query = random.Random(QUERY_SEED)
    query_rows = [rng_query.choice(by_smiles[smiles]) for smiles in sorted(test_smiles)]
    chosen_source_rows = {int(row["source_row"]) for row in query_rows}
    library_smiles: list[str] = []
    for smiles in sorted(by_smiles):
        for row in by_smiles[smiles]:
            if int(row["source_row"]) not in chosen_source_rows:
                library_smiles.append(smiles)
    return pd.DataFrame(query_rows), library_smiles


def exact_top1(
    query_embedding: np.ndarray,
    library_embedding: np.ndarray,
    query_smiles: list[str],
    library_smiles: list[str],
    device: str,
    query_block: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = torch.as_tensor(query_embedding, dtype=torch.float32)
    l = torch.as_tensor(library_embedding, dtype=torch.float32)
    q = torch.nn.functional.normalize(q, dim=1)
    l = torch.nn.functional.normalize(l, dim=1).to(device)
    library_smiles_array = np.asarray(library_smiles, dtype=object)
    scores = np.empty(len(q), dtype=np.float32)
    indices = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), query_block):
        stop = min(start + query_block, len(q))
        similarity = q[start:stop].to(device) @ l.T
        block_scores, block_indices = similarity.max(dim=1)
        scores[start:stop] = block_scores.cpu().numpy()
        indices[start:stop] = block_indices.cpu().numpy()
    correct = library_smiles_array[indices] == np.asarray(query_smiles, dtype=object)
    return scores, indices, correct


def frontier(scores: np.ndarray, correct: np.ndarray, n_positive: int) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_correct = correct[order].astype(np.int64)
    rows = []
    accepted = 0
    cumulative_correct = 0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        accepted = stop
        cumulative_correct += int(ordered_correct[start:stop].sum())
        empirical_fdr = (
            (accepted - cumulative_correct) / accepted if accepted >= MIN_ACCEPTED else np.nan
        )
        rows.append(
            {
                "threshold": float(ordered_scores[start]),
                "accepted": int(accepted),
                "correct": int(cumulative_correct),
                "empirical_fdr": float(empirical_fdr),
                "identification_recall": cumulative_correct / n_positive if n_positive else 0.0,
            }
        )
        start = stop
    return pd.DataFrame(rows)


def working_point(frame: pd.DataFrame, target: float) -> dict[str, object]:
    valid = frame[np.isfinite(frame.empirical_fdr) & (frame.empirical_fdr <= target)]
    if valid.empty:
        return {
            "threshold": np.nan,
            "empirical_fdr": np.nan,
            "identification_recall_at_5pct_fdr": 0.0,
            "accepted": 0,
            "correct": 0,
        }
    row = valid.sort_values(
        ["identification_recall", "empirical_fdr", "threshold"],
        ascending=[False, True, False],
        kind="stable",
    ).iloc[0]
    return {
        "threshold": float(row.threshold),
        "empirical_fdr": float(row.empirical_fdr),
        "identification_recall_at_5pct_fdr": float(row.identification_recall),
        "accepted": int(row.accepted),
        "correct": int(row.correct),
    }


def mean_low_fdr_recall(frame: pd.DataFrame, target: float) -> float:
    fdr = frame.empirical_fdr.to_numpy(dtype=float)
    recall = frame.identification_recall.to_numpy(dtype=float)
    valid = np.isfinite(fdr) & (fdr >= 0) & (fdr <= target)
    breaks = np.unique(np.r_[0.0, fdr[valid], target])
    breaks.sort()
    area = 0.0
    for left, right in zip(breaks[:-1], breaks[1:]):
        eligible = np.isfinite(fdr) & (fdr <= left)
        area += (right - left) * (float(recall[eligible].max()) if eligible.any() else 0.0)
    return area / target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-block", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_queries = []
    all_frontiers = []
    summaries = []
    audit: dict[str, object] = {
        "status": "complete",
        "protocol": "cached composite-polarity library search stratified by replayed query adduct",
        "split_seed": SEED,
        "query_seed": QUERY_SEED,
        "target_fdr": TARGET_FDR,
        "minimum_accepted": MIN_ACCEPTED,
        "csv": str(args.csv),
        "csv_sha256": sha256(args.csv),
        "modes": {},
    }
    for mode in ("pos", "neg"):
        query_meta, replayed_library_smiles = replay_query_metadata(args.csv, mode)
        saved_query_smiles = json.loads(
            (args.embedding_dir / f"emb_msnlib_{mode}_query_smis.json").read_text()
        )
        saved_library_smiles = json.loads(
            (args.embedding_dir / f"emb_msnlib_{mode}_lib_smis.json").read_text()
        )
        if query_meta.smiles.tolist() != saved_query_smiles:
            raise RuntimeError(f"{mode}: replayed query SMILES do not match cached embedding order")
        if replayed_library_smiles != saved_library_smiles:
            raise RuntimeError(f"{mode}: replayed library SMILES do not match cached embedding order")
        library_counts = Counter(saved_library_smiles)
        has_positive = query_meta.smiles.map(lambda value: library_counts[value] > 0).to_numpy()
        mode_audit = {
            "n_query": len(query_meta),
            "n_library": len(saved_library_smiles),
            "query_adduct_counts": query_meta.adduct.value_counts().sort_index().to_dict(),
            "zero_library_positive": int((~has_positive).sum()),
        }
        for method, file_token in (("UltraMS", "ultra"), ("DreaMS", "dreams")):
            q_path = args.embedding_dir / f"emb_msnlib_{mode}_query_{file_token}.npy"
            l_path = args.embedding_dir / f"emb_msnlib_{mode}_lib_{file_token}.npy"
            query_embedding = np.load(q_path, mmap_mode="r")
            library_embedding = np.load(l_path, mmap_mode="r")
            if len(query_embedding) != len(query_meta) or len(library_embedding) != len(saved_library_smiles):
                raise RuntimeError(f"{method}/{mode}: embedding shape does not match replayed metadata")
            scores, indices, correct = exact_top1(
                query_embedding,
                library_embedding,
                saved_query_smiles,
                saved_library_smiles,
                args.device,
                args.query_block,
            )
            per_query = query_meta.copy()
            per_query["mode"] = mode
            per_query["method"] = method
            per_query["has_library_positive"] = has_positive
            per_query["top1_score"] = scores
            per_query["top1_library_index"] = indices
            per_query["top1_correct"] = correct
            all_queries.append(per_query)
            for adduct, group in per_query.groupby("adduct", sort=True):
                if adduct in PRIMARY_ADDUCTS:
                    continue
                local_scores = group.top1_score.to_numpy(dtype=np.float32)
                local_correct = group.top1_correct.to_numpy(dtype=bool)
                n_positive = int(group.has_library_positive.sum())
                local_frontier = frontier(local_scores, local_correct, n_positive)
                local_frontier.insert(0, "method", method)
                local_frontier.insert(1, "mode", mode)
                local_frontier.insert(2, "adduct", adduct)
                all_frontiers.append(local_frontier)
                summaries.append(
                    {
                        "method": method,
                        "mode": mode,
                        "adduct": adduct,
                        "n_query": len(group),
                        "n_library_positive": n_positive,
                        "full_library_top1_accuracy": float(local_correct.mean()),
                        "mean_identification_recall_fdr_0_5pct": mean_low_fdr_recall(
                            local_frontier, TARGET_FDR
                        ),
                        **working_point(local_frontier, TARGET_FDR),
                    }
                )
        audit["modes"][mode] = mode_audit

    query_path = args.output_dir / "other_adduct_per_query.csv.gz"
    frontier_path = args.output_dir / "other_adduct_exact_fdr_frontiers.csv.gz"
    summary_path = args.output_dir / "other_adduct_low_fdr_summary.csv"
    pd.concat(all_queries, ignore_index=True).to_csv(query_path, index=False)
    pd.concat(all_frontiers, ignore_index=True).to_csv(frontier_path, index=False)
    summary = pd.DataFrame(summaries).sort_values(["mode", "adduct", "method"], kind="stable")
    summary.to_csv(summary_path, index=False)
    audit["outputs"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (query_path, frontier_path, summary_path)
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
