#!/usr/bin/env python3
"""Shared, audit-oriented utilities for the Figure 3 simple baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


POSITIVE_ADDUCTS = frozenset(
    {"[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+H-H2O]+"}
)
NEGATIVE_ADDUCTS = frozenset(
    {"[M-H]-", "[M+Cl]-", "[M+FA]-", "[M+FA-H]-", "[M+CH3COO]-"}
)
FOLD_TO_CODE = {"train": 0, "val": 1, "test": 2}
CODE_TO_FOLD = {v: k for k, v in FOLD_TO_CODE.items()}


def sha256_file(path: str | os.PathLike[str], block_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    first = True
    for value in values:
        if not first:
            digest.update(b"\n")
        digest.update(str(value).encode("utf-8"))
        first = False
    return digest.hexdigest()


def stable_uint32(text: str, base_seed: int = 0) -> int:
    payload = f"{base_seed}\0{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def atomic_write_json(path: str | os.PathLike[str], payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_peak_strings(mzs_text: object, intensities_text: object) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.fromstring(str(mzs_text), dtype=np.float32, sep=",")
    intensities = np.fromstring(str(intensities_text), dtype=np.float32, sep=",")
    n = min(mzs.size, intensities.size)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mzs = mzs[:n]
    intensities = intensities[:n]
    valid = np.isfinite(mzs) & np.isfinite(intensities) & (mzs > 0) & (intensities >= 0)
    return mzs[valid], intensities[valid]


def fingerprint_ffn_bins(
    mzs: np.ndarray,
    intensities: np.ndarray,
    *,
    max_mz: int = 1005,
    bin_width: float = 1.0,
) -> np.ndarray:
    """MassSpecGym-style sum binning followed by maximum normalization."""
    n_bins = int(math.ceil(max_mz / bin_width))
    out = np.zeros(n_bins, dtype=np.float32)
    if mzs.size == 0:
        return out
    idx = np.floor(mzs / bin_width).astype(np.int64)
    keep = (idx >= 0) & (idx < n_bins) & (intensities > 0)
    if keep.any():
        np.add.at(out, idx[keep], intensities[keep])
        maximum = float(out.max())
        if maximum > 0:
            out /= maximum
    return out


def raw_binned_sparse_row(
    mzs: np.ndarray,
    intensities: np.ndarray,
    *,
    max_peaks: int = 150,
    bin_width: float = 0.02,
    max_mz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted sparse indices/values for the frozen raw-cosine protocol."""
    if mzs.size < 3 or intensities.size < 3:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    maximum = float(intensities.max(initial=0.0))
    if maximum <= 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    scaled = np.clip(intensities / maximum, 0.0, 1.0)
    if mzs.size > max_peaks:
        chosen = np.argpartition(scaled, -max_peaks)[-max_peaks:]
        mzs = mzs[chosen]
        scaled = scaled[chosen]
    idx = np.rint(mzs / bin_width).astype(np.int64)
    keep = (idx >= 0) & (scaled > 0)
    if max_mz is not None:
        keep &= idx < int(math.ceil(max_mz / bin_width))
    idx = idx[keep]
    vals = np.sqrt(scaled[keep], dtype=np.float32)
    if idx.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    unique_idx, first = np.unique(idx, return_index=True)
    summed = np.add.reduceat(vals, first).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(summed))
    if norm > 0:
        summed /= norm
    return unique_idx.astype(np.int32, copy=False), summed


def exact_tie_aware_fdr_curve(
    top1_score: Sequence[float],
    top1_correct: Sequence[bool],
    *,
    denominator: int,
    min_hits: int = 10,
) -> dict[str, np.ndarray]:
    """Accept equal-score observations as one block and sweep every observed score."""
    scores = np.asarray(top1_score, dtype=np.float64)
    correct = np.asarray(top1_correct, dtype=bool)
    if scores.ndim != 1 or scores.shape != correct.shape:
        raise ValueError("score/correct arrays must be aligned one-dimensional arrays")
    if denominator <= 0:
        raise ValueError("recall denominator must be positive")
    finite = np.isfinite(scores)
    scores, correct = scores[finite], correct[finite]
    order = np.argsort(-scores, kind="stable")
    scores, correct = scores[order], correct[order]
    if scores.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"threshold": empty, "accepted": empty.astype(np.int64), "correct": empty.astype(np.int64), "fdr": empty, "recall": empty}
    block_end = np.r_[scores[1:] != scores[:-1], True]
    accepted_all = np.arange(1, scores.size + 1, dtype=np.int64)
    correct_all = np.cumsum(correct, dtype=np.int64)
    accepted = accepted_all[block_end]
    n_correct = correct_all[block_end]
    thresholds = scores[block_end]
    fdr = np.full(accepted.shape, np.nan, dtype=np.float64)
    eligible = accepted >= int(min_hits)
    fdr[eligible] = (accepted[eligible] - n_correct[eligible]) / accepted[eligible]
    recall = n_correct.astype(np.float64) / float(denominator)
    return {
        "threshold": thresholds,
        "accepted": accepted,
        "correct": n_correct,
        "fdr": fdr,
        "recall": recall,
    }


def fdr_envelope(curve: dict[str, np.ndarray], *, maximum: float = 0.35, n_grid: int = 500) -> dict[str, np.ndarray]:
    fdr = np.asarray(curve["fdr"], dtype=np.float64)
    recall = np.asarray(curve["recall"], dtype=np.float64)
    valid = np.isfinite(fdr)
    grid = np.linspace(0.0, maximum, n_grid)
    envelope = np.zeros_like(grid)
    for idx, budget in enumerate(grid):
        eligible = valid & (fdr <= budget)
        if eligible.any():
            envelope[idx] = recall[eligible].max()
    return {"fdr_grid": grid, "recall_envelope": envelope}


def fdr_working_point(curve: dict[str, np.ndarray], target: float = 0.05) -> dict[str, float | int | None]:
    fdr = np.asarray(curve["fdr"], dtype=np.float64)
    recall = np.asarray(curve["recall"], dtype=np.float64)
    valid = np.isfinite(fdr) & (fdr <= target)
    if not valid.any():
        return {"threshold": None, "fdr": None, "recall": 0.0, "accepted": 0, "correct": 0}
    candidates = np.where(valid)[0]
    # Highest recall, then lower FDR, then stricter threshold.
    best = sorted(candidates, key=lambda i: (-recall[i], fdr[i], -curve["threshold"][i]))[0]
    return {
        "threshold": float(curve["threshold"][best]),
        "fdr": float(fdr[best]),
        "recall": float(recall[best]),
        "accepted": int(curve["accepted"][best]),
        "correct": int(curve["correct"][best]),
    }


def threshold_metrics(
    top1_score: Sequence[float],
    top1_correct: Sequence[bool],
    threshold: float,
    *,
    denominator: int,
) -> dict[str, float | int]:
    score = np.asarray(top1_score, dtype=np.float64)
    correct = np.asarray(top1_correct, dtype=bool)
    accepted = np.isfinite(score) & (score >= float(threshold))
    n_accepted = int(accepted.sum())
    n_correct = int((accepted & correct).sum())
    return {
        "threshold": float(threshold),
        "accepted": n_accepted,
        "correct": n_correct,
        "coverage": n_accepted / len(score) if len(score) else 0.0,
        "fdr": (n_accepted - n_correct) / n_accepted if n_accepted else float("nan"),
        "recall": n_correct / denominator if denominator else 0.0,
    }


def rank_metrics(ranks: Sequence[int], ks: Sequence[int] = (1, 5, 10, 20)) -> dict[str, float | int]:
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or (values < 1).any():
        raise ValueError("ranks must be a nonempty vector of positive integers")
    result: dict[str, float | int] = {"n_queries": int(values.size)}
    for k in ks:
        result[f"top{k}"] = float(np.mean(values <= k))
    result["mrr"] = float(np.mean(1.0 / values))
    result["median_rank"] = float(np.median(values))
    return result


def candidate_consensus(
    top_candidates: Sequence[Sequence[str]],
    top_scores: Sequence[Sequence[float]],
    *,
    depth: int = 50,
) -> list[str]:
    """Frozen hard-plurality ranking with normalized-Borda tie support."""
    votes: Counter[str] = Counter()
    support: defaultdict[str, float] = defaultdict(float)
    for candidates, _scores in zip(top_candidates, top_scores):
        row = list(candidates[:depth])
        if row:
            votes[row[0]] += 1
        denom = max(1, len(row))
        for rank0, candidate in enumerate(row):
            support[candidate] += (denom - rank0) / denom
    return sorted(support, key=lambda c: (-votes[c], -support[c], c))


def bootstrap_cluster_delta(
    baseline: Sequence[float],
    comparator: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260821,
) -> dict[str, object]:
    """Paired cluster bootstrap of the mean comparator-minus-baseline difference."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(comparator, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=object)
    if a.shape != b.shape or a.shape != clusters.shape:
        raise ValueError("paired arrays and cluster ids must align")
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sum = np.bincount(inverse, weights=b - a)
    cluster_n = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draws[idx] = cluster_sum[sampled].sum() / cluster_n[sampled].sum()
    return {
        "estimate": float(np.mean(b - a)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_clusters": int(len(unique)),
        "n_observations": int(len(a)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "draws": draws,
    }


def read_csv_rows(path: str | os.PathLike[str]) -> Iterable[dict[str, str]]:
    with open(path, newline="") as handle:
        yield from csv.DictReader(handle)


def split_compounds(smiles: Sequence[str], *, seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15) -> dict[str, list[str]]:
    ordered = sorted(smiles)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = int(len(ordered) * test_ratio)
    n_val = int(len(ordered) * val_ratio)
    return {
        "test": ordered[:n_test],
        "val": ordered[n_test : n_test + n_val],
        "train": ordered[n_test + n_val :],
    }


def choose_query_row_ids(
    row_ids_by_smiles: dict[str, list[int]],
    compounds: Sequence[str],
    *,
    seed: int = 119,
) -> tuple[list[int], list[str]]:
    rng = random.Random(seed)
    rows: list[int] = []
    smiles: list[str] = []
    for smi in sorted(compounds):
        rows.append(rng.choice(row_ids_by_smiles[smi]))
        smiles.append(smi)
    return rows, smiles


def build_library_row_ids(
    row_ids_by_smiles: dict[str, list[int]],
    excluded_query_rows: Sequence[int],
    *,
    allowed_compounds: set[str] | None = None,
) -> tuple[list[int], list[str]]:
    excluded = set(int(x) for x in excluded_query_rows)
    rows: list[int] = []
    smiles: list[str] = []
    for smi in sorted(row_ids_by_smiles):
        if allowed_compounds is not None and smi not in allowed_compounds:
            continue
        for row_id in row_ids_by_smiles[smi]:
            if row_id not in excluded:
                rows.append(row_id)
                smiles.append(smi)
    return rows, smiles
