#!/usr/bin/env python3
"""Exact low-FDR library search for non-primary strict MSnLib adducts.

The cached composite-polarity embeddings cover every eligible spectrum once
across the saved query and library arrays. This script reconstructs their
source-row order, freezes a separate compound-disjoint task for each exact
adduct, and evaluates UltraMS and DreaMS with the same tie-aware FDR evaluator
used for the primary strict [M+H]+ and [M-H]- panels.
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
    valid_spectrum,
    working_point,
)


def replay_composite_metadata(csv_path: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    chosen = {int(row["source_row"]) for row in query_rows}
    library_rows = [
        row
        for smiles in sorted(by_smiles)
        for row in by_smiles[smiles]
        if int(row["source_row"]) not in chosen
    ]
    return pd.DataFrame(query_rows), pd.DataFrame(library_rows)


def build_strict_task(
    all_rows: pd.DataFrame,
    adduct: str,
    frozen_composite_test_smiles: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = all_rows[all_rows.adduct == adduct].copy()
    groups = {
        smiles: group.sort_values("source_row", kind="stable")
        for smiles, group in frame.groupby("smiles", sort=True)
    }
    # Reuse the exact compound identities that were held out when the cached
    # polarity-specific task-adapted encoders were trained. Re-splitting the
    # exact-adduct subset here would leak many queries into the old train set.
    test_smiles = sorted(
        smiles
        for smiles in frozen_composite_test_smiles
        if smiles in groups and len(groups[smiles]) >= 2
    )
    rng_query = random.Random(QUERY_SEED)
    query_source_rows = []
    for smiles in sorted(test_smiles):
        rows = groups[smiles].source_row.tolist()
        query_source_rows.append(rng_query.choice(rows))
    query_source_rows_set = set(query_source_rows)
    query = (
        frame[frame.source_row.isin(query_source_rows_set)]
        .set_index("source_row")
        .loc[query_source_rows]
        .reset_index()
    )
    library = frame[~frame.source_row.isin(query_source_rows_set)].copy()
    library = library.sort_values(["smiles", "source_row"], kind="stable").reset_index(drop=True)
    return query, library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-block", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_queries = []
    frontiers = []
    task_rows = []
    audit: dict[str, object] = {
        "status": "complete",
        "protocol": "strict exact-adduct library search using the frozen composite-polarity held-out compound identities",
        "split_seed": SEED,
        "split_reuse": "query compounds are a subset of the exact composite-polarity test compounds used by the cached task-adapted encoders",
        "query_seed": QUERY_SEED,
        "target_fdr": TARGET_FDR,
        "minimum_accepted": MIN_ACCEPTED,
        "csv_sha256": sha256(args.csv),
        "tasks": {},
    }
    for mode in ("pos", "neg"):
        cached_query_rows, cached_library_rows = replay_composite_metadata(args.csv, mode)
        saved_query_smiles = json.loads(
            (args.embedding_dir / f"emb_msnlib_{mode}_query_smis.json").read_text()
        )
        saved_library_smiles = json.loads(
            (args.embedding_dir / f"emb_msnlib_{mode}_lib_smis.json").read_text()
        )
        if cached_query_rows.smiles.tolist() != saved_query_smiles:
            raise RuntimeError(f"{mode}: cached query replay mismatch")
        if cached_library_rows.smiles.tolist() != saved_library_smiles:
            raise RuntimeError(f"{mode}: cached library replay mismatch")
        all_rows = pd.concat([cached_query_rows, cached_library_rows], ignore_index=True)
        frozen_composite_test_smiles = set(cached_query_rows.smiles)
        if all_rows.source_row.duplicated().any():
            raise RuntimeError(f"{mode}: duplicate source rows in cached embedding coverage")
        for method, token in (("UltraMS", "ultra"), ("DreaMS", "dreams")):
            q_emb = np.load(args.embedding_dir / f"emb_msnlib_{mode}_query_{token}.npy", mmap_mode="r")
            l_emb = np.load(args.embedding_dir / f"emb_msnlib_{mode}_lib_{token}.npy", mmap_mode="r")
            if len(q_emb) != len(cached_query_rows) or len(l_emb) != len(cached_library_rows):
                raise RuntimeError(f"{method}/{mode}: cached embedding length mismatch")
            source_rows = np.r_[
                cached_query_rows.source_row.to_numpy(dtype=np.int64),
                cached_library_rows.source_row.to_numpy(dtype=np.int64),
            ]
            embeddings = np.concatenate([np.asarray(q_emb), np.asarray(l_emb)], axis=0)
            row_to_embedding = {int(row): index for index, row in enumerate(source_rows)}
            target_adducts = sorted((POSITIVE_ADDUCTS if mode == "pos" else NEGATIVE_ADDUCTS) - PRIMARY_ADDUCTS)
            for adduct in target_adducts:
                query, library = build_strict_task(
                    all_rows,
                    adduct,
                    frozen_composite_test_smiles,
                )
                if len(query) < MIN_ACCEPTED or library.empty:
                    continue
                query_indices = np.asarray([row_to_embedding[int(row)] for row in query.source_row], dtype=np.int64)
                library_indices = np.asarray([row_to_embedding[int(row)] for row in library.source_row], dtype=np.int64)
                query_embedding = embeddings[query_indices]
                library_embedding = embeddings[library_indices]
                library_counts = Counter(library.smiles)
                has_positive = query.smiles.map(lambda value: library_counts[value] > 0).to_numpy()
                scores, top1_indices, correct = exact_top1(
                    query_embedding,
                    library_embedding,
                    query.smiles.tolist(),
                    library.smiles.tolist(),
                    args.device,
                    args.query_block,
                )
                per_query = query.copy()
                per_query["mode"] = mode
                per_query["method"] = method
                per_query["has_library_positive"] = has_positive
                per_query["top1_score"] = scores
                per_query["top1_library_source_row"] = library.source_row.to_numpy()[top1_indices]
                per_query["top1_correct"] = correct
                per_queries.append(per_query)
                n_positive = int(has_positive.sum())
                local_frontier = frontier(scores, correct, n_positive)
                local_frontier.insert(0, "method", method)
                local_frontier.insert(1, "mode", mode)
                local_frontier.insert(2, "adduct", adduct)
                frontiers.append(local_frontier)
                summaries.append(
                    {
                        "method": method,
                        "mode": mode,
                        "adduct": adduct,
                        "n_query": len(query),
                        "n_library": len(library),
                        "n_library_positive": n_positive,
                        "full_library_top1_accuracy": float(correct.mean()),
                        "mean_identification_recall_fdr_0_5pct": mean_low_fdr_recall(
                            local_frontier, TARGET_FDR
                        ),
                        **working_point(local_frontier, TARGET_FDR),
                    }
                )
                task_rows.append(
                    pd.DataFrame(
                        {
                            "mode": mode,
                            "adduct": adduct,
                            "role": "query",
                            "source_row": query.source_row,
                            "smiles": query.smiles,
                        }
                    )
                )
                task_rows.append(
                    pd.DataFrame(
                        {
                            "mode": mode,
                            "adduct": adduct,
                            "role": "library",
                            "source_row": library.source_row,
                            "smiles": library.smiles,
                        }
                    )
                )
                audit["tasks"].setdefault(
                    adduct,
                    {
                        "mode": mode,
                        "n_query": len(query),
                        "n_library": len(library),
                        "n_library_positive": n_positive,
                    },
                )

    outputs = {
        "other_strict_adduct_per_query.csv.gz": pd.concat(per_queries, ignore_index=True),
        "other_strict_adduct_exact_fdr_frontiers.csv.gz": pd.concat(frontiers, ignore_index=True),
        "other_strict_adduct_low_fdr_summary.csv": pd.DataFrame(summaries).sort_values(
            ["mode", "adduct", "method"], kind="stable"
        ),
        "other_strict_adduct_task_rows.csv.gz": pd.concat(task_rows, ignore_index=True).drop_duplicates(
            ["mode", "adduct", "role", "source_row"], keep="first"
        ),
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
