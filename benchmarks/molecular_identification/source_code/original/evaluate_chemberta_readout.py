#!/usr/bin/env python3
"""Exact joint evaluation of matched ChemBERTa-readout Figure 3 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from baseline_common import atomic_write_json, rank_metrics, sha256_file
from train_chemberta_readout import MODEL_NAMES, make_model, model_inputs, set_determinism


DISPLAY_NAMES = {
    "linear": "Linear",
    "binned_ffn": "Binned FFN",
    "deepsets": "DeepSets",
    "deepsets_fourier": "DeepSets + Fourier",
    "shallow_deepsets": "Shallow DeepSets",
    "fourier_projection": "Fourier projection",
    "ultrams_codebook": "UltraMS codebook",
}
EXPECTED_QUERY_SHA = "4cec8e332cb5cf5ba24ce6994795182fd5cdaceaef1c65c3dc674234db0413df"
EXPECTED_CANDIDATE_COUNT_SHA = "0590af48d96981c9c11df3f10cf3d20e09fd45d06952bb2270a45cbe67c0f05c"


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def stable_score_order(scores: np.ndarray) -> np.ndarray:
    """Descending score with frozen candidate position as the deterministic tie break."""
    return np.lexsort((np.arange(len(scores), dtype=np.int64), -scores))


@torch.no_grad()
def predict_fold(
    model: torch.nn.Module,
    name: str,
    bins: np.ndarray,
    peaks: np.ndarray,
    codebook: np.ndarray | None,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    force: bool,
) -> np.ndarray:
    if output_path.exists() and not force:
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape != (len(indices), 768) or existing.dtype != np.float32:
            raise RuntimeError(f"invalid existing prediction cache: {output_path}")
        return existing
    model.eval()
    output = np.lib.format.open_memmap(output_path, "w+", dtype=np.float32, shape=(len(indices), 768))
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        x = torch.from_numpy(model_inputs(name, bins, peaks, indices[start:end], codebook)).to(device)
        output[start:end] = F.normalize(model(x).float(), dim=1).cpu().numpy()
        if start == 0 or end % 10_000 < batch_size or end == len(indices):
            print(f"prediction method={name} rows={end:,}/{len(indices):,}", flush=True)
    output.flush()
    return np.load(output_path, mmap_mode="r")


def evaluate_fold_joint(
    *,
    fold: str,
    global_indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dirs: dict[str, Path],
    metadata: pd.DataFrame,
    candidates: dict[str, list[str]],
    molecule_cache: dict,
    device: torch.device,
    max_query_groups: int | None,
    reference_test_jsonl: str | None,
) -> dict[str, dict]:
    groups_by_smiles: defaultdict[str, list[int]] = defaultdict(list)
    for local_idx, global_idx in enumerate(global_indices):
        groups_by_smiles[str(metadata.iloc[int(global_idx)].smiles)].append(local_idx)
    groups = sorted(groups_by_smiles)
    if max_query_groups is not None:
        groups = groups[:max_query_groups]

    n_rows = len(global_indices)
    payloads = {method: [None] * n_rows for method in predictions}
    ranks = {method: np.zeros(n_rows, np.int64) for method in predictions}
    gt_scores = {method: np.full(n_rows, np.nan, np.float32) for method in predictions}
    top1_scores = {method: np.full(n_rows, np.nan, np.float32) for method in predictions}
    top20_scores = {method: np.full((n_rows, 20), np.nan, np.float32) for method in predictions}
    counts = np.zeros(n_rows, np.int64)
    tie_rows = {method: 0 for method in predictions}
    zero_rows = {method: 0 for method in predictions}
    started = time.time()
    processed = 0

    for group_number, gt_smiles in enumerate(groups, start=1):
        candidate_list = list(candidates[gt_smiles])
        if gt_smiles not in candidate_list:
            candidate_list.append(gt_smiles)
        missing = [smiles for smiles in candidate_list if smiles not in molecule_cache]
        if missing:
            raise RuntimeError(f"{len(missing)} candidates are absent from the frozen ChemBERTa cache")
        candidate_tensor = F.normalize(
            torch.stack([molecule_cache[smiles].float() for smiles in candidate_list]).to(device), dim=1
        )
        gt_position = candidate_list.index(gt_smiles)
        local_rows = np.asarray(groups_by_smiles[gt_smiles], dtype=np.int64)
        counts[local_rows] = len(candidate_list)

        for method, method_predictions in predictions.items():
            query_np = np.asarray(method_predictions[local_rows], dtype=np.float32)
            nonzero = np.linalg.norm(query_np, axis=1) > 0
            zero_rows[method] += int((~nonzero).sum())
            score_matrix = np.full((len(local_rows), len(candidate_list)), -1.0, np.float32)
            if nonzero.any():
                query_tensor = torch.from_numpy(query_np[nonzero]).to(device)
                score_matrix[nonzero] = (query_tensor @ candidate_tensor.T).cpu().numpy()
            for offset, local_idx in enumerate(local_rows):
                values = score_matrix[offset]
                ranked = stable_score_order(values)
                rank = int(np.where(ranked == gt_position)[0][0]) + 1
                top_position = int(ranked[0])
                maximum = float(values[top_position])
                n_top_ties = int(np.sum(values == maximum))
                tie_rows[method] += int(n_top_ties > 1)
                selected = ranked[:50]
                global_idx = int(global_indices[local_idx])
                row = metadata.iloc[global_idx]
                payloads[method][int(local_idx)] = {
                    "sample_idx": global_idx,
                    "fold": fold,
                    "smiles": gt_smiles,
                    "adduct": None if pd.isna(row.adduct) else str(row.adduct),
                    "precursor_type": None,
                    "ionmode": None,
                    "precursor_mz": float(row.precursor_mz),
                    "collision_energy": None if pd.isna(row.collision_energy) else float(row.collision_energy),
                    "instrument_type": None,
                    "spectrum_id": None,
                    "rank": rank,
                    "n_candidates": len(candidate_list),
                    "gt_score": float(values[gt_position]),
                    "top1_smiles": candidate_list[top_position],
                    "top1_score": maximum,
                    "top_candidates": [candidate_list[int(position)] for position in selected],
                    "top_scores": [float(values[int(position)]) for position in selected],
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "score_semantics": "spectrum-encoder-to-ChemBERTa cosine",
                    "zero_spectrum_embedding": int(not nonzero[offset]),
                    "tie_break_rule": "score descending, then frozen candidate position ascending",
                    "n_top_score_ties": n_top_ties,
                }
                ranks[method][int(local_idx)] = rank
                gt_scores[method][int(local_idx)] = values[gt_position]
                top1_scores[method][int(local_idx)] = maximum
                top20_scores[method][int(local_idx), : min(20, len(ranked))] = values[ranked[:20]]
        processed += len(local_rows)
        if group_number == 1 or group_number % 100 == 0 or group_number == len(groups):
            print(
                f"fold={fold} groups={group_number:,}/{len(groups):,} queries={processed:,}",
                flush=True,
            )

    summaries: dict[str, dict] = {}
    for method in predictions:
        selected = np.asarray([idx for idx, payload in enumerate(payloads[method]) if payload is not None], np.int64)
        selected_global = global_indices[selected]
        query_sha = array_sha256(selected_global.astype("<i8", copy=False))
        count_sha = array_sha256(counts[selected].astype("<i8", copy=False))
        regression = {
            "query_indices_sha256_i64le": query_sha,
            "candidate_counts_sha256_i64le": count_sha,
        }
        if fold == "test" and max_query_groups is None:
            if query_sha != EXPECTED_QUERY_SHA:
                raise RuntimeError(f"query order regression failed for {method}")
            if count_sha != EXPECTED_CANDIDATE_COUNT_SHA:
                raise RuntimeError(f"candidate-count regression failed for {method}")
            if reference_test_jsonl:
                with open(reference_test_jsonl) as reference:
                    reference_rows = list(reference)
                if len(reference_rows) != len(selected):
                    raise RuntimeError("reference JSONL row count differs")
                for row_number, (line, local_idx) in enumerate(zip(reference_rows, selected, strict=True)):
                    old = json.loads(line)
                    new = payloads[method][int(local_idx)]
                    identity = ("sample_idx", "smiles", "n_candidates")
                    if tuple(old[key] for key in identity) != tuple(new[key] for key in identity):
                        raise RuntimeError(f"reference mismatch for {method} at row {row_number}")
                regression["reference_test_jsonl_sha256"] = sha256_file(reference_test_jsonl)

        output = output_dirs[method]
        jsonl = output / f"per_spectrum_msnlib_mass_{method}_chemberta_{fold}.jsonl"
        with jsonl.open("w") as handle:
            for local_idx in selected:
                handle.write(json.dumps(payloads[method][int(local_idx)], separators=(",", ":")) + "\n")
        npz = output / f"raw_scores_msnlib_mass_{method}_chemberta_{fold}.npz"
        np.savez_compressed(
            npz,
            gt_sims=gt_scores[method][selected],
            top1_sims=top1_scores[method][selected],
            topn_sims=top20_scores[method][selected],
            ranks=ranks[method][selected],
            query_indices=selected_global,
            score_semantics=np.asarray("spectrum-encoder-to-ChemBERTa cosine"),
        )
        summary = {
            **rank_metrics(ranks[method][selected]),
            "fold": fold,
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "n_query_groups": len(groups),
            "n_zero_spectrum_embeddings": zero_rows[method],
            "n_rows_with_top_score_tie": tie_rows[method],
            "candidate_count": {
                "mean": float(counts[selected].mean()),
                "median": float(np.median(counts[selected])),
                "minimum": int(counts[selected].min()),
                "maximum": int(counts[selected].max()),
            },
            "frozen_order_regression": regression,
            "joint_runtime_seconds": time.time() - started,
            "artifacts": {
                jsonl.name: {"sha256": sha256_file(jsonl)},
                npz.name: {"sha256": sha256_file(npz)},
            },
        }
        atomic_write_json(output / f"metrics_{fold}.json", summary)
        summaries[method] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--chemberta-cache", required=True)
    parser.add_argument("--ffn-cache-dir", required=True)
    parser.add_argument("--peak-cache-dir", required=True)
    parser.add_argument("--codebook-cache-dir")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reference-test-jsonl")
    parser.add_argument("--max-query-groups", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("the preregistered main comparison uses seed 0 only")
    set_determinism(args.seed)

    run_dir = Path(args.run_dir)
    active_methods = []
    output_dirs: dict[str, Path] = {}
    for method in args.models:
        output = run_dir / method / f"seed_{args.seed}" / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "DONE.json").exists() and not args.force:
            print(f"already complete: {output}")
        else:
            active_methods.append(method)
            output_dirs[method] = output
    if not active_methods:
        return

    cache = Path(args.ffn_cache_dir)
    peak_cache = Path(args.peak_cache_dir)
    bins = np.load(cache / "ffn_bins_float16.npy", mmap_mode="r")
    peaks = np.load(peak_cache / "peaksets_float32.npy", mmap_mode="r")
    codebook = None
    if "ultrams_codebook" in active_methods:
        if not args.codebook_cache_dir:
            raise RuntimeError("--codebook-cache-dir is required for ultrams_codebook")
        codebook = np.load(
            Path(args.codebook_cache_dir) / "ultrams_codebook_pool_float32.npy", mmap_mode="r"
        )
        if codebook.shape != (len(bins), 1024) or codebook.dtype != np.float32:
            raise RuntimeError(f"invalid UltraMS codebook cache: {codebook.shape} {codebook.dtype}")
    folds = np.load(cache / "fold_codes_uint8.npy", mmap_mode="r")
    metadata = pd.read_csv(args.csv, usecols=["smiles", "precursor_mz", "adduct", "collision_energy", "fold"])
    if len(metadata) != len(folds):
        raise RuntimeError("metadata and frozen cache row counts differ")
    fold_indices = {"val": np.where(folds == 1)[0], "test": np.where(folds == 2)[0]}
    device = torch.device(args.device)

    predictions_by_fold: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    checkpoint_hashes: dict[str, str] = {}
    for method in active_methods:
        checkpoint_path = run_dir / method / f"seed_{args.seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = make_model(method, out_dim=768).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_hashes[method] = sha256_file(checkpoint_path)
        for fold, indices in fold_indices.items():
            prediction_path = output_dirs[method] / f"predicted_chemberta_{fold}_float32.npy"
            predictions_by_fold[fold][method] = predict_fold(
                model,
                method,
                bins,
                peaks,
                codebook,
                indices,
                device,
                args.prediction_batch_size,
                prediction_path,
                args.force,
            )
        del model, checkpoint
        torch.cuda.empty_cache()

    with open(args.candidates) as handle:
        candidates = json.load(handle)
    print(f"loading frozen ChemBERTa cache: {args.chemberta_cache}", flush=True)
    molecule_cache = torch.load(args.chemberta_cache, map_location="cpu", weights_only=False)
    fold_summaries: dict[str, dict[str, dict]] = {}
    for fold in ("val", "test"):
        fold_summaries[fold] = evaluate_fold_joint(
            fold=fold,
            global_indices=fold_indices[fold],
            predictions=predictions_by_fold[fold],
            output_dirs=output_dirs,
            metadata=metadata,
            candidates=candidates,
            molecule_cache=molecule_cache,
            device=device,
            max_query_groups=args.max_query_groups,
            reference_test_jsonl=args.reference_test_jsonl,
        )

    for method in active_methods:
        record = {
            "status": "complete",
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "seed": args.seed,
            "score_semantics": "spectrum-encoder-to-ChemBERTa cosine",
            "checkpoint_sha256": checkpoint_hashes[method],
            "candidate_manifest_sha256": sha256_file(args.candidates),
            "chemberta_cache_sha256": sha256_file(args.chemberta_cache),
            "summaries": {fold: fold_summaries[fold][method] for fold in ("val", "test")},
        }
        atomic_write_json(output_dirs[method] / "METHOD.json", record)
        atomic_write_json(output_dirs[method] / "DONE.json", record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
