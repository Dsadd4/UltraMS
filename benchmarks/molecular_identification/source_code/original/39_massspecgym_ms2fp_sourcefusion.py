"""MassSpecGym spectrum-to-molecule MS2FP source-fusion evaluation.

This extends the audited NPLIB1 MS2FP paradigm to MassSpecGym:

  spectrum representation -> Morgan fingerprint head -> candidate CE -> candidate ranking

The spectrum sources mirror the final NPLIB1 comparison:
  - DreaMS global embedding
  - DreaMS peak intensity-weighted token summary
  - UltraMS u5 projected CLS
  - UltraMS u5 projected CLS + peak mean/intensity-weighted token summary
  - UltraMS raw hard-negative checkpoint projected CLS + peak mean/intensity-weighted token summary

MassSpecGym-specific work is kept in this file: CSV loading, official formula/mass
candidate conversion to spec_id keys, data audit, and source fusion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(HERE))


def import_script(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ms2fp = import_script(HERE / "17_nplib1_ms2fp_retrieval.py", "msgym_ms2fp_base")
probe = ms2fp.probe
triplet34 = import_script(HERE / "34_nplib1_ms2fp_contrastive_model_compare.py", "msgym_triplet34")
triplet36 = import_script(HERE / "36_nplib1_ms2fp_triplet_token_summary.py", "msgym_triplet36")
token22 = import_script(HERE / "22_nplib1_token_summary_ms2fp.py", "msgym_token22")


OUT_BASE = TRAIN_DIR / "output" / "comparison" / "massspecgym_ms2fp_sourcefusion"
DATA_DIR = ROOT / "datasets" / "MassSpecGym"
DREAMS_EMBED_CKPT = TRAIN_DIR / "comparison" / "resources" / "DreaMS_Check" / "embedding_model.ckpt"

SCENARIOS = {
    "all": {"desc": "All MassSpecGym spectra with at least 3 peaks.", "min_peaks": 3},
    "min20": {"desc": "MassSpecGym spectra with at least 20 peaks.", "min_peaks": 20},
    "min50": {"desc": "MassSpecGym spectra with at least 50 peaks.", "min_peaks": 50},
    "hplus": {"desc": "Only [M+H]+ spectra with at least 3 peaks.", "min_peaks": 3, "adducts": {"[M+H]+"}},
    "na": {"desc": "Only [M+Na]+ spectra with at least 3 peaks.", "min_peaks": 3, "adducts": {"[M+Na]+"}},
}

DEFAULT_SOURCE_CONFIGS = {
    "dreams_embedding": {
        "kind": "dreams_embedding",
        "ckpt": str(DREAMS_EMBED_CKPT),
    },
    "dreams_peak_iw": {
        "kind": "dreams_token",
        "summary_parts": ["peak_iw_mean"],
    },
    "u5elr8_rank_src": {
        "kind": "ultra_triplet",
        "ckpt": str(TRAIN_DIR / "output" / "comparison" / "dreams_contrastive_ultrams_finetune"
                    / "u5_fp005_t01_elr8e6_e3_top100_seed3407_20260702e" / "best.pt"),
        "use_projection": True,
    },
    "u5elr8_proj_mean_iw": {
        "kind": "ultra_triplet_token",
        "ckpt": str(TRAIN_DIR / "output" / "comparison" / "dreams_contrastive_ultrams_finetune"
                    / "u5_fp005_t01_elr8e6_e3_top100_seed3407_20260702e" / "best.pt"),
        "summary_parts": ["peak_mean", "peak_iw_mean"],
        "include_projected_cls": True,
    },
    "hard4_raw_m010_best_projmean": {
        "kind": "ultra_triplet_token",
        "ckpt": str(TRAIN_DIR / "output" / "comparison" / "dreams_contrastive_ultrams_finetune"
                    / "raw_hard4_elr8e6_m010_e3_top100_seed3407_20260703h" / "best.pt"),
        "summary_parts": ["peak_mean", "peak_iw_mean"],
        "include_projected_cls": True,
    },
}

DEFAULT_ENSEMBLES = {
    "dreams_global_iw": ["dreams_embedding", "dreams_peak_iw"],
    "u5_cls_projmean": ["u5elr8_rank_src", "u5elr8_proj_mean_iw"],
    "u5_cls_projmean_rawproj": [
        "u5elr8_rank_src",
        "u5elr8_proj_mean_iw",
        "hard4_raw_m010_best_projmean",
    ],
}


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(":", "_").replace(" ", "_")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_spectrum(mzs_text, intens_text) -> np.ndarray:
    try:
        mz = np.asarray([float(x) for x in str(mzs_text).split(",") if str(x).strip()], dtype=np.float32)
        inten = np.asarray([float(x) for x in str(intens_text).split(",") if str(x).strip()], dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)
    n = min(len(mz), len(inten))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    mz = mz[:n]
    inten = inten[:n]
    mask = np.isfinite(mz) & np.isfinite(inten) & (mz > 0) & (inten > 0)
    if not mask.any():
        return np.zeros((0, 2), dtype=np.float32)
    spec = np.stack([mz[mask], inten[mask]], axis=-1)
    order = np.argsort(spec[:, 0], kind="mergesort")
    return spec[order].astype(np.float32)


def load_massspecgym(debug_load_limit_per_fold: int = 0):
    csv_path = DATA_DIR / "MassSpecGym.csv"
    usecols = [
        "identifier", "mzs", "intensities", "smiles", "inchikey", "formula",
        "precursor_formula", "parent_mass", "precursor_mz", "adduct",
        "instrument_type", "collision_energy", "fold", "simulation_challenge",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    if debug_load_limit_per_fold > 0:
        parts = []
        for fold, sub in df.groupby("fold", sort=False):
            parts.append(sub.head(debug_load_limit_per_fold))
        df = pd.concat(parts, axis=0).reset_index(drop=True)

    samples = []
    folds = []
    skipped = defaultdict(int)
    for row in df.itertuples(index=False):
        spec = parse_spectrum(row.mzs, row.intensities)
        try:
            precursor_mz = float(row.precursor_mz)
        except Exception:
            precursor_mz = 0.0
        if len(spec) == 0 or precursor_mz <= 0:
            skipped["bad_spectrum_or_precursor"] += 1
        above_parent = int(((spec[:, 0] > precursor_mz + 1.0) & (spec[:, 1] > 0)).sum()) if len(spec) else 0
        smi = str(row.smiles) if str(row.smiles) != "nan" else ""
        spec_id = str(row.identifier) if str(row.identifier) != "nan" else f"MassSpecGym:{len(samples)}"
        samples.append({
            "spec_id": spec_id,
            "parent_spec_id": str(row.inchikey) if str(row.inchikey) != "nan" else smi,
            "spectrum": spec,
            "precursor_mz": precursor_mz,
            "smiles": smi,
            "inchikey": str(row.inchikey) if str(row.inchikey) != "nan" else "",
            "formula": str(row.formula) if str(row.formula) != "nan" else "",
            "precursor_formula": str(row.precursor_formula) if str(row.precursor_formula) != "nan" else "",
            "parent_mass": float(row.parent_mass) if pd.notna(row.parent_mass) else np.nan,
            "adduct": str(row.adduct) if str(row.adduct) != "nan" else "",
            "instrument_type": str(row.instrument_type) if str(row.instrument_type) != "nan" else "",
            "collision_energy": float(row.collision_energy) if pd.notna(row.collision_energy) else np.nan,
            "simulation_challenge": bool(row.simulation_challenge),
            "block_type": "massspecgym",
            "n_peaks_model": int(len(spec)),
            "n_peaks_original": int(len(spec)),
            "n_peaks_above_precursor_plus1": above_parent,
            "fold": str(row.fold),
        })
        folds.append(str(row.fold))

    folds = np.asarray(folds)
    train_idx = np.where(folds == "train")[0].astype(np.int64)
    val_idx = np.where(folds == "val")[0].astype(np.int64)
    test_idx = np.where(folds == "test")[0].astype(np.int64)
    audit = audit_loaded_samples(samples, train_idx, val_idx, test_idx, csv_path, skipped)
    print(
        f"[MassSpecGym] samples={len(samples)} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)} skipped={dict(skipped)}"
    )
    return samples, train_idx, val_idx, test_idx, audit


def audit_loaded_samples(samples, train_idx, val_idx, test_idx, csv_path: Path, skipped: dict) -> dict:
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    out = {
        "csv": str(csv_path),
        "n_samples": int(len(samples)),
        "splits": {k: int(len(v)) for k, v in split_map.items()},
        "skipped": dict(skipped),
        "overlap": {},
        "peak_count": {},
        "adduct_counts": {},
        "instrument_counts": {},
    }
    for field in ("spec_id", "smiles", "inchikey"):
        sets = {
            k: {str(samples[int(i)].get(field, "")) for i in idx if str(samples[int(i)].get(field, ""))}
            for k, idx in split_map.items()
        }
        keys = list(split_map)
        out["overlap"][field] = {}
        for a_i, a in enumerate(keys):
            for b in keys[a_i + 1:]:
                out["overlap"][field][f"{a}-{b}"] = int(len(sets[a] & sets[b]))
    for name, idx in split_map.items():
        peaks = np.asarray([samples[int(i)]["n_peaks_original"] for i in idx], dtype=np.float64)
        out["peak_count"][name] = {
            "min": int(peaks.min()) if len(peaks) else 0,
            "p05": float(np.percentile(peaks, 5)) if len(peaks) else 0,
            "median": float(np.median(peaks)) if len(peaks) else 0,
            "p95": float(np.percentile(peaks, 95)) if len(peaks) else 0,
            "max": int(peaks.max()) if len(peaks) else 0,
            "ge20": int((peaks >= 20).sum()),
            "ge50": int((peaks >= 50).sum()),
        }
        adduct_counts = defaultdict(int)
        inst_counts = defaultdict(int)
        for i in idx:
            adduct_counts[samples[int(i)].get("adduct", "")] += 1
            inst_counts[samples[int(i)].get("instrument_type", "")] += 1
        out["adduct_counts"][name] = dict(sorted(adduct_counts.items()))
        out["instrument_counts"][name] = dict(sorted(inst_counts.items()))
    return out


def load_candidates(task: str, samples: list[dict]) -> tuple[dict, dict]:
    if task not in {"formula", "mass"}:
        raise ValueError(f"unknown task: {task}")
    path = DATA_DIR / f"MassSpecGym_candidates_{task}.json"
    raw = json.loads(path.read_text())
    candidates_by_spec = {}
    lens = []
    missing_key = 0
    gt_in = 0
    gt_missing = 0
    duplicate_rows = 0
    for s in samples:
        smi = s.get("smiles", "")
        cand = raw.get(smi)
        if cand is None:
            missing_key += 1
            continue
        seen = set()
        dedup = []
        for c in cand:
            c = str(c)
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        if len(dedup) != len(cand):
            duplicate_rows += 1
        if smi in dedup:
            gt_in += 1
        else:
            gt_missing += 1
        candidates_by_spec[s["spec_id"]] = dedup
        lens.append(len(dedup))
    arr = np.asarray(lens, dtype=np.float64)
    audit = {
        "task": task,
        "path": str(path),
        "raw_keys": int(len(raw)),
        "spec_keys": int(len(candidates_by_spec)),
        "missing_key": int(missing_key),
        "gt_in": int(gt_in),
        "gt_missing": int(gt_missing),
        "duplicate_rows": int(duplicate_rows),
        "candidate_count": {
            "min": int(arr.min()) if len(arr) else 0,
            "p05": float(np.percentile(arr, 5)) if len(arr) else 0,
            "median": float(np.median(arr)) if len(arr) else 0,
            "mean": float(arr.mean()) if len(arr) else 0,
            "p95": float(np.percentile(arr, 95)) if len(arr) else 0,
            "max": int(arr.max()) if len(arr) else 0,
        },
    }
    print(
        f"[candidates:{task}] spec_keys={audit['spec_keys']} gt_missing={gt_missing} "
        f"median={audit['candidate_count']['median']:.0f}"
    )
    return candidates_by_spec, audit


def scenario_mask(samples: list[dict], indices, scenario: dict) -> np.ndarray:
    min_peaks = int(scenario.get("min_peaks", 3))
    adducts = scenario.get("adducts")
    keep = []
    for i in indices:
        s = samples[int(i)]
        if int(s.get("n_peaks_model", len(s.get("spectrum", [])))) < min_peaks:
            continue
        if adducts and s.get("adduct", "") not in adducts:
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def maybe_limit_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(np.asarray(indices, dtype=np.int64), size=limit, replace=False))
    return chosen.astype(np.int64)


def collect_smiles_for_protocol(samples, train_idx, val_idx, test_idx, candidates_by_spec, include_train_candidates):
    smiles = set()
    smiles |= ms2fp.collect_smiles(samples, train_idx, candidates_by_spec if include_train_candidates else None)
    smiles |= ms2fp.collect_smiles(samples, val_idx, candidates_by_spec)
    smiles |= ms2fp.collect_smiles(samples, test_idx, candidates_by_spec)
    return smiles


def smiles_digest(smiles) -> str:
    h = hashlib.sha256()
    for sm in sorted(set(smiles)):
        h.update(str(sm).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def fingerprint_cache_path(out_dir: Path, task: str, scenario_name: str, smiles, args) -> Path:
    fp_dir = out_dir / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)
    digest = smiles_digest(smiles)
    stem = (
        f"fpdict_{safe_name(task)}_{safe_name(scenario_name)}"
        f"_bits{args.fp_bits}_r{args.radius}_chir{int(args.fp_use_chirality)}"
        f"_traincand{int(args.include_train_candidates_in_fp)}_{digest}"
    )
    return fp_dir / f"{stem}.pkl"


def build_or_load_fp_dict(out_dir: Path, task: str, scenario_name: str, smiles, args) -> tuple[dict, str]:
    smiles_list = sorted(set(smiles))
    path = fingerprint_cache_path(out_dir, task, scenario_name, smiles_list, args)
    if path.exists() and not args.force_fp:
        print(f"Loading cached Morgan fingerprints: {path} n_smiles={len(smiles_list)}", flush=True)
        with path.open("rb") as fh:
            fp_dict = pickle.load(fh)
        return fp_dict, str(path)

    print(
        f"Building Morgan fingerprints: task={task} scenario={scenario_name} "
        f"n_smiles={len(smiles_list)} cache={path}",
        flush=True,
    )
    fp_dict = {}
    for i, sm in enumerate(smiles_list, 1):
        fp_dict[sm] = ms2fp.morgan_fp(
            sm,
            n_bits=args.fp_bits,
            radius=args.radius,
            use_chirality=args.fp_use_chirality,
        )
        if i == 1 or i % 10000 == 0 or i == len(smiles_list):
            print(f"  Morgan fingerprints: {i}/{len(smiles_list)}", flush=True)
    with path.open("wb") as fh:
        pickle.dump(fp_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return fp_dict, str(path)


def build_ms2fp_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        fp_bits=args.fp_bits,
        radius=args.radius,
        fp_use_chirality=args.fp_use_chirality,
        tanimoto_loss_weight=args.tanimoto_loss_weight,
        score=args.score,
        epochs=args.epochs,
        val_every=args.val_every,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        hidden=args.hidden,
        dropout=args.dropout,
        pos_weight_clip=args.pos_weight_clip,
        calibrate_pos_weight=args.calibrate_pos_weight,
        candidate_ce_epochs=args.candidate_ce_epochs,
        candidate_ce_batch_size=args.candidate_ce_batch_size,
        candidate_ce_lr=args.candidate_ce_lr,
        candidate_ce_temperature=args.candidate_ce_temperature,
        balanced=args.balanced,
        seed=args.seed,
    )


def fingerprint_score_matrix(pred_mat: np.ndarray, cand_mat: np.ndarray, mode: str) -> np.ndarray:
    eps = 1e-6
    if mode in {"tanimoto", "soft_tanimoto"}:
        inter = pred_mat @ cand_mat.T
        denom = pred_mat.sum(axis=1, keepdims=True) + cand_mat.sum(axis=1)[None, :] - inter
        return inter / np.maximum(denom, 1e-8)
    if mode == "cosine":
        denom = np.linalg.norm(pred_mat, axis=1, keepdims=True) * np.linalg.norm(cand_mat, axis=1)[None, :]
        return (pred_mat @ cand_mat.T) / np.maximum(denom, 1e-8)
    if mode == "dot":
        return pred_mat @ cand_mat.T
    if mode == "pos_mean":
        return (pred_mat @ cand_mat.T) / np.maximum(cand_mat.sum(axis=1)[None, :], 1.0)
    if mode == "bernoulli":
        p = np.clip(pred_mat, eps, 1.0 - eps)
        log_p = np.log(p)
        log_not_p = np.log(1.0 - p)
        return log_p @ cand_mat.T + log_not_p @ (1.0 - cand_mat).T
    if mode == "hard_tanimoto":
        hard = (pred_mat >= 0.5).astype(np.float32)
        inter = hard @ cand_mat.T
        denom = hard.sum(axis=1, keepdims=True) + cand_mat.sum(axis=1)[None, :] - inter
        return inter / np.maximum(denom, 1e-8)
    raise ValueError(f"unknown score mode: {mode}")


def fast_evaluate(head, spec_emb, samples, eval_idx, candidates_by_spec, fp_dict, device, score_mode,
                  logit_shift=None):
    """MassSpecGym-scale replacement for ms2fp.evaluate with identical row fields.

    NPLIB1 evaluates each query independently. MassSpecGym has many spectra per
    molecule, so many queries share the same candidate list. This function groups
    those queries and scores them in matrices. The ranking protocol is unchanged.
    """
    skipped = defaultdict(int)
    items = []
    groups = defaultdict(list)
    cand_cache = {}

    for qi in eval_idx:
        s = samples[int(qi)]
        smi = s.get("smiles", "")
        spec_id = s["spec_id"]
        if not smi or smi == "nan":
            skipped["bad_smiles"] += 1
            continue
        cand = list(candidates_by_spec.get(spec_id, []))
        if not cand:
            skipped["no_candidates"] += 1
            continue
        if smi not in cand:
            cand.append(smi)
            skipped["gt_appended"] += 1
        seen = set()
        cand = [c for c in cand if not (c in seen or seen.add(c))]
        cand_raw_count = len(cand)
        cand = [c for c in cand if c in fp_dict and fp_dict[c].sum() > 0]
        if smi not in cand:
            skipped["missing_gt_fp"] += 1
            continue
        key = smi
        if key not in cand_cache:
            cand_cache[key] = {
                "cand": cand,
                "cand_raw_count": int(cand_raw_count),
                "cand_mat": np.stack([fp_dict[c] for c in cand]).astype(np.float32),
            }
        item = {
            "idx": int(qi),
            "key": key,
            "gt_pos": int(cand_cache[key]["cand"].index(smi)),
            "sample": s,
        }
        groups[key].append(len(items))
        items.append(item)

    if not items:
        return [], dict(skipped)

    head.eval()
    eval_indices = np.asarray([it["idx"] for it in items], dtype=np.int64)
    pred = np.zeros((len(items), next(iter(fp_dict.values())).shape[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(eval_indices), 2048):
            idx = eval_indices[start:start + 2048]
            q_np = np.asarray(spec_emb[idx], dtype=np.float32)
            q = torch.from_numpy(q_np).float().to(device)
            logits = head(q)
            if logit_shift is not None:
                logits = logits - logit_shift.to(device)
            pred[start:start + len(idx)] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    rows = []
    for key, item_positions in groups.items():
        cache = cand_cache[key]
        cand = cache["cand"]
        cand_mat = cache["cand_mat"]
        pred_mat = pred[np.asarray(item_positions, dtype=np.int64)]
        score_mat = fingerprint_score_matrix(pred_mat, cand_mat, score_mode)
        for local_i, item_pos in enumerate(item_positions):
            item = items[item_pos]
            scores = score_mat[local_i]
            order = np.argsort(-scores)
            gt_pos = item["gt_pos"]
            rank = int(np.where(order == gt_pos)[0][0] + 1)
            top_pos = int(order[0])
            s = item["sample"]
            rows.append({
                "spec_id": s["spec_id"],
                "parent_spec_id": s.get("parent_spec_id", ""),
                "smiles": s.get("smiles", ""),
                "block_type": s.get("block_type", ""),
                "adduct": s.get("adduct", ""),
                "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                "candidate_count": int(len(cand)),
                "candidate_count_raw": int(cache["cand_raw_count"]),
                "candidate_count_scored": int(len(cand)),
                "rank": rank,
                "hit1": int(rank <= 1),
                "hit5": int(rank <= 5),
                "hit10": int(rank <= 10),
                "hit20": int(rank <= 20),
                "rr": float(1.0 / rank),
                "gt_score": float(scores[gt_pos]),
                "top1_score": float(scores[top_pos]),
                "top1_is_gt": int(top_pos == gt_pos),
                "top1_smiles": cand[top_pos],
                "pred_on_mass": float(pred[item_pos].sum()),
            })
    return rows, dict(skipped)


def torch_fingerprint_score_matrix(pred_t: torch.Tensor, cand_t: torch.Tensor, mask_t: torch.Tensor,
                                   mode: str) -> torch.Tensor:
    eps = 1e-8
    if mode in {"tanimoto", "soft_tanimoto"}:
        inter = torch.einsum("bd,bcd->bc", pred_t, cand_t)
        denom = pred_t.sum(dim=1, keepdim=True) + cand_t.sum(dim=2) - inter
        scores = inter / denom.clamp_min(eps)
    elif mode == "cosine":
        inter = torch.einsum("bd,bcd->bc", pred_t, cand_t)
        denom = pred_t.norm(dim=1, keepdim=True) * cand_t.norm(dim=2)
        scores = inter / denom.clamp_min(eps)
    elif mode == "dot":
        scores = torch.einsum("bd,bcd->bc", pred_t, cand_t)
    elif mode == "pos_mean":
        inter = torch.einsum("bd,bcd->bc", pred_t, cand_t)
        scores = inter / cand_t.sum(dim=2).clamp_min(1.0)
    elif mode == "bernoulli":
        p = pred_t.clamp(1e-6, 1.0 - 1e-6)
        log_p = p.log()
        log_not_p = (1.0 - p).log()
        scores = (
            torch.einsum("bd,bcd->bc", log_p, cand_t)
            + torch.einsum("bd,bcd->bc", log_not_p, 1.0 - cand_t)
        )
    elif mode == "hard_tanimoto":
        hard = (pred_t >= 0.5).to(cand_t.dtype)
        inter = torch.einsum("bd,bcd->bc", hard, cand_t)
        denom = hard.sum(dim=1, keepdim=True) + cand_t.sum(dim=2) - inter
        scores = inter / denom.clamp_min(eps)
    else:
        raise ValueError(f"unknown score mode: {mode}")
    return scores.masked_fill(~mask_t, -1e9)


def gpu_evaluate(head, spec_emb, samples, eval_idx, candidates_by_spec, fp_dict, device, score_mode,
                 logit_shift=None):
    """GPU equivalent of fast_evaluate for MassSpecGym-scale validation/test.

    The protocol and row fields are intentionally identical to fast_evaluate;
    only the candidate fingerprint scoring is batched on GPU.
    """
    if not torch.cuda.is_available() or str(device).startswith("cpu"):
        return fast_evaluate(
            head, spec_emb, samples, eval_idx, candidates_by_spec, fp_dict, device,
            score_mode, logit_shift=logit_shift,
        )

    skipped = defaultdict(int)
    records = []
    for qi in eval_idx:
        s = samples[int(qi)]
        smi = s.get("smiles", "")
        spec_id = s["spec_id"]
        if not smi or smi == "nan":
            skipped["bad_smiles"] += 1
            continue
        cand = list(candidates_by_spec.get(spec_id, []))
        if not cand:
            skipped["no_candidates"] += 1
            continue
        if smi not in cand:
            cand.append(smi)
            skipped["gt_appended"] += 1
        seen = set()
        cand = [c for c in cand if not (c in seen or seen.add(c))]
        cand_raw_count = len(cand)
        cand = [c for c in cand if c in fp_dict and fp_dict[c].sum() > 0]
        if smi not in cand:
            skipped["missing_gt_fp"] += 1
            continue
        records.append({
            "idx": int(qi),
            "sample": s,
            "cand": cand,
            "gt_pos": int(cand.index(smi)),
            "cand_raw_count": int(cand_raw_count),
        })

    if not records:
        return [], dict(skipped)

    batch_size = int(os.environ.get("MASSGYM_EVAL_BATCH_SIZE", "64"))
    fp_dim = int(next(iter(fp_dict.values())).shape[0])
    rows = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            max_c = max(len(r["cand"]) for r in batch)
            idx_np = np.asarray([r["idx"] for r in batch], dtype=np.int64)
            cand_np = np.zeros((len(batch), max_c, fp_dim), dtype=np.float32)
            mask_np = np.zeros((len(batch), max_c), dtype=bool)
            for bi, rec in enumerate(batch):
                n_c = len(rec["cand"])
                cand_np[bi, :n_c] = np.stack([fp_dict[c] for c in rec["cand"]]).astype(np.float32)
                mask_np[bi, :n_c] = True

            q_np = np.asarray(spec_emb[idx_np], dtype=np.float32)
            q_t = torch.from_numpy(q_np).float().to(device)
            cand_t = torch.from_numpy(cand_np).to(device=device, dtype=torch.float32)
            mask_t = torch.from_numpy(mask_np).to(device)
            logits = head(q_t)
            if logit_shift is not None:
                logits = logits - logit_shift.to(device)
            pred_t = torch.sigmoid(logits)
            scores_t = torch_fingerprint_score_matrix(pred_t, cand_t, mask_t, score_mode)
            scores_np = scores_t.detach().cpu().numpy()
            pred_sum_np = pred_t.sum(dim=1).detach().cpu().numpy()

            for bi, rec in enumerate(batch):
                cand = rec["cand"]
                gt_pos = rec["gt_pos"]
                scores = scores_np[bi, :len(cand)]
                order = np.argsort(-scores)
                rank = int(np.where(order == gt_pos)[0][0] + 1)
                top_pos = int(order[0])
                s = rec["sample"]
                rows.append({
                    "spec_id": s["spec_id"],
                    "parent_spec_id": s.get("parent_spec_id", ""),
                    "smiles": s.get("smiles", ""),
                    "block_type": s.get("block_type", ""),
                    "adduct": s.get("adduct", ""),
                    "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                    "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                    "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                    "candidate_count": int(len(cand)),
                    "candidate_count_raw": int(rec["cand_raw_count"]),
                    "candidate_count_scored": int(len(cand)),
                    "rank": rank,
                    "hit1": int(rank <= 1),
                    "hit5": int(rank <= 5),
                    "hit10": int(rank <= 10),
                    "hit20": int(rank <= 20),
                    "rr": float(1.0 / rank),
                    "gt_score": float(scores[gt_pos]),
                    "top1_score": float(scores[top_pos]),
                    "top1_is_gt": int(top_pos == gt_pos),
                    "top1_smiles": cand[top_pos],
                    "pred_on_mass": float(pred_sum_np[bi]),
                })

            del q_t, cand_t, mask_t, logits, pred_t, scores_t
            if torch.cuda.is_available() and start % (batch_size * 40) == 0:
                torch.cuda.empty_cache()
    return rows, dict(skipped)


ms2fp.evaluate = gpu_evaluate if torch.cuda.is_available() else fast_evaluate


def source_feature_tag(source_key: str, cfg: dict) -> tuple[str, str | None]:
    kind = cfg["kind"]
    source_sha = None
    if cfg.get("ckpt"):
        source_sha = sha256_file(Path(cfg["ckpt"]))
    if kind == "dreams_embedding":
        tag = "dreams_embedding"
    elif kind == "dreams_token":
        tag = "dreams_token_" + "-".join(cfg.get("summary_parts", []))
    elif kind == "ultra_triplet":
        tag = "ultra_triplet_proj" if cfg.get("use_projection", True) else "ultra_triplet_raw"
    elif kind == "ultra_triplet_token":
        tag = "ultra_triplet_token_" + "-".join(cfg.get("summary_parts", []))
        if not cfg.get("include_projected_cls", True):
            tag += "_rawonly"
    else:
        raise ValueError(kind)
    return f"{safe_name(source_key)}_{safe_name(tag)}", source_sha


def feature_cache_paths(out_dir: Path, source_key: str, cfg: dict) -> tuple[Path, Path]:
    tag, source_sha = source_feature_tag(source_key, cfg)
    sha = source_sha[:12] if source_sha else "nosha"
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    stem = f"massspecgym_features_{tag}_{sha}"
    return emb_dir / f"{stem}.npy", emb_dir / f"{stem}.json"


def get_features(source_key: str, cfg: dict, out_dir: Path, samples: list[dict],
                 device: torch.device, args) -> tuple[np.ndarray, str, str, dict]:
    emb_path, meta_path = feature_cache_paths(out_dir, source_key, cfg)
    if emb_path.exists() and meta_path.exists() and not args.force_emb:
        emb = np.load(emb_path)
        meta = json.loads(meta_path.read_text())
        if emb.shape[0] != len(samples):
            raise RuntimeError(f"feature row mismatch for {emb_path}: {emb.shape[0]} != {len(samples)}")
        return emb, str(emb_path), str(meta_path), meta

    kind = cfg["kind"]
    print(f"Extracting MassSpecGym features: source={source_key} kind={kind}", flush=True)
    if kind == "dreams_embedding":
        emb = triplet34.extract_dreams_embedding_ckpt(
            samples,
            device,
            batch_size=args.embed_batch_size,
            ckpt_path=Path(cfg["ckpt"]),
        )
        meta = {"extractor": "dreams_embedding_checkpoint", "checkpoint": cfg["ckpt"]}
    elif kind == "dreams_token":
        emb, meta = token22.extract_dreams_token_summary(
            samples,
            device,
            batch_size=args.embed_batch_size,
            summary_parts=list(cfg.get("summary_parts", ["peak_iw_mean"])),
        )
    elif kind == "ultra_triplet":
        emb = triplet34.extract_ultra_triplet_embeddings(
            Path(cfg["ckpt"]),
            samples,
            device,
            batch_size=args.embed_batch_size,
            use_projection=bool(cfg.get("use_projection", True)),
        )
        meta = {
            "extractor": "ultrams_triplet_projected_cls",
            "checkpoint": cfg["ckpt"],
            "use_projection": bool(cfg.get("use_projection", True)),
        }
    elif kind == "ultra_triplet_token":
        emb, meta = triplet36.extract_ultra_triplet_token_summary(
            Path(cfg["ckpt"]),
            samples,
            device,
            batch_size=args.embed_batch_size,
            summary_parts=list(cfg.get("summary_parts", ["peak_mean", "peak_iw_mean"])),
            include_projected_cls=bool(cfg.get("include_projected_cls", True)),
        )
    else:
        raise ValueError(kind)

    if emb.shape[0] != len(samples):
        raise RuntimeError(f"extracted feature row mismatch for {source_key}: {emb.shape[0]} != {len(samples)}")
    emb = emb.astype(np.float32)
    np.save(emb_path, emb)
    source_sha = sha256_file(Path(cfg["ckpt"])) if cfg.get("ckpt") else None
    meta.update({
        "source_key": source_key,
        "source_config": cfg,
        "source_sha256": source_sha,
        "n_samples": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "path": str(emb_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=json_default) + "\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb, str(emb_path), str(meta_path), meta


def write_rows_extra(path: Path, rows, extra_fields=None) -> None:
    extra_fields = list(extra_fields or [])
    fields = [
        "candidate_task", "split", "scenario", "model", "phase", "spec_id",
        "parent_spec_id", "smiles", "block_type", "adduct", "n_peaks_model",
        "n_peaks_original", "n_peaks_above_precursor_plus1", "candidate_count",
        "candidate_count_raw", "candidate_count_scored", "rank", "hit1",
        "hit5", "hit10", "hit20", "rr", "gt_score", "top1_score",
        "top1_is_gt", "top1_smiles", "pred_on_mass", "n_members",
    ] + extra_fields
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def train_one_source(source_key: str, cfg: dict, out_dir: Path, samples, train_idx, val_idx, test_idx,
                     candidates_by_spec, fp_dict, device, args, task: str, scenario_name: str) -> dict:
    spec_emb, emb_path, meta_path, emb_meta = get_features(source_key, cfg, out_dir, samples, device, args)
    ms_args = build_ms2fp_args(args)
    print(
        f"\n[{source_key}] task={task} scenario={scenario_name} "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} dim={spec_emb.shape[1]}",
        flush=True,
    )
    head, best, history, fp_stats, logit_shift = ms2fp.train_head(
        spec_emb, samples, train_idx, val_idx, candidates_by_spec, fp_dict, ms_args, device
    )
    rows, skipped = ms2fp.evaluate(
        head, spec_emb, samples, test_idx, candidates_by_spec, fp_dict, device,
        args.score, logit_shift=logit_shift,
    )
    for row in rows:
        row.update({
            "candidate_task": task,
            "split": task,
            "scenario": scenario_name,
            "model": source_key,
            "phase": "test",
        })

    stem = f"{safe_name(task)}_{safe_name(scenario_name)}_{safe_name(source_key)}"
    row_path = out_dir / f"query_rows_{stem}.csv"
    hist_path = out_dir / f"history_{stem}.json"
    head_path = out_dir / f"ms2fp_head_{stem}_seed{args.seed}.pt"
    write_rows_extra(row_path, rows)
    hist_path.write_text(json.dumps(history, indent=2, sort_keys=True, default=json_default) + "\n")
    torch.save(head.state_dict(), head_path)
    metrics = {
        "block_micro": probe.summarize_rows(rows),
        "molecule_macro": probe.macro_summary(rows, "smiles"),
        "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
        "stratified": probe.stratified_summary(rows),
        "best_val": best,
        "fp_stats": fp_stats,
        "skipped": skipped,
        "feature_path": emb_path,
        "feature_meta_path": meta_path,
        "feature_meta": emb_meta,
        "query_rows": str(row_path),
        "history": str(hist_path),
        "head_path": str(head_path),
        "source_config": cfg,
    }
    block = metrics["block_micro"]
    print(
        f"    TEST n={block.get('n_queries', 0)} Top1={block.get('top1', 0):.2f} "
        f"Top5={block.get('top5', 0):.2f} MRR={block.get('mrr', 0):.4f}",
        flush=True,
    )
    del head, spec_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def normalize_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = scores.astype(np.float64)
    if mode == "none":
        return scores
    if mode == "zscore":
        std = float(scores.std())
        if std < 1e-12:
            return scores * 0.0
        return (scores - float(scores.mean())) / std
    if mode == "rank":
        order = np.argsort(np.argsort(scores))
        return order.astype(np.float64) / max(len(scores) - 1, 1)
    if mode == "minmax":
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-12:
            return scores * 0.0
        return (scores - lo) / (hi - lo)
    raise ValueError(mode)


def simplex_weights(n: int, step: float):
    if n == 1:
        yield np.ones(1, dtype=np.float64)
        return
    units = int(round(1.0 / step))
    if units < 1 or abs(units * step - 1.0) > 1e-6:
        raise ValueError("--weight-step must divide 1.0 exactly")
    cur = [0] * n

    def rec(pos: int, remaining: int):
        if pos == n - 1:
            cur[pos] = remaining
            yield np.asarray(cur, dtype=np.float64) / units
            return
        for value in range(remaining + 1):
            cur[pos] = value
            yield from rec(pos + 1, remaining - value)

    yield from rec(0, units)


def build_context(source_key: str, metrics: dict, samples, train_idx, fp_dict, device, args) -> dict:
    spec_emb = np.load(metrics["feature_path"], mmap_mode="r")
    head = ms2fp.FingerprintHead(
        spec_emb.shape[1],
        fp_dim=args.fp_bits,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    head.load_state_dict(torch.load(metrics["head_path"], map_location=device))
    head.eval()
    logit_shift = None
    if args.calibrate_pos_weight:
        pw, _ = ms2fp.compute_pos_weight(samples, train_idx, fp_dict, args.fp_bits, args.pos_weight_clip)
        logit_shift = torch.log(pw.clamp_min(1e-6))
    return {
        "source_key": source_key,
        "spec_emb": spec_emb,
        "head": head,
        "fp_dict": fp_dict,
        "logit_shift": logit_shift,
    }


def candidate_scores_for_context(ctx, sample_idx: int, candidates, device, score_mode: str):
    fp_dict = ctx["fp_dict"]
    cand = [c for c in candidates if c in fp_dict and fp_dict[c].sum() > 0]
    if not cand:
        return None, None
    with torch.no_grad():
        q_np = np.asarray(ctx["spec_emb"][sample_idx:sample_idx + 1], dtype=np.float32)
        q = torch.from_numpy(q_np).float().to(device)
        logits = ctx["head"](q).squeeze(0)
        if ctx["logit_shift"] is not None:
            logits = logits - ctx["logit_shift"].to(device)
        pred = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    cand_mat = np.stack([fp_dict[c] for c in cand]).astype(np.float32)
    scores = ms2fp.fingerprint_scores(pred, cand_mat, score_mode)
    return cand, scores


def build_score_cache(contexts, samples, eval_idx, candidates_by_spec, device, args, phase: str):
    cache = []
    skipped = defaultdict(int)
    total = len(eval_idx)
    for n_done, qi in enumerate(eval_idx, 1):
        if n_done == 1 or n_done % 500 == 0 or n_done == total:
            print(f"      fusion cache {phase}: {n_done}/{total}", flush=True)
        s = samples[int(qi)]
        smi = s.get("smiles", "")
        spec_id = s["spec_id"]
        if not smi or smi == "nan":
            skipped["bad_smiles"] += 1
            continue
        cand = list(candidates_by_spec.get(spec_id, []))
        if not cand:
            skipped["no_candidates"] += 1
            continue
        gt_appended = 0
        if smi not in cand:
            cand.append(smi)
            gt_appended = 1
            skipped["gt_appended"] += 1
        seen = set()
        cand = [c for c in cand if not (c in seen or seen.add(c))]
        if smi not in cand:
            skipped["missing_gt_candidate"] += 1
            continue
        pos = {c: i for i, c in enumerate(cand)}
        score_mat = np.full((len(contexts), len(cand)), np.nan, dtype=np.float64)
        for mi, ctx in enumerate(contexts):
            scored_cand, scores = candidate_scores_for_context(ctx, int(qi), cand, device, args.score)
            if scored_cand is None:
                continue
            norm = normalize_scores(scores, args.normalize)
            for c, sc in zip(scored_cand, norm):
                if c in pos:
                    score_mat[mi, pos[c]] = float(sc)
        gt_pos = cand.index(smi)
        valid_any = np.isfinite(score_mat).any(axis=0)
        if not valid_any.any() or not valid_any[gt_pos]:
            skipped["missing_gt_score"] += 1
            continue
        cache.append({
            "score_mat": score_mat,
            "gt_pos": gt_pos,
            "cand": cand,
            "meta": {
                "candidate_task": "",
                "split": "",
                "scenario": "",
                "model": "",
                "phase": phase,
                "spec_id": spec_id,
                "parent_spec_id": s.get("parent_spec_id", ""),
                "smiles": smi,
                "block_type": s.get("block_type", ""),
                "adduct": s.get("adduct", ""),
                "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                "gt_appended": int(gt_appended),
            },
        })
    return cache, dict(skipped)


def rows_from_cache(cache, weights: np.ndarray):
    rows = []
    weights = weights.astype(np.float64)
    for item in cache:
        score_mat = item["score_mat"]
        valid_mat = np.isfinite(score_mat)
        score_weight = valid_mat.astype(np.float64) * weights[:, None]
        score_sum = np.where(valid_mat, score_mat, 0.0) * weights[:, None]
        score_sum = score_sum.sum(axis=0)
        score_weight = score_weight.sum(axis=0)
        valid = score_weight > 0
        cand = item["cand"]
        gt_pos = item["gt_pos"]
        ensemble = np.full(len(cand), -1e9, dtype=np.float64)
        ensemble[valid] = score_sum[valid] / np.maximum(score_weight[valid], 1e-12)
        order = np.argsort(-ensemble)
        rank = int(np.where(order == gt_pos)[0][0] + 1)
        top_pos = int(order[0])
        row = dict(item["meta"])
        row.update({
            "candidate_count": int(len(cand)),
            "candidate_count_raw": int(len(cand)),
            "candidate_count_scored": int(valid.sum()),
            "rank": rank,
            "hit1": int(rank <= 1),
            "hit5": int(rank <= 5),
            "hit10": int(rank <= 10),
            "hit20": int(rank <= 20),
            "rr": float(1.0 / rank),
            "gt_score": float(ensemble[gt_pos]),
            "top1_score": float(ensemble[top_pos]),
            "top1_is_gt": int(top_pos == gt_pos),
            "top1_smiles": cand[top_pos],
            "n_members": int(score_mat.shape[0]),
        })
        rows.append(row)
    return rows


def choose_weights_from_cache(cache, skipped, n_members: int, args):
    grid_rows = []
    best = None
    for weights in simplex_weights(n_members, args.weight_step):
        rows = rows_from_cache(cache, weights)
        block = probe.summarize_rows(rows)
        item = {
            "weights": weights.tolist(),
            "top1": float(block.get("top1", 0.0)),
            "top5": float(block.get("top5", 0.0)),
            "mrr": float(block.get("mrr", 0.0)),
            "n_queries": int(block.get("n_queries", 0)),
            "skipped": skipped,
        }
        grid_rows.append(item)
        key = (item["top1"], item["mrr"], item["top5"])
        if best is None or key > best[0]:
            best = (key, item)
    assert best is not None
    return np.asarray(best[1]["weights"], dtype=np.float64), best[1], grid_rows


def write_weight_grid(path: Path, rows, member_names) -> None:
    fields = ["rank", "top1", "top5", "mrr", "n_queries"] + [f"w_{safe_name(m)}" for m in member_names]
    ranked = sorted(rows, key=lambda r: (r["top1"], r["mrr"], r["top5"]), reverse=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(ranked, 1):
            out = {
                "rank": i,
                "top1": row["top1"],
                "top5": row["top5"],
                "mrr": row["mrr"],
                "n_queries": row["n_queries"],
            }
            for name, weight in zip(member_names, row["weights"]):
                out[f"w_{safe_name(name)}"] = weight
            writer.writerow(out)


def run_ensemble(ens_name: str, member_keys: list[str], source_metrics: dict, out_dir: Path,
                 samples, train_idx, val_idx, test_idx, candidates_by_spec, fp_dict,
                 device, args, task: str, scenario_name: str) -> dict:
    missing = [m for m in member_keys if m not in source_metrics]
    if missing:
        raise KeyError(f"ensemble {ens_name} missing members: {missing}")
    print(f"\n[ensemble:{ens_name}] task={task} scenario={scenario_name} members={member_keys}", flush=True)
    contexts = [
        build_context(m, source_metrics[m], samples, train_idx, fp_dict, device, args)
        for m in member_keys
    ]
    val_cache, val_skipped = build_score_cache(contexts, samples, val_idx, candidates_by_spec, device, args, phase="val")
    if len(contexts) == 1:
        weights = np.ones(1, dtype=np.float64)
        val_rows = rows_from_cache(val_cache, weights)
        val_block = probe.summarize_rows(val_rows)
        best_weight = {
            "weights": weights.tolist(),
            "top1": float(val_block.get("top1", 0.0)),
            "top5": float(val_block.get("top5", 0.0)),
            "mrr": float(val_block.get("mrr", 0.0)),
            "n_queries": int(val_block.get("n_queries", 0)),
            "skipped": val_skipped,
        }
        grid_rows = [best_weight]
    else:
        weights, best_weight, grid_rows = choose_weights_from_cache(val_cache, val_skipped, len(contexts), args)

    test_cache, test_skipped = build_score_cache(contexts, samples, test_idx, candidates_by_spec, device, args, phase="test")
    rows = rows_from_cache(test_cache, weights)
    for row in rows:
        row.update({
            "candidate_task": task,
            "split": task,
            "scenario": scenario_name,
            "model": ens_name,
            "phase": "test",
        })
    stem = f"{safe_name(task)}_{safe_name(scenario_name)}_{safe_name(ens_name)}"
    row_path = out_dir / f"query_rows_{stem}.csv"
    grid_path = out_dir / f"weight_grid_{stem}.csv"
    write_rows_extra(row_path, rows, extra_fields=["gt_appended"])
    write_weight_grid(grid_path, grid_rows, member_keys)
    metrics = {
        "block_micro": probe.summarize_rows(rows),
        "molecule_macro": probe.macro_summary(rows, "smiles"),
        "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
        "stratified": probe.stratified_summary(rows),
        "skipped": test_skipped,
        "query_rows": str(row_path),
        "weight_grid": str(grid_path),
        "selected_val": best_weight,
        "selected_weights": weights.tolist(),
        "members": member_keys,
    }
    block = metrics["block_micro"]
    print(
        f"    val Top1={best_weight['top1']:.2f} weights={weights.tolist()} | "
        f"TEST n={block.get('n_queries', 0)} Top1={block.get('top1', 0):.2f} "
        f"Top5={block.get('top5', 0):.2f} MRR={block.get('mrr', 0):.4f}",
        flush=True,
    )
    for ctx in contexts:
        del ctx["head"], ctx["spec_emb"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_metrics_table(summary: dict, out_dir: Path) -> None:
    rows = []
    for task, task_sum in summary.get("tasks", {}).items():
        for scenario_name, sc in task_sum.get("scenarios", {}).items():
            for group in ("sources", "ensembles"):
                for model, metrics in sc.get(group, {}).items():
                    block = metrics.get("block_micro", {})
                    mol = metrics.get("molecule_macro", {})
                    parent = metrics.get("parent_macro", {})
                    rows.append({
                        "candidate_task": task,
                        "scenario": scenario_name,
                        "group": group,
                        "model": model,
                        "n_queries": block.get("n_queries", 0),
                        "top1": block.get("top1", 0),
                        "top5": block.get("top5", 0),
                        "top10": block.get("top10", 0),
                        "top20": block.get("top20", 0),
                        "mrr": block.get("mrr", 0),
                        "avg_candidates": block.get("avg_candidates", 0),
                        "molecule_top1": mol.get("top1", 0),
                        "parent_top1": parent.get("top1", 0),
                        "query_rows": metrics.get("query_rows", ""),
                        "selected_weights": json.dumps(metrics.get("selected_weights", [])),
                    })
    fields = [
        "candidate_task", "scenario", "group", "model", "n_queries", "top1", "top5",
        "top10", "top20", "mrr", "avg_candidates", "molecule_top1", "parent_top1",
        "query_rows", "selected_weights",
    ]
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=json_default) + "\n")


def write_embedding_manifest(source_configs: dict, out_dir: Path) -> None:
    fields = ["source", "kind", "path", "meta_path", "exists", "shape0", "shape1"]
    rows = []
    for key, cfg in source_configs.items():
        emb_path, meta_path = feature_cache_paths(out_dir, key, cfg)
        shape0 = shape1 = ""
        if emb_path.exists():
            arr = np.load(emb_path, mmap_mode="r")
            shape0, shape1 = int(arr.shape[0]), int(arr.shape[1]) if arr.ndim == 2 else ""
            del arr
        rows.append({
            "source": key,
            "kind": cfg["kind"],
            "path": str(emb_path),
            "meta_path": str(meta_path),
            "exists": int(emb_path.exists()),
            "shape0": shape0,
            "shape1": shape1,
        })
    with (out_dir / "audit_embedding_manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> None:
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source_configs = {k: DEFAULT_SOURCE_CONFIGS[k] for k in args.sources}
    ensembles = {
        name: members for name, members in DEFAULT_ENSEMBLES.items()
        if all(m in source_configs for m in members)
    }
    if args.ensembles:
        ensembles = {name: DEFAULT_ENSEMBLES[name] for name in args.ensembles}
    config = {
        "run_name": args.run_name,
        "dataset": "MassSpecGym",
        "data_csv": str(DATA_DIR / "MassSpecGym.csv"),
        "tasks": args.tasks,
        "scenarios": args.scenarios,
        "sources": source_configs,
        "ensembles": ensembles,
        "seed": args.seed,
        "score": args.score,
        "normalize": args.normalize,
        "weight_step": args.weight_step,
        "include_train_candidates_in_fp": bool(args.include_train_candidates_in_fp),
        "debug_load_limit_per_fold": int(args.debug_load_limit_per_fold),
        "debug_index_limit": int(args.debug_index_limit),
        "ms2fp_args": {
            "fp_bits": args.fp_bits,
            "radius": args.radius,
            "fp_use_chirality": args.fp_use_chirality,
            "epochs": args.epochs,
            "candidate_ce_epochs": args.candidate_ce_epochs,
            "candidate_ce_lr": args.candidate_ce_lr,
            "candidate_ce_temperature": args.candidate_ce_temperature,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "score": args.score,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=json_default) + "\n")

    samples, train_idx, val_idx, test_idx, data_audit = load_massspecgym(args.debug_load_limit_per_fold)
    summary = {"config": config, "data_audit": data_audit, "candidate_audit": {}, "tasks": {}}

    for task in args.tasks:
        candidates_by_spec, cand_audit = load_candidates(task, samples)
        summary["candidate_audit"][task] = cand_audit
        summary["tasks"].setdefault(task, {"scenarios": {}})
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            train_f = scenario_mask(samples, train_idx, scenario)
            val_f = scenario_mask(samples, val_idx, scenario)
            test_f = scenario_mask(samples, test_idx, scenario)
            train_f = maybe_limit_indices(train_f, args.debug_index_limit, args.seed + 1)
            val_f = maybe_limit_indices(val_f, args.debug_index_limit, args.seed + 2)
            test_f = maybe_limit_indices(test_f, args.debug_index_limit, args.seed + 3)
            smiles = collect_smiles_for_protocol(
                samples, train_f, val_f, test_f, candidates_by_spec,
                include_train_candidates=bool(args.include_train_candidates_in_fp),
            )
            fp_dict, fp_cache = build_or_load_fp_dict(out_dir, task, scenario_name, smiles, args)
            train_f = ms2fp.valid_indices(samples, train_f, fp_dict)
            val_f = ms2fp.valid_indices(samples, val_f, fp_dict)
            test_f = ms2fp.valid_indices(samples, test_f, fp_dict)
            if len(train_f) == 0 or len(val_f) == 0 or len(test_f) == 0:
                raise RuntimeError(
                    f"empty split after filtering task={task} scenario={scenario_name}: "
                    f"train={len(train_f)} val={len(val_f)} test={len(test_f)}"
                )
            print(
                "\n" + "=" * 92 + "\n"
                f"Task={task} scenario={scenario_name} train={len(train_f)} "
                f"val={len(val_f)} test={len(test_f)} fps={len(fp_dict)}\n"
                + "=" * 92,
                flush=True,
            )
            sc_sum = {
                "description": scenario.get("desc", ""),
                "n_train": int(len(train_f)),
                "n_val": int(len(val_f)),
                "n_test": int(len(test_f)),
                "n_fingerprints": int(len(fp_dict)),
                "fingerprint_cache": fp_cache,
                "sources": {},
                "ensembles": {},
            }
            for source_key, cfg in source_configs.items():
                metrics = train_one_source(
                    source_key, cfg, out_dir, samples, train_f, val_f, test_f,
                    candidates_by_spec, fp_dict, device, args, task, scenario_name,
                )
                sc_sum["sources"][source_key] = metrics
                summary["tasks"][task]["scenarios"][scenario_name] = sc_sum
                (out_dir / "summary.partial.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n"
                )
                write_metrics_table(summary, out_dir)
                write_embedding_manifest(source_configs, out_dir)

            for ens_name, members in ensembles.items():
                metrics = run_ensemble(
                    ens_name, members, sc_sum["sources"], out_dir, samples, train_f, val_f,
                    test_f, candidates_by_spec, fp_dict, device, args, task, scenario_name,
                )
                sc_sum["ensembles"][ens_name] = metrics

            if "dreams_global_iw" in sc_sum["ensembles"]:
                base_rows = []
                base_path = Path(sc_sum["ensembles"]["dreams_global_iw"]["query_rows"])
                if base_path.exists():
                    with base_path.open() as fh:
                        base_rows = list(csv.DictReader(fh))
                        for r in base_rows:
                            for key in ("rank", "hit1", "rr", "candidate_count", "n_peaks_model", "n_peaks_above_precursor_plus1"):
                                if key in r and r[key] != "":
                                    r[key] = float(r[key]) if key == "rr" else int(float(r[key]))
                for ens_name, metrics in list(sc_sum["ensembles"].items()):
                    if ens_name == "dreams_global_iw":
                        continue
                    qpath = Path(metrics["query_rows"])
                    if qpath.exists() and base_rows:
                        with qpath.open() as fh:
                            rows = list(csv.DictReader(fh))
                            for r in rows:
                                for key in ("rank", "hit1", "rr", "candidate_count", "n_peaks_model", "n_peaks_above_precursor_plus1"):
                                    if key in r and r[key] != "":
                                        r[key] = float(r[key]) if key == "rr" else int(float(r[key]))
                        metrics[f"{ens_name}_minus_dreams_global_iw_bootstrap"] = probe.bootstrap_delta(
                            rows, base_rows, seed=args.seed + 991
                        )
            summary["tasks"][task]["scenarios"][scenario_name] = sc_sum
            (out_dir / "summary.partial.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n"
            )
            write_metrics_table(summary, out_dir)
            write_embedding_manifest(source_configs, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n")
    write_metrics_table(summary, out_dir)
    write_embedding_manifest(source_configs, out_dir)
    print("\nSaved", out_dir / "summary.json")
    print((out_dir / "metrics.csv").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="sourcefusion")
    ap.add_argument("--tasks", nargs="+", default=["formula", "mass"], choices=["formula", "mass"])
    ap.add_argument("--scenarios", nargs="+", default=["all"], choices=sorted(SCENARIOS))
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCE_CONFIGS), choices=sorted(DEFAULT_SOURCE_CONFIGS))
    ap.add_argument("--ensembles", nargs="*", default=None, choices=sorted(DEFAULT_ENSEMBLES))
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fp-use-chirality", action="store_true")
    ap.add_argument("--include-train-candidates-in-fp", action="store_true")
    ap.add_argument("--tanimoto-loss-weight", type=float, default=0.0)
    ap.add_argument(
        "--score",
        choices=["tanimoto", "soft_tanimoto", "cosine", "dot", "pos_mean", "bernoulli", "hard_tanimoto"],
        default="tanimoto",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--pos-weight-clip", type=float, default=50.0)
    ap.add_argument("--calibrate-pos-weight", action="store_true")
    ap.add_argument("--candidate-ce-epochs", type=int, default=10)
    ap.add_argument("--candidate-ce-batch-size", type=int, default=64)
    ap.add_argument("--candidate-ce-lr", type=float, default=5e-5)
    ap.add_argument("--candidate-ce-temperature", type=float, default=24.0)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--normalize", choices=["zscore", "rank", "minmax", "none"], default="zscore")
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--force-emb", action="store_true")
    ap.add_argument("--force-fp", action="store_true")
    ap.add_argument("--debug-load-limit-per-fold", type=int, default=0)
    ap.add_argument("--debug-index-limit", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
