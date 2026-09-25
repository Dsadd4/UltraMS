#!/usr/bin/env python
"""Evaluate atom-query attention as an element-specific top-N subspectrum.

For an element E, the predicted subspectrum is made from the model's top-N
per-peak scores for E. For UltraMS/DreaMS those scores are atom-query attention
weights. For classical baselines they are positive peak-bin contributions from
models trained on raw binned MS2 features.

The reference subspectrum is the MAGMa-filtered MS2 spectrum containing peaks
whose matched fragment chemically contains E.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


TARGET_ATOMS = ["S", "Cl", "F", "Br"]
MAX_ULTRA_PEAKS = 150
DEFAULT_TOP_N = [5, 10, 20]
METHOD_ORDER = ["UltraMS", "DreaMS", "Linear SVM", "Random forest", "XGBoost"]
def project_root() -> Path:
    return Path(os.environ.get('ULTRAMS_EXPERIMENT_ROOT', Path(__file__).resolve().parents[4]))


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def split_floats(text: Any) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(x) for x in text.split(",") if x], dtype=np.float32)


def normalize_spectrum(mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    mz = np.asarray(mz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    ok = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[ok], intensity[ok]
    if len(mz) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    mx = float(np.max(intensity))
    if mx > 0:
        intensity = intensity / mx
    order = np.argsort(mz)
    return np.stack([mz[order], np.clip(intensity[order], 0, 1)], axis=1).astype(np.float32)


def parse_top_n(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or min(values) < 1:
        raise ValueError("--top-n must contain positive integers")
    if min(values) < 5:
        raise ValueError("This experiment requires top-N >= 5.")
    return values


def parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(formula, str):
        return counts
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def count_atoms_from_smiles(smiles: str, atoms: list[str]) -> dict[str, int]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {a: 0 for a in atoms}
    return {a: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == a) for a in atoms}


def sample_from_row(row: pd.Series, test_index: int) -> dict[str, Any]:
    spec = normalize_spectrum(split_floats(row["mzs"]), split_floats(row["intensities"]))
    return {
        "sample_id": f"test_{int(test_index)}",
        "test_index": int(test_index),
        "spectrum": spec,
        "precursor_mz": float(row["precursor_mz"]),
        "smiles": str(row["smiles"]),
        "adduct": str(row["adduct"]),
        "collision_energy": None if pd.isna(row.get("collision_energy")) else float(row.get("collision_energy")),
        "fold": "test",
    }


def choose_eval_indices(
    test_df: pd.DataFrame,
    atoms: list[str],
    repeats: int,
    samples_per_repeat: int,
    seed: int,
) -> tuple[list[list[int]], dict[int, dict[str, int]]]:
    atom_counts: dict[int, dict[str, int]] = {}
    eligible: list[int] = []
    groups = {a: [] for a in atoms}
    for idx, row in test_df.iterrows():
        mz = split_floats(row["mzs"])
        if len(mz) < 3:
            continue
        counts = count_atoms_from_smiles(str(row["smiles"]), atoms)
        atom_counts[int(idx)] = counts
        if any(counts[a] > 0 for a in atoms):
            eligible.append(int(idx))
            for atom in atoms:
                if counts[atom] > 0:
                    groups[atom].append(int(idx))

    if len(eligible) < samples_per_repeat:
        raise ValueError(f"Only {len(eligible)} eligible test spectra for requested {samples_per_repeat}.")

    repeats_idx: list[list[int]] = []
    base_rng = np.random.default_rng(seed)
    for rep in range(repeats):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        selected: list[int] = []
        per_atom = max(1, samples_per_repeat // max(len(atoms), 1))
        for atom in atoms:
            pool = np.asarray(groups[atom], dtype=int)
            if len(pool) == 0:
                continue
            take = min(per_atom, len(pool))
            selected.extend(rng.choice(pool, size=take, replace=False).tolist())
        selected = list(dict.fromkeys(selected))
        if len(selected) < samples_per_repeat:
            rest = np.setdiff1d(np.asarray(eligible, dtype=int), np.asarray(selected, dtype=int))
            selected.extend(rng.choice(rest, size=samples_per_repeat - len(selected), replace=False).tolist())
        rng.shuffle(selected)
        repeats_idx.append([int(x) for x in selected[:samples_per_repeat]])
    return repeats_idx, atom_counts


def _decode_fragment_atoms(fe: Any, frag_id: Any) -> list[int]:
    entry = fe.frag_to_entry.get(int(frag_id))
    if entry is None:
        return []
    frag_bits = entry["frag"]
    atom_indices = []
    bits = int(frag_bits)
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        if idx < fe.natoms:
            atom_indices.append(idx)
        bits ^= lsb
    return atom_indices


def _fragment_smiles(mol: Chem.Mol, atom_indices: list[int]) -> str | None:
    if not atom_indices:
        return None
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_indices), canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _fragment_formula(frag_smiles: str | None) -> str | None:
    if not frag_smiles:
        return None
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


def magma_annotate_detailed(
    smiles: str,
    mzs: np.ndarray,
    intensities: np.ndarray,
    adduct: str,
    atom_names: list[str],
    ppm: float,
) -> tuple[np.ndarray, dict[int, dict[str, Any]], str | None]:
    try:
        from fragments.magma.op import fragmentation_op_v8 as v8
        from fragments.magma.op.Magma4MassSpecGYm import match_peaks
    except ImportError as exc:
        return np.zeros((len(mzs), len(atom_names)), dtype=bool), {}, f"import_error: {exc}"

    try:
        fe = v8.FragmentEngine(
            mol_str=smiles,
            max_tree_depth=3,
            max_broken_bonds=6,
            skip_canonicalization=False,
        )
        fe.generate_fragments()
        results = match_peaks(fe, mzs, intensities, ppm, adduct)
        mol = fe.mol
    except Exception as exc:
        return np.zeros((len(mzs), len(atom_names)), dtype=bool), {}, f"magma_error: {exc}"

    annot = np.zeros((len(mzs), len(atom_names)), dtype=bool)
    details: dict[int, dict[str, Any]] = {}
    for row in results.itertuples(index=False):
        rowd = row._asdict()
        frag_id = rowd.get("frag_id")
        peak_id = rowd.get("peak_id")
        if peak_id is None or frag_id is None:
            continue
        try:
            if np.isnan(float(frag_id)):
                continue
        except Exception:
            pass
        peak_id = int(peak_id)
        atom_indices = _decode_fragment_atoms(fe, frag_id)
        if not atom_indices:
            continue
        frag_smiles = _fragment_smiles(mol, atom_indices)
        frag_formula = rowd.get("frag_formula") or rowd.get("formula") or _fragment_formula(frag_smiles)
        for ai, atom in enumerate(atom_names):
            if any(mol.GetAtomWithIdx(idx).GetSymbol() == atom for idx in atom_indices):
                annot[peak_id, ai] = True
        details[peak_id] = {
            "peak_id": peak_id,
            "mz": float(mzs[peak_id]),
            "intensity": float(intensities[peak_id]),
            "frag_id": int(frag_id),
            "frag_atom_indices": [int(i) for i in atom_indices],
            "frag_smiles": frag_smiles,
            "frag_formula": frag_formula,
            "score": None if rowd.get("score") is None else float(rowd.get("score")),
            "ppm_diff": None if rowd.get("ppm_diff") is None else float(rowd.get("ppm_diff")),
        }
    return annot, details, None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_magma_annotations(
    samples: list[dict[str, Any]],
    atom_names: list[str],
    out_dir: Path,
    ppm: float,
) -> dict[str, dict[str, Any]]:
    cache_path = out_dir / "attn_subspectrum_eval_v1.magma_cache.json"
    cache = load_json(cache_path, {"samples": {}})
    sample_cache = cache.setdefault("samples", {})
    for i, sample in enumerate(samples):
        key = sample["sample_id"]
        if key in sample_cache:
            continue
        spec = sample["spectrum"]
        annot, details, error = magma_annotate_detailed(
            sample["smiles"],
            spec[:, 0],
            spec[:, 1],
            sample["adduct"],
            atom_names,
            ppm=ppm,
        )
        sample_cache[key] = {
            "sample_id": key,
            "test_index": int(sample["test_index"]),
            "success": error is None,
            "error": error,
            "magma_mask": annot.astype(bool).tolist(),
            "details_by_peak": {str(k): v for k, v in details.items()},
        }
        if (i + 1) % 5 == 0 or i + 1 == len(samples):
            write_json(cache_path, cache)
        print(f"[MAGMa] {i + 1}/{len(samples)} {key} success={error is None}", flush=True)
    write_json(cache_path, cache)
    return sample_cache


def add_project_imports(root: Path) -> None:
    source_train = Path(os.environ.get("ULTRAMS_SOURCE_TRAIN_ROOT", root / "train"))
    for path in [root, source_train, source_train.parent]:
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)


def add_magma_import(root: Path) -> None:
    magma_root = Path(os.environ.get(
        "ULTRAGO_DIR", str(Path(__file__).resolve().parents[2] / "magma_support")
    ))
    if magma_root.is_dir():
        p = str(magma_root)
        if p not in sys.path:
            sys.path.insert(0, p)


def load_ultra_probe(root: Path, ckpt_path: Path, ultra_ckpt: Path | None, device: torch.device):
    add_project_imports(root)
    from showcase.models_atom_query import load_probe_from_ckpt

    model, atom_names, norm_mean, norm_std = load_probe_from_ckpt(
        str(ckpt_path),
        device,
        ultra_ckpt=None if ultra_ckpt is None else str(ultra_ckpt),
    )
    model.eval()
    return model, atom_names, norm_mean, norm_std


def load_patched_dreams_loader(root: Path):
    add_project_imports(root)
    path = root / "train/comparison/dreams_loader.py"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "support/comparison/dreams_loader.py"
    spec = importlib.util.spec_from_file_location("comparison.dreams_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DreaMS loader from {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.os = os
    sys.modules["comparison.dreams_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dreams_probe(root: Path, ckpt_path: Path, device: torch.device, dreams_max_peaks: int):
    add_project_imports(root)
    dreams_loader = load_patched_dreams_loader(root)
    dreams_loader.N_HIGHEST_PEAKS = int(dreams_max_peaks)
    from showcase.models_atom_query import DreamsAtomQueryProbe

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    atom_names = ckpt["atom_names"]
    enc = dreams_loader.load_dreams_encoder(device)
    if hasattr(enc, "spec_preproc") and hasattr(enc.spec_preproc, "n_highest_peaks"):
        enc.spec_preproc.n_highest_peaks = int(dreams_max_peaks)
    model = DreamsAtomQueryProbe(enc, atom_names).to(device)
    model.atom_queries.data = ckpt["atom_queries"].to(device)
    model.k_proj.load_state_dict(ckpt["k_proj_state"])
    model.v_proj.load_state_dict(ckpt["v_proj_state"])
    for head, state in zip(model.heads, ckpt["heads_state"]):
        head.load_state_dict(state)
    model.eval()
    return model, atom_names, dreams_loader, np.asarray(ckpt["norm_mean"]), np.asarray(ckpt["norm_std"])


@torch.no_grad()
def get_ultra_attention(model: torch.nn.Module, samples: list[dict[str, Any]], device: torch.device, batch_size: int):
    logits_rows = []
    attn_rows = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        logits, attn = model(batch, device, return_attn=True)
        logits_rows.append(logits.detach().cpu().numpy())
        attn_rows.append(attn.detach().cpu().numpy())
        print(f"[UltraMS] attention {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return np.concatenate(logits_rows, axis=0), np.concatenate(attn_rows, axis=0)


@torch.no_grad()
def get_dreams_attention(
    model: torch.nn.Module,
    dreams_loader: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
):
    logits_rows = []
    attn_rows = []
    peak_rows = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        logits, attn = model(batch, device, return_attn=True)
        peaks = dreams_loader.build_dreams_batch(model.dreams, batch, device)[:, 1:, :].detach().cpu().numpy()
        logits_rows.append(logits.detach().cpu().numpy())
        attn_rows.append(attn.detach().cpu().numpy())
        peak_rows.extend([p.astype(np.float32) for p in peaks])
        print(f"[DreaMS] attention {min(start + batch_size, len(samples))}/{len(samples)}", flush=True)
    return np.concatenate(logits_rows, axis=0), np.concatenate(attn_rows, axis=0), peak_rows


def make_binned_matrix(df: pd.DataFrame, n_bins: int, bin_size: float) -> np.ndarray:
    x = np.zeros((len(df), n_bins), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        mz = split_floats(getattr(row, "mzs"))
        inten = split_floats(getattr(row, "intensities"))
        spec = normalize_spectrum(mz, inten)
        if len(spec) == 0:
            continue
        bins = np.floor(spec[:, 0] / bin_size).astype(int)
        ok = (bins >= 0) & (bins < n_bins)
        if np.any(ok):
            np.maximum.at(x[i], bins[ok], spec[:, 1][ok])
    return x


def choose_classifier_train_df(
    csv_path: Path,
    atoms: list[str],
    max_train: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    cols = ["mzs", "intensities", "formula", "fold"]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df[df["fold"] == "train"].reset_index(drop=True)
    labels = np.zeros((len(df), len(atoms)), dtype=np.int8)
    for i, formula in enumerate(df["formula"].fillna("").astype(str)):
        counts = parse_formula(formula)
        labels[i] = [1 if counts.get(atom, 0) > 0 else 0 for atom in atoms]

    if max_train > 0 and len(df) > max_train:
        rng = np.random.default_rng(seed)
        any_pos = labels.sum(axis=1) > 0
        pos_idx = np.where(any_pos)[0]
        neg_idx = np.where(~any_pos)[0]
        n_pos = min(len(pos_idx), max_train // 2)
        n_neg = min(len(neg_idx), max_train - n_pos)
        chosen = []
        if n_pos:
            chosen.extend(rng.choice(pos_idx, size=n_pos, replace=False).tolist())
        if n_neg:
            chosen.extend(rng.choice(neg_idx, size=n_neg, replace=False).tolist())
        if len(chosen) < max_train:
            rest = np.setdiff1d(np.arange(len(df)), np.asarray(chosen, dtype=int))
            chosen.extend(rng.choice(rest, size=max_train - len(chosen), replace=False).tolist())
        chosen = np.asarray(chosen, dtype=int)
        rng.shuffle(chosen)
        df = df.iloc[chosen].reset_index(drop=True)
        labels = labels[chosen]
    return df, labels


def train_classical_baselines(
    csv_path: Path,
    atoms: list[str],
    max_train: int,
    mz_max: float,
    bin_size: float,
    seed: int,
    include_xgboost: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC

    n_bins = int(math.ceil(mz_max / bin_size)) + 1
    train_df, y_all = choose_classifier_train_df(csv_path, atoms, max_train, seed)
    print(f"[classical] training rows={len(train_df):,} bins={n_bins}", flush=True)
    x = make_binned_matrix(train_df, n_bins, bin_size)
    models: dict[str, dict[str, Any]] = {m: {} for m in ["Linear SVM", "Random forest"]}
    summary: dict[str, Any] = {
        "training_rows": int(len(train_df)),
        "n_bins": int(n_bins),
        "bin_size": float(bin_size),
        "mz_max": float(mz_max),
        "models": {},
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            xgb_available = True
        except Exception as exc:
            XGBClassifier = None
            xgb_available = False
            summary["xgboost_unavailable"] = repr(exc)
    else:
        XGBClassifier = None
        xgb_available = False
        summary["xgboost_excluded"] = True

    if xgb_available:
        models["XGBoost"] = {}

    for ai, atom in enumerate(atoms):
        y = y_all[:, ai]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        summary["models"][atom] = {"positives": pos, "negatives": neg}
        if len(np.unique(y)) < 2:
            continue
        svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=6000, random_state=seed)
        svm.fit(x, y)
        models["Linear SVM"][atom] = svm

        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        rf.fit(x, y)
        models["Random forest"][atom] = rf

        if xgb_available and XGBClassifier is not None:
            scale_pos_weight = float(neg / max(pos, 1))
            xgb = XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                n_jobs=4,
                random_state=seed,
                scale_pos_weight=scale_pos_weight,
            )
            xgb.fit(x, y)
            models["XGBoost"][atom] = xgb
        print(f"[classical] trained {atom}: pos={pos} neg={neg}", flush=True)
    return models, summary


def peak_bins(spec: np.ndarray, n_bins: int, bin_size: float) -> np.ndarray:
    bins = np.floor(spec[:, 0] / bin_size).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def classical_peak_scores(
    models: dict[str, dict[str, Any]],
    spec: np.ndarray,
    atoms: list[str],
    n_bins: int,
    bin_size: float,
) -> dict[str, dict[str, list[float]]]:
    bins = peak_bins(spec, n_bins, bin_size)
    inten = spec[:, 1].astype(float)
    out: dict[str, dict[str, list[float]]] = {}
    for method, atom_models in models.items():
        out[method] = {}
        for atom in atoms:
            model = atom_models.get(atom)
            if model is None:
                out[method][atom] = [0.0 for _ in range(len(spec))]
                continue
            if method == "Linear SVM":
                coef = np.asarray(model.coef_[0], dtype=float)
                pos_coef = np.maximum(coef, 0.0)
                if float(pos_coef.max(initial=0.0)) <= 0:
                    pos_coef = np.abs(coef)
                score = pos_coef[bins] * inten
            else:
                if hasattr(model, "feature_importances_"):
                    imp = np.asarray(model.feature_importances_, dtype=float)
                else:
                    booster = model.get_booster()
                    fmap = booster.get_score(importance_type="gain")
                    imp = np.zeros(n_bins, dtype=float)
                    for key, value in fmap.items():
                        if key.startswith("f"):
                            idx = int(key[1:])
                            if 0 <= idx < n_bins:
                                imp[idx] = float(value)
                score = imp[bins] * inten
            out[method][atom] = np.asarray(score, dtype=float).tolist()
    return out


def align_dreams_to_original(dreams_peaks: np.ndarray, original_spec: np.ndarray, mz_tol: float) -> list[int]:
    mapping: list[int] = []
    orig_mz = original_spec[:, 0]
    for mz in dreams_peaks[:, 0]:
        if mz <= 0:
            mapping.append(-1)
            continue
        delta = np.abs(orig_mz - float(mz))
        idx = int(delta.argmin()) if len(delta) else -1
        mapping.append(idx if idx >= 0 and float(delta[idx]) <= mz_tol else -1)
    return mapping


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def evaluate_one(
    scores: np.ndarray,
    truth_mask: np.ndarray,
    reference_intensity: np.ndarray,
    top_n_values: list[int],
) -> list[dict[str, float]]:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    truth_mask = np.asarray(truth_mask, dtype=bool)
    reference_intensity = np.asarray(reference_intensity, dtype=float)
    n = min(len(scores), len(truth_mask), len(reference_intensity))
    scores = scores[:n]
    truth_mask = truth_mask[:n]
    reference_intensity = reference_intensity[:n]
    out = []
    n_element = int(truth_mask.sum())
    total_element_intensity = float(reference_intensity[truth_mask].sum())
    for top_n in top_n_values:
        n_eff = min(int(top_n), len(scores))
        order = np.argsort(-scores)[:n_eff]
        pred = np.zeros_like(scores, dtype=float)
        pred[order] = np.maximum(scores[order], 0.0)
        truth = np.where(truth_mask, reference_intensity, 0.0)
        hits = int(truth_mask[order].sum()) if n_eff else 0
        captured = float(reference_intensity[order][truth_mask[order]].sum()) if n_eff else 0.0
        out.append(
            {
                "top_n": int(top_n),
                "n_visible_peaks": int(len(scores)),
                "n_element_peaks": n_element,
                "topn_hits": hits,
                "precision_at_n": float(hits / max(n_eff, 1)),
                "element_peak_recall_at_n": float(hits / max(n_element, 1)),
                "element_intensity_capture_at_n": float(captured / total_element_intensity)
                if total_element_intensity > 0
                else float("nan"),
                "cosine": cosine(pred, truth),
            }
        )
    return out


def aggregate_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    valid = df[df["n_element_peaks"] > 0].copy()
    grouped = (
        valid.groupby(["method", "atom", "top_n"], dropna=False)
        .agg(
            n=("sample_id", "count"),
            cosine_mean=("cosine", "mean"),
            cosine_sem=("cosine", lambda x: float(np.nanstd(x, ddof=1) / math.sqrt(max(np.isfinite(x).sum(), 1)))),
            precision_mean=("precision_at_n", "mean"),
            precision_sem=("precision_at_n", lambda x: float(np.nanstd(x, ddof=1) / math.sqrt(max(np.isfinite(x).sum(), 1)))),
            capture_mean=("element_intensity_capture_at_n", "mean"),
            capture_sem=(
                "element_intensity_capture_at_n",
                lambda x: float(np.nanstd(x, ddof=1) / math.sqrt(max(np.isfinite(x).sum(), 1))),
            ),
        )
        .reset_index()
    )
    return grouped


def save_peak_scores_csv(path: Path, full_records: list[dict[str, Any]], atoms: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHOD_ORDER if any(m in rec["method_scores"] for rec in full_records)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "test_index",
                "peak_id",
                "mz",
                "intensity",
                "atom",
                "magma_hit",
                "frag_formula",
                "frag_smiles",
                *[f"{method}_score" for method in methods],
            ]
        )
        for rec in full_records:
            details = rec["magma_details_by_peak"]
            for peak_id, (mz, intensity) in enumerate(rec["spectrum"]):
                detail = details.get(str(peak_id), {})
                for atom in atoms:
                    row = [
                        rec["sample_id"],
                        rec["test_index"],
                        peak_id,
                        float(mz),
                        float(intensity),
                        atom,
                        bool(rec["magma_mask_by_atom"][atom][peak_id]),
                        detail.get("frag_formula", ""),
                        detail.get("frag_smiles", ""),
                    ]
                    for method in methods:
                        scores = rec["method_scores"].get(method, {}).get(atom, [])
                        row.append(scores[peak_id] if peak_id < len(scores) else "")
                    writer.writerow(row)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="datasets/MassSpecGym/MassSpecGym_test.csv")
    parser.add_argument("--all-csv", default="datasets/MassSpecGym/MassSpecGym.csv")
    parser.add_argument("--out-dir", default="main_figure/fig2/output/attn_subspectrum_eval_v1")
    parser.add_argument("--ultra-ckpt", default="train/output/showcase/atomq_ultra_S_Cl_F_Br_0413_1448/last_ultra.pt")
    parser.add_argument("--dreams-ckpt", default="train/output/showcase/atomq_dreams_S_Cl_F_Br_0413_2123/last_dreams.pt")
    parser.add_argument("--ultra-backbone-ckpt", default="train/output/phase2_rt_only/stage_d_epoch_11.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples-per-repeat", type=int, default=100)
    parser.add_argument("--top-n", default=",".join(str(x) for x in DEFAULT_TOP_N))
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--mz-tol", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dreams-max-peaks", type=int, default=150)
    parser.add_argument("--classifier-max-train", type=int, default=50000)
    parser.add_argument("--mz-max", type=float, default=1200.0)
    parser.add_argument("--bin-size", type=float, default=1.0)
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--without-xgboost", action="store_true",
                        help="evaluate the four methods in the published element-peak result")
    parser.add_argument("--force-attention", action="store_true")
    args = parser.parse_args()

    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    add_magma_import(root)
    top_n_values = parse_top_n(args.top_n)
    test_csv = resolve_path(root, args.test_csv)
    all_csv = resolve_path(root, args.all_csv)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    if test_csv.exists():
        test_df = pd.read_csv(test_csv).reset_index(drop=True)
    else:
        full_df = pd.read_csv(all_csv)
        test_df = full_df[full_df["fold"] == "test"].reset_index(drop=True)
    repeat_indices, atom_counts = choose_eval_indices(
        test_df,
        TARGET_ATOMS,
        repeats=args.repeats,
        samples_per_repeat=args.samples_per_repeat,
        seed=args.seed,
    )
    unique_indices = sorted({idx for rep in repeat_indices for idx in rep})
    samples = [sample_from_row(test_df.iloc[idx], idx) for idx in unique_indices]
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    index_to_position = {idx: pos for pos, idx in enumerate(unique_indices)}
    print(
        f"[setup] repeats={args.repeats} samples/repeat={args.samples_per_repeat} "
        f"unique_test_spectra={len(samples)} device={device}",
        flush=True,
    )

    magma_cache = ensure_magma_annotations(samples, TARGET_ATOMS, out_dir, ppm=args.ppm)

    attention_cache_path = out_dir / "attn_subspectrum_eval_v1.attention_cache.pkl"
    if attention_cache_path.exists() and not args.force_attention:
        with attention_cache_path.open("rb") as f:
            attention_payload = pickle.load(f)
        cache_ok = (
            attention_payload.get("dreams_max_peaks") == int(args.dreams_max_peaks)
            and attention_payload.get("sample_ids") == [sample["sample_id"] for sample in samples]
        )
        if not cache_ok:
            attention_payload = None
    else:
        attention_payload = None

    if attention_payload is None:
        ultra_ckpt = resolve_path(root, args.ultra_ckpt)
        dreams_ckpt = resolve_path(root, args.dreams_ckpt)
        ultra_backbone = resolve_path(root, args.ultra_backbone_ckpt)
        ultra_model, ultra_atoms, ultra_mean, ultra_std = load_ultra_probe(root, ultra_ckpt, ultra_backbone, device)
        dreams_model, dreams_atoms, dreams_loader, dreams_mean, dreams_std = load_dreams_probe(
            root,
            dreams_ckpt,
            device,
            dreams_max_peaks=args.dreams_max_peaks,
        )
        if ultra_atoms != TARGET_ATOMS or dreams_atoms != TARGET_ATOMS:
            raise ValueError(f"Expected atoms {TARGET_ATOMS}, got Ultra={ultra_atoms}, DreaMS={dreams_atoms}")
        ultra_logits_z, ultra_attn = get_ultra_attention(ultra_model, samples, device, args.batch_size)
        dreams_logits_z, dreams_attn, dreams_peaks = get_dreams_attention(
            dreams_model,
            dreams_loader,
            samples,
            device,
            args.batch_size,
        )
        attention_payload = {
            "ultra_logits_z": ultra_logits_z,
            "ultra_attn": ultra_attn,
            "ultra_norm_mean": ultra_mean,
            "ultra_norm_std": ultra_std,
            "dreams_logits_z": dreams_logits_z,
            "dreams_attn": dreams_attn,
            "dreams_peaks": dreams_peaks,
            "dreams_norm_mean": dreams_mean,
            "dreams_norm_std": dreams_std,
            "dreams_max_peaks": int(args.dreams_max_peaks),
            "atom_names": TARGET_ATOMS,
            "sample_ids": [sample["sample_id"] for sample in samples],
        }
        with attention_cache_path.open("wb") as f:
            pickle.dump(attention_payload, f)

    if args.skip_classical:
        classical_models, classical_summary = {}, {"skipped": True}
    else:
        classical_models, classical_summary = train_classical_baselines(
            all_csv,
            TARGET_ATOMS,
            max_train=args.classifier_max_train,
            mz_max=args.mz_max,
            bin_size=args.bin_size,
            seed=args.seed,
            include_xgboost=not args.without_xgboost,
        )

    n_bins = int(math.ceil(args.mz_max / args.bin_size)) + 1
    full_records = []
    metric_rows = []
    for pos, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        spec = sample["spectrum"]
        mag = magma_cache[sample_id]
        magma_mask_array = np.asarray(mag["magma_mask"], dtype=bool)
        magma_by_atom = {
            atom: magma_mask_array[: len(spec), ai].astype(bool).tolist() for ai, atom in enumerate(TARGET_ATOMS)
        }
        method_scores = {"UltraMS": {}, "DreaMS": {}}
        for ai, atom in enumerate(TARGET_ATOMS):
            method_scores["UltraMS"][atom] = attention_payload["ultra_attn"][pos, ai, : len(spec)].astype(float).tolist()

        dreams_peaks = np.asarray(attention_payload["dreams_peaks"][pos], dtype=np.float32)
        dreams_valid = dreams_peaks[:, 0] > 0
        dreams_peaks_valid = dreams_peaks[dreams_valid]
        dreams_map = align_dreams_to_original(dreams_peaks_valid, spec, args.mz_tol)
        dreams_truth_by_atom: dict[str, list[bool]] = {}
        dreams_intensity = dreams_peaks_valid[:, 1].astype(float) if len(dreams_peaks_valid) else np.zeros(0)
        for ai, atom in enumerate(TARGET_ATOMS):
            method_scores["DreaMS"][atom] = attention_payload["dreams_attn"][pos, ai, : len(dreams_peaks_valid)].astype(float).tolist()
            dreams_truth_by_atom[atom] = [
                bool(magma_mask_array[orig_idx, ai]) if orig_idx >= 0 and orig_idx < len(magma_mask_array) else False
                for orig_idx in dreams_map
            ]

        classical_scores = classical_peak_scores(classical_models, spec, TARGET_ATOMS, n_bins, args.bin_size)
        method_scores.update(classical_scores)

        full_records.append(
            {
                "sample_id": sample_id,
                "test_index": int(sample["test_index"]),
                "smiles": sample["smiles"],
                "adduct": sample["adduct"],
                "precursor_mz": float(sample["precursor_mz"]),
                "target_atom_counts": atom_counts[int(sample["test_index"])],
                "spectrum": spec.astype(float).tolist(),
                "magma_success": bool(mag["success"]),
                "magma_error": mag.get("error"),
                "magma_mask_by_atom": magma_by_atom,
                "magma_details_by_peak": mag.get("details_by_peak", {}),
                "dreams_peaks": dreams_peaks_valid.astype(float).tolist(),
                "dreams_peak_to_original_peak": dreams_map,
                "dreams_magma_mask_by_atom": dreams_truth_by_atom,
                "method_scores": method_scores,
            }
        )

        for atom in TARGET_ATOMS:
            truth = np.asarray(magma_by_atom[atom], dtype=bool)
            ref_intensity = spec[:, 1].astype(float)
            for method, scores_by_atom in method_scores.items():
                if method == "DreaMS":
                    method_truth = np.asarray(dreams_truth_by_atom[atom], dtype=bool)
                    method_intensity = dreams_intensity
                else:
                    method_truth = truth
                    method_intensity = ref_intensity
                scores = np.asarray(scores_by_atom.get(atom, []), dtype=float)
                if len(scores) == 0 or len(method_truth) == 0:
                    continue
                for metrics in evaluate_one(scores, method_truth, method_intensity, top_n_values):
                    metric_rows.append(
                        {
                            "sample_id": sample_id,
                            "test_index": int(sample["test_index"]),
                            "atom": atom,
                            "method": method,
                            "molecule_atom_count": int(atom_counts[int(sample["test_index"])][atom]),
                            **metrics,
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    summary_df = aggregate_metrics(metric_rows)
    metrics_path = out_dir / "attn_subspectrum_eval_v1.metrics.csv"
    summary_path = out_dir / "attn_subspectrum_eval_v1.summary_metrics.csv"
    peak_scores_path = out_dir / "attn_subspectrum_eval_v1.peak_scores.csv"
    full_cache_path = out_dir / "attn_subspectrum_eval_v1.full_cache.pkl"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    save_peak_scores_csv(peak_scores_path, full_records, TARGET_ATOMS)
    with full_cache_path.open("wb") as f:
        pickle.dump(
            {
                "records": full_records,
                "metrics": metric_rows,
                "summary": summary_df.to_dict(orient="records"),
                "repeat_indices": repeat_indices,
                "config": vars(args),
            },
            f,
        )

    manifest = {
        "stage": "fig2_attn_subspectrum_eval_v1",
        "description": "Element-specific top-N attention subspectra compared with MAGMa-filtered MS2 peaks.",
        "atoms": TARGET_ATOMS,
        "top_n_values": top_n_values,
        "repeat_indices": repeat_indices,
        "unique_test_indices": unique_indices,
        "n_unique_samples": len(samples),
        "n_metric_rows": int(len(metrics_df)),
        "classical_summary": classical_summary,
        "paths": {
            "metrics_csv": str(metrics_path),
            "summary_metrics_csv": str(summary_path),
            "peak_scores_csv": str(peak_scores_path),
            "magma_cache_json": str(out_dir / "attn_subspectrum_eval_v1.magma_cache.json"),
            "attention_cache_pkl": str(attention_cache_path),
            "full_cache_pkl": str(full_cache_path),
        },
        "config": vars(args),
    }
    write_json(out_dir / "attn_subspectrum_eval_v1.plot_data.json", manifest)
    write_json(out_dir / "attn_subspectrum_eval_v1.summary.json", summary_df.to_dict(orient="records"))

    print(f"Saved metrics and data under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
