#!/usr/bin/env python
"""Extract matched peak-level embeddings from UltraMS and DreaMS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MAX_ULTRA_PEAKS = 150


def project_root() -> Path:
    return Path(os.environ.get('ULTRAMS_EXPERIMENT_ROOT', Path(__file__).resolve().parents[4]))


def split_floats(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(x) for x in text.split(",") if x], dtype=np.float32)


def prep_spectrum(spec_np: np.ndarray, max_peaks: int = MAX_ULTRA_PEAKS) -> np.ndarray:
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    mx = float(np.nanmax(inten))
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0, 1)
    if len(mz) > max_peaks:
        idx = np.argsort(inten)[-max_peaks:]
        idx = np.sort(idx)
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


def parse_formula(formula: str) -> dict[str, int]:
    import re
    counts: dict[str, int] = {}
    if not formula:
        return counts
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def fragment_label(formula: str) -> str:
    c = parse_formula(formula)
    if not c:
        return "Unassigned"
    if any(c.get(x, 0) for x in ("F", "Cl", "Br", "I")):
        return "Halogenated"
    if c.get("S", 0) or c.get("P", 0):
        return "S/P-containing"
    if c.get("N", 0):
        return "N-containing"
    if c.get("O", 0) >= 2:
        return "O-rich"
    if set(c).issubset({"C", "H"}):
        return "Hydrocarbon"
    return "Other"


def load_annotations(path: Path, min_match_rate: float) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for sample in payload.get("samples", []):
        if not sample.get("success"):
            continue
        if float(sample.get("match_rate", 0.0)) < min_match_rate:
            continue
        out[str(sample["spec_id"])] = sample
    return out


def nearest_annotation(sample_ann: dict, mz: float, tol: float) -> dict | None:
    best = None
    best_delta = float("inf")
    for ann in sample_ann.get("annotations", []):
        delta = abs(float(ann["mz_observed"]) - float(mz))
        if delta < best_delta:
            best = ann
            best_delta = delta
    if best is None or best_delta > tol:
        return None
    return best


def load_ultra(root: Path, device: torch.device, ckpt_path: Path):
    source_train = Path(os.environ.get("ULTRAMS_SOURCE_TRAIN_ROOT", root / "train"))
    sys.path.insert(0, str(source_train.parent))
    sys.path.insert(0, str(source_train))
    sys.path.insert(0, str(root))
    from train_ue_multiscale_v9_mlm import CONFIG, UltraExplorerMLM

    model = UltraExplorerMLM(CONFIG)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("base_state_dict") or ckpt.get("model_state_dict")
    if any(k.startswith("base.") for k in state):
        state = {k.replace("base.", "", 1): v for k, v in state.items() if k.startswith("base.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[UltraMS] checkpoint={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def load_dreams_loader(root: Path):
    # The historical loader forgot to import os in one migrated snapshot; inject it.
    path = root / "train/comparison/dreams_loader.py"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "support/comparison/dreams_loader.py"
    spec = importlib.util.spec_from_file_location("fig2_dreams_loader", path)
    mod = importlib.util.module_from_spec(spec)
    mod.os = os
    assert spec and spec.loader
    sys.modules["fig2_dreams_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dreams(root: Path, device: torch.device):
    loader = load_dreams_loader(root)
    model = loader.load_dreams_encoder(device)
    return model.eval(), loader


@torch.no_grad()
def ultra_batch_embeddings(model, batch: list[dict], device: torch.device) -> list[dict]:
    bsz = len(batch)
    padded = np.zeros((bsz, MAX_ULTRA_PEAKS, 2), dtype=np.float32)
    attn = np.zeros((bsz, MAX_ULTRA_PEAKS), dtype=np.int64)
    processed = []
    for i, sample in enumerate(batch):
        sp = prep_spectrum(sample["spectrum"], MAX_ULTRA_PEAKS)
        processed.append(sp)
        k = min(len(sp), MAX_ULTRA_PEAKS)
        if k:
            padded[i, :k] = sp[:k]
            attn[i, :k] = 1
    peaks_t = torch.from_numpy(padded).to(device)
    attn_t = torch.from_numpy(attn).to(device)
    pmz_t = torch.tensor([x["precursor_mz"] for x in batch], dtype=torch.float32, device=device)

    ms_emb = model.peak_encoder(peaks_t)
    cls_emb = model.cls_emb.expand(bsz, 1, -1)
    pi = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], dim=-1).unsqueeze(1)
    prec_emb = model.peak_encoder(pi) + model.precursor_type_emb
    full = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
    pos = torch.arange(full.shape[1], device=device).unsqueeze(0)
    full = model.dropout(full + model.pos_emb(pos))
    mask = torch.cat([torch.ones(bsz, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1)
    out = model.encoder(inputs_embeds=full, attention_mask=mask).last_hidden_state[:, 2:, :].detach().cpu().numpy()

    results = []
    for i, sp in enumerate(processed):
        k = len(sp)
        results.append({"mzs": sp[:, 0].copy(), "intensities": sp[:, 1].copy(), "emb": out[i, :k].copy()})
    return results


@torch.no_grad()
def dreams_batch_embeddings(model, loader, batch: list[dict], device: torch.device) -> list[dict]:
    peaks = loader.build_dreams_batch(model, batch, device)
    out = model(peaks).detach().cpu().numpy()
    peaks_np = peaks.detach().cpu().numpy()
    results = []
    for i in range(len(batch)):
        peak_rows = peaks_np[i, 1:, :]
        mask = peak_rows[:, 0] > 0
        results.append({
            "mzs": peak_rows[mask, 0].astype(np.float32),
            "intensities": peak_rows[mask, 1].astype(np.float32),
            "emb": out[i, 1:][mask].copy(),
        })
    return results


def main() -> None:
    root = project_root()
    data_dir = root / "main_figure/fig2/data/task5_peak_embedding_umap"
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-csv", type=Path, default=data_dir / "msnlib_magma_subset.csv")
    ap.add_argument("--annotations", type=Path, default=data_dir / "msnlib_magma_annotations.json")
    ap.add_argument("--output-dir", type=Path, default=data_dir)
    ap.add_argument("--ultra-ckpt", type=Path, default=root / "train/output/phase2_rt_only/stage_d_epoch_11.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-spectra", type=int, default=0, help="Optional cap after annotation filtering; 0 keeps all spectra.")
    ap.add_argument("--min-match-rate", type=float, default=0.65)
    ap.add_argument("--mz-tol", type=float, default=0.015)
    ap.add_argument("--max-per-fragment-label", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"device={device}")

    anns = load_annotations(args.annotations, args.min_match_rate)
    df = pd.read_csv(args.subset_csv)
    df = df[df["spec_id"].astype(str).isin(anns)].copy()
    if args.max_spectra and len(df) > args.max_spectra:
        df = df.sample(args.max_spectra, random_state=args.seed)
    df = df.sort_values("spec_id").reset_index(drop=True)
    print(f"selected spectra={len(df)} from annotated={len(anns)}")

    samples = []
    for _, row in df.iterrows():
        mzs = split_floats(row["mzs"])
        ints = split_floats(row["intensities"])
        samples.append({
            "spec_id": str(row["spec_id"]),
            "spectrum": np.stack([mzs, ints], axis=-1).astype(np.float32),
            "precursor_mz": float(row["precursor_mz"]),
            "smiles": str(row["smiles"]),
            "adduct": str(row["adduct"]),
            "molecule_label": str(row.get("molecule_label", "Other")),
        })

    ultra = load_ultra(root, device, args.ultra_ckpt)
    dreams, dreams_loader = load_dreams(root, device)

    meta_rows = []
    ultra_rows = []
    dreams_rows = []
    label_counts: dict[str, int] = {}

    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        u_out = ultra_batch_embeddings(ultra, batch, device)
        d_out = dreams_batch_embeddings(dreams, dreams_loader, batch, device)
        for sample, ue, de in zip(batch, u_out, d_out):
            ann_sample = anns[sample["spec_id"]]
            for j, mz in enumerate(de["mzs"]):
                ann = nearest_annotation(ann_sample, float(mz), args.mz_tol)
                if ann is None or not ann.get("assigned") or not ann.get("frag_formula"):
                    continue
                delta = np.abs(ue["mzs"] - float(mz))
                ui = int(delta.argmin()) if len(delta) else -1
                if ui < 0 or float(delta[ui]) > args.mz_tol:
                    continue
                label = fragment_label(str(ann["frag_formula"]))
                if label == "Unassigned":
                    continue
                if label_counts.get(label, 0) >= args.max_per_fragment_label:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
                meta_rows.append({
                    "peak_uid": f"{sample['spec_id']}::{len(meta_rows):05d}",
                    "spec_id": sample["spec_id"],
                    "mz": float(mz),
                    "intensity": float(de["intensities"][j]),
                    "frag_formula": str(ann["frag_formula"]),
                    "frag_smiles": str(ann.get("frag_smiles") or ""),
                    "frag_atom_indices": json.dumps(ann.get("frag_atom_indices") or [], separators=(",", ":")),
                    "frag_label": label,
                    "molecule_label": sample["molecule_label"],
                    "smiles": sample["smiles"],
                    "adduct": sample["adduct"],
                    "match_rate": float(ann_sample["match_rate"]),
                })
                ultra_rows.append(ue["emb"][ui])
                dreams_rows.append(de["emb"][j])
        print(f"processed {min(start + args.batch_size, len(samples))}/{len(samples)} peaks={len(meta_rows)}", flush=True)

    if not meta_rows:
        raise RuntimeError("No matched peak embeddings were extracted.")

    # Stable shuffle prevents label-block ordering from affecting downstream sampled plots.
    order = rng.permutation(len(meta_rows))
    meta = pd.DataFrame(meta_rows).iloc[order].reset_index(drop=True)
    ultra_arr = np.asarray(ultra_rows, dtype=np.float32)[order]
    dreams_arr = np.asarray(dreams_rows, dtype=np.float32)[order]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.output_dir / "peak_embedding_metadata.csv", index=False)
    np.save(args.output_dir / "peak_embeddings_ultra.npy", ultra_arr)
    np.save(args.output_dir / "peak_embeddings_dreams.npy", dreams_arr)
    summary = {
        "n_spectra": int(len(df)),
        "n_peaks": int(len(meta)),
        "fragment_label_counts": meta["frag_label"].value_counts().to_dict(),
        "molecule_label_counts": meta["molecule_label"].value_counts().to_dict(),
        "ultra_embedding_shape": list(ultra_arr.shape),
        "dreams_embedding_shape": list(dreams_arr.shape),
    }
    with open(args.output_dir / "peak_embedding_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
