"""
NPLIB1 spectrum-to-molecule retrieval with peak-token spectrum summaries.

This script keeps the corrected NPLIB1 MS2-block protocol used by
17_nplib1_ms2fp_retrieval.py:

  token-level spectrum summary -> Morgan fingerprint logits -> candidate CE
  rank the same candidate molecules with the same Top1 query set.

The only intended change is the spectrum representation.  Instead of using a
single global embedding, UltraMS is summarized from encoder last_hidden_state
tokens (CLS, precursor, peak mean/max/std/intensity-weighted mean).  DreaMS is
summarized from its full token output when the current dreams_loader exposes it.
If DreaMS full-token output is unavailable, this script raises a clear error
rather than silently falling back to pooled embeddings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAIN_DIR = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve() / "train"

MS2FP_PATH = HERE / "17_nplib1_ms2fp_retrieval.py"
OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_token_summary_ms2fp"

SUMMARY_PART_CHOICES = (
    "cls",
    "precursor",
    "peak_mean",
    "peak_max",
    "peak_std",
    "peak_iw_mean",
)
DEFAULT_SUMMARY_PARTS = [
    "cls",
    "precursor",
    "peak_mean",
    "peak_max",
    "peak_std",
    "peak_iw_mean",
]


def import_script(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found: {path}. Run this from train/comparison "
            "after copying the companion NPLIB1 scripts."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ms2fp = import_script(MS2FP_PATH, "nplib1_ms2fp")
runner = ms2fp.runner
probe = ms2fp.probe
cm10 = runner.cm10


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
    masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    out = masked.max(dim=1).values
    empty = ~mask.any(dim=1)
    if empty.any():
        out[empty] = 0
    return out


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
    mean = masked_mean(x, mask)
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    var = (((x - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / denom
    return torch.sqrt(var.clamp_min(0.0) + 1e-8)


def masked_weighted_mean(x: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 0:
        return torch.zeros(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
    w = weights.to(dtype=x.dtype).clamp_min(0.0) * mask.to(dtype=x.dtype)
    no_weight = (w.sum(dim=1, keepdim=True) <= 0) & (mask.sum(dim=1, keepdim=True) > 0)
    if no_weight.any():
        w = torch.where(no_weight, mask.to(dtype=x.dtype), w)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * w.unsqueeze(-1)).sum(dim=1) / denom


def build_token_summary(
    hidden: torch.Tensor,
    peak_mask: torch.Tensor,
    peak_weights: torch.Tensor,
    summary_parts: list[str],
    cls_index: int | None,
    precursor_index: int,
    peak_start: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected token hidden states with shape [B, L, D], got {tuple(hidden.shape)}")
    if precursor_index >= hidden.shape[1]:
        raise RuntimeError(
            f"Cannot read precursor token index {precursor_index}; hidden length is {hidden.shape[1]}"
        )

    peak_hidden = hidden[:, peak_start:, :]
    if peak_hidden.shape[1] != peak_mask.shape[1]:
        common = min(peak_hidden.shape[1], peak_mask.shape[1])
        peak_hidden = peak_hidden[:, :common, :]
        peak_mask = peak_mask[:, :common]
        peak_weights = peak_weights[:, :common]

    feats = []
    used = []
    skipped = []
    for part in summary_parts:
        if part == "cls":
            if cls_index is None:
                skipped.append(part)
                continue
            if cls_index >= hidden.shape[1]:
                raise RuntimeError(f"Cannot read CLS token index {cls_index}; hidden length is {hidden.shape[1]}")
            feats.append(hidden[:, cls_index, :])
        elif part == "precursor":
            feats.append(hidden[:, precursor_index, :])
        elif part == "peak_mean":
            feats.append(masked_mean(peak_hidden, peak_mask))
        elif part == "peak_max":
            feats.append(masked_max(peak_hidden, peak_mask))
        elif part == "peak_std":
            feats.append(masked_std(peak_hidden, peak_mask))
        elif part == "peak_iw_mean":
            feats.append(masked_weighted_mean(peak_hidden, peak_mask, peak_weights))
        else:
            raise ValueError(f"Unknown summary part: {part}")
        used.append(part)

    if not feats:
        raise RuntimeError(f"No usable summary parts from requested parts={summary_parts}; skipped={skipped}")
    return torch.cat(feats, dim=-1), used, skipped


def finalize_token_stats(counts: list[np.ndarray], extra: dict | None = None) -> dict:
    stats = dict(extra or {})
    if counts:
        arr = np.concatenate(counts).astype(np.float64)
        stats.update({
            "n_spectra": int(arr.size),
            "peak_tokens_min": int(arr.min()),
            "peak_tokens_mean": float(arr.mean()),
            "peak_tokens_median": float(np.median(arr)),
            "peak_tokens_max": int(arr.max()),
        })
    else:
        stats.update({
            "n_spectra": 0,
            "peak_tokens_min": 0,
            "peak_tokens_mean": 0.0,
            "peak_tokens_median": 0.0,
            "peak_tokens_max": 0,
        })
    return stats


@torch.no_grad()
def extract_ultrams_token_summary(model, samples, device, batch_size: int, summary_parts: list[str]):
    model = model.to(device)
    model.eval()
    max_peaks = int(model.max_peaks)
    features = []
    peak_counts = []
    used_parts = None
    skipped_parts = None

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        bsz = len(batch_samples)
        spectra = [cm10.prep_spectrum(s["spectrum"], max_peaks) for s in batch_samples]
        pmzs = torch.tensor([s["precursor_mz"] for s in batch_samples], dtype=torch.float32, device=device)
        padded = np.zeros((bsz, max_peaks, 2), dtype=np.float32)
        attn = np.zeros((bsz, max_peaks), dtype=np.int64)
        for i, sp in enumerate(spectra):
            k = min(len(sp), max_peaks)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i, :k] = 1

        peaks_t = torch.from_numpy(padded).to(device)
        peak_mask = torch.from_numpy(attn.astype(bool)).to(device)
        ms_emb = model.peak_encoder(peaks_t)
        cls_emb = model.cls_emb.expand(bsz, 1, -1)
        precursor_input = torch.stack([pmzs, torch.full_like(pmzs, 1.1)], dim=-1).unsqueeze(1)
        precursor_emb = model.peak_encoder(precursor_input) + model.precursor_type_emb
        full = torch.cat([cls_emb, precursor_emb, ms_emb], dim=1)
        seq_len = full.shape[1]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        full = model.dropout(full + model.pos_emb(pos))
        full_attn = torch.cat(
            [torch.ones(bsz, 2, dtype=torch.long, device=device), peak_mask.to(torch.long)], dim=1
        )
        out = model.encoder(inputs_embeds=full, attention_mask=full_attn)
        hidden = out.last_hidden_state
        feat, used, skipped = build_token_summary(
            hidden=hidden,
            peak_mask=peak_mask,
            peak_weights=peaks_t[:, :, 1],
            summary_parts=summary_parts,
            cls_index=0,
            precursor_index=1,
            peak_start=2,
        )
        features.append(feat.detach().cpu().numpy().astype(np.float32))
        peak_counts.append(peak_mask.sum(dim=1).detach().cpu().numpy())
        used_parts = used
        skipped_parts = skipped

    emb = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    meta = finalize_token_stats(peak_counts, {
        "extractor": "ultrams_last_hidden_state",
        "summary_parts_requested": summary_parts,
        "summary_parts_used": used_parts or [],
        "summary_parts_skipped": skipped_parts or [],
        "token_layout": {"cls": 0, "precursor": 1, "peaks_start": 2},
        "max_peaks": max_peaks,
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
    })
    return emb, meta


def import_dreams_loader():
    candidates = [
        HERE / "dreams_loader.py",
        TRAIN_DIR / "comparison" / "dreams_loader.py",
    ]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            loader = import_script(path, "current_dreams_loader")
            missing = [name for name in ("load_dreams_encoder", "build_dreams_batch") if not hasattr(loader, name)]
            if missing:
                raise AttributeError(
                    f"{path} is missing required DreaMS full-token helpers: {missing}. "
                    "This experiment needs the current dreams_loader, not a pooled-only fallback."
                )
            return loader, path
    raise FileNotFoundError(
        "Could not find dreams_loader.py next to this script. DreaMS token-summary extraction "
        "requires train/comparison/dreams_loader.py with load_dreams_encoder and build_dreams_batch."
    )


def load_dreams_model(loader, device):
    try:
        loaded = loader.load_dreams_encoder(device)
    except TypeError as first_error:
        try:
            loaded = loader.load_dreams_encoder()
        except TypeError as second_error:
            raise TypeError(
                "dreams_loader.load_dreams_encoder must accept either (device) or no arguments."
            ) from second_error
        if first_error is None:
            raise

    if isinstance(loaded, tuple):
        model = loaded[0]
        reported_dim = loaded[1] if len(loaded) > 1 else None
    else:
        model = loaded
        reported_dim = None
    if model is None:
        raise RuntimeError("dreams_loader.load_dreams_encoder returned None")
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, reported_dim


def build_dreams_batch(loader, model, batch_samples, device):
    try:
        return loader.build_dreams_batch(model, batch_samples, device)
    except TypeError as first_error:
        try:
            return loader.build_dreams_batch(batch_samples, device)
        except TypeError as second_error:
            raise TypeError(
                "dreams_loader.build_dreams_batch must accept either "
                "(model, batch_samples, device) or (batch_samples, device)."
            ) from second_error
        if first_error is None:
            raise


def unpack_dreams_token_output(raw_output):
    if isinstance(raw_output, dict):
        for key in ("last_hidden_state", "hidden_states", "tokens", "embeddings"):
            if key in raw_output:
                raw_output = raw_output[key]
                break
        else:
            raise RuntimeError(
                f"DreaMS model output is a dict but has no token tensor key; keys={sorted(raw_output)}"
            )
    if isinstance(raw_output, (tuple, list)):
        if not raw_output:
            raise RuntimeError("DreaMS model returned an empty tuple/list")
        raw_output = raw_output[0]
    if not torch.is_tensor(raw_output):
        raise RuntimeError(f"DreaMS model output is not a tensor after unpacking: {type(raw_output)!r}")
    if raw_output.ndim != 3:
        raise RuntimeError(
            "DreaMS full-token output unavailable: expected model(peaks) to return [B, L, D], "
            f"got shape {tuple(raw_output.shape)}. Refusing to fall back to pooled embeddings."
        )
    return raw_output


def dreams_peak_mask_and_weights(peaks: torch.Tensor, hidden_len: int):
    if not torch.is_tensor(peaks):
        raise RuntimeError(f"build_dreams_batch returned non-tensor peaks: {type(peaks)!r}")
    if peaks.ndim != 3:
        raise RuntimeError(f"Expected DreaMS input tensor [B, L, C], got shape {tuple(peaks.shape)}")
    if peaks.shape[1] < 2:
        raise RuntimeError(f"DreaMS input has no peak tokens after precursor row: shape={tuple(peaks.shape)}")

    peak_inputs = peaks[:, 1:min(peaks.shape[1], hidden_len), :]
    if peak_inputs.shape[1] == 0:
        peak_mask = torch.zeros(peaks.shape[0], 0, dtype=torch.bool, device=peaks.device)
        weights = torch.zeros(peaks.shape[0], 0, dtype=peaks.dtype, device=peaks.device)
        return peak_mask, weights

    nonzero = peak_inputs.abs().sum(dim=-1) > 0
    if peak_inputs.shape[-1] >= 2:
        mz_positive = peak_inputs[:, :, 0] > 0
        intensity = peak_inputs[:, :, 1].clamp_min(0)
        intensity_positive = intensity > 0
        peak_mask = nonzero & (mz_positive | intensity_positive)
        weights = intensity
    else:
        peak_mask = nonzero
        weights = nonzero.to(dtype=peaks.dtype)
    return peak_mask, weights


@torch.no_grad()
def extract_dreams_token_summary(samples, device, batch_size: int, summary_parts: list[str]):
    loader, loader_path = import_dreams_loader()
    model, reported_dim = load_dreams_model(loader, device)
    features = []
    peak_counts = []
    used_parts = None
    skipped_parts = None
    token_mismatch_batches = 0

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        peaks = build_dreams_batch(loader, model, batch_samples, device)
        peaks = peaks.to(device)
        raw_output = model(peaks)
        hidden = unpack_dreams_token_output(raw_output)
        if hidden.shape[0] != len(batch_samples):
            raise RuntimeError(
                f"DreaMS token output batch mismatch: expected {len(batch_samples)}, got {hidden.shape[0]}"
            )
        if hidden.shape[1] != peaks.shape[1]:
            token_mismatch_batches += 1

        peak_mask, peak_weights = dreams_peak_mask_and_weights(peaks, hidden.shape[1])
        feat, used, skipped = build_token_summary(
            hidden=hidden,
            peak_mask=peak_mask,
            peak_weights=peak_weights,
            summary_parts=summary_parts,
            cls_index=None,
            precursor_index=0,
            peak_start=1,
        )
        features.append(feat.detach().cpu().numpy().astype(np.float32))
        peak_counts.append(peak_mask.sum(dim=1).detach().cpu().numpy())
        used_parts = used
        skipped_parts = skipped

    emb = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    meta = finalize_token_stats(peak_counts, {
        "extractor": "dreams_loader_full_token_output",
        "dreams_loader_path": str(loader_path),
        "reported_dim": int(reported_dim) if reported_dim is not None else None,
        "summary_parts_requested": summary_parts,
        "summary_parts_used": used_parts or [],
        "summary_parts_skipped": skipped_parts or [],
        "token_layout": {"cls": None, "precursor": 0, "peaks_start": 1},
        "token_length_mismatch_batches": int(token_mismatch_batches),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
    })
    return emb, meta


def summary_cache_paths(split_id: int, model_name: str, scenario_name: str, summary_parts: list[str]):
    part_tag = "-".join(summary_parts)
    safe_model = model_name.replace("/", "_")
    emb_dir = OUT_BASE / "token_summary_embeddings" / scenario_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    stem = f"token_summary_split{split_id}_{safe_model}_{part_tag}"
    return emb_dir / f"{stem}.npy", emb_dir / f"{stem}.json"


def get_token_summary_embeddings(model_name: str, split_id: int, samples, scenario_name: str,
                                 device, args):
    emb_path, meta_path = summary_cache_paths(split_id, model_name, scenario_name, args.summary_parts)
    if emb_path.exists() and meta_path.exists() and not args.force_emb:
        emb = np.load(emb_path)
        meta = json.loads(meta_path.read_text())
        if emb.shape[0] != len(samples):
            raise RuntimeError(
                f"Cached token summary row mismatch for {emb_path}: "
                f"cached={emb.shape[0]} samples={len(samples)}. Use --force-emb to rebuild."
            )
        return emb, int(emb.shape[1]), str(emb_path), str(meta_path), meta

    print(f"Extracting token summaries: model={model_name} split={split_id} scenario={scenario_name}")
    if model_name == "rt_only_d11":
        model, _ = cm10.load_model(model_name, device)
        emb, meta = extract_ultrams_token_summary(
            model, samples, device, args.extract_batch_size, args.summary_parts
        )
        del model
    elif model_name == "dreams":
        emb, meta = extract_dreams_token_summary(
            samples, device, args.extract_batch_size, args.summary_parts
        )
    else:
        raise ValueError(f"Unsupported model for token summary: {model_name}")

    if emb.shape[0] != len(samples):
        raise RuntimeError(
            f"Extracted token summary row mismatch: model={model_name} split={split_id} "
            f"features={emb.shape[0]} samples={len(samples)}"
        )
    np.save(emb_path, emb.astype(np.float32))
    meta.update({
        "model": model_name,
        "split": int(split_id),
        "scenario": scenario_name,
        "path": str(emb_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2, default=json_default))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb, int(emb.shape[1]), str(emb_path), str(meta_path), meta


def run(args):
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "config": vars(args),
        "summary_note": (
            "Fixed NPLIB1 spectrum-to-molecule Top1 protocol; only spectrum representation "
            "is changed to peak-token summaries."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_default))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = {"config": vars(args), "scenarios": {}}

    for scenario_name in args.scenarios:
        scenario = probe.SCENARIOS[scenario_name]
        sc_sum = {"splits": {}, "description": scenario.get("desc", "")}
        all_rows_by_model = defaultdict(list)
        print("\n" + "=" * 88)
        print(f"Scenario {scenario_name}: {scenario.get('desc', '')}")

        for split_id in args.splits:
            samples, train_idx, val_idx, test_idx, _ = probe.load_samples_with_meta(split_id, scenario)
            with open(runner.candidates_path(split_id)) as f:
                candidates_by_block = json.load(f)

            train_f = probe.scenario_mask(samples, train_idx, scenario)
            val_f = probe.scenario_mask(samples, val_idx, scenario)
            test_f = probe.scenario_mask(samples, test_idx, scenario)

            smiles = set()
            smiles |= ms2fp.collect_smiles(samples, train_f)
            smiles |= ms2fp.collect_smiles(samples, val_f, candidates_by_block)
            smiles |= ms2fp.collect_smiles(samples, test_f, candidates_by_block)
            fp_dict = ms2fp.build_fp_dict(smiles, args.fp_bits, args.radius, args.fp_use_chirality)
            train_f = ms2fp.valid_indices(samples, train_f, fp_dict)
            val_f = ms2fp.valid_indices(samples, val_f, fp_dict)
            test_f = ms2fp.valid_indices(samples, test_f, fp_dict)
            if len(train_f) == 0 or len(val_f) == 0 or len(test_f) == 0:
                raise RuntimeError(
                    f"Empty split after filtering: scenario={scenario_name} split={split_id} "
                    f"train={len(train_f)} val={len(val_f)} test={len(test_f)}"
                )

            print(
                f"split {split_id}: train={len(train_f)} val={len(val_f)} "
                f"test={len(test_f)} fps={len(fp_dict)}"
            )
            sc_sum["splits"].setdefault(str(split_id), {})

            for model_name in args.models:
                print(f"\n[{model_name}] split {split_id}")
                spec_emb, spec_dim, emb_path, meta_path, emb_meta = get_token_summary_embeddings(
                    model_name, split_id, samples, scenario_name, device, args
                )
                print(
                    f"    token summary dim={spec_dim} path={emb_path} "
                    f"parts={emb_meta.get('summary_parts_used', [])}"
                )
                head, best, history, fp_stats, logit_shift = ms2fp.train_head(
                    spec_emb, samples, train_f, val_f, candidates_by_block, fp_dict, args, device
                )
                rows, skipped = ms2fp.evaluate(
                    head, spec_emb, samples, test_f, candidates_by_block, fp_dict, device,
                    args.score, logit_shift=logit_shift,
                )
                for row in rows:
                    row.update({"split": split_id, "scenario": scenario_name, "model": model_name})
                all_rows_by_model[model_name].extend(rows)

                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{model_name}.csv"
                hist_path = out_dir / f"history_{scenario_name}_split{split_id}_{model_name}.json"
                head_path = out_dir / f"ms2fp_head_{scenario_name}_split{split_id}_{model_name}_seed{args.seed}.pt"
                ms2fp.write_rows(row_path, rows)
                hist_path.write_text(json.dumps(history, indent=2, default=json_default))
                torch.save(head.state_dict(), head_path)

                metrics = {
                    "block_micro": probe.summarize_rows(rows),
                    "molecule_macro": probe.macro_summary(rows, "smiles"),
                    "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                    "stratified": probe.stratified_summary(rows),
                    "best_val": best,
                    "fp_stats": fp_stats,
                    "calibrate_pos_weight": bool(args.calibrate_pos_weight),
                    "skipped": skipped,
                    "token_summary_path": emb_path,
                    "token_summary_meta_path": meta_path,
                    "token_summary_meta": emb_meta,
                    "query_rows": str(row_path),
                    "history": str(hist_path),
                    "head_path": str(head_path),
                }
                sc_sum["splits"][str(split_id)][model_name] = metrics
                b = metrics["block_micro"]
                print(
                    f"    TEST n={b.get('n_queries', 0)} Top1={b.get('top1', 0):.2f} "
                    f"Top5={b.get('top5', 0):.2f} MRR={b.get('mrr', 0):.4f}"
                )
                del head
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aggregate = {}
        for model_name in args.models:
            rows = all_rows_by_model[model_name]
            aggregate[model_name] = {
                "block_micro": probe.summarize_rows(rows),
                "molecule_macro": probe.macro_summary(rows, "smiles"),
                "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                "stratified": probe.stratified_summary(rows),
            }
        if "rt_only_d11" in args.models and "dreams" in args.models:
            aggregate["ultrams_minus_dreams_bootstrap"] = probe.bootstrap_delta(
                all_rows_by_model["rt_only_d11"], all_rows_by_model["dreams"], seed=args.seed + 2201
            )
        sc_sum["aggregate"] = aggregate
        summary["scenarios"][scenario_name] = sc_sum
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2, default=json_default))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default))
    print("\nSaved", out_dir / "summary.json")
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for model_name in args.models:
            b = sc["aggregate"][model_name]["block_micro"]
            m = sc["aggregate"][model_name]["molecule_macro"]
            print(
                f"  {model_name}: block Top1={b.get('top1', 0):.2f} "
                f"MRR={b.get('mrr', 0):.4f}; mol Top1={m.get('top1', 0):.2f}"
            )
        print("  delta", sc["aggregate"].get("ultrams_minus_dreams_bootstrap", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="token_summary_ms2fp")
    ap.add_argument("--splits", nargs="+", type=int, default=[1])
    ap.add_argument("--models", nargs="+", default=["rt_only_d11", "dreams"], choices=["rt_only_d11", "dreams"])
    ap.add_argument("--scenarios", nargs="+", default=["ms2peaks_min50"], choices=sorted(probe.SCENARIOS))
    ap.add_argument("--summary-parts", nargs="+", default=DEFAULT_SUMMARY_PARTS, choices=SUMMARY_PART_CHOICES)
    ap.add_argument("--extract-batch-size", type=int, default=128)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fp-use-chirality", action="store_true")
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
    ap.add_argument("--candidate-ce-epochs", type=int, default=0)
    ap.add_argument("--candidate-ce-batch-size", type=int, default=64)
    ap.add_argument("--candidate-ce-lr", type=float, default=1e-4)
    ap.add_argument("--candidate-ce-temperature", type=float, default=32.0)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force-emb", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
