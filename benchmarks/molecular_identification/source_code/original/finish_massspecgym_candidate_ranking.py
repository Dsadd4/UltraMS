"""Finish MassSpecGym source-fusion ensembles from existing source heads.

This is a lightweight follow-up for a partially completed
39_massspecgym_ms2fp_sourcefusion.py run. It does not retrain source
fingerprint heads. It loads the saved source head paths/features from the
source run summary and only evaluates the fair two-source ensembles:

  DreaMS  = dreams_embedding + dreams_peak_iw
  UltraMS = u5elr8_rank_src + u5elr8_proj_mean_iw
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
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(HERE))


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


m39 = import_script(HERE / "39_massspecgym_ms2fp_sourcefusion.py", "msgym_sourcefusion39_finish")
ms2fp = m39.ms2fp


OUT_BASE = TRAIN_DIR / "output" / "comparison" / "massspecgym_ms2fp_sourcefusion"
FAIR_ENSEMBLES = {
    "dreams_global_iw": m39.DEFAULT_ENSEMBLES["dreams_global_iw"],
    "u5_cls_projmean": m39.DEFAULT_ENSEMBLES["u5_cls_projmean"],
}


def load_source_summary(run_name: str) -> dict:
    run_dir = OUT_BASE / run_name
    for name in ("summary.json", "summary.partial.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"missing summary in {run_dir}")


def infer_hidden(cli, source_summary: dict) -> int:
    if cli.hidden is not None:
        return int(cli.hidden)
    return int(source_summary.get("config", {}).get("hidden", 2048))


def build_args(cli, source_summary: dict) -> argparse.Namespace:
    hidden = infer_hidden(cli, source_summary)
    return argparse.Namespace(
        fp_bits=2048,
        radius=2,
        fp_use_chirality=False,
        include_train_candidates_in_fp=False,
        tanimoto_loss_weight=0.0,
        score=cli.score,
        epochs=30,
        val_every=5,
        batch_size=512,
        lr=5e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
        hidden=hidden,
        dropout=0.15,
        pos_weight_clip=50.0,
        calibrate_pos_weight=cli.calibrate_pos_weight,
        candidate_ce_epochs=10,
        candidate_ce_batch_size=64,
        candidate_ce_lr=5e-5,
        candidate_ce_temperature=24.0,
        balanced=False,
        normalize=cli.normalize,
        weight_step=cli.weight_step,
        seed=2026,
        device=cli.device,
        embed_batch_size=128,
        force_emb=False,
        force_fp=False,
        debug_load_limit_per_fold=0,
        debug_index_limit=0,
        fusion_batch_size=cli.fusion_batch_size,
    )


def predict_context(ctx: dict, eval_indices: np.ndarray, device, fp_dim: int) -> np.ndarray:
    pred = np.zeros((len(eval_indices), fp_dim), dtype=np.float32)
    ctx["head"].eval()
    with torch.no_grad():
        for start in range(0, len(eval_indices), 2048):
            idx = eval_indices[start:start + 2048]
            q_np = np.asarray(ctx["spec_emb"][idx], dtype=np.float32)
            q = torch.from_numpy(q_np).float().to(device)
            logits = ctx["head"](q)
            if ctx["logit_shift"] is not None:
                logits = logits - ctx["logit_shift"].to(device)
            pred[start:start + len(idx)] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return pred


def normalize_score_matrix(raw: np.ndarray, mode: str) -> np.ndarray:
    raw = raw.astype(np.float64, copy=False)
    if mode == "none":
        return raw
    if mode == "zscore":
        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True)
        return np.where(std < 1e-12, 0.0, (raw - mean) / np.maximum(std, 1e-12))
    if mode == "minmax":
        lo = raw.min(axis=1, keepdims=True)
        hi = raw.max(axis=1, keepdims=True)
        span = hi - lo
        return np.where(span < 1e-12, 0.0, (raw - lo) / np.maximum(span, 1e-12))
    if mode == "rank":
        order = np.argsort(np.argsort(raw, axis=1), axis=1)
        denom = max(raw.shape[1] - 1, 1)
        return order.astype(np.float64) / denom
    raise ValueError(mode)


def fast_build_score_cache(contexts, samples, eval_idx, candidates_by_spec, device, args, phase: str):
    skipped = defaultdict(int)
    items = []
    groups = defaultdict(list)
    cand_cache = {}
    fp_dict = contexts[0]["fp_dict"]

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
        gt_appended = 0
        if smi not in cand:
            cand.append(smi)
            gt_appended = 1
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
            "gt_appended": int(gt_appended),
        }
        groups[key].append(len(items))
        items.append(item)

    if not items:
        return [], dict(skipped)

    eval_indices = np.asarray([it["idx"] for it in items], dtype=np.int64)
    fp_dim = int(next(iter(fp_dict.values())).shape[0])
    print(
        f"      fast fusion cache {phase}: items={len(items)} groups={len(groups)} "
        f"contexts={len(contexts)}",
        flush=True,
    )
    preds = []
    for ci, ctx in enumerate(contexts, 1):
        print(f"      fast fusion cache {phase}: predict context {ci}/{len(contexts)}", flush=True)
        preds.append(predict_context(ctx, eval_indices, device, fp_dim))

    cache = []
    n_done = 0
    total_groups = len(groups)
    for gi, (key, item_positions) in enumerate(groups.items(), 1):
        if gi == 1 or gi % 1000 == 0 or gi == total_groups:
            print(f"      fast fusion cache {phase}: score groups {gi}/{total_groups}", flush=True)
        cache_item_base = cand_cache[key]
        cand = cache_item_base["cand"]
        cand_mat = cache_item_base["cand_mat"]
        pos_arr = np.asarray(item_positions, dtype=np.int64)
        per_context_scores = []
        for pred in preds:
            raw = m39.fingerprint_score_matrix(pred[pos_arr], cand_mat, args.score)
            per_context_scores.append(normalize_score_matrix(raw, args.normalize))
        stacked = np.stack(per_context_scores, axis=1)  # n_items, n_contexts, n_candidates
        for local_i, item_pos in enumerate(item_positions):
            item = items[item_pos]
            s = item["sample"]
            cache.append({
                "score_mat": stacked[local_i],
                "gt_pos": item["gt_pos"],
                "cand": cand,
                "meta": {
                    "candidate_task": "",
                    "split": "",
                    "scenario": "",
                    "model": "",
                    "phase": phase,
                    "spec_id": s["spec_id"],
                    "parent_spec_id": s.get("parent_spec_id", ""),
                    "smiles": s.get("smiles", ""),
                    "block_type": s.get("block_type", ""),
                    "adduct": s.get("adduct", ""),
                    "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                    "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                    "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                    "gt_appended": int(item["gt_appended"]),
                },
            })
            n_done += 1
    print(f"      fast fusion cache {phase}: rows={n_done} skipped={dict(skipped)}", flush=True)
    return cache, dict(skipped)


def torch_score_matrix(pred_t: torch.Tensor, cand_t: torch.Tensor, mask_t: torch.Tensor, mode: str) -> torch.Tensor:
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
        raise ValueError(f"unsupported GPU fusion score mode: {mode}")
    return scores.masked_fill(~mask_t, -1e9)


def torch_normalize_scores(scores: torch.Tensor, mask_t: torch.Tensor, mode: str) -> torch.Tensor:
    masked = scores.masked_fill(~mask_t, float("nan"))
    if mode == "none":
        return scores.masked_fill(~mask_t, -1e9)
    if mode == "zscore":
        count = mask_t.sum(dim=1, keepdim=True).clamp_min(1).to(scores.dtype)
        clean = scores.masked_fill(~mask_t, 0.0)
        mean = clean.sum(dim=1, keepdim=True) / count
        var = ((clean - mean).masked_fill(~mask_t, 0.0).pow(2).sum(dim=1, keepdim=True) / count).clamp_min(0.0)
        std = var.sqrt()
        return torch.where(std < 1e-12, torch.zeros_like(scores), (scores - mean) / std.clamp_min(1e-12)).masked_fill(~mask_t, -1e9)
    if mode == "minmax":
        lo = masked.nan_to_num(float("inf")).min(dim=1, keepdim=True).values
        hi = masked.nan_to_num(float("-inf")).max(dim=1, keepdim=True).values
        span = hi - lo
        return torch.where(span < 1e-12, torch.zeros_like(scores), (scores - lo) / span.clamp_min(1e-12)).masked_fill(~mask_t, -1e9)
    if mode == "rank":
        fill = scores.masked_fill(~mask_t, float("-inf"))
        ranks = torch.argsort(torch.argsort(fill, dim=1), dim=1).to(scores.dtype)
        denom = (mask_t.sum(dim=1, keepdim=True).to(scores.dtype) - 1.0).clamp_min(1.0)
        return (ranks / denom).masked_fill(~mask_t, -1e9)
    raise ValueError(mode)


def gpu_build_score_cache(contexts, samples, eval_idx, candidates_by_spec, device, args, phase: str):
    fp_dict = contexts[0]["fp_dict"]
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
        gt_appended = 0
        if smi not in cand:
            cand.append(smi)
            gt_appended = 1
            skipped["gt_appended"] += 1
        seen = set()
        cand = [c for c in cand if not (c in seen or seen.add(c))]
        cand = [c for c in cand if c in fp_dict and fp_dict[c].sum() > 0]
        if smi not in cand:
            skipped["missing_gt_fp"] += 1
            continue
        records.append({
            "idx": int(qi),
            "sample": s,
            "cand": cand,
            "gt_pos": cand.index(smi),
            "gt_appended": int(gt_appended),
        })

    if not records:
        return [], dict(skipped)

    batch_size = int(getattr(args, "fusion_batch_size", 64))
    fp_dim = int(next(iter(fp_dict.values())).shape[0])
    print(
        f"      gpu fusion cache {phase}: records={len(records)} contexts={len(contexts)} "
        f"batch_size={batch_size}",
        flush=True,
    )
    cache = []
    for start in range(0, len(records), batch_size):
        if start == 0 or start % (batch_size * 20) == 0 or start + batch_size >= len(records):
            print(f"      gpu fusion cache {phase}: queries {start + 1}/{len(records)}", flush=True)
        batch = records[start:start + batch_size]
        max_c = max(len(r["cand"]) for r in batch)
        cand_np = np.zeros((len(batch), max_c, fp_dim), dtype=np.float32)
        mask_np = np.zeros((len(batch), max_c), dtype=bool)
        idx_np = np.asarray([r["idx"] for r in batch], dtype=np.int64)
        for bi, rec in enumerate(batch):
            n_c = len(rec["cand"])
            cand_np[bi, :n_c] = np.stack([fp_dict[c] for c in rec["cand"]]).astype(np.float32)
            mask_np[bi, :n_c] = True
        cand_t = torch.from_numpy(cand_np).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(mask_np).to(device)
        context_scores = []
        with torch.no_grad():
            for ctx in contexts:
                q_np = np.asarray(ctx["spec_emb"][idx_np], dtype=np.float32)
                q_t = torch.from_numpy(q_np).float().to(device)
                logits = ctx["head"](q_t)
                if ctx["logit_shift"] is not None:
                    logits = logits - ctx["logit_shift"].to(device)
                pred_t = torch.sigmoid(logits)
                raw = torch_score_matrix(pred_t, cand_t, mask_t, args.score)
                norm = torch_normalize_scores(raw, mask_t, args.normalize)
                context_scores.append(norm.detach().cpu().numpy().astype(np.float64))
        stacked = np.stack(context_scores, axis=1)  # batch, contexts, max_c
        for bi, rec in enumerate(batch):
            s = rec["sample"]
            n_c = len(rec["cand"])
            cache.append({
                "score_mat": stacked[bi, :, :n_c],
                "gt_pos": int(rec["gt_pos"]),
                "cand": rec["cand"],
                "meta": {
                    "candidate_task": "",
                    "split": "",
                    "scenario": "",
                    "model": "",
                    "phase": phase,
                    "spec_id": s["spec_id"],
                    "parent_spec_id": s.get("parent_spec_id", ""),
                    "smiles": s.get("smiles", ""),
                    "block_type": s.get("block_type", ""),
                    "adduct": s.get("adduct", ""),
                    "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                    "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                    "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                    "gt_appended": int(rec["gt_appended"]),
                },
            })
        del cand_t, mask_t
        if torch.cuda.is_available() and start % (batch_size * 40) == 0:
            torch.cuda.empty_cache()
    print(f"      gpu fusion cache {phase}: rows={len(cache)} skipped={dict(skipped)}", flush=True)
    return cache, dict(skipped)


def run(cli) -> None:
    m39.build_score_cache = gpu_build_score_cache
    source_summary = load_source_summary(cli.source_run)
    ensembles = FAIR_ENSEMBLES
    if cli.ensembles:
        ensembles = {name: FAIR_ENSEMBLES[name] for name in cli.ensembles}
    src_sc = source_summary["tasks"][cli.task]["scenarios"][cli.scenario]
    source_metrics = src_sc["sources"]
    missing = sorted(
        m for members in ensembles.values() for m in members
        if m not in source_metrics
    )
    if missing:
        raise KeyError(f"missing source metrics in source run: {missing}")

    out_dir = OUT_BASE / cli.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    args = build_args(cli, source_summary)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    samples, train_idx, val_idx, test_idx, data_audit = m39.load_massspecgym(0)
    candidates_by_spec, cand_audit = m39.load_candidates(cli.task, samples)
    scenario_cfg = m39.SCENARIOS[cli.scenario]
    train_f = m39.scenario_mask(samples, train_idx, scenario_cfg)
    val_f = m39.scenario_mask(samples, val_idx, scenario_cfg)
    test_f = m39.scenario_mask(samples, test_idx, scenario_cfg)

    fp_path = Path(src_sc["fingerprint_cache"])
    if not fp_path.exists():
        raise FileNotFoundError(fp_path)
    print(f"Loading source fingerprint cache: {fp_path}", flush=True)
    import pickle
    with fp_path.open("rb") as fh:
        fp_dict = pickle.load(fh)
    train_f = ms2fp.valid_indices(samples, train_f, fp_dict)
    val_f = ms2fp.valid_indices(samples, val_f, fp_dict)
    test_f = ms2fp.valid_indices(samples, test_f, fp_dict)

    config = {
        "run_name": cli.run_name,
        "source_run": cli.source_run,
        "task": cli.task,
        "scenario": cli.scenario,
        "ensembles": ensembles,
        "score": args.score,
        "normalize": args.normalize,
        "weight_step": args.weight_step,
        "hidden": args.hidden,
        "calibrate_pos_weight": bool(args.calibrate_pos_weight),
        "source_fingerprint_cache": str(fp_path),
        "note": "ensemble-only run; source heads/features are loaded from source_run",
    }
    summary = {
        "config": config,
        "data_audit": data_audit,
        "tasks": {
            cli.task: {
                "candidate_audit": cand_audit,
                "scenarios": {
                    cli.scenario: {
                        "description": scenario_cfg.get("desc", ""),
                        "n_train": int(len(train_f)),
                        "n_val": int(len(val_f)),
                        "n_test": int(len(test_f)),
                        "fingerprint_cache": str(fp_path),
                        "sources_from": cli.source_run,
                        "sources": source_metrics,
                        "ensembles": {},
                    }
                },
            }
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=m39.json_default) + "\n")

    sc_sum = summary["tasks"][cli.task]["scenarios"][cli.scenario]
    print(
        f"Finish ensembles task={cli.task} scenario={cli.scenario} "
        f"train={len(train_f)} val={len(val_f)} test={len(test_f)}",
        flush=True,
    )
    for ens_name, members in ensembles.items():
        metrics = m39.run_ensemble(
            ens_name, members, source_metrics, out_dir,
            samples, train_f, val_f, test_f, candidates_by_spec, fp_dict,
            device, args, cli.task, cli.scenario,
        )
        sc_sum["ensembles"][ens_name] = metrics
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=m39.json_default) + "\n"
        )
        m39.write_metrics_table(summary, out_dir)

    if "dreams_global_iw" in sc_sum["ensembles"] and "u5_cls_projmean" in sc_sum["ensembles"]:
        d = sc_sum["ensembles"]["dreams_global_iw"]["block_micro"].get("top1", 0.0)
        u = sc_sum["ensembles"]["u5_cls_projmean"]["block_micro"].get("top1", 0.0)
        print(f"PRIMARY DELTA sourcefusion mass/all: UltraMS - DreaMS Top1={u - d:.4f} pp", flush=True)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=m39.json_default) + "\n"
    )
    m39.write_metrics_table(summary, out_dir)
    print("\nSaved", out_dir / "summary.json", flush=True)
    print((out_dir / "metrics.csv").read_text(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", default="sourcefusion_all_formula_mass_20260703a")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--task", default="mass", choices=["formula", "mass"])
    ap.add_argument("--scenario", default="all", choices=sorted(m39.SCENARIOS))
    ap.add_argument("--ensembles", nargs="*", default=None, choices=sorted(FAIR_ENSEMBLES))
    ap.add_argument("--score", default="tanimoto", choices=["tanimoto", "soft_tanimoto", "cosine", "dot", "pos_mean", "bernoulli", "hard_tanimoto"])
    ap.add_argument("--normalize", default="zscore", choices=["zscore", "rank", "minmax", "none"])
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--fusion-batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--calibrate-pos-weight", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
