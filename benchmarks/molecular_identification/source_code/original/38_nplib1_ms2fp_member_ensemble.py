"""
Validation-tuned ensemble for NPLIB1 MS2->fingerprint retrieval runs.

This script keeps the existing NPLIB1 protocol intact: same MS2 queries, same
candidate lists, same MS2FP heads, and the same Top1/Top5/MRR metrics. It only
combines already-trained run outputs at score level. Fusion weights are selected
on validation queries and then frozen for test evaluation.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRAIN_DIR = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve() / "train"

MS2FP_PATH = HERE / "17_nplib1_ms2fp_retrieval.py"
spec = importlib.util.spec_from_file_location("ms2fp", MS2FP_PATH)
ms2fp = importlib.util.module_from_spec(spec)
sys.modules["ms2fp"] = ms2fp
spec.loader.exec_module(ms2fp)

probe = ms2fp.probe
runner = ms2fp.runner
OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2fp_member_ensemble"


@dataclass(frozen=True)
class MemberSpec:
    run_dir: Path
    model_key: str


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    members: tuple[MemberSpec, ...]


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)


def parse_member(text: str) -> MemberSpec:
    if "::" in text:
        run_dir, model_key = text.rsplit("::", 1)
    elif ":" in text:
        run_dir, model_key = text.rsplit(":", 1)
    else:
        raise ValueError(f"Member must be RUN_DIR::MODEL_KEY: {text}")
    return MemberSpec(Path(run_dir), model_key)


def parse_ensemble(text: str) -> EnsembleSpec:
    if "=" not in text:
        raise ValueError(f"Ensemble must be NAME=RUN_DIR::MODEL_KEY[,..]: {text}")
    name, rest = text.split("=", 1)
    members = tuple(parse_member(part) for part in rest.split(",") if part)
    if not members:
        raise ValueError(f"Ensemble has no members: {text}")
    return EnsembleSpec(name, members)


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


def ms2fp_cfg(summary: dict) -> dict:
    cfg = summary.get("config", {})
    merged = dict(cfg)
    merged.update(cfg.get("ms2fp_args", {}) or {})
    return merged


def metric_for(summary: dict, scenario_name: str, split_id: int, model_key: str) -> dict:
    return summary["scenarios"][scenario_name]["splits"][str(split_id)][model_key]


def embedding_path(metrics: dict) -> str:
    for key in ("embedding_path", "feature_path", "token_summary_path"):
        if metrics.get(key):
            return metrics[key]
    raise KeyError(f"No embedding/feature path in metrics keys={sorted(metrics)}")


def build_run_context(member: MemberSpec, scenario_name: str, split_id: int, samples,
                      train_idx, val_idx, test_idx, candidates_by_block, device):
    summary = json.loads((member.run_dir / "summary.json").read_text())
    cfg = ms2fp_cfg(summary)
    scenario = probe.SCENARIOS[scenario_name]
    metrics = metric_for(summary, scenario_name, split_id, member.model_key)
    fp_bits = int(cfg.get("fp_bits", 2048))
    radius = int(cfg.get("radius", 2))
    use_chirality = bool(cfg.get("fp_use_chirality", False))

    train_f = probe.scenario_mask(samples, train_idx, scenario)
    val_f = probe.scenario_mask(samples, val_idx, scenario)
    test_f = probe.scenario_mask(samples, test_idx, scenario)
    smiles = set()
    smiles |= ms2fp.collect_smiles(samples, train_f)
    smiles |= ms2fp.collect_smiles(samples, val_f, candidates_by_block)
    smiles |= ms2fp.collect_smiles(samples, test_f, candidates_by_block)
    fp_dict = ms2fp.build_fp_dict(smiles, fp_bits, radius, use_chirality)
    train_f = ms2fp.valid_indices(samples, train_f, fp_dict)
    spec_emb = np.load(embedding_path(metrics))
    head = ms2fp.FingerprintHead(
        spec_emb.shape[1],
        fp_dim=fp_bits,
        hidden=int(cfg.get("hidden", 2048)),
        dropout=float(cfg.get("dropout", 0.15)),
    ).to(device)
    head.load_state_dict(torch.load(metrics["head_path"], map_location=device))
    head.eval()

    logit_shift = None
    if cfg.get("calibrate_pos_weight", False):
        pw, _ = ms2fp.compute_pos_weight(
            samples, train_f, fp_dict, fp_bits, float(cfg.get("pos_weight_clip", 50.0))
        )
        logit_shift = torch.log(pw.clamp_min(1e-6))

    return {
        "member": f"{member.run_dir}::{member.model_key}",
        "run_dir": str(member.run_dir),
        "model_key": member.model_key,
        "cfg": cfg,
        "fp_dict": fp_dict,
        "spec_emb": spec_emb,
        "head": head,
        "logit_shift": logit_shift,
    }


def candidate_scores_for_context(ctx, sample_idx: int, candidates, device, score_mode: str):
    fp_dict = ctx["fp_dict"]
    cand = [c for c in candidates if c in fp_dict and fp_dict[c].sum() > 0]
    if not cand:
        return None, None
    with torch.no_grad():
        q = torch.from_numpy(ctx["spec_emb"][sample_idx:sample_idx + 1]).float().to(device)
        logits = ctx["head"](q).squeeze(0)
        if ctx["logit_shift"] is not None:
            logits = logits - ctx["logit_shift"].to(device)
        pred = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    cand_mat = np.stack([fp_dict[c] for c in cand]).astype(np.float32)
    scores = ms2fp.fingerprint_scores(pred, cand_mat, score_mode)
    return cand, scores


def simplex_weights(n: int, step: float):
    if n == 1:
        yield np.ones(1, dtype=np.float64)
        return
    units = int(round(1.0 / step))
    if units < 1 or abs(units * step - 1.0) > 1e-6:
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.1 or 0.05")

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


def build_score_cache(contexts, samples, eval_idx, candidates_by_block, args, device, phase: str):
    cache = []
    skipped = defaultdict(int)
    total = len(eval_idx)
    for n_done, qi in enumerate(eval_idx, 1):
        if n_done == 1 or n_done % 200 == 0 or n_done == total:
            print(f"      cache {phase}: {n_done}/{total}", flush=True)
        s = samples[int(qi)]
        sm = s.get("smiles", "")
        block_id = s["spec_id"]
        if not sm or sm == "nan":
            skipped["bad_smiles"] += 1
            continue
        if block_id not in candidates_by_block:
            skipped["no_candidate_key"] += 1
            continue
        cand = list(candidates_by_block.get(block_id, []))
        if not cand:
            skipped["no_candidates"] += 1
            continue
        if sm not in cand:
            skipped["gt_appended"] += 1
            cand.append(sm)
        seen = set()
        cand = [c for c in cand if not (c in seen or seen.add(c))]
        if sm not in cand:
            skipped["missing_gt_candidate"] += 1
            continue

        pos = {c: i for i, c in enumerate(cand)}
        score_mat = np.full((len(contexts), len(cand)), np.nan, dtype=np.float64)
        for mi, ctx in enumerate(contexts):
            scored_cand, scores = candidate_scores_for_context(ctx, int(qi), cand, device, args.score_mode)
            if scored_cand is None:
                continue
            norm = normalize_scores(scores, args.normalize)
            for c, sc in zip(scored_cand, norm):
                if c in pos:
                    score_mat[mi, pos[c]] = float(sc)
        gt_pos = cand.index(sm)
        valid_any = np.isfinite(score_mat).any(axis=0)
        if not valid_any.any() or not valid_any[gt_pos]:
            skipped["missing_gt_score"] += 1
            continue
        cache.append({
            "score_mat": score_mat,
            "gt_pos": gt_pos,
            "cand": cand,
            "meta": {
                "spec_id": block_id,
                "parent_spec_id": s.get("parent_spec_id", ""),
                "smiles": sm,
                "block_type": s.get("block_type", ""),
                "adduct": s.get("adduct", ""),
                "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                "phase": phase,
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

def write_rows(path: Path, rows):
    fields = [
        "split", "scenario", "model", "phase", "spec_id", "parent_spec_id", "smiles",
        "block_type", "adduct", "n_peaks_model", "n_peaks_original",
        "n_peaks_above_precursor_plus1", "candidate_count", "candidate_count_raw",
        "candidate_count_scored", "rank", "hit1", "hit5", "hit10", "hit20",
        "rr", "gt_score", "top1_score", "top1_is_gt", "top1_smiles", "n_members",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_grid(path: Path, rows, member_names):
    fields = ["rank", "top1", "top5", "mrr", "n_queries"] + [f"w_{safe_name(m)}" for m in member_names]
    ranked = sorted(rows, key=lambda r: (r["top1"], r["mrr"], r["top5"]), reverse=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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


def close_contexts(contexts):
    for ctx in contexts:
        del ctx["head"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args) -> None:
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensembles = [parse_ensemble(item) for item in args.ensembles]
    config = {
        "run_name": args.run_name,
        "ensembles": [
            {
                "name": e.name,
                "members": [{"run_dir": str(m.run_dir), "model_key": m.model_key} for m in e.members],
            }
            for e in ensembles
        ],
        "splits": args.splits,
        "scenarios": args.scenarios,
        "score_mode": args.score_mode,
        "normalize": args.normalize,
        "weight_step": args.weight_step,
        "baseline": args.baseline,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    summary = {"config": config, "scenarios": {}}

    for scenario_name in args.scenarios:
        scenario = probe.SCENARIOS[scenario_name]
        sc_sum = {"splits": {}, "description": scenario.get("desc", "")}
        rows_by_model = defaultdict(list)
        for split_id in args.splits:
            samples, train_idx, val_idx, test_idx, _ = probe.load_samples_with_meta(split_id, scenario)
            with open(runner.candidates_path(split_id)) as f:
                candidates_by_block = json.load(f)
            val_f = probe.scenario_mask(samples, val_idx, scenario)
            test_f = probe.scenario_mask(samples, test_idx, scenario)
            sc_sum["splits"].setdefault(str(split_id), {})

            for ens in ensembles:
                print(f"\n[{ens.name}] scenario={scenario_name} split={split_id}")
                contexts = [
                    build_run_context(member, scenario_name, split_id, samples, train_idx, val_idx,
                                      test_idx, candidates_by_block, device)
                    for member in ens.members
                ]
                member_names = [ctx["model_key"] for ctx in contexts]
                val_cache, val_skipped = build_score_cache(
                    contexts, samples, val_f, candidates_by_block, args, device, phase="val"
                )
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
                    weights, best_weight, grid_rows = choose_weights_from_cache(
                        val_cache, val_skipped, len(contexts), args
                    )

                test_cache, test_skipped = build_score_cache(
                    contexts, samples, test_f, candidates_by_block, args, device, phase="test"
                )
                test_rows = rows_from_cache(test_cache, weights)
                for row in test_rows:
                    row.update({"split": split_id, "scenario": scenario_name, "model": ens.name})
                rows_by_model[ens.name].extend(test_rows)

                safe = safe_name(ens.name)
                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{safe}.csv"
                grid_path = out_dir / f"weight_grid_{scenario_name}_split{split_id}_{safe}.csv"
                write_rows(row_path, test_rows)
                write_grid(grid_path, grid_rows, member_names)

                metrics = {
                    "block_micro": probe.summarize_rows(test_rows),
                    "molecule_macro": probe.macro_summary(test_rows, "smiles"),
                    "parent_macro": probe.macro_summary(test_rows, "parent_spec_id"),
                    "stratified": probe.stratified_summary(test_rows),
                    "skipped": test_skipped,
                    "query_rows": str(row_path),
                    "weight_grid": str(grid_path),
                    "selected_val": best_weight,
                    "selected_weights": weights.tolist(),
                    "members": [ctx["member"] for ctx in contexts],
                }
                sc_sum["splits"][str(split_id)][ens.name] = metrics
                block = metrics["block_micro"]
                print(
                    f"    val Top1={best_weight['top1']:.2f} weights={weights.tolist()} | "
                    f"TEST Top1={block.get('top1', 0):.2f} Top5={block.get('top5', 0):.2f} "
                    f"MRR={block.get('mrr', 0):.4f}"
                )
                close_contexts(contexts)

        aggregate = {}
        for ens in ensembles:
            rows = rows_by_model[ens.name]
            aggregate[ens.name] = {
                "block_micro": probe.summarize_rows(rows),
                "molecule_macro": probe.macro_summary(rows, "smiles"),
                "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                "stratified": probe.stratified_summary(rows),
            }
        if args.baseline in rows_by_model:
            base_rows = rows_by_model[args.baseline]
            for ens in ensembles:
                if ens.name == args.baseline:
                    continue
                aggregate[f"{ens.name}_minus_{args.baseline}_bootstrap"] = probe.bootstrap_delta(
                    rows_by_model[ens.name], base_rows, seed=args.seed + 991
                )
        sc_sum["aggregate"] = aggregate
        summary["scenarios"][scenario_name] = sc_sum
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("\nSaved", out_path)
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for ens in ensembles:
            b = sc["aggregate"][ens.name]["block_micro"]
            print(f"  {ens.name}: Top1={b.get('top1', 0):.2f} Top5={b.get('top5', 0):.2f} "
                  f"MRR={b.get('mrr', 0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="member_ensemble")
    ap.add_argument("--ensembles", nargs="+", required=True,
                    help="NAME=RUN_DIR::MODEL_KEY[,RUN_DIR::MODEL_KEY...]")
    ap.add_argument("--splits", nargs="+", type=int, default=[1])
    ap.add_argument("--scenarios", nargs="+", default=["ms2peaks_min50"], choices=sorted(probe.SCENARIOS))
    ap.add_argument("--score-mode", choices=[
        "tanimoto", "soft_tanimoto", "cosine", "dot", "pos_mean", "bernoulli", "hard_tanimoto"
    ], default="tanimoto")
    ap.add_argument("--normalize", choices=["zscore", "rank", "minmax", "none"], default="zscore")
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--baseline", default="dreams_embedding")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
