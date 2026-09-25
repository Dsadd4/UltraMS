"""
NPLIB1 spectrum-to-molecule retrieval through molecular fingerprint prediction.

Protocol:
  spectrum embedding -> Morgan fingerprint logits
  candidate SMILES -> Morgan fingerprint bits
  rank candidates by soft Tanimoto(predicted fingerprint, candidate fingerprint)

This tests whether UltraMS carries more directly recoverable structural
fingerprint information than DreaMS on the corrected NPLIB1 MS2-block data.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


HERE = Path(__file__).resolve().parent
TRAIN_DIR = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve() / "train"

RUNNER_PATH = HERE / "11_nplib1_cross_modal_existing_interface_ms2blocks.py"
PROBE_PATH = HERE / "13_nplib1_ms2blocks_advantage_probe.py"

spec = importlib.util.spec_from_file_location("nplib1_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules["nplib1_runner"] = runner
spec.loader.exec_module(runner)

probe_spec = importlib.util.spec_from_file_location("nplib1_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(probe_spec)
sys.modules["nplib1_probe"] = probe
probe_spec.loader.exec_module(probe)

OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2fp_retrieval"


class FingerprintHead(nn.Module):
    def __init__(self, in_dim: int, fp_dim: int = 2048, hidden: int = 2048, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, fp_dim),
        )

    def forward(self, x):
        return self.net(x)


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2, use_chirality: bool = False) -> np.ndarray:
    arr = np.zeros(n_bits, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return arr
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, useChirality=use_chirality
        )
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr


def build_fp_dict(smiles_list, n_bits: int, radius: int, use_chirality: bool = False):
    return {
        sm: morgan_fp(sm, n_bits=n_bits, radius=radius, use_chirality=use_chirality)
        for sm in sorted(set(smiles_list))
    }


def collect_smiles(samples, indices, candidates_by_block=None):
    smiles = set()
    for i in indices:
        s = samples[int(i)]
        sm = s.get("smiles", "")
        if sm and sm != "nan":
            smiles.add(sm)
        if candidates_by_block is not None:
            smiles.update(candidates_by_block.get(s["spec_id"], []))
    return smiles


def valid_indices(samples, indices, fp_dict):
    keep = []
    for i in indices:
        sm = samples[int(i)].get("smiles", "")
        if sm and sm != "nan" and sm in fp_dict and fp_dict[sm].sum() > 0:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def make_epoch_indices(samples, train_idx, balanced, seed):
    if not balanced:
        return np.asarray(train_idx, dtype=np.int64)
    groups = defaultdict(list)
    for i in train_idx:
        groups[samples[int(i)]["smiles"]].append(int(i))
    rng = np.random.default_rng(seed)
    out = [int(rng.choice(v)) for v in groups.values()]
    rng.shuffle(out)
    return np.asarray(out, dtype=np.int64)


def compute_pos_weight(samples, train_idx, fp_dict, fp_dim, clip):
    mat = np.stack([fp_dict[samples[int(i)]["smiles"]] for i in train_idx]).astype(np.float32)
    pos = mat.sum(axis=0)
    neg = mat.shape[0] - pos
    weight = (neg + 1.0) / (pos + 1.0)
    weight = np.clip(weight, 1.0, clip)
    return torch.from_numpy(weight.astype(np.float32)), {
        "n_train": int(mat.shape[0]),
        "mean_on_bits": float(mat.sum(axis=1).mean()),
        "median_on_bits": float(np.median(mat.sum(axis=1))),
        "mean_pos_weight": float(weight.mean()),
        "max_pos_weight": float(weight.max()),
    }


def soft_tanimoto_scores(pred_prob: np.ndarray, cand_mat: np.ndarray):
    inter = cand_mat @ pred_prob
    denom = pred_prob.sum() + cand_mat.sum(axis=1) - inter
    return inter / np.maximum(denom, 1e-8)


def soft_tanimoto_loss(logits: torch.Tensor, target: torch.Tensor):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1) - inter
    return (1.0 - inter / denom.clamp_min(1e-8)).mean()


def cosine_scores(pred_prob: np.ndarray, cand_mat: np.ndarray):
    denom = np.linalg.norm(cand_mat, axis=1) * max(float(np.linalg.norm(pred_prob)), 1e-8)
    return (cand_mat @ pred_prob) / np.maximum(denom, 1e-8)


def fingerprint_scores(pred_prob: np.ndarray, cand_mat: np.ndarray, mode: str):
    eps = 1e-6
    if mode in {"tanimoto", "soft_tanimoto"}:
        return soft_tanimoto_scores(pred_prob, cand_mat)
    if mode == "cosine":
        return cosine_scores(pred_prob, cand_mat)
    if mode == "dot":
        return cand_mat @ pred_prob
    if mode == "pos_mean":
        return (cand_mat @ pred_prob) / np.maximum(cand_mat.sum(axis=1), 1.0)
    if mode == "bernoulli":
        p = np.clip(pred_prob, eps, 1.0 - eps)
        return cand_mat @ np.log(p) + (1.0 - cand_mat) @ np.log(1.0 - p)
    if mode == "hard_tanimoto":
        hard = (pred_prob >= 0.5).astype(np.float32)
        return soft_tanimoto_scores(hard, cand_mat)
    raise ValueError(f"unknown score mode: {mode}")


def make_candidate_examples(samples, indices, candidates_by_block, fp_dict):
    examples = []
    skipped = defaultdict(int)
    for qi in indices:
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
        cand = [c for c in cand if c in fp_dict and fp_dict[c].sum() > 0]
        if sm not in cand:
            skipped["missing_gt_fp"] += 1
            continue
        examples.append({"idx": int(qi), "candidates": cand, "target": int(cand.index(sm))})
    return examples, dict(skipped)


def candidate_ce_scores(logits: torch.Tensor, cand_t: torch.Tensor, mask_t: torch.Tensor,
                        logit_shift: torch.Tensor | None, temperature: float):
    if logit_shift is not None:
        logits = logits - logit_shift.to(logits.device)
    log_p = -F.softplus(-logits)
    log_not_p = -F.softplus(logits)
    scores = (cand_t * log_p[:, None, :] + (1.0 - cand_t) * log_not_p[:, None, :]).sum(dim=-1)
    scores = scores.masked_fill(~mask_t, -1e9)
    return scores / max(float(temperature), 1e-6)


def finetune_candidate_ce(head, spec_emb, samples, train_idx, val_idx, candidates_by_block, fp_dict,
                          args, device, logit_shift, best, history):
    train_examples, train_skip = make_candidate_examples(samples, train_idx, candidates_by_block, fp_dict)
    if not train_examples:
        print(f"      candidate CE skipped: no usable train examples; skipped={train_skip}")
        return head, best, history, {"train_skipped": train_skip, "val_skipped": {}}

    opt = torch.optim.AdamW(head.parameters(), lr=args.candidate_ce_lr, weight_decay=args.weight_decay)
    spec_t = torch.from_numpy(spec_emb).float().to(device)
    rng = np.random.default_rng(args.seed + 9001)
    ce_best = {
        "phase": best.get("phase", "bce"),
        "epoch": best.get("epoch", -1),
        "val_top1": best.get("val_top1", -1.0),
        "val_mrr": best.get("val_mrr", -1.0),
        "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }
    last_val_skip = {}
    print(f"      candidate CE train examples={len(train_examples)} skipped={train_skip}")

    for ce_epoch in range(1, args.candidate_ce_epochs + 1):
        head.train()
        order = np.arange(len(train_examples))
        rng.shuffle(order)
        total = 0.0
        n_seen = 0
        for start in range(0, len(order), args.candidate_ce_batch_size):
            batch_ids = order[start:start + args.candidate_ce_batch_size]
            batch = [train_examples[int(i)] for i in batch_ids]
            idx = np.asarray([b["idx"] for b in batch], dtype=np.int64)
            max_c = max(len(b["candidates"]) for b in batch)
            cand_np = np.zeros((len(batch), max_c, args.fp_bits), dtype=np.float32)
            mask_np = np.zeros((len(batch), max_c), dtype=bool)
            target_np = np.asarray([b["target"] for b in batch], dtype=np.int64)
            for bi, ex in enumerate(batch):
                for ci, sm in enumerate(ex["candidates"]):
                    cand_np[bi, ci] = fp_dict[sm]
                    mask_np[bi, ci] = True
            logits = head(spec_t[idx])
            cand_t = torch.from_numpy(cand_np).to(device)
            mask_t = torch.from_numpy(mask_np).to(device)
            target_t = torch.from_numpy(target_np).to(device)
            scores = candidate_ce_scores(
                logits, cand_t, mask_t, logit_shift, args.candidate_ce_temperature
            )
            loss = F.cross_entropy(scores, target_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item()) * len(batch)
            n_seen += len(batch)

        val_rows, last_val_skip = evaluate(
            head, spec_emb, samples, val_idx, candidates_by_block, fp_dict, device,
            args.score, logit_shift=logit_shift,
        )
        val = probe.summarize_rows(val_rows)
        row = {
            "phase": "candidate_ce",
            "epoch": ce_epoch,
            "train_loss": total / max(n_seen, 1),
            "val_top1": val.get("top1", 0.0),
            "val_mrr": val.get("mrr", 0.0),
            "val_n": val.get("n_queries", 0),
            "val_skipped": last_val_skip,
        }
        better = (row["val_top1"] > ce_best["val_top1"]) or (
            row["val_top1"] == ce_best["val_top1"] and row["val_mrr"] > ce_best["val_mrr"]
        )
        if better:
            ce_best = {
                "phase": "candidate_ce",
                "epoch": ce_epoch,
                "val_top1": row["val_top1"],
                "val_mrr": row["val_mrr"],
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
        print(f"      candidate CE epoch {ce_epoch}: train={row['train_loss']:.4f} "
              f"val_top1={row['val_top1']:.2f} val_mrr={row['val_mrr']:.4f}"
              f"{' *best' if better else ''}")
        history.append(row)

    head.load_state_dict(ce_best["state"])
    clean_best = {k: v for k, v in ce_best.items() if k != "state"}
    return head, clean_best, history, {"train_skipped": train_skip, "val_skipped": last_val_skip}


def evaluate(head, spec_emb, samples, eval_idx, candidates_by_block, fp_dict, device, score_mode,
             logit_shift=None):
    rows = []
    skipped = defaultdict(int)
    head.eval()
    with torch.no_grad():
        for qi in eval_idx:
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
                cand.append(sm)
            seen = set()
            cand = [c for c in cand if not (c in seen or seen.add(c))]
            cand_raw_count = len(cand)
            cand = [c for c in cand if c in fp_dict and fp_dict[c].sum() > 0]
            if sm not in cand:
                skipped["missing_gt_fp"] += 1
                continue
            q = torch.from_numpy(spec_emb[int(qi):int(qi) + 1]).float().to(device)
            logits = head(q).squeeze(0)
            if logit_shift is not None:
                logits = logits - logit_shift.to(device)
            pred = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            cand_mat = np.stack([fp_dict[c] for c in cand]).astype(np.float32)
            scores = fingerprint_scores(pred, cand_mat, score_mode)
            order = np.argsort(-scores)
            gt_pos = cand.index(sm)
            rank = int(np.where(order == gt_pos)[0][0] + 1)
            top_pos = int(order[0])
            rows.append({
                "spec_id": block_id,
                "parent_spec_id": s.get("parent_spec_id", ""),
                "smiles": sm,
                "block_type": s.get("block_type", ""),
                "adduct": s.get("adduct", ""),
                "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                "candidate_count": int(len(cand)),
                "candidate_count_raw": int(cand_raw_count),
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
                "pred_on_mass": float(pred.sum()),
            })
    return rows, dict(skipped)


def train_head(spec_emb, samples, train_idx, val_idx, candidates_by_block, fp_dict, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    head = FingerprintHead(spec_emb.shape[1], fp_dim=args.fp_bits, hidden=args.hidden, dropout=args.dropout).to(device)
    pw, fp_stats = compute_pos_weight(samples, train_idx, fp_dict, args.fp_bits, args.pos_weight_clip)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    spec_t = torch.from_numpy(spec_emb).float().to(device)
    best = {"epoch": -1, "val_top1": -1.0, "val_mrr": -1.0, "state": None}
    history = []
    print(f"    train spectra={len(train_idx)} fp_stats={fp_stats}")

    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_idx = make_epoch_indices(samples, train_idx, args.balanced, args.seed + epoch)
        rng = np.random.default_rng(args.seed + epoch * 17)
        rng.shuffle(epoch_idx)
        total = 0.0
        n_seen = 0
        for start in range(0, len(epoch_idx), args.batch_size):
            idx = epoch_idx[start:start + args.batch_size]
            target = np.stack([fp_dict[samples[int(i)]["smiles"]] for i in idx]).astype(np.float32)
            target_t = torch.from_numpy(target).to(device)
            logits = head(spec_t[idx])
            bce_loss = criterion(logits, target_t)
            loss = bce_loss
            if args.tanimoto_loss_weight > 0:
                loss = loss + args.tanimoto_loss_weight * soft_tanimoto_loss(logits, target_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.item()) * len(idx)
            n_seen += len(idx)
        sched.step()
        row = {"epoch": epoch, "train_loss": total / max(n_seen, 1), "lr": sched.get_last_lr()[0]}
        if epoch % args.val_every == 0 or epoch == 1 or epoch == args.epochs:
            logit_shift = torch.log(pw.clamp_min(1e-6)) if args.calibrate_pos_weight else None
            val_rows, val_skip = evaluate(
                head, spec_emb, samples, val_idx, candidates_by_block, fp_dict, device,
                args.score, logit_shift=logit_shift,
            )
            val = probe.summarize_rows(val_rows)
            row.update({
                "val_top1": val.get("top1", 0.0),
                "val_mrr": val.get("mrr", 0.0),
                "val_n": val.get("n_queries", 0),
                "val_skipped": val_skip,
            })
            better = (row["val_top1"] > best["val_top1"]) or (
                row["val_top1"] == best["val_top1"] and row["val_mrr"] > best["val_mrr"]
            )
            if better:
                best = {
                    "epoch": epoch,
                    "val_top1": row["val_top1"],
                    "val_mrr": row["val_mrr"],
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
            print(f"      epoch {epoch}: train={row['train_loss']:.4f} "
                  f"val_top1={row['val_top1']:.2f} val_mrr={row['val_mrr']:.4f}"
                  f"{' *best' if better else ''}")
        history.append(row)
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    logit_shift = torch.log(pw.clamp_min(1e-6)) if args.calibrate_pos_weight else None
    if args.candidate_ce_epochs > 0:
        head, best, history, ce_stats = finetune_candidate_ce(
            head, spec_emb, samples, train_idx, val_idx, candidates_by_block, fp_dict,
            args, device, logit_shift, {**best, "phase": "bce"}, history
        )
        fp_stats["candidate_ce"] = ce_stats
    return head, {k: v for k, v in best.items() if k != "state"}, history, fp_stats, logit_shift


def write_rows(path: Path, rows):
    fields = [
        "split", "scenario", "model", "spec_id", "parent_spec_id", "smiles",
        "block_type", "adduct", "n_peaks_model", "n_peaks_original",
        "n_peaks_above_precursor_plus1", "candidate_count", "candidate_count_raw",
        "candidate_count_scored", "rank", "hit1",
        "hit5", "hit10", "hit20", "rr", "gt_score", "top1_score",
        "top1_is_gt", "top1_smiles", "pred_on_mass",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def run(args):
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
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
            smiles |= collect_smiles(samples, train_f)
            smiles |= collect_smiles(samples, val_f, candidates_by_block)
            smiles |= collect_smiles(samples, test_f, candidates_by_block)
            fp_dict = build_fp_dict(smiles, args.fp_bits, args.radius, args.fp_use_chirality)
            train_f = valid_indices(samples, train_f, fp_dict)
            val_f = valid_indices(samples, val_f, fp_dict)
            test_f = valid_indices(samples, test_f, fp_dict)
            print(f"split {split_id}: train={len(train_f)} val={len(val_f)} test={len(test_f)} fps={len(fp_dict)}")
            sc_sum["splits"].setdefault(str(split_id), {})
            for model_name in args.models:
                spec_emb, _, emb_path = probe.get_spec_embeddings(
                    model_name, split_id, samples, scenario_name, scenario, device, args.force_emb
                )
                print(f"\n[{model_name}] split {split_id}")
                head, best, history, fp_stats, logit_shift = train_head(
                    spec_emb, samples, train_f, val_f, candidates_by_block, fp_dict, args, device
                )
                rows, skipped = evaluate(
                    head, spec_emb, samples, test_f, candidates_by_block, fp_dict, device,
                    args.score, logit_shift=logit_shift,
                )
                for r in rows:
                    r.update({"split": split_id, "scenario": scenario_name, "model": model_name})
                all_rows_by_model[model_name].extend(rows)
                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{model_name}.csv"
                write_rows(row_path, rows)
                hist_path = out_dir / f"history_{scenario_name}_split{split_id}_{model_name}.json"
                hist_path.write_text(json.dumps(history, indent=2))
                head_path = out_dir / f"ms2fp_head_{scenario_name}_split{split_id}_{model_name}_seed{args.seed}.pt"
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
                    "embedding_path": emb_path,
                    "query_rows": str(row_path),
                    "history": str(hist_path),
                    "head_path": str(head_path),
                }
                sc_sum["splits"][str(split_id)][model_name] = metrics
                b = metrics["block_micro"]
                print(f"    TEST n={b.get('n_queries', 0)} Top1={b.get('top1', 0):.2f} "
                      f"Top5={b.get('top5', 0):.2f} MRR={b.get('mrr', 0):.4f}")
                del head
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        ag = {}
        for model_name in args.models:
            rows = all_rows_by_model[model_name]
            ag[model_name] = {
                "block_micro": probe.summarize_rows(rows),
                "molecule_macro": probe.macro_summary(rows, "smiles"),
                "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                "stratified": probe.stratified_summary(rows),
            }
        if "rt_only_d11" in args.models and "dreams" in args.models:
            ag["ultrams_minus_dreams_bootstrap"] = probe.bootstrap_delta(
                all_rows_by_model["rt_only_d11"], all_rows_by_model["dreams"], seed=args.seed + 177
            )
        sc_sum["aggregate"] = ag
        summary["scenarios"][scenario_name] = sc_sum
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved", out_dir / "summary.json")
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for model_name in args.models:
            b = sc["aggregate"][model_name]["block_micro"]
            m = sc["aggregate"][model_name]["molecule_macro"]
            print(f"  {model_name}: block Top1={b.get('top1', 0):.2f} MRR={b.get('mrr', 0):.4f}; "
                  f"mol Top1={m.get('top1', 0):.2f}")
        print("  delta", sc["aggregate"].get("ultrams_minus_dreams_bootstrap", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ms2fp")
    ap.add_argument("--splits", nargs="+", type=int, default=[1])
    ap.add_argument("--models", nargs="+", default=["rt_only_d11", "dreams"], choices=["rt_only_d11", "dreams"])
    ap.add_argument("--scenarios", nargs="+", default=["ms2peaks_min50"], choices=sorted(probe.SCENARIOS))
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fp-use-chirality", action="store_true")
    ap.add_argument("--tanimoto-loss-weight", type=float, default=0.0)
    ap.add_argument("--score", choices=["tanimoto", "soft_tanimoto", "cosine", "dot", "pos_mean", "bernoulli", "hard_tanimoto"], default="tanimoto")
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
