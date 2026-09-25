
"""
NPLIB1/CANOPUS MS2-block spectrum-to-molecule retrieval runner.

Primary data rule:
  one ms2peaks/collision block = one spectrum sample;
  molecule annotation inherited from labels.tsv via parent_spec_id;
  no MS2 block merging.

This reuses evaluate_molecule_identification.py for model loading,
embedding extraction, molecule embeddings, projection training, and metrics style,
but evaluates candidates by block_id instead of SMILES key.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"
NPLIB1_ROOT = ROOT / "datasets" / "NPLIB1"
OUT_DIR = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_existing_interface"
CM_PATH = HERE / "evaluate_molecule_identification.py"

spec = importlib.util.spec_from_file_location("cm10", CM_PATH)
cm10 = importlib.util.module_from_spec(spec)
sys.modules["cm10"] = cm10
spec.loader.exec_module(cm10)


def fixed_dreams_loader():
    src = HERE / "dreams_loader.py"
    sp = importlib.util.spec_from_file_location("dreams_loader", src)
    mod = importlib.util.module_from_spec(sp)
    mod.os = os
    sys.modules["dreams_loader"] = mod
    sp.loader.exec_module(mod)
    return mod


cm10._load_dreams_loader = fixed_dreams_loader


def csv_path(split_id: int) -> Path:
    return NPLIB1_ROOT / f"NPLIB1_ms2blocks_split{split_id}.csv"


def candidates_path(split_id: int) -> Path:
    return NPLIB1_ROOT / f"NPLIB1_ms2blocks_candidates_by_block_split{split_id}.json"


def load_ms2block_dataset(split_id: int):
    import pandas as pd
    path = csv_path(split_id)
    df = pd.read_csv(path)
    samples, folds = [], []
    for _, row in df.iterrows():
        mzs = np.array([float(x) for x in str(row["mzs"]).split(",") if x != ""], dtype=np.float32)
        ints = np.array([float(x) for x in str(row["intensities"]).split(",") if x != ""], dtype=np.float32)
        if len(mzs) != len(ints):
            raise ValueError(f"m/z and intensity length mismatch for {row['spec_id']}")
        spectrum = np.stack([mzs, ints], axis=-1)
        sm = str(row.get("smiles", ""))
        samples.append({
            "spec_id": str(row["spec_id"]),
            "parent_spec_id": str(row.get("parent_spec_id", "")),
            "block_index": int(row.get("block_index", -1)),
            "block_header": str(row.get("block_header", "")),
            "spectrum": spectrum,
            "precursor_mz": float(row["precursor_mz"]),
            "smiles": sm if sm != "nan" else "",
        })
        folds.append(row.get("fold", "test"))
    folds = np.array(folds)
    train_idx = np.where(folds == "train")[0]
    val_idx = np.where(folds == "val")[0]
    test_idx = np.where(folds == "test")[0]
    print(f"[nplib1_ms2blocks split {split_id}] {len(samples)} total "
          f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    return samples, train_idx, val_idx, test_idx, {"csv": str(path), "candidates_mass": str(candidates_path(split_id))}


def cache_has_all(cache_path: Path, smiles: list[str]) -> bool:
    if not cache_path.exists():
        return False
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return all(sm in cached for sm in smiles)
    except Exception:
        return False


def evaluate_retrieval_by_block(spec_emb, proj, mol_emb_dict, samples, test_idx,
                                candidates_by_block, device, top_k=(1, 5, 10, 20),
                                stratified=False, collect_raw=False):
    ranks, n_cands_list = [], []
    gt_sims, top1_sims, topn_sims = [], [], []
    skipped = {"no_candidate_key": 0, "bad_smiles": 0, "missing_mol_embedding": 0}

    for qi in test_idx:
        sample = samples[int(qi)]
        block_id = sample["spec_id"]
        sm = sample["smiles"]
        if not sm or sm == "nan":
            skipped["bad_smiles"] += 1
            continue
        if block_id not in candidates_by_block:
            skipped["no_candidate_key"] += 1
            continue
        cand_list = list(candidates_by_block[block_id])
        if sm not in cand_list:
            cand_list.append(sm)
        available_cands = [c for c in cand_list if c in mol_emb_dict]
        if sm not in available_cands:
            skipped["missing_mol_embedding"] += 1
            continue

        q_emb = torch.from_numpy(spec_emb[int(qi):int(qi)+1]).float().to(device)
        q_proj = F.normalize(proj(q_emb), dim=-1) if proj is not None else F.normalize(q_emb, dim=-1)
        c_embs = torch.stack([mol_emb_dict[c] for c in available_cands]).to(device)
        c_embs = F.normalize(c_embs, dim=-1)
        sims = (q_proj @ c_embs.T).squeeze(0)
        sorted_idx = torch.argsort(sims, descending=True)
        gt_pos = available_cands.index(sm)
        rank = (sorted_idx == gt_pos).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        n_cands_list.append(len(available_cands))

        if collect_raw:
            sims_np = sims.detach().cpu().float().numpy()
            sorted_sims = sims_np[sorted_idx.detach().cpu().numpy()]
            gt_sims.append(float(sims_np[gt_pos]))
            top1_sims.append(float(sorted_sims[0]))
            n_take = min(cm10.N_RAW_RANKS, len(sorted_sims))
            row = np.full(cm10.N_RAW_RANKS, np.nan, dtype=np.float32)
            row[:n_take] = sorted_sims[:n_take]
            topn_sims.append(row)

    if len(ranks) < 10:
        print(f"    Only {len(ranks)} evaluable; skipped={skipped}")
        return {}

    ranks = np.array(ranks)
    n_cands_arr = np.array(n_cands_list)
    results = {
        "n_queries": int(len(ranks)),
        "avg_candidates": float(np.mean(n_cands_arr)),
        "skipped": skipped,
    }
    for k in top_k:
        results[f"top{k}"] = float((ranks <= k).mean() * 100)
    results["mrr"] = float((1.0 / ranks).mean())
    results["median_rank"] = int(np.median(ranks))

    print(f"    n={len(ranks)}, avg_cands={np.mean(n_cands_arr):.0f}: "
          f"Top1={results['top1']:.2f}%  Top5={results['top5']:.2f}%  "
          f"Top10={results['top10']:.2f}%  Top20={results['top20']:.2f}%  "
          f"MRR={results['mrr']:.4f}  skipped={skipped}")

    if stratified:
        results["stratified"] = {}
        for lo, hi in cm10.CAND_BINS:
            mask = (n_cands_arr >= lo) & (n_cands_arr < hi)
            cnt = int(mask.sum())
            if cnt < 5:
                continue
            bin_ranks = ranks[mask]
            label = f"{lo}-{int(hi) if hi != float('inf') else 'inf'}"
            bin_r = {"n": cnt, "avg_cands": float(n_cands_arr[mask].mean())}
            for k in top_k:
                bin_r[f"top{k}"] = float((bin_ranks <= k).mean() * 100)
            bin_r["mrr"] = float((1.0 / bin_ranks).mean())
            results["stratified"][label] = bin_r
            print(f"      [{label}] n={cnt}, top1={bin_r['top1']:.2f}%  mrr={bin_r['mrr']:.4f}")

    if collect_raw:
        results["_raw"] = {
            "gt_sims": np.array(gt_sims, dtype=np.float32),
            "top1_sims": np.array(top1_sims, dtype=np.float32),
            "topn_sims": np.array(topn_sims, dtype=np.float32),
            "ranks": ranks,
        }
    return results


def build_smiles_universe(samples, train_idx, val_idx, test_idx, candidates_by_block):
    eval_smiles = set()
    for idx_set in (val_idx, test_idx):
        for i in idx_set:
            sample = samples[int(i)]
            block_id = sample["spec_id"]
            sm = sample["smiles"]
            if block_id in candidates_by_block:
                eval_smiles.update(candidates_by_block[block_id])
            if sm and sm != "nan":
                eval_smiles.add(sm)
    train_val_gt = {
        samples[int(i)]["smiles"]
        for i in list(train_idx) + list(val_idx)
        if samples[int(i)]["smiles"] and samples[int(i)]["smiles"] != "nan"
    }
    return sorted(eval_smiles | train_val_gt)


def write_metrics_csv(summary, path: Path):
    rows = []
    for split_id, models in summary.get("splits", {}).items():
        for model_name, r in models.items():
            if not r:
                continue
            rows.append({
                "split": split_id,
                "model": model_name,
                "n_queries": r.get("n_queries"),
                "avg_candidates": r.get("avg_candidates"),
                "top1": r.get("top1"),
                "top5": r.get("top5"),
                "top10": r.get("top10"),
                "top20": r.get("top20"),
                "mrr": r.get("mrr"),
                "median_rank": r.get("median_rank"),
                "temperature": r.get("temperature"),
            })
    fields = ["split", "model", "n_queries", "avg_candidates", "top1", "top5", "top10", "top20", "mrr", "median_rank", "temperature"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = {
        "dataset": "NPLIB1/CANOPUS MS2-block interface",
        "dataset_root": str(NPLIB1_ROOT),
        "data_protocol": str(TRAIN_DIR / "experience" / "nplib1_ms2block_protocol_20260701.md"),
        "splits": args.splits,
        "models": args.models,
        "temperature": args.temperature,
        "proj_epochs": args.proj_epochs,
        "early_stop": args.early_stop,
        "seed": args.seed,
        "candidate_key": "block_id/spec_id",
        "model_backbones": {
            "rt_only_d11": str(TRAIN_DIR / "output" / "phase2_rt_only" / "stage_d_epoch_11.pt"),
            "dreams": str(HERE / "resources" / "DreaMS_Check" / "ssl_model.ckpt"),
            "chemberta": str(ROOT / "model" / "feature" / "ChemBERTa-100M-MLM"),
        },
        "reused_logic": str(CM_PATH),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))
    summary = {"config": config, "splits": {}}
    mol_encoder = None

    for split_id in args.splits:
        print("\n" + "=" * 80)
        print(f"NPLIB1 MS2-block split {split_id}")
        print("=" * 80)
        samples, train_idx, val_idx, test_idx, cfg = load_ms2block_dataset(split_id)
        with open(candidates_path(split_id)) as f:
            candidates_by_block = json.load(f)
        print(f"Candidates: {len(candidates_by_block)} block keys from {candidates_path(split_id)}")

        all_smiles = build_smiles_universe(samples, train_idx, val_idx, test_idx, candidates_by_block)
        cache_tag = f"nplib1_split{split_id}_mass_chemberta_existing_interface"
        cache_path = Path(cm10.MOL_EMB_CACHE_DIR) / f"{cache_tag}.pt"
        if mol_encoder is None and not cache_has_all(cache_path, all_smiles):
            print("Loading ChemBERTa molecule encoder...")
            mol_encoder = cm10.MolEncoder(cm10.CHEMBERTA_PATH).to(device).eval()
        mol_emb_dict = cm10.precompute_mol_embeddings(mol_encoder, all_smiles, device, cache_tag=cache_tag)
        mol_dim = 768 if mol_encoder is None else int(mol_encoder.hidden_dim)

        emb_cache = {}
        for model_name in args.models:
            emb_path = OUT_DIR / f"spec_embeddings_split{split_id}_{model_name}.npy"
            if emb_path.exists() and not args.force_emb:
                spec_emb = np.load(emb_path)
                spec_dim = int(spec_emb.shape[1])
                print(f"Loaded cached spectrum embeddings: {emb_path} {spec_emb.shape}")
            else:
                print(f"\nLoading/extracting spectrum model: {model_name}")
                model, spec_dim = cm10.load_model(model_name, device)
                spec_emb = cm10.EXTRACTORS[model_name](model, samples, device)
                np.save(emb_path, spec_emb)
                print(f"Saved spectrum embeddings: {emb_path} {spec_emb.shape}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            emb_cache[model_name] = (spec_emb, spec_dim)

        summary["splits"].setdefault(str(split_id), {})
        for model_name in args.models:
            spec_emb, spec_dim = emb_cache[model_name]
            print(f"\n[{model_name}] split {split_id}: train projection")
            proj = cm10.train_projection(
                spec_emb, mol_emb_dict, samples, train_idx, spec_dim, mol_dim,
                device, epochs=args.proj_epochs, seed=args.seed,
                temperature=args.temperature, val_idx=val_idx if args.early_stop else None,
            )
            proj_path = OUT_DIR / f"proj_nplib1_ms2blocks_split{split_id}_mass_{model_name}_seed{args.seed}.pt"
            if args.save_proj:
                torch.save(proj.state_dict(), proj_path)
                print(f"Saved projection: {proj_path}")
            print(f"[{model_name}] split {split_id}: evaluate test")
            result = evaluate_retrieval_by_block(
                spec_emb, proj, mol_emb_dict, samples, test_idx, candidates_by_block,
                device, stratified=True, collect_raw=args.save_raw_scores,
            )
            result["temperature"] = args.temperature
            result["projection_path"] = str(proj_path) if args.save_proj else None
            result["candidate_file"] = str(candidates_path(split_id))
            result["csv_file"] = cfg["csv"]
            if args.save_raw_scores and "_raw" in result:
                raw = result.pop("_raw")
                raw_path = OUT_DIR / f"raw_scores_split{split_id}_{model_name}.npz"
                np.savez_compressed(raw_path, **raw)
                result["raw_scores"] = str(raw_path)
            summary["splits"][str(split_id)][model_name] = result
            (OUT_DIR / "summary.partial.json").write_text(json.dumps(summary, indent=2))
            del proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = {}
    for model_name in args.models:
        vals = [summary["splits"].get(str(split_id), {}).get(model_name, {}) for split_id in args.splits]
        vals = [r for r in vals if r]
        if vals:
            aggregate[model_name] = {
                "n_splits": len(vals),
                "top1_mean": float(np.mean([r["top1"] for r in vals])),
                "top1_std": float(np.std([r["top1"] for r in vals])),
                "top5_mean": float(np.mean([r["top5"] for r in vals])),
                "top10_mean": float(np.mean([r["top10"] for r in vals])),
                "top20_mean": float(np.mean([r["top20"] for r in vals])),
                "mrr_mean": float(np.mean([r["mrr"] for r in vals])),
                "mrr_std": float(np.std([r["mrr"] for r in vals])),
                "n_queries_total": int(sum(r["n_queries"] for r in vals)),
            }
    summary["aggregate"] = aggregate
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    write_metrics_csv(summary, OUT_DIR / "nplib1_ms2blocks_cross_modal_metrics.csv")
    print(f"\nSaved {out}")
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["rt_only_d11", "dreams"], choices=["rt_only_d11", "dreams"])
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proj-epochs", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--save-raw-scores", action="store_true")
    parser.add_argument("--save-proj", action="store_true")
    parser.add_argument("--force-emb", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
