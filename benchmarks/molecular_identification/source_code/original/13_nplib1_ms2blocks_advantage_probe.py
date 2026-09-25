"""
NPLIB1 MS2-block spectrum-to-molecule advantage probe.

This script does not replace the primary corrected MS2-block runner.  It is a
diagnostic layer for two questions:

1. With the same full-data projection, where does UltraMS beat DreaMS?
2. If we use a defensible filtered protocol, do the same trends remain?

Outputs are intentionally query-level CSV + JSON summaries so the result can be
audited and replotted without rerunning models.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"

RUNNER_PATH = HERE / "11_nplib1_cross_modal_existing_interface_ms2blocks.py"
DIAG_PATH = HERE / "12_nplib1_ms2blocks_unique_mol_projection.py"

spec = importlib.util.spec_from_file_location("nplib1_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules["nplib1_runner"] = runner
spec.loader.exec_module(runner)
cm10 = runner.cm10

diag_spec = importlib.util.spec_from_file_location("nplib1_diag", DIAG_PATH)
diag = importlib.util.module_from_spec(diag_spec)
sys.modules["nplib1_diag"] = diag
diag_spec.loader.exec_module(diag)

OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_advantage_probe"
BASE_EMB_DIR = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_existing_interface"
BASE_PROJ_DIR = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_unique_mol_projection"


SCENARIOS = {
    "all": {
        "desc": "All valid MS2 blocks; full-data projection; post-hoc query stratification.",
    },
    "ms2peaks_only": {
        "desc": "Only canonical ms2peaks blocks.",
        "block_types": {"ms2peaks"},
    },
    "collision_only": {
        "desc": "Only collision blocks.",
        "block_types": {"collision"},
    },
    "protonated_only": {
        "desc": "Only [M+H]+ blocks.",
        "adducts": {"[M+H]+"},
    },
    "ms2peaks_protonated": {
        "desc": "Canonical ms2peaks blocks with [M+H]+ adduct.",
        "block_types": {"ms2peaks"},
        "adducts": {"[M+H]+"},
    },
    "min20": {
        "desc": "Blocks with at least 20 model-facing peaks.",
        "min_peaks": 20,
    },
    "min50": {
        "desc": "Blocks with at least 50 model-facing peaks.",
        "min_peaks": 50,
    },
    "ms2peaks_min50": {
        "desc": "Canonical ms2peaks blocks with at least 50 model-facing peaks.",
        "block_types": {"ms2peaks"},
        "min_peaks": 50,
    },
    "parent_filter_all": {
        "desc": "All blocks after filtering peaks with m/z > precursor_mz + 1.",
        "parent_filter": True,
    },
    "parent_filter_ms2peaks": {
        "desc": "ms2peaks blocks after filtering peaks with m/z > precursor_mz + 1.",
        "block_types": {"ms2peaks"},
        "parent_filter": True,
    },
}


def block_type_from_header(header: str) -> str:
    return str(header or "").strip().lower().split()[0] if str(header or "").strip() else ""


def load_samples_with_meta(split_id: int, scenario: dict):
    samples, train_idx, val_idx, test_idx, cfg = runner.load_ms2block_dataset(split_id)
    import pandas as pd

    df = pd.read_csv(runner.csv_path(split_id))
    meta_by_id = {str(r["spec_id"]): r for _, r in df.iterrows()}
    out = []
    for s in samples:
        ns = dict(s)
        row = meta_by_id[ns["spec_id"]]
        spectrum = np.asarray(ns["spectrum"], dtype=np.float32).copy()
        pmz = float(ns["precursor_mz"])
        raw_n = int(len(spectrum))
        above_parent = int(((spectrum[:, 0] > pmz + 1.0) & (spectrum[:, 1] > 0)).sum()) if raw_n else 0
        if scenario.get("parent_filter") and raw_n:
            spectrum = spectrum[(spectrum[:, 0] <= pmz + 1.0) & (spectrum[:, 1] > 0)]
        if scenario.get("sqrt_intensity") and len(spectrum):
            spectrum[:, 1] = np.sqrt(np.maximum(spectrum[:, 1], 0))
        ns["spectrum"] = spectrum.astype(np.float32)
        ns["block_type"] = block_type_from_header(ns.get("block_header", ""))
        ns["adduct"] = str(row.get("adduct", ""))
        ns["inchikey"] = str(row.get("inchikey", ""))
        ns["formula"] = str(row.get("formula", ""))
        ns["n_peaks_model"] = int(len(spectrum))
        ns["n_peaks_original"] = raw_n
        ns["n_peaks_above_precursor_plus1"] = above_parent
        out.append(ns)
    return out, train_idx, val_idx, test_idx, cfg


def scenario_mask(samples, indices, scenario: dict):
    keep = []
    block_types = scenario.get("block_types")
    adducts = scenario.get("adducts")
    min_peaks = int(scenario.get("min_peaks", 3))
    for i in indices:
        s = samples[int(i)]
        if len(s["spectrum"]) < max(3, min_peaks):
            continue
        if block_types and s.get("block_type") not in block_types:
            continue
        if adducts and s.get("adduct") not in adducts:
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def ensure_mol_embeddings(split_id: int, all_smiles: list[str], device):
    cache_tag = f"nplib1_split{split_id}_mass_chemberta_existing_interface"
    cache_path = Path(cm10.MOL_EMB_CACHE_DIR) / f"{cache_tag}.pt"
    if runner.cache_has_all(cache_path, all_smiles):
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return {sm: cached[sm] for sm in all_smiles}
    print("Loading ChemBERTa molecule encoder for missing cache entries...")
    mol_encoder = cm10.MolEncoder(cm10.CHEMBERTA_PATH).to(device).eval()
    return cm10.precompute_mol_embeddings(mol_encoder, all_smiles, device, cache_tag=cache_tag)


def get_spec_embeddings(model_name: str, split_id: int, samples, scenario_name: str,
                        scenario: dict, device, force_emb: bool):
    use_base = not scenario.get("parent_filter") and not scenario.get("sqrt_intensity")
    if use_base:
        base_path = BASE_EMB_DIR / f"spec_embeddings_split{split_id}_{model_name}.npy"
        if base_path.exists() and not force_emb:
            emb = np.load(base_path)
            return emb, int(emb.shape[1]), str(base_path)

    emb_dir = OUT_BASE / "embeddings" / scenario_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"spec_embeddings_split{split_id}_{model_name}.npy"
    if emb_path.exists() and not force_emb:
        emb = np.load(emb_path)
        return emb, int(emb.shape[1]), str(emb_path)

    print(f"Extracting {model_name} embeddings for split {split_id}, scenario={scenario_name}")
    model, spec_dim = cm10.load_model(model_name, device)
    emb = cm10.EXTRACTORS[model_name](model, samples, device)
    np.save(emb_path, emb)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb, int(spec_dim), str(emb_path)


def load_base_projection(model_name: str, split_id: int, spec_dim: int, mol_dim: int, device):
    path = BASE_PROJ_DIR / f"proj_unique_mol_split{split_id}_{model_name}_seed42.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    proj = cm10.ResidualProjection(spec_dim, mol_dim).to(device)
    state = torch.load(path, map_location=device)
    proj.load_state_dict(state)
    proj.eval()
    return proj, str(path)


def evaluate_rows(spec_emb, proj, mol_emb_dict, samples, eval_idx, candidates_by_block, device):
    rows = []
    skipped = defaultdict(int)
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
            cand_list = list(candidates_by_block[block_id])
            if sm not in cand_list:
                cand_list.append(sm)
            cand_list = [c for c in cand_list if c in mol_emb_dict]
            if sm not in cand_list:
                skipped["missing_gt_embedding"] += 1
                continue
            q = torch.from_numpy(spec_emb[int(qi):int(qi) + 1]).float().to(device)
            q = F.normalize(proj(q), dim=-1)
            c = F.normalize(torch.stack([mol_emb_dict[cand] for cand in cand_list]).to(device), dim=-1)
            sims = (q @ c.T).squeeze(0)
            order = torch.argsort(sims, descending=True)
            gt_pos = cand_list.index(sm)
            rank = int((order == gt_pos).nonzero(as_tuple=True)[0].item() + 1)
            top_pos = int(order[0].item())
            rows.append({
                "spec_id": block_id,
                "parent_spec_id": s.get("parent_spec_id", ""),
                "smiles": sm,
                "inchikey": s.get("inchikey", ""),
                "formula": s.get("formula", ""),
                "block_type": s.get("block_type", ""),
                "adduct": s.get("adduct", ""),
                "n_peaks_model": int(s.get("n_peaks_model", len(s.get("spectrum", [])))),
                "n_peaks_original": int(s.get("n_peaks_original", len(s.get("spectrum", [])))),
                "n_peaks_above_precursor_plus1": int(s.get("n_peaks_above_precursor_plus1", 0)),
                "candidate_count": int(len(cand_list)),
                "rank": rank,
                "hit1": int(rank <= 1),
                "hit5": int(rank <= 5),
                "hit10": int(rank <= 10),
                "hit20": int(rank <= 20),
                "rr": float(1.0 / rank),
                "gt_sim": float(sims[gt_pos].detach().cpu().item()),
                "top1_sim": float(sims[top_pos].detach().cpu().item()),
                "top1_is_gt": int(top_pos == gt_pos),
                "top1_smiles": cand_list[top_pos],
            })
    return rows, dict(skipped)


def summarize_rows(rows):
    if not rows:
        return {"n_queries": 0}
    arr = np.asarray([r["rank"] for r in rows], dtype=np.float64)
    cands = np.asarray([r["candidate_count"] for r in rows], dtype=np.float64)
    out = {
        "n_queries": int(len(rows)),
        "avg_candidates": float(cands.mean()),
        "median_candidates": float(np.median(cands)),
        "median_rank": float(np.median(arr)),
        "top1": float(np.mean(arr <= 1) * 100.0),
        "top5": float(np.mean(arr <= 5) * 100.0),
        "top10": float(np.mean(arr <= 10) * 100.0),
        "top20": float(np.mean(arr <= 20) * 100.0),
        "mrr": float(np.mean(1.0 / arr)),
    }
    return out


def macro_summary(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "")].append(r)
    vals = []
    for gid, rs in groups.items():
        if not gid:
            continue
        vals.append(summarize_rows(rs))
    if not vals:
        return {"n_groups": 0}
    return {
        "n_groups": int(len(vals)),
        "top1": float(np.mean([v["top1"] for v in vals])),
        "top5": float(np.mean([v["top5"] for v in vals])),
        "top10": float(np.mean([v["top10"] for v in vals])),
        "top20": float(np.mean([v["top20"] for v in vals])),
        "mrr": float(np.mean([v["mrr"] for v in vals])),
    }


def stratified_summary(rows):
    strata = {}
    bins = [
        ("cand_1_9", lambda r: r["candidate_count"] < 10),
        ("cand_10_49", lambda r: 10 <= r["candidate_count"] < 50),
        ("cand_50", lambda r: r["candidate_count"] >= 50),
        ("peaks_3_19", lambda r: 3 <= r["n_peaks_model"] < 20),
        ("peaks_20_49", lambda r: 20 <= r["n_peaks_model"] < 50),
        ("peaks_50_149", lambda r: 50 <= r["n_peaks_model"] < 150),
        ("peaks_150", lambda r: r["n_peaks_model"] >= 150),
        ("has_parent_plus1_peak", lambda r: r["n_peaks_above_precursor_plus1"] > 0),
        ("no_parent_plus1_peak", lambda r: r["n_peaks_above_precursor_plus1"] == 0),
    ]
    for name, pred in bins:
        subset = [r for r in rows if pred(r)]
        if len(subset) >= 10:
            strata[name] = summarize_rows(subset)
    for field in ("block_type", "adduct"):
        counts = defaultdict(list)
        for r in rows:
            counts[str(r.get(field, ""))].append(r)
        for val, subset in counts.items():
            if len(subset) >= 10:
                strata[f"{field}:{val}"] = summarize_rows(subset)
    return strata


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "scenario", "model", "spec_id", "parent_spec_id", "smiles",
        "inchikey", "formula", "block_type", "adduct", "n_peaks_model",
        "n_peaks_original", "n_peaks_above_precursor_plus1", "candidate_count",
        "rank", "hit1", "hit5", "hit10", "hit20", "rr", "gt_sim", "top1_sim",
        "top1_is_gt", "top1_smiles",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def bootstrap_delta(rows_u, rows_d, n_boot=2000, seed=13):
    by_u = {(r["split"], r["spec_id"]): r for r in rows_u}
    by_d = {(r["split"], r["spec_id"]): r for r in rows_d}
    keys = sorted(set(by_u) & set(by_d))
    if len(keys) < 20:
        return {"n_pairs": len(keys)}
    u_hit = np.asarray([by_u[k]["hit1"] for k in keys], dtype=np.float64)
    d_hit = np.asarray([by_d[k]["hit1"] for k in keys], dtype=np.float64)
    u_rr = np.asarray([by_u[k]["rr"] for k in keys], dtype=np.float64)
    d_rr = np.asarray([by_d[k]["rr"] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(keys)
    idx = rng.integers(0, n, size=(n_boot, n))
    top1 = (u_hit[idx] - d_hit[idx]).mean(axis=1) * 100.0
    mrr = (u_rr[idx] - d_rr[idx]).mean(axis=1)
    return {
        "n_pairs": int(n),
        "top1_delta_pp": float((u_hit - d_hit).mean() * 100.0),
        "top1_ci95": [float(np.quantile(top1, 0.025)), float(np.quantile(top1, 0.975))],
        "mrr_delta": float((u_rr - d_rr).mean()),
        "mrr_ci95": [float(np.quantile(mrr, 0.025)), float(np.quantile(mrr, 0.975))],
    }


def aggregate_scenario(summary, all_rows, models):
    aggregate = {}
    for model in models:
        rows = all_rows.get(model, [])
        aggregate[model] = {
            "block_micro": summarize_rows(rows),
            "molecule_macro": macro_summary(rows, "smiles"),
            "parent_macro": macro_summary(rows, "parent_spec_id"),
            "stratified": stratified_summary(rows),
        }
    if set(models) >= {"rt_only_d11", "dreams"}:
        aggregate["ultrams_minus_dreams_bootstrap"] = bootstrap_delta(
            all_rows.get("rt_only_d11", []), all_rows.get("dreams", [])
        )
    summary["aggregate"] = aggregate


def run(args):
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    selected = args.scenarios
    bad = [s for s in selected if s not in SCENARIOS]
    if bad:
        raise ValueError(f"unknown scenarios: {bad}; valid={sorted(SCENARIOS)}")

    config = {
        "run_name": args.run_name,
        "mode": args.mode,
        "splits": args.splits,
        "models": args.models,
        "scenarios": selected,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "seed": args.seed,
        "scenario_definitions": {k: {kk: sorted(vv) if isinstance(vv, set) else vv for kk, vv in SCENARIOS[k].items()} for k in selected},
        "base_projection_dir": str(BASE_PROJ_DIR),
        "base_embedding_dir": str(BASE_EMB_DIR),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    summary = {"config": config, "scenarios": {}}

    for scenario_name in selected:
        scenario = copy.deepcopy(SCENARIOS[scenario_name])
        print("\n" + "=" * 88)
        print(f"Scenario: {scenario_name} | {scenario.get('desc', '')}")
        scenario_summary = {"splits": {}, "description": scenario.get("desc", "")}
        scenario_rows_by_model = defaultdict(list)

        for split_id in args.splits:
            samples, train_idx, val_idx, test_idx, _ = load_samples_with_meta(split_id, scenario)
            with open(runner.candidates_path(split_id)) as f:
                candidates_by_block = json.load(f)
            train_f = scenario_mask(samples, train_idx, scenario)
            val_f = scenario_mask(samples, val_idx, scenario)
            test_f = scenario_mask(samples, test_idx, scenario)
            print(f"split {split_id}: train={len(train_f)} val={len(val_f)} test={len(test_f)}")
            if len(train_f) < 100 or len(test_f) < 20:
                scenario_summary["splits"].setdefault(str(split_id), {})["skipped"] = {
                    "reason": "too_few_filtered_samples",
                    "train": int(len(train_f)),
                    "val": int(len(val_f)),
                    "test": int(len(test_f)),
                }
                continue
            all_smiles = runner.build_smiles_universe(samples, train_f, val_f, test_f, candidates_by_block)
            mol_emb_dict = ensure_mol_embeddings(split_id, all_smiles, device)
            mol_dim = 768
            scenario_summary["splits"].setdefault(str(split_id), {})

            for model_name in args.models:
                spec_emb, spec_dim, emb_path = get_spec_embeddings(
                    model_name, split_id, samples, scenario_name, scenario, device, args.force_emb
                )
                if args.mode == "posthoc":
                    if scenario.get("parent_filter") or scenario.get("sqrt_intensity"):
                        print(f"  skip {model_name}: posthoc cannot use transformed spectra for {scenario_name}")
                        continue
                    proj, proj_path = load_base_projection(model_name, split_id, spec_dim, mol_dim, device)
                    best = {"mode": "loaded_full_data_projection"}
                elif args.mode == "train_filtered":
                    proj, best, history = diag.train_unique_mol_projection(
                        spec_emb, mol_emb_dict, samples, train_f, val_f, candidates_by_block,
                        spec_dim, mol_dim, device, epochs=args.epochs, temperature=args.temperature,
                        seed=args.seed, balanced=True, val_every=args.val_every,
                    )
                    hist_path = out_dir / f"history_{scenario_name}_split{split_id}_{model_name}.json"
                    hist_path.write_text(json.dumps(history, indent=2))
                    proj_path = out_dir / f"proj_{scenario_name}_split{split_id}_{model_name}_seed{args.seed}.pt"
                    torch.save(proj.state_dict(), proj_path)
                    proj_path = str(proj_path)
                else:
                    raise ValueError(args.mode)

                rows, skipped = evaluate_rows(spec_emb, proj, mol_emb_dict, samples, test_f, candidates_by_block, device)
                for r in rows:
                    r["split"] = split_id
                    r["scenario"] = scenario_name
                    r["model"] = model_name
                scenario_rows_by_model[model_name].extend(rows)
                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{model_name}.csv"
                write_rows(row_path, rows)
                metrics = {
                    "block_micro": summarize_rows(rows),
                    "molecule_macro": macro_summary(rows, "smiles"),
                    "parent_macro": macro_summary(rows, "parent_spec_id"),
                    "stratified": stratified_summary(rows),
                    "skipped": skipped,
                    "embedding_path": emb_path,
                    "projection_path": proj_path,
                    "best_val": best,
                    "query_rows": str(row_path),
                }
                scenario_summary["splits"][str(split_id)][model_name] = metrics
                print(f"  {model_name}: n={metrics['block_micro'].get('n_queries', 0)} "
                      f"Top1={metrics['block_micro'].get('top1', 0):.2f} "
                      f"MRR={metrics['block_micro'].get('mrr', 0):.4f}")
                del proj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aggregate_scenario(scenario_summary, scenario_rows_by_model, args.models)
        summary["scenarios"][scenario_name] = scenario_summary
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved", out_dir / "summary.json")
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for model_name in args.models:
            b = sc.get("aggregate", {}).get(model_name, {}).get("block_micro", {})
            m = sc.get("aggregate", {}).get(model_name, {}).get("molecule_macro", {})
            print(f"  {model_name}: block Top1={b.get('top1', 0):.2f}, MRR={b.get('mrr', 0):.4f}; "
                  f"mol Top1={m.get('top1', 0):.2f}, MRR={m.get('mrr', 0):.4f}")
        print("  delta", sc.get("aggregate", {}).get("ultrams_minus_dreams_bootstrap", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="probe")
    ap.add_argument("--mode", choices=["posthoc", "train_filtered"], default="posthoc")
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["rt_only_d11", "dreams"], choices=["rt_only_d11", "dreams"])
    ap.add_argument("--scenarios", nargs="+", default=["all", "ms2peaks_only", "collision_only", "protonated_only", "ms2peaks_protonated", "min20", "min50"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--force-emb", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
