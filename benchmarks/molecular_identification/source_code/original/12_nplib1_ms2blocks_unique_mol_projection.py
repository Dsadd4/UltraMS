"""
NPLIB1 MS2-block projection diagnostic.

This keeps the corrected MS2-block data and cached spectrum embeddings, but changes
projection training to avoid treating spectra from the same molecule as negatives.

Training objective:
  - fixed molecule vectors are unique train SMILES only
  - each query spectrum is classified to its molecule vector among unique train molecules
  - optional molecule-balanced epochs sample at most one MS2 block per molecule per epoch
  - best epoch is selected by validation retrieval Top1 using block-level candidates
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"
RUNNER_PATH = HERE / "11_nplib1_cross_modal_existing_interface_ms2blocks.py"

spec = importlib.util.spec_from_file_location("ms2runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules["ms2runner"] = runner
spec.loader.exec_module(runner)
cm10 = runner.cm10

OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_unique_mol_projection"
EMB_DIR = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2blocks_existing_interface"


def load_mol_embeddings(split_id: int, all_smiles: list[str], device):
    cache_tag = f"nplib1_split{split_id}_mass_chemberta_existing_interface"
    cache_path = Path(cm10.MOL_EMB_CACHE_DIR) / f"{cache_tag}.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    missing = [sm for sm in all_smiles if sm not in cached]
    if missing:
        raise RuntimeError(f"molecule cache missing {len(missing)} SMILES for split {split_id}; first={missing[:5]}")
    return {sm: cached[sm] for sm in all_smiles}


def build_train_groups(samples, train_idx, mol_emb_dict):
    groups = defaultdict(list)
    for i in train_idx:
        sm = samples[int(i)]["smiles"]
        if sm and sm != "nan" and sm in mol_emb_dict:
            groups[sm].append(int(i))
    return dict(groups)


def evaluate_val_top1(spec_emb, proj, mol_emb_dict, samples, val_idx, candidates_by_block, device):
    r = runner.evaluate_retrieval_by_block(
        spec_emb, proj, mol_emb_dict, samples, val_idx, candidates_by_block,
        device, stratified=False, collect_raw=False,
    )
    return float(r.get("top1", -1.0)), float(r.get("mrr", -1.0)), r


def train_unique_mol_projection(spec_emb, mol_emb_dict, samples, train_idx, val_idx,
                                candidates_by_block, spec_dim, mol_dim, device,
                                epochs=50, lr=5e-4, batch_size=512, temperature=0.05,
                                seed=42, balanced=True, val_every=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_groups = build_train_groups(samples, train_idx, mol_emb_dict)
    train_smiles = sorted(train_groups)
    label_map = {sm: j for j, sm in enumerate(train_smiles)}
    mol_mat = np.stack([mol_emb_dict[sm].numpy() for sm in train_smiles]).astype(np.float32)
    mol_t = F.normalize(torch.from_numpy(mol_mat).float().to(device), dim=-1)
    spec_t = torch.from_numpy(spec_emb).float().to(device)

    proj = cm10.ResidualProjection(spec_dim, mol_dim).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    rng = np.random.RandomState(seed)

    all_pairs = [(int(i), samples[int(i)]["smiles"]) for i in train_idx
                 if samples[int(i)]["smiles"] in label_map]
    best = {"top1": -1.0, "mrr": -1.0, "epoch": -1, "state": None, "val_result": None}
    history = []
    print(f"    unique train molecules={len(train_smiles)} train spectra={len(all_pairs)} balanced={balanced}")

    for epoch in range(1, epochs + 1):
        proj.train()
        if balanced:
            epoch_pairs = [(rng.choice(idxs), sm) for sm, idxs in train_groups.items()]
        else:
            epoch_pairs = list(all_pairs)
        rng.shuffle(epoch_pairs)
        total_loss = 0.0
        n_seen = 0
        for start in range(0, len(epoch_pairs), batch_size):
            batch = epoch_pairs[start:start + batch_size]
            idx = torch.tensor([i for i, _ in batch], dtype=torch.long, device=device)
            labels = torch.tensor([label_map[sm] for _, sm in batch], dtype=torch.long, device=device)
            q = F.normalize(proj(spec_t[idx]), dim=-1)
            logits = q @ mol_t.T / temperature
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(batch)
            n_seen += len(batch)
        sched.step()
        train_loss = total_loss / max(n_seen, 1)
        row = {"epoch": epoch, "train_loss": train_loss, "lr": sched.get_last_lr()[0]}
        if epoch % val_every == 0 or epoch == 1 or epoch == epochs:
            proj.eval()
            with torch.no_grad():
                top1, mrr, val_result = evaluate_val_top1(
                    spec_emb, proj, mol_emb_dict, samples, val_idx, candidates_by_block, device
                )
            row.update({"val_top1": top1, "val_mrr": mrr})
            better = (top1 > best["top1"]) or (top1 == best["top1"] and mrr > best["mrr"])
            if better:
                best = {
                    "top1": top1,
                    "mrr": mrr,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()},
                    "val_result": val_result,
                }
            if epoch == 1 or epoch % 5 == 0 or better:
                print(f"      epoch {epoch}: train={train_loss:.4f} val_top1={top1:.2f} val_mrr={mrr:.4f} lr={sched.get_last_lr()[0]:.2e}{' *best' if better else ''}")
        elif epoch % 5 == 0:
            print(f"      epoch {epoch}: train={train_loss:.4f} lr={sched.get_last_lr()[0]:.2e}")
        history.append(row)

    if best["state"] is not None:
        proj.load_state_dict(best["state"])
        print(f"      restored best epoch {best['epoch']} val_top1={best['top1']:.2f} val_mrr={best['mrr']:.4f}")
    proj.eval()
    return proj, {k: v for k, v in best.items() if k != "state"}, history


def run(args):
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary = {
        "config": {
            "splits": args.splits,
            "models": args.models,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "balanced": args.balanced,
            "seed": args.seed,
            "selection": "best validation retrieval Top1, MRR tie-break",
            "objective": "spectrum to unique train molecule vectors; no same-SMILES false negatives",
        },
        "splits": {},
    }

    for split_id in args.splits:
        print("\n" + "=" * 80)
        print(f"Unique-molecule projection split {split_id}")
        samples, train_idx, val_idx, test_idx, _ = runner.load_ms2block_dataset(split_id)
        with open(runner.candidates_path(split_id)) as f:
            candidates_by_block = json.load(f)
        all_smiles = runner.build_smiles_universe(samples, train_idx, val_idx, test_idx, candidates_by_block)
        mol_emb_dict = load_mol_embeddings(split_id, all_smiles, device)
        mol_dim = 768
        summary["splits"].setdefault(str(split_id), {})

        for model_name in args.models:
            print(f"\n[{model_name}] split {split_id}")
            emb_path = EMB_DIR / f"spec_embeddings_split{split_id}_{model_name}.npy"
            spec_emb = np.load(emb_path)
            spec_dim = int(spec_emb.shape[1])
            proj, best, history = train_unique_mol_projection(
                spec_emb, mol_emb_dict, samples, train_idx, val_idx, candidates_by_block,
                spec_dim, mol_dim, device, epochs=args.epochs, temperature=args.temperature,
                seed=args.seed, balanced=args.balanced, val_every=args.val_every,
            )
            test_result = runner.evaluate_retrieval_by_block(
                spec_emb, proj, mol_emb_dict, samples, test_idx, candidates_by_block,
                device, stratified=True, collect_raw=args.save_raw_scores,
            )
            test_result["best_val"] = best
            test_result["temperature"] = args.temperature
            test_result["balanced"] = args.balanced
            test_result["objective"] = "unique_train_molecule_cross_entropy"
            hist_path = OUT_BASE / f"history_split{split_id}_{model_name}.json"
            hist_path.write_text(json.dumps(history, indent=2))
            test_result["history"] = str(hist_path)
            proj_path = OUT_BASE / f"proj_unique_mol_split{split_id}_{model_name}_seed{args.seed}.pt"
            if args.save_proj:
                torch.save(proj.state_dict(), proj_path)
                test_result["projection_path"] = str(proj_path)
            if args.save_raw_scores and "_raw" in test_result:
                raw = test_result.pop("_raw")
                raw_path = OUT_BASE / f"raw_scores_split{split_id}_{model_name}.npz"
                np.savez_compressed(raw_path, **raw)
                test_result["raw_scores"] = str(raw_path)
            summary["splits"][str(split_id)][model_name] = test_result
            (OUT_BASE / "summary.partial.json").write_text(json.dumps(summary, indent=2))
            del proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = {}
    for model_name in args.models:
        vals = [summary["splits"].get(str(sid), {}).get(model_name, {}) for sid in args.splits]
        vals = [v for v in vals if v]
        if vals:
            aggregate[model_name] = {
                "n_splits": len(vals),
                "top1_mean": float(np.mean([v["top1"] for v in vals])),
                "top1_std": float(np.std([v["top1"] for v in vals])),
                "top5_mean": float(np.mean([v["top5"] for v in vals])),
                "top10_mean": float(np.mean([v["top10"] for v in vals])),
                "top20_mean": float(np.mean([v["top20"] for v in vals])),
                "mrr_mean": float(np.mean([v["mrr"] for v in vals])),
                "mrr_std": float(np.std([v["mrr"] for v in vals])),
            }
    summary["aggregate"] = aggregate
    out = OUT_BASE / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\nSaved", out)
    print(json.dumps(aggregate, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", type=int, default=[1])
    ap.add_argument("--models", nargs="+", default=["rt_only_d11", "dreams"], choices=["rt_only_d11", "dreams"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--save-proj", action="store_true")
    ap.add_argument("--save-raw-scores", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
