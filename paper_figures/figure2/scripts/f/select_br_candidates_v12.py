"""
Select Br-containing case-study candidates for the Figure 2 atom-attention panel.

The selector runs model attention once over Br-containing val/test rows, then
checks a smaller shortlist with MAGMa. It writes a complete JSON report so the
chosen rows can be redrawn without repeating the search.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from atom_attention_viz_magma_v2 import (
    CSV_TEST,
    CSV_VAL,
    _PROJECT_ROOT,
    _load_rows_by_global_idx,
    get_attention_for_samples,
    load_probe_from_ckpt,
)
from atom_attention_viz_magma_v11_lower_inset_more import (
    magma_annotate_detailed,
    molecular_formula,
    top_peak_summary,
)

RDLogger.DisableLog("rdApp.*")


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(_PROJECT_ROOT) / path


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def atom_counts(smiles, atom_names):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return {
        atom: sum(1 for at in mol.GetAtoms() if at.GetSymbol() == atom)
        for atom in atom_names
    }


def collect_br_rows(atom_names):
    val_df = pd.read_csv(CSV_VAL).reset_index(drop=True)
    test_df = pd.read_csv(CSV_TEST).reset_index(drop=True)
    rows = []
    for fold, df, offset in (("val", val_df, 0), ("test", test_df, len(val_df))):
        for local_idx, row in df.iterrows():
            counts = atom_counts(row["smiles"], atom_names)
            if not counts or counts.get("Br", 0) <= 0:
                continue
            can = canonical_smiles(row["smiles"])
            if can is None:
                continue
            pattern = [atom for atom in atom_names if counts.get(atom, 0) > 0]
            rows.append({
                "global_index": int(offset + local_idx),
                "local_index": int(local_idx),
                "fold": fold,
                "smiles": str(row["smiles"]),
                "canonical_smiles": can,
                "formula": molecular_formula(str(row["smiles"])),
                "adduct": str(row["adduct"]),
                "precursor_mz": float(row["precursor_mz"]),
                "counts": {k: int(v) for k, v in counts.items()},
                "pattern": pattern,
            })
    return rows


def unique_best_by_smiles(records):
    best = {}
    for rec in records:
        key = (rec["canonical_smiles"], rec["adduct"])
        old = best.get(key)
        if old is None or rec["pre_score"] > old["pre_score"]:
            best[key] = rec
    return list(best.values())


def prefilter_records(rows, samples, targets, logits, attn_maps, atom_names):
    br_i = atom_names.index("Br")
    records = []
    for pos, row in enumerate(rows):
        spec = samples[pos]["spectrum"]
        mz = spec[:, 0]
        inten = spec[:, 1]
        weights = attn_maps[pos, br_i, :len(mz)]
        order = np.argsort(-weights)
        top_idx = int(order[0])
        true_br = float(targets[pos, br_i])
        pred_br = float(logits[pos, br_i])
        br_err = abs(pred_br - true_br)
        top_attn = float(weights[top_idx])
        top_intensity = float(inten[top_idx])
        # Prefer accurate Br regression, visible top-attention peaks and simpler rows.
        pre_score = (
            12.0 / (1.0 + br_err)
            + 5.0 * top_intensity
            + 2.0 * top_attn
            - 0.18 * max(0, len(row["pattern"]) - 1)
        )
        rec = dict(row)
        rec.update({
            "row_pos": int(pos),
            "true_br": true_br,
            "pred_br": pred_br,
            "br_abs_error": float(br_err),
            "top_attention_peak_id": top_idx,
            "top_attention_mz": float(mz[top_idx]),
            "top_attention_intensity": top_intensity,
            "top_attention": top_attn,
            "pre_score": float(pre_score),
        })
        records.append(rec)
    return records


def magma_evaluate(records, samples, targets, logits, attn_maps, atom_names, ppm, top_k):
    br_i = atom_names.index("Br")
    evaluated = []
    for rec in records:
        pos = rec["row_pos"]
        sample = samples[pos]
        spec = sample["spectrum"]
        mz = spec[:, 0]
        inten = spec[:, 1]
        annot, details = magma_annotate_detailed(
            sample["smiles"],
            mz,
            inten,
            sample["adduct"],
            atom_names,
            ppm=ppm,
        )
        weights = attn_maps[pos, br_i, :len(mz)]
        mask = annot[:len(mz), br_i]
        top_records = top_peak_summary(mz, inten, weights, mask, details, top_k=top_k)
        first_supported = next((r for r in top_records if r["magma_hit"] and r.get("frag_smiles")), None)
        supported_rank = int(first_supported["rank"]) if first_supported else None
        top_hit = bool(top_records and top_records[0]["magma_hit"])
        callout_count = sum(1 for r in top_records if r["magma_hit"] and r.get("frag_smiles"))
        br_hit_count = int(mask.sum())
        score = (
            (120.0 if top_hit else 0.0)
            + (35.0 / supported_rank if supported_rank else 0.0)
            + 4.0 * min(callout_count, 6)
            + 1.5 * min(br_hit_count, 12)
            + rec["pre_score"]
            - 2.0 * rec["br_abs_error"]
        )
        out = dict(rec)
        out.update({
            "magma_br_hit_count": br_hit_count,
            "top_br_attention_peak_magma_hit": top_hit,
            "first_supported_rank": supported_rank,
            "callout_count_top_k": int(callout_count),
            "selection_score": float(score),
            "top_br_records": top_records,
        })
        evaluated.append(out)
        print(
            f"  global={rec['global_index']} pattern={'+'.join(rec['pattern'])} "
            f"predBr={rec['pred_br']:.2f} err={rec['br_abs_error']:.2f} "
            f"topHit={top_hit} rank={supported_rank} BrHits={br_hit_count} score={score:.1f}",
            flush=True,
        )
    return evaluated


def choose_final(evaluated, n_select):
    ranked = sorted(evaluated, key=lambda r: r["selection_score"], reverse=True)
    selected = []
    seen_smiles = set()
    preferred_patterns = [
        ("Br",),
        ("F", "Br"),
        ("Cl", "Br"),
        ("S", "Br"),
        ("S", "F", "Br"),
        ("S", "Cl", "Br"),
        ("Cl", "F", "Br"),
    ]

    def add(rec):
        if len(selected) >= n_select:
            return False
        key = rec["canonical_smiles"]
        if key in seen_smiles:
            return False
        if rec["magma_br_hit_count"] <= 0 or rec["callout_count_top_k"] <= 0:
            return False
        selected.append(rec)
        seen_smiles.add(key)
        return True

    for pattern in preferred_patterns:
        pool = [r for r in ranked if tuple(r["pattern"]) == pattern and r["top_br_attention_peak_magma_hit"]]
        if pool:
            add(pool[0])
    for rec in ranked:
        if rec["top_br_attention_peak_magma_hit"]:
            add(rec)
    for rec in ranked:
        add(rec)
    return selected[:n_select], ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="train/output/showcase/atomq_ultra_S_Cl_F_Br_0413_1448/last_ultra.pt")
    parser.add_argument("--ultra-ckpt", default="train/output/phase2_rt_only/stage_d_epoch_11.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--magma-scan", type=int, default=140)
    parser.add_argument("--n-select", type=int, default=8)
    parser.add_argument(
        "--out-json",
        default=os.path.join(
            _PROJECT_ROOT,
            "main_figure",
            "fig2",
            "output",
            "case_study_br_candidates_v12",
            "br_candidate_selection_v12.json",
        ),
    )
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    ckpt = resolve_project_path(args.ckpt)
    ultra_ckpt = resolve_project_path(args.ultra_ckpt) if args.ultra_ckpt else None
    print(f"Loading model: {ckpt}", flush=True)
    model, atom_names, norm_mean, norm_std = load_probe_from_ckpt(
        str(ckpt),
        device,
        ultra_ckpt=None if ultra_ckpt is None else str(ultra_ckpt),
    )

    rows = collect_br_rows(atom_names)
    print(f"Found Br rows: {len(rows)}", flush=True)
    global_indices = [r["global_index"] for r in rows]
    samples, targets, _ = _load_rows_by_global_idx(global_indices, atom_names)

    print("Running Br attention inference over all Br rows...", flush=True)
    logits_z, attn_maps = get_attention_for_samples(model, samples, device, batch_size=64)
    logits = logits_z * norm_std + norm_mean

    pre_records = prefilter_records(rows, samples, targets, logits, attn_maps, atom_names)
    unique_records = unique_best_by_smiles(pre_records)
    shortlisted = sorted(unique_records, key=lambda r: r["pre_score"], reverse=True)[:args.magma_scan]
    print(f"Unique Br molecules/adducts: {len(unique_records)}; MAGMa scan: {len(shortlisted)}", flush=True)

    evaluated = magma_evaluate(
        shortlisted,
        samples,
        targets,
        logits,
        attn_maps,
        atom_names,
        ppm=args.ppm,
        top_k=args.top_k,
    )
    selected, ranked = choose_final(evaluated, args.n_select)
    report = {
        "stage": "fig2_case_study_br_candidate_selection_v12",
        "ppm": args.ppm,
        "top_k": args.top_k,
        "magma_scan": args.magma_scan,
        "n_br_rows": len(rows),
        "n_unique_br_molecules_adducts": len(unique_records),
        "selected_global_indices": [int(r["global_index"]) for r in selected],
        "selected": selected,
        "ranked": ranked,
    }
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected: {report['selected_global_indices']}", flush=True)
    print(f"Wrote: {out_json}", flush=True)


if __name__ == "__main__":
    main()
