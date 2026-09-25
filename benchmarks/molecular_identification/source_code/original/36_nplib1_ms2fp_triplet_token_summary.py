"""Evaluate MoNA-finetuned UltraMS token summaries on NPLIB1 MS2FP.

This keeps the same NPLIB1 spectrum-to-molecule protocol as
34_nplib1_ms2fp_contrastive_model_compare.py:

  spectrum features -> Morgan fingerprint head -> candidate CE -> Top1/Top5/MRR

The added representation is for UltraMS triplet checkpoints only.  It extracts
projected CLS plus last-layer token summaries from the fine-tuned encoder, so
the final metric stays identical while testing whether UltraMS peak-level
features help the molecule retrieval task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ms2fp = import_script(HERE / "17_nplib1_ms2fp_retrieval.py", "nplib1_ms2fp_for_triplet_token")
probe = ms2fp.probe
runner = ms2fp.runner
triplet34 = import_script(HERE / "34_nplib1_ms2fp_contrastive_model_compare.py", "nplib1_triplet34_for_token")
token22 = import_script(HERE / "22_nplib1_token_summary_ms2fp.py", "nplib1_token22_for_triplet")

from appliedmodel.common import get_backbone_hidden_dim, load_ultra_backbone, ultra_prepare  # noqa: E402


OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2fp_triplet_token_summary"
DREAMS_EMBED_CKPT = triplet34.DREAMS_EMBED_CKPT


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_key_path(item: str) -> tuple[str, Path]:
    if "=" in item:
        key, path = item.split("=", 1)
    else:
        path = item
        key = Path(path).parent.name
    return key, Path(path)


def safe_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(" ", "_")


def cache_paths(out_dir: Path, model_key: str, scenario_name: str, split_id: int,
                feature_tag: str, source_sha: str | None) -> tuple[Path, Path]:
    emb_dir = out_dir / "embeddings" / scenario_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    sha_tag = source_sha[:12] if source_sha else "nosha"
    stem = f"spec_features_split{split_id}_{safe_name(model_key)}_{safe_name(feature_tag)}_{sha_tag}"
    return emb_dir / f"{stem}.npy", emb_dir / f"{stem}.json"


@torch.no_grad()
def extract_ultra_triplet_token_summary(
    ckpt_path: Path,
    samples: list[dict],
    device: torch.device,
    batch_size: int,
    summary_parts: list[str],
    include_projected_cls: bool,
) -> tuple[np.ndarray, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    max_peaks = int(config.get("max_peaks", 100))
    encoder = load_ultra_backbone(
        device,
        checkpoint_path=config.get("ultra_ckpt") or None,
        state_key=config.get("state_key", "base_state_dict"),
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    encoder.to(device).eval()

    projection = None
    projected_dim = 0
    if include_projected_cls:
        in_dim = get_backbone_hidden_dim(encoder)
        projection = triplet34.build_projection_from_state(
            in_dim,
            ckpt["projection_state_dict"],
            config.get("projection", "auto"),
        ).to(device).eval()
        projected_dim = int(in_dim)

    features = []
    peak_counts = []
    used_parts = None
    skipped_parts = None

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        peaks_t, attn_t, pmz_t = ultra_prepare(batch_samples, device, max_peaks=max_peaks)
        bsz = peaks_t.shape[0]

        ms_emb = encoder.peak_encoder(peaks_t)
        cls_emb = encoder.cls_emb.expand(bsz, 1, -1)
        precursor_input = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], dim=-1).unsqueeze(1)
        precursor_emb = encoder.peak_encoder(precursor_input) + encoder.precursor_type_emb
        full = torch.cat([cls_emb, precursor_emb, ms_emb], dim=1)
        seq_len = full.shape[1]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        full = encoder.dropout(full + encoder.pos_emb(pos))
        peak_mask = attn_t.bool()
        full_attn = torch.cat(
            [torch.ones(bsz, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1
        )
        out = encoder.encoder(inputs_embeds=full, attention_mask=full_attn)
        hidden = out.last_hidden_state

        raw_feat, used, skipped = token22.build_token_summary(
            hidden=hidden,
            peak_mask=peak_mask,
            peak_weights=peaks_t[:, :, 1],
            summary_parts=summary_parts,
            cls_index=0,
            precursor_index=1,
            peak_start=2,
        )
        parts = []
        if projection is not None:
            parts.append(projection(hidden[:, 0, :]))
        parts.append(raw_feat)
        feat = torch.cat(parts, dim=-1)
        features.append(feat.detach().cpu().numpy().astype(np.float32))
        peak_counts.append(peak_mask.sum(dim=1).detach().cpu().numpy())
        used_parts = used
        skipped_parts = skipped
        if (start // batch_size) % 25 == 0:
            print(f"    Ultra-triplet-token embed: {start}/{len(samples)}", flush=True)

    emb = np.concatenate(features, axis=0) if features else np.zeros((0, 0), dtype=np.float32)
    peak_stats = token22.finalize_token_stats(peak_counts, {
        "extractor": "ultrams_triplet_last_hidden_state",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "source_config": config,
        "summary_parts_requested": summary_parts,
        "summary_parts_used": used_parts or [],
        "summary_parts_skipped": skipped_parts or [],
        "include_projected_cls": bool(include_projected_cls),
        "projected_cls_dim": int(projected_dim),
        "raw_summary_dim": int(emb.shape[1] - projected_dim) if emb.ndim == 2 else 0,
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "token_layout": {"cls": 0, "precursor": 1, "peaks_start": 2},
        "max_peaks": int(max_peaks),
    })

    del encoder, projection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb, peak_stats


def model_configs(args) -> dict[str, dict]:
    cfg = {}
    if args.dreams_embedding:
        cfg["dreams_embedding"] = {"kind": "dreams_embedding", "ckpt": str(args.dreams_embedding)}
    for item in args.ultra_triplet:
        key, path = parse_key_path(item)
        cfg[key] = {"kind": "ultra_triplet", "ckpt": str(path)}
    for item in args.ultra_triplet_token:
        key, path = parse_key_path(item)
        cfg[key] = {"kind": "ultra_triplet_token", "ckpt": str(path)}
    if not cfg:
        raise ValueError("No models specified.")
    return cfg


def get_features(model_key: str, cfg: dict, out_dir: Path, samples: list[dict],
                 scenario_name: str, split_id: int, device: torch.device,
                 args) -> tuple[np.ndarray, str, str, dict]:
    kind = cfg["kind"]
    ckpt = Path(cfg["ckpt"])
    source_sha = sha256_file(ckpt) if ckpt.exists() else None
    if kind == "ultra_triplet_token":
        tag = "triplet_token_" + "-".join(args.summary_parts)
        if args.no_token_projected_cls:
            tag += "_rawonly"
    elif kind == "ultra_triplet":
        tag = "triplet_projected_cls"
    elif kind == "dreams_embedding":
        tag = "dreams_embedding"
    else:
        raise ValueError(kind)

    emb_path, meta_path = cache_paths(out_dir, model_key, scenario_name, split_id, tag, source_sha)
    if emb_path.exists() and meta_path.exists() and not args.force_emb:
        emb = np.load(emb_path)
        meta = json.loads(meta_path.read_text())
        if emb.shape[0] != len(samples):
            raise RuntimeError(
                f"Cached feature row mismatch for {emb_path}: cached={emb.shape[0]} samples={len(samples)}"
            )
        return emb, str(emb_path), str(meta_path), meta

    print(f"Extracting features: model={model_key} kind={kind} split={split_id}")
    if kind == "ultra_triplet_token":
        emb, meta = extract_ultra_triplet_token_summary(
            ckpt,
            samples,
            device,
            args.embed_batch_size,
            args.summary_parts,
            include_projected_cls=not args.no_token_projected_cls,
        )
    elif kind == "ultra_triplet":
        emb = triplet34.extract_ultra_triplet_embeddings(
            ckpt,
            samples,
            device,
            batch_size=args.embed_batch_size,
            use_projection=not args.no_triplet_projection,
        )
        meta = {
            "extractor": "ultrams_triplet_projected_cls",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": source_sha,
            "use_projection": not args.no_triplet_projection,
            "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        }
    elif kind == "dreams_embedding":
        emb = triplet34.extract_dreams_embedding_ckpt(
            samples,
            device,
            batch_size=args.embed_batch_size,
            ckpt_path=ckpt,
        )
        meta = {
            "extractor": "dreams_embedding_checkpoint",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": source_sha,
            "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        }

    if emb.shape[0] != len(samples):
        raise RuntimeError(
            f"Extracted feature row mismatch: model={model_key} features={emb.shape[0]} samples={len(samples)}"
        )
    np.save(emb_path, emb.astype(np.float32))
    meta.update({
        "model_key": model_key,
        "kind": kind,
        "split": int(split_id),
        "scenario": scenario_name,
        "path": str(emb_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=json_default) + "\n")
    return emb, str(emb_path), str(meta_path), meta


def write_metric_tables(summary: dict, out_dir: Path) -> None:
    per_split_fields = [
        "scenario", "split", "model", "n_queries", "top1", "top5", "mrr",
        "molecule_top1", "parent_top1", "best_phase", "best_epoch",
        "best_val_top1", "best_val_mrr", "feature_path", "query_rows",
        "history", "skipped_json",
    ]
    agg_fields = [
        "scenario", "model", "n_queries", "top1", "top5", "mrr",
        "molecule_top1", "parent_top1",
    ]
    per_rows = []
    agg_rows = []
    for scenario_name, sc in summary["scenarios"].items():
        for split_id, split_models in sc.get("splits", {}).items():
            for model_key, metrics in split_models.items():
                block = metrics.get("block_micro", {})
                mol = metrics.get("molecule_macro", {})
                parent = metrics.get("parent_macro", {})
                best = metrics.get("best_val", {})
                per_rows.append({
                    "scenario": scenario_name,
                    "split": split_id,
                    "model": model_key,
                    "n_queries": block.get("n_queries", 0),
                    "top1": block.get("top1", 0),
                    "top5": block.get("top5", 0),
                    "mrr": block.get("mrr", 0),
                    "molecule_top1": mol.get("top1", 0),
                    "parent_top1": parent.get("top1", 0),
                    "best_phase": best.get("phase", ""),
                    "best_epoch": best.get("epoch", ""),
                    "best_val_top1": best.get("val_top1", ""),
                    "best_val_mrr": best.get("val_mrr", ""),
                    "feature_path": metrics.get("feature_path", ""),
                    "query_rows": metrics.get("query_rows", ""),
                    "history": metrics.get("history", ""),
                    "skipped_json": json.dumps(metrics.get("skipped", {}), sort_keys=True),
                })
        for model_key, metrics in sc.get("aggregate", {}).items():
            if model_key.endswith("_minus_dreams_embedding_bootstrap"):
                continue
            block = metrics.get("block_micro", {})
            mol = metrics.get("molecule_macro", {})
            parent = metrics.get("parent_macro", {})
            agg_rows.append({
                "scenario": scenario_name,
                "model": model_key,
                "n_queries": block.get("n_queries", 0),
                "top1": block.get("top1", 0),
                "top5": block.get("top5", 0),
                "mrr": block.get("mrr", 0),
                "molecule_top1": mol.get("top1", 0),
                "parent_top1": parent.get("top1", 0),
            })

    with (out_dir / "per_split_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=per_split_fields)
        writer.writeheader()
        writer.writerows(per_rows)
    with (out_dir / "aggregate_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(agg_rows, indent=2, sort_keys=True, default=json_default) + "\n"
    )


def run(args) -> None:
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = model_configs(args)
    config = {
        "run_name": args.run_name,
        "models": cfg,
        "splits": args.splits,
        "scenarios": args.scenarios,
        "summary_parts": args.summary_parts,
        "no_token_projected_cls": args.no_token_projected_cls,
        "seed": args.seed,
        "force_emb_requested": bool(args.force_emb),
        "ms2fp_args": {
            "epochs": args.epochs,
            "candidate_ce_epochs": args.candidate_ce_epochs,
            "candidate_ce_lr": args.candidate_ce_lr,
            "candidate_ce_temperature": args.candidate_ce_temperature,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "score": args.score,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=json_default) + "\n")

    ms_args = argparse.Namespace(
        fp_bits=args.fp_bits,
        radius=args.radius,
        fp_use_chirality=False,
        tanimoto_loss_weight=args.tanimoto_loss_weight,
        score=args.score,
        epochs=args.epochs,
        val_every=args.val_every,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        hidden=args.hidden,
        dropout=args.dropout,
        pos_weight_clip=args.pos_weight_clip,
        calibrate_pos_weight=args.calibrate_pos_weight,
        candidate_ce_epochs=args.candidate_ce_epochs,
        candidate_ce_batch_size=args.candidate_ce_batch_size,
        candidate_ce_lr=args.candidate_ce_lr,
        candidate_ce_temperature=args.candidate_ce_temperature,
        balanced=args.balanced,
        seed=args.seed,
    )

    summary = {"config": config, "scenarios": {}}
    for scenario_name in args.scenarios:
        scenario = probe.SCENARIOS[scenario_name]
        print("\n" + "=" * 88)
        print(f"Scenario {scenario_name}: {scenario.get('desc', '')}")
        scenario_summary = {"splits": {}, "description": scenario.get("desc", "")}
        rows_by_model = defaultdict(list)

        for split_id in args.splits:
            samples, train_idx, val_idx, test_idx, _ = probe.load_samples_with_meta(split_id, scenario)
            with open(runner.candidates_path(split_id)) as fh:
                candidates_by_block = json.load(fh)

            train_f = probe.scenario_mask(samples, train_idx, scenario)
            val_f = probe.scenario_mask(samples, val_idx, scenario)
            test_f = probe.scenario_mask(samples, test_idx, scenario)
            smiles = set()
            smiles |= ms2fp.collect_smiles(samples, train_f)
            smiles |= ms2fp.collect_smiles(samples, val_f, candidates_by_block)
            smiles |= ms2fp.collect_smiles(samples, test_f, candidates_by_block)
            fp_dict = ms2fp.build_fp_dict(smiles, args.fp_bits, args.radius, False)
            train_f = ms2fp.valid_indices(samples, train_f, fp_dict)
            val_f = ms2fp.valid_indices(samples, val_f, fp_dict)
            test_f = ms2fp.valid_indices(samples, test_f, fp_dict)
            print(f"split {split_id}: train={len(train_f)} val={len(val_f)} test={len(test_f)} fps={len(fp_dict)}")
            scenario_summary["splits"].setdefault(str(split_id), {})

            for model_key, model_cfg in cfg.items():
                print(f"\n[{model_key}] split {split_id}")
                spec_emb, emb_path, meta_path, emb_meta = get_features(
                    model_key, model_cfg, out_dir, samples, scenario_name, split_id, device, args
                )
                print(
                    f"    feature dim={spec_emb.shape[1]} path={emb_path} "
                    f"extractor={emb_meta.get('extractor', '')}"
                )
                head, best, history, fp_stats, logit_shift = ms2fp.train_head(
                    spec_emb, samples, train_f, val_f, candidates_by_block, fp_dict, ms_args, device
                )
                rows, skipped = ms2fp.evaluate(
                    head, spec_emb, samples, test_f, candidates_by_block, fp_dict,
                    device, args.score, logit_shift=logit_shift,
                )
                for row in rows:
                    row.update({"split": split_id, "scenario": scenario_name, "model": model_key})
                rows_by_model[model_key].extend(rows)

                safe = safe_name(model_key)
                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{safe}.csv"
                hist_path = out_dir / f"history_{scenario_name}_split{split_id}_{safe}.json"
                head_path = out_dir / f"ms2fp_head_{scenario_name}_split{split_id}_{safe}_seed{args.seed}.pt"
                triplet34.write_rows(row_path, rows)
                hist_path.write_text(json.dumps(history, indent=2, sort_keys=True, default=json_default) + "\n")
                torch.save(head.state_dict(), head_path)

                metrics = {
                    "block_micro": probe.summarize_rows(rows),
                    "molecule_macro": probe.macro_summary(rows, "smiles"),
                    "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                    "stratified": probe.stratified_summary(rows),
                    "best_val": best,
                    "fp_stats": fp_stats,
                    "skipped": skipped,
                    "feature_path": emb_path,
                    "feature_meta_path": meta_path,
                    "feature_meta": emb_meta,
                    "query_rows": str(row_path),
                    "history": str(hist_path),
                    "head_path": str(head_path),
                }
                scenario_summary["splits"][str(split_id)][model_key] = metrics
                block = metrics["block_micro"]
                print(
                    f"    TEST n={block.get('n_queries', 0)} Top1={block.get('top1', 0):.2f} "
                    f"Top5={block.get('top5', 0):.2f} MRR={block.get('mrr', 0):.4f}"
                )
                del head
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aggregate = {}
        for model_key, rows in rows_by_model.items():
            aggregate[model_key] = {
                "block_micro": probe.summarize_rows(rows),
                "molecule_macro": probe.macro_summary(rows, "smiles"),
                "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                "stratified": probe.stratified_summary(rows),
            }
        if "dreams_embedding" in rows_by_model:
            for model_key in rows_by_model:
                if model_key != "dreams_embedding":
                    aggregate[f"{model_key}_minus_dreams_embedding_bootstrap"] = probe.bootstrap_delta(
                        rows_by_model[model_key],
                        rows_by_model["dreams_embedding"],
                        seed=args.seed + 3601,
                    )
        scenario_summary["aggregate"] = aggregate
        summary["scenarios"][scenario_name] = scenario_summary
        (out_dir / "summary.partial.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n")
    write_metric_tables(summary, out_dir)
    print("\nSaved", out_dir / "summary.json")
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for model_key in cfg:
            block = sc["aggregate"][model_key]["block_micro"]
            print(
                f"  {model_key}: Top1={block.get('top1', 0):.2f} "
                f"Top5={block.get('top5', 0):.2f} MRR={block.get('mrr', 0):.4f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="triplet_token_summary")
    ap.add_argument("--ultra-triplet", nargs="*", default=[], help="key=/path/best.pt projected-CLS baseline")
    ap.add_argument("--ultra-triplet-token", nargs="*", default=[], help="key=/path/best.pt token summary features")
    ap.add_argument("--dreams-embedding", type=Path, default=DREAMS_EMBED_CKPT)
    ap.add_argument("--no-dreams-embedding", action="store_true",
                    help="Disable the default DreaMS embedding baseline for UltraMS-only sweeps.")
    ap.add_argument("--no-triplet-projection", action="store_true")
    ap.add_argument("--summary-parts", nargs="+", default=token22.DEFAULT_SUMMARY_PARTS,
                    choices=token22.SUMMARY_PART_CHOICES)
    ap.add_argument("--no-token-projected-cls", action="store_true")
    ap.add_argument("--splits", nargs="+", type=int, default=[1])
    ap.add_argument("--scenarios", nargs="+", default=["ms2peaks_min50"], choices=sorted(probe.SCENARIOS))
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
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
    ap.add_argument("--candidate-ce-epochs", type=int, default=10)
    ap.add_argument("--candidate-ce-batch-size", type=int, default=64)
    ap.add_argument("--candidate-ce-lr", type=float, default=5e-5)
    ap.add_argument("--candidate-ce-temperature", type=float, default=24.0)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--force-emb", action="store_true")
    args = ap.parse_args()
    if args.no_dreams_embedding:
        args.dreams_embedding = None
    run(args)


if __name__ == "__main__":
    main()
