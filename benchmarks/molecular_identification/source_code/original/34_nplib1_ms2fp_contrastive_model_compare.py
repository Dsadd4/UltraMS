"""Evaluate MoNA-triplet fine-tuned UltraMS against DreaMS on NPLIB1 MS2FP.

This reuses the existing NPLIB1 MS2-block protocol and MS2FP candidate-CE
retrieval logic from comparison/17_nplib1_ms2fp_retrieval.py.  The only new
piece is spectrum embedding extraction from a saved UltraMS triplet checkpoint
or from the public DreaMS contrastive embedding checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("LIGHT_ULTRA_ROOT", str(HERE.parent.parent))).resolve()
TRAIN_DIR = ROOT / "train"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(HERE))

MS2FP_PATH = HERE / "17_nplib1_ms2fp_retrieval.py"
PROBE_PATH = HERE / "13_nplib1_ms2blocks_advantage_probe.py"

spec = importlib.util.spec_from_file_location("ms2fp_mod", MS2FP_PATH)
ms2fp = importlib.util.module_from_spec(spec)
sys.modules["ms2fp_mod"] = ms2fp
spec.loader.exec_module(ms2fp)

probe_spec = importlib.util.spec_from_file_location("nplib1_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(probe_spec)
sys.modules["nplib1_probe_for_triplet_eval"] = probe
probe_spec.loader.exec_module(probe)

from appliedmodel.common import get_backbone_hidden_dim, load_ultra_backbone  # noqa: E402
import dreams_loader  # noqa: E402


OUT_BASE = TRAIN_DIR / "output" / "comparison" / "nplib1_ms2fp_contrastive_model_compare"
DREAMS_EMBED_CKPT = TRAIN_DIR / "comparison" / "resources" / "DreaMS_Check" / "embedding_model.ckpt"


class ProjectionWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, projection: nn.Module, max_peaks: int):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.max_peaks = int(max_peaks)


def build_projection_from_state(in_dim: int, state: dict, kind_hint: str = "auto") -> nn.Module:
    if kind_hint == "linear" or (
        kind_hint == "auto" and set(state.keys()) == {"weight", "bias"}
    ):
        proj = nn.Linear(in_dim, in_dim, bias=True)
        proj.load_state_dict(state)
        return proj
    if kind_hint == "mlp" or kind_hint == "auto":
        # State keys follow Sequential: 0 LayerNorm, 1 Linear, 4 Linear.
        if "1.weight" not in state or "4.weight" not in state:
            raise ValueError(f"Cannot infer MLP projection from state keys: {sorted(state)[:20]}")
        hidden = int(state["1.weight"].shape[0])
        proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, in_dim),
        )
        proj.load_state_dict(state, strict=False)
        return proj
    raise ValueError(f"unknown projection kind: {kind_hint}")


@torch.no_grad()
def extract_ultra_triplet_embeddings(ckpt_path: Path, samples: list[dict], device: torch.device,
                                     batch_size: int = 128, use_projection: bool = True) -> np.ndarray:
    from appliedmodel.common import ultra_forward_tokens, ultra_prepare

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    encoder = load_ultra_backbone(
        device,
        checkpoint_path=config.get("ultra_ckpt") or None,
        state_key=config.get("state_key", "base_state_dict"),
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
    encoder.eval()
    max_peaks = int(config.get("max_peaks", 100))
    in_dim = get_backbone_hidden_dim(encoder)
    projection = build_projection_from_state(in_dim, ckpt["projection_state_dict"], config.get("projection", "auto"))
    projection.to(device).eval()
    outs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        peaks_t, attn_t, pmz_t = ultra_prepare(batch, device, max_peaks=max_peaks)
        cls, _ = ultra_forward_tokens(encoder, peaks_t, attn_t, pmz_t)
        emb = projection(cls) if use_projection else cls
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"    Ultra-triplet embed: {start}/{len(samples)}", flush=True)
    del encoder, projection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def patch_dreams_msml_model_aliases() -> None:
    """Map legacy DreaMS checkpoint module names onto the vendored dreams package."""
    # dreams_loader._patch_msml creates msml as a simple module.  Old public
    # embedding checkpoints also reference msml.models.*, so make it package-like.
    msml = sys.modules.get("msml")
    if msml is not None and not hasattr(msml, "__path__"):
        msml.__path__ = []
    aliases = {
        "msml.models": "dreams.models",
        "msml.models.heads": "dreams.models.heads",
        "msml.models.heads.heads": "dreams.models.heads.heads",
        "msml.models.dreams": "dreams.models.dreams",
        "msml.models.dreams.dreams": "dreams.models.dreams.dreams",
    }
    for legacy, current in aliases.items():
        sys.modules[legacy] = importlib.import_module(current)
    if "msml" in sys.modules:
        sys.modules["msml"].models = sys.modules["msml.models"]
    sys.modules["msml.models"].heads = sys.modules["msml.models.heads"]
    sys.modules["msml.models"].dreams = sys.modules["msml.models.dreams"]


def patch_dreams_preprocessor_defaults(backbone: nn.Module) -> None:
    """Fill attributes added to SpectrumPreprocessor after old checkpoints were saved."""
    spec_preproc = getattr(backbone, "spec_preproc", None)
    if spec_preproc is None:
        return
    defaults = {
        "prec_intens": 1.1,
        "n_highest_peaks": dreams_loader.N_HIGHEST_PEAKS,
        "spec_entropy_cleaning": False,
        "normalize_mzs": False,
        "to_relative_intensities": True,
        "precision": 32,
        "mz_shift_aug_p": 0,
        "mz_shift_aug_max": 0,
    }
    for key, value in defaults.items():
        if not hasattr(spec_preproc, key):
            setattr(spec_preproc, key, value)


@torch.no_grad()
def extract_dreams_embedding_ckpt(samples: list[dict], device: torch.device, batch_size: int = 128,
                                  ckpt_path: Path = DREAMS_EMBED_CKPT) -> np.ndarray:
    """Load public DreaMS ContrastiveHead checkpoint and return projected embeddings."""
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    # Ensure the loader uses the same DreaMS source tree as the project copy.
    dreams_root = TRAIN_DIR / "comparison" / "resources" / "dreams" / "DreaMS"
    if dreams_root.exists():
        sys.path.insert(0, str(dreams_root))
    dreams_loader._patch_msml()
    from dreams.models.heads.heads import ContrastiveHead

    backbone_pth = TRAIN_DIR / "comparison" / "resources" / "DreaMS_Check" / "ssl_model.ckpt"
    patch_dreams_msml_model_aliases()
    model = ContrastiveHead.load_from_checkpoint(
        ckpt_path,
        backbone_pth=backbone_pth,
        map_location=device,
    ).to(device).eval()
    patch_dreams_preprocessor_defaults(model.backbone)
    model.backbone.ff_out = None
    model.backbone.ro_out = None
    outs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        peaks = dreams_loader.build_dreams_batch(model.backbone, batch, device)
        emb = model(peaks, charge=None)
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        if (start // batch_size) % 25 == 0:
            print(f"    DreaMS-embedding embed: {start}/{len(samples)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def write_rows(path: Path, rows):
    fields = [
        "split", "scenario", "model", "spec_id", "parent_spec_id", "smiles",
        "block_type", "adduct", "n_peaks_model", "n_peaks_original",
        "n_peaks_above_precursor_plus1", "candidate_count", "candidate_count_raw",
        "candidate_count_scored", "rank", "hit1", "hit5", "hit10", "hit20",
        "rr", "gt_score", "top1_score", "top1_is_gt", "top1_smiles",
        "pred_on_mass",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def embedding_cache_path(out_dir: Path, model_key: str, scenario_name: str, split_id: int) -> Path:
    emb_dir = out_dir / "embeddings" / scenario_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    safe = model_key.replace("/", "_").replace(":", "_")
    return emb_dir / f"spec_embeddings_split{split_id}_{safe}.npy"


def get_embeddings(model_key: str, model_cfg: dict, out_dir: Path, samples: list[dict],
                   scenario_name: str, split_id: int, device: torch.device,
                   force: bool, batch_size: int) -> tuple[np.ndarray, str]:
    path = embedding_cache_path(out_dir, model_key, scenario_name, split_id)
    if path.exists() and not force:
        return np.load(path), str(path)
    kind = model_cfg["kind"]
    if kind == "ultra_triplet":
        emb = extract_ultra_triplet_embeddings(
            Path(model_cfg["ckpt"]),
            samples,
            device,
            batch_size=batch_size,
            use_projection=bool(model_cfg.get("use_projection", True)),
        )
    elif kind == "dreams_embedding":
        emb = extract_dreams_embedding_ckpt(
            samples,
            device,
            batch_size=batch_size,
            ckpt_path=Path(model_cfg.get("ckpt", DREAMS_EMBED_CKPT)),
        )
    elif kind == "baseline":
        emb, _, emb_path = probe.get_spec_embeddings(
            model_cfg["name"], split_id, samples, scenario_name, probe.SCENARIOS[scenario_name],
            device, force_emb=force,
        )
        return emb, emb_path
    else:
        raise ValueError(kind)
    np.save(path, emb)
    return emb, str(path)


def load_model_configs(args) -> dict[str, dict]:
    cfg = {}
    if args.include_baselines:
        cfg["rt_only_d11_ssl"] = {"kind": "baseline", "name": "rt_only_d11"}
        cfg["dreams_ssl"] = {"kind": "baseline", "name": "dreams"}
    if args.dreams_embedding:
        cfg["dreams_embedding"] = {"kind": "dreams_embedding", "ckpt": str(args.dreams_embedding)}
    for item in args.ultra_triplet:
        # key=path or path; default key is checkpoint parent name.
        if "=" in item:
            key, path = item.split("=", 1)
        else:
            path = item
            key = Path(path).parent.name
        cfg[key] = {"kind": "ultra_triplet", "ckpt": path, "use_projection": not args.no_triplet_projection}
    if not cfg:
        raise ValueError("No models specified.")
    return cfg


def run(args) -> None:
    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_configs = load_model_configs(args)
    config = {
        "run_name": args.run_name,
        "models": model_configs,
        "splits": args.splits,
        "scenarios": args.scenarios,
        "seed": args.seed,
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
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    summary = {"config": config, "scenarios": {}}

    # Reuse the original argparse namespace expected by ms2fp.train_head/evaluate.
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

    for scenario_name in args.scenarios:
        scenario = probe.SCENARIOS[scenario_name]
        print("\n" + "=" * 88)
        print(f"Scenario {scenario_name}: {scenario.get('desc', '')}")
        scenario_summary = {"splits": {}, "description": scenario.get("desc", "")}
        rows_by_model = defaultdict(list)
        for split_id in args.splits:
            samples, train_idx, val_idx, test_idx, _ = probe.load_samples_with_meta(split_id, scenario)
            with open(ms2fp.runner.candidates_path(split_id)) as fh:
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
            for model_key, model_cfg in model_configs.items():
                print(f"\n[{model_key}] split {split_id}")
                spec_emb, emb_path = get_embeddings(
                    model_key, model_cfg, out_dir, samples, scenario_name, split_id,
                    device, args.force_emb, args.embed_batch_size,
                )
                head, best, history, fp_stats, logit_shift = ms2fp.train_head(
                    spec_emb, samples, train_f, val_f, candidates_by_block, fp_dict,
                    ms_args, device,
                )
                rows, skipped = ms2fp.evaluate(
                    head, spec_emb, samples, test_f, candidates_by_block, fp_dict,
                    device, args.score, logit_shift=logit_shift,
                )
                for row in rows:
                    row.update({"split": split_id, "scenario": scenario_name, "model": model_key})
                rows_by_model[model_key].extend(rows)
                safe = model_key.replace("/", "_").replace(":", "_")
                row_path = out_dir / f"query_rows_{scenario_name}_split{split_id}_{safe}.csv"
                hist_path = out_dir / f"history_{scenario_name}_split{split_id}_{safe}.json"
                head_path = out_dir / f"ms2fp_head_{scenario_name}_split{split_id}_{safe}_seed{args.seed}.pt"
                write_rows(row_path, rows)
                hist_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
                torch.save(head.state_dict(), head_path)
                metrics = {
                    "block_micro": probe.summarize_rows(rows),
                    "molecule_macro": probe.macro_summary(rows, "smiles"),
                    "parent_macro": probe.macro_summary(rows, "parent_spec_id"),
                    "stratified": probe.stratified_summary(rows),
                    "best_val": best,
                    "fp_stats": fp_stats,
                    "skipped": skipped,
                    "embedding_path": emb_path,
                    "query_rows": str(row_path),
                    "history": str(hist_path),
                    "head_path": str(head_path),
                }
                scenario_summary["splits"][str(split_id)][model_key] = metrics
                b = metrics["block_micro"]
                print(
                    f"    TEST n={b.get('n_queries', 0)} Top1={b.get('top1', 0):.2f} "
                    f"Top5={b.get('top5', 0):.2f} MRR={b.get('mrr', 0):.4f}"
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
        # Compute deltas against DreaMS embedding when available.
        if "dreams_embedding" in rows_by_model:
            for model_key in rows_by_model:
                if model_key != "dreams_embedding":
                    aggregate[f"{model_key}_minus_dreams_embedding_bootstrap"] = probe.bootstrap_delta(
                        rows_by_model[model_key], rows_by_model["dreams_embedding"], seed=args.seed + 177
                    )
        summary["scenarios"][scenario_name] = {
            **scenario_summary,
            "aggregate": aggregate,
        }
        (out_dir / "summary.partial.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("\nSaved", out_dir / "summary.json")
    for scenario_name, sc in summary["scenarios"].items():
        print("\n" + scenario_name)
        for model_key in model_configs:
            b = sc["aggregate"][model_key]["block_micro"]
            print(f"  {model_key}: Top1={b.get('top1', 0):.2f} Top5={b.get('top5', 0):.2f} MRR={b.get('mrr', 0):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="triplet_compare")
    ap.add_argument("--ultra-triplet", nargs="*", default=[], help="key=/path/best.pt or /path/best.pt")
    ap.add_argument("--dreams-embedding", type=Path, default=DREAMS_EMBED_CKPT)
    ap.add_argument("--no-dreams-embedding", action="store_true",
                    help="Disable the default DreaMS embedding baseline for UltraMS-only sweeps.")
    ap.add_argument("--include-baselines", action="store_true")
    ap.add_argument("--no-triplet-projection", action="store_true")
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
