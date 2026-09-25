"""
Heteroatom per-atom query pooling — training script.

Trains UltraAtomQueryProbe / DreamsAtomQueryProbe on heteroatom count
regression.  Default atoms: N O S Cl F Br.  Use --atoms to restrict.

Usage:
  python -u showcase/train_atom_query.py --backbone ultra --device cuda:0
  python -u showcase/train_atom_query.py --backbone dreams --device cuda:1
  python -u showcase/train_atom_query.py --backbone ultra --atoms N O S Cl --device cuda:0
"""
import os, sys, math, json, time, argparse, random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score

_HERE       = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', os.path.dirname(_HERE))
_PROJ_ROOT  = os.path.dirname(_TRAIN_ROOT)
sys.path.insert(0, _TRAIN_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from showcase.data import load_fold, PARQUET_PATH
from showcase.models_atom_query import (
    UltraAtomQueryProbe, DreamsAtomQueryProbe,
    HETEROATOM_NAMES, HETEROATOM_COLS, FORMULA_IDX,
)
from element.models import freeze_all, unfreeze_ultra_last_n, unfreeze_dreams_last_n

_ROOT = _TRAIN_ROOT


def _patch_phase2_import():
    """Pre-register phase2.train_phase2_rt_only from .pyc when source is missing."""
    import importlib.util
    key = 'phase2.train_phase2_rt_only'
    if key in sys.modules:
        return
    pyc = os.path.join(_TRAIN_ROOT, 'phase2', '__pycache__',
                       'train_phase2_rt_only.cpython-311.pyc')
    if not os.path.isfile(pyc):
        return   # source may be present, let normal import work
    spec = importlib.util.spec_from_file_location(key, pyc)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)


class ShowcaseDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels  = labels
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i], self.labels[i]


def collate(batch):
    samples = [b[0] for b in batch]
    labels  = torch.from_numpy(np.stack([b[1] for b in batch]))
    return samples, labels


def compute_r2(preds: np.ndarray, targets: np.ndarray, names) -> dict:
    m, per = {}, []
    for i, name in enumerate(names):
        r2 = r2_score(targets[:, i], preds[:, i])
        m[f'r2_{name}'] = float(r2)
        per.append(r2)
    m['mean_r2'] = float(np.mean(per))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone',      choices=('ultra', 'dreams'), required=True)
    ap.add_argument('--atoms',         nargs='+', default=None,
                    help='Atoms to predict, e.g. --atoms N O S Cl  (default: all 6)')
    ap.add_argument('--device',        default='auto')
    ap.add_argument('--epochs',        type=int,   default=15)
    ap.add_argument('--batch-size',    type=int,   default=512)
    ap.add_argument('--head-lr',       type=float, default=3e-4)
    ap.add_argument('--encoder-lr',    type=float, default=0.0)
    ap.add_argument('--unfreeze-last', type=int,   default=0)
    ap.add_argument('--weight-decay',  type=float, default=0.01)
    ap.add_argument('--grad-clip',     type=float, default=1.0)
    ap.add_argument('--warmup-ratio',  type=float, default=0.1)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--max-train',     type=int,   default=None)
    ap.add_argument('--max-val',       type=int,   default=None)
    ap.add_argument('--ultra-ckpt',    default=None)
    ap.add_argument('--log-interval',  type=int,   default=20)
    ap.add_argument('--out-dir',       default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    device = torch.device(device_name)
    args.device = device_name

    # ── Atom selection ─────────────────────────────────────────────────
    if args.atoms:
        atom_names = [a for a in HETEROATOM_NAMES if a in args.atoms]
    else:
        atom_names = HETEROATOM_NAMES     # N O S Cl F Br

    # column indices in formula_count labels (C H N O S Cl F Br)
    col_idx = [FORMULA_IDX[n] for n in atom_names]

    ts = time.strftime('%m%d_%H%M')
    tag_atoms = '_'.join(atom_names)
    if args.out_dir is None:
        args.out_dir = os.path.join(_ROOT, 'output', 'showcase',
                                    f'atomq_{args.backbone}_{tag_atoms}_{ts}')
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f'[atomq/{args.backbone}]'
    print(f'{tag} atoms={atom_names}', flush=True)
    print(f'{tag} Loading data...', flush=True)

    # Load formula_count (has all C/H/N/O/S/Cl/F/Br cols) then slice
    tr_s, tr_y_all, _ = load_fold(PARQUET_PATH, 'train', 'formula_count', args.max_train)
    va_s, va_y_all, _ = load_fold(PARQUET_PATH, 'val',   'formula_count', args.max_val)

    tr_y = tr_y_all[:, col_idx].astype(np.float32)
    va_y = va_y_all[:, col_idx].astype(np.float32)

    # z-score on training set
    norm_mean = tr_y.mean(0)
    norm_std  = tr_y.std(0).clip(min=1e-6)
    tr_y_z = ((tr_y - norm_mean) / norm_std).astype(np.float32)

    tr_loader = DataLoader(ShowcaseDataset(tr_s, tr_y_z), batch_size=args.batch_size,
                           shuffle=True, collate_fn=collate, num_workers=0)
    va_loader = DataLoader(ShowcaseDataset(va_s, va_y),   batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate, num_workers=0)

    print(f'{tag} train={len(tr_s):,}  val={len(va_s):,}', flush=True)

    # ── Build model ────────────────────────────────────────────────────
    if args.backbone == 'ultra':
        _patch_phase2_import()
        from selection.selection_data_build import load_phase2_stage_d_for_fusion
        enc = load_phase2_stage_d_for_fusion(device, checkpoint_path=args.ultra_ckpt)
        freeze_all(enc)
        unfreeze_ultra_last_n(enc, args.unfreeze_last)
        model = UltraAtomQueryProbe(enc, atom_names).to(device)
    else:
        from comparison.dreams_loader import load_dreams_encoder
        enc = load_dreams_encoder(device)
        freeze_all(enc)
        unfreeze_dreams_last_n(enc, args.unfreeze_last)
        model = DreamsAtomQueryProbe(enc, atom_names).to(device)

    enc_module = model.phase2 if args.backbone == 'ultra' else model.dreams
    enc_pid    = {id(p) for p in enc_module.parameters()}
    enc_params = [p for p in enc_module.parameters() if p.requires_grad]
    probe_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in enc_pid]

    param_groups = [{'params': probe_params, 'lr': args.head_lr}]
    if enc_params:
        param_groups.append({'params': enc_params, 'lr': args.encoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps  = len(tr_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────
    best_score = -1e9
    best_ckpt  = os.path.join(args.out_dir, f'best_{args.backbone}.pt')
    last_ckpt  = os.path.join(args.out_dir, f'last_{args.backbone}.pt')
    history    = []

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for bi, (batch, labels) in enumerate(tr_loader, 1):
            labels = labels.float().to(device)
            logits = model(batch, device)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if bi % args.log_interval == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'{tag} ep{ep}/{args.epochs} batch{bi}/{len(tr_loader)} '
                      f'loss={loss.item():.4f} avg={running_loss/bi:.4f} lr={lr_now:.2e}',
                      flush=True)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch, labels in va_loader:
                logits   = model(batch, device)
                pred_np  = logits.cpu().float().numpy()
                pred_np  = pred_np * norm_std + norm_mean   # inverse z-score
                all_preds.append(pred_np)
                all_targets.append(labels.numpy())

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        m       = compute_r2(preds, targets, atom_names)
        score   = m['mean_r2']

        train_loss = running_loss / len(tr_loader)
        per_str = '  '.join(f'r2_{n}={m[f"r2_{n}"]:.3f}' for n in atom_names)
        print(f'{tag} ep{ep}  train_loss={train_loss:.4f}  mean_r2={score:.4f}  [{per_str}]',
              flush=True)
        history.append({'ep': ep, 'train_loss': train_loss, **m})

        ckpt_data = {
            'epoch':        ep,
            'atom_queries': model.atom_queries.detach().cpu(),
            'k_proj_state': model.k_proj.state_dict(),
            'v_proj_state': model.v_proj.state_dict(),
            'heads_state':  [h.state_dict() for h in model.heads],
            'norm_mean':    norm_mean.tolist(),
            'norm_std':     norm_std.tolist(),
            'atom_names':   atom_names,
            'backbone':     args.backbone,
            'val_mean_r2':  score,
        }

        if score > best_score:
            best_score = score
            torch.save(ckpt_data, best_ckpt)
            print(f'  ★ best saved  mean_r2={best_score:.4f}', flush=True)

        # always overwrite last checkpoint
        torch.save(ckpt_data, last_ckpt)

    meta = {**vars(args), 'atom_names': atom_names, 'history': history, 'best_val_r2': best_score}
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Done. best_mean_r2={best_score:.4f}  -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
