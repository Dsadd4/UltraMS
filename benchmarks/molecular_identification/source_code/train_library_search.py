"""Train and evaluate UltraMS or DreaMS spectrum-library-search encoders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_PUBLIC_MODEL_ROOT = Path(__file__).resolve().parents[3] / "training" / "ultrams_training" / "model"
_PROJECT_ROOT = Path.cwd()
_TRAIN_DIR = str(_PUBLIC_MODEL_ROOT / "train")
_COMP_DIR = str(_PROJECT_ROOT / "train" / "comparison")
MSNLIB_CSV = str(_PROJECT_ROOT / "datasets" / "MSnLib" / "MSnLib.csv")
ULTRA_CKPT = str(_PROJECT_ROOT / "train" / "output" / "phase2_rt_only" / "stage_d_epoch_11.pt")
DREAMS_LOADER = str(Path(__file__).resolve().parent / "original" / "dreams_loader.py")
OUT_DIR = str(_PROJECT_ROOT / "train" / "comparison" / "output")

MAX_PEAKS = 150
SEED = 42
D_PROJ = 256
TEMPERATURE = 0.07
CHUNK = 256
POSITIVE_ADDUCTS = {'[M+H]+', '[M+Na]+', '[M+NH4]+', '[M+K]+', '[M+H-H2O]+'}
NEGATIVE_ADDUCTS = {'[M-H]-', '[M+Cl]-', '[M+FA]-', '[M+FA-H]-', '[M+CH3COO]-'}
ADDUCT_SCOPE = 'composite'


def _load_dreams_loader():
    path = Path(DREAMS_LOADER).resolve()
    spec = importlib.util.spec_from_file_location('dreams_loader', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load DreaMS loader: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['dreams_loader'] = module
    spec.loader.exec_module(module)
    return module


def parse_spectrum(mzs_str, ints_str, max_peaks=MAX_PEAKS):
    try:
        mz    = np.fromstring(str(mzs_str),  dtype=np.float32, sep=',')
        inten = np.fromstring(str(ints_str), dtype=np.float32, sep=',')
    except Exception:
        return None
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return None
    mx = inten.max()
    if mx <= 0:
        return None
    inten = np.clip(inten / mx, 0, 1)
    if len(mz) > max_peaks:
        idx = np.sort(np.argsort(inten)[-max_peaks:])
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


# ── data loading ──────────────────────────────────────────────────────────────
def load_adduct(csv_path, adduct_mode, progress_step=100_000):
    """
    Load MSnLib for one adduct mode ('pos'=[M+H]+, 'neg'=[M-H]-).
    Returns: {smiles: [{'spectrum':..., 'precursor_mz':...}, ...]}
    """
    if adduct_mode == 'pos':
        keep_adducts = {'[M+H]+'} if ADDUCT_SCOPE == 'strict' else POSITIVE_ADDUCTS
        adduct_label = '[M+H]+'
    else:
        keep_adducts = {'[M-H]-'} if ADDUCT_SCOPE == 'strict' else NEGATIVE_ADDUCTS
        adduct_label = '[M-H]-'

    t0 = time.time()
    df = pd.read_csv(csv_path,
                     usecols=['mzs', 'intensities', 'smiles',
                               'precursor_mz', 'adduct'])
    df = df[df['adduct'].isin(keep_adducts)].reset_index(drop=True)
    print(f'  {adduct_label} rows: {len(df):,}')

    if ADDUCT_SCOPE == 'strict':
        from rdkit import Chem

    compound_data = defaultdict(list)
    skip = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % progress_step == 0 and i > 0:
            print(f'  {i:,}/{len(df):,}', flush=True)
        spec = parse_spectrum(row.mzs, row.intensities)
        smi  = str(row.smiles).strip()
        if ADDUCT_SCOPE == 'strict' and smi not in ('', 'nan'):
            molecule = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ''
        if spec is None or smi in ('', 'nan'):
            skip += 1
            continue
        compound_data[smi].append({
            'spectrum':     spec,
            'precursor_mz': float(row.precursor_mz),
            'smiles':       smi,
        })

    # keep only compounds with ≥2 spectra
    valid = {s: v for s, v in compound_data.items() if len(v) >= 2}
    print(f'  Valid compounds (≥2 spectra): {len(valid):,}  skipped={skip:,}  ({time.time()-t0:.1f}s)')
    return valid


def split_compounds(compound_data, val_r=0.15, test_r=0.15, seed=SEED):
    smis = sorted(compound_data.keys())
    rng  = random.Random(seed)
    rng.shuffle(smis)
    n       = len(smis)
    n_test  = int(n * test_r)
    n_val   = int(n * val_r)
    splits  = {
        'test':  smis[:n_test],
        'val':   smis[n_test:n_test + n_val],
        'train': smis[n_test + n_val:],
    }
    for k, v in splits.items():
        print(f'  {k}: {len(v):,}')
    return splits


# ── dataset ───────────────────────────────────────────────────────────────────
class LibSearchDataset(Dataset):
    """Each sample: two random spectra from the same compound (positive pair)."""
    def __init__(self, compound_data, smis, seed=SEED):
        self.smis = sorted(smis)
        self.data = {s: compound_data[s] for s in self.smis}
        self.rng  = random.Random(seed)

    def __len__(self):
        return len(self.smis)

    def __getitem__(self, idx):
        smi = self.smis[idx]
        specs = self.data[smi]
        a, b = self.rng.sample(specs, 2)
        return a, b


def collate_pairs(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# ── projection head ───────────────────────────────────────────────────────────
class UL2Proj(nn.Module):
    """Two-layer MLP projection head (UL2-style)."""
    def __init__(self, d_in: int, d_out: int = D_PROJ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── backbone encoding ─────────────────────────────────────────────────────────
def encode_ultra(model, samples, device, no_grad=True):
    import contextlib
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        B  = len(samples)
        mp = model.max_peaks
        spectra = []
        for s in samples:
            sp = np.asarray(s['spectrum'], dtype=np.float32)
            if sp.ndim != 2 or sp.shape[1] != 2:
                sp = np.zeros((0, 2), dtype=np.float32)
            mz, inten = sp[:, 0], sp[:, 1]
            valid = mz > 0
            mz, inten = mz[valid], inten[valid]
            if len(mz) < 3:
                sp = np.zeros((0, 2), dtype=np.float32)
            else:
                mx = inten.max()
                if mx > 0:
                    inten = inten / mx
                inten = np.clip(inten, 0, 1)
                if len(mz) > mp:
                    idx = np.sort(np.argsort(inten)[-mp:])
                    mz, inten = mz[idx], inten[idx]
                sp = np.stack([mz[np.argsort(mz)], inten[np.argsort(mz)]], axis=-1).astype(np.float32)
            spectra.append(sp)

        pmzs   = torch.tensor([s['precursor_mz'] for s in samples], dtype=torch.float32, device=device)
        padded = np.zeros((B, mp, 2), dtype=np.float32)
        attn   = np.zeros((B, mp),    dtype=np.int64)
        for i, sp in enumerate(spectra):
            k = min(len(sp), mp)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i,   :k] = 1
        peaks_t = torch.from_numpy(padded).to(device)
        attn_t  = torch.from_numpy(attn).to(device)

        ms_emb   = model.peak_encoder(peaks_t)
        cls_emb  = model.cls_emb.expand(B, 1, -1)
        pi       = torch.stack([pmzs, torch.full_like(pmzs, 1.1)], dim=-1).unsqueeze(1)
        prec_emb = model.peak_encoder(pi) + model.precursor_type_emb
        full     = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
        S        = full.shape[1]
        pos      = torch.arange(S, device=device).unsqueeze(0)
        full     = model.dropout(full + model.pos_emb(pos))
        fa       = torch.cat([torch.ones(B, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1)
        out      = model.encoder(inputs_embeds=full, attention_mask=fa)
        return out.last_hidden_state[:, 0, :]


def encode_dreams(model, samples, device, no_grad=True):
    import contextlib
    mod = _load_dreams_loader()
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        peaks_t = mod.build_dreams_batch(model, samples, device)
        hs      = model(peaks_t)
        return hs[:, 0, :]


# ── model loading ─────────────────────────────────────────────────────────────
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_ultra_last_n(model, n=4):
    if n <= 0:
        return
    layers = model.encoder.encoder.layer
    L      = len(layers)
    for i in range(max(0, L - n), L):
        for p in layers[i].parameters():
            p.requires_grad_(True)


def unfreeze_dreams_last_n(model, n=4):
    if n <= 0:
        return
    L     = model.n_layers
    start = max(0, L - n)
    if model.vanilla_transformer:
        for i in range(start, L):
            for p in model.transformer_encoder.layers[i].parameters():
                p.requires_grad_(True)
        return
    te = model.transformer_encoder
    for i in range(start, L):
        for p in te.atts[i].parameters():
            p.requires_grad_(True)
        for p in te.ffs[i].parameters():
            p.requires_grad_(True)
    if hasattr(te, 'scales') and te.scales is not None:
        for p in te.scales[max(0, L - n):].parameters():
            p.requires_grad_(True)


def count_trainable(model):
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    return frozen, total


def load_ultra_backbone(device, n_unfreeze=4):
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG
    model = UltraExplorerMLM(CONFIG)
    ckpt  = torch.load(ULTRA_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['base_state_dict'], strict=False)
    ep    = ckpt.get('epoch', '?')
    step  = ckpt.get('global_step', '?')
    print(f'  [Ultra Phase2 Stage-D] {ULTRA_CKPT}')
    print(f'    ep={ep}, step={step}')
    model = model.to(device)
    freeze_all(model)
    unfreeze_ultra_last_n(model, n_unfreeze)
    frozen, total = count_trainable(model)
    print(f'  [Freeze] {frozen}/{total} frozen, {total-frozen} trainable')
    return model, CONFIG['d_model']


def load_dreams_backbone(device, n_unfreeze=4):
    mod = _load_dreams_loader()
    model = mod.load_dreams_encoder(device)
    freeze_all(model)
    unfreeze_dreams_last_n(model, n_unfreeze)
    frozen, total = count_trainable(model)
    print(f'  [Freeze] {frozen}/{total} frozen, {total-frozen} trainable')
    return model, int(model.d_model)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────
def info_nce(z_a, z_b, temperature=TEMPERATURE):
    B = z_a.shape[0]
    logits_ab = z_a @ z_b.T / temperature
    logits_ba = z_b @ z_a.T / temperature
    labels    = torch.arange(B, device=z_a.device)
    loss = (F.cross_entropy(logits_ab, labels) +
            F.cross_entropy(logits_ba, labels)) / 2
    acc  = ((logits_ab.argmax(1) == labels).float().mean() +
            (logits_ba.argmax(1) == labels).float().mean()) / 2
    return loss, acc.item()


# ── embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, proj, samples, device, backbone, bs=CHUNK):
    model.eval(); proj.eval()
    all_emb = []
    encode  = encode_ultra if backbone == 'ultra' else encode_dreams
    for i in range(0, len(samples), bs):
        batch = samples[i:i + bs]
        h     = encode(model, batch, device, no_grad=True)
        z     = proj(h)
        all_emb.append(z.cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


# ── retrieval evaluation ──────────────────────────────────────────────────────
def retrieval_metrics(q_emb, l_emb, query_smis, lib_smis, top_k=(1, 5, 10)):
    q_n  = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
    l_n  = l_emb / (np.linalg.norm(l_emb, axis=1, keepdims=True) + 1e-8)
    lib_arr = np.array(lib_smis)
    ranks = []
    for qi, qsmi in enumerate(query_smis):
        sims  = q_n[qi] @ l_n.T
        order = np.argsort(-sims)
        pos   = np.where(lib_arr[order] == qsmi)[0]
        if len(pos):
            ranks.append(int(pos[0]) + 1)
    ranks = np.array(ranks)
    res = {}
    for k in top_k:
        res[f'R@{k}'] = float((ranks <= k).mean())
    res['R@10'] = float((ranks <= 10).mean())
    res['MRR']  = float((1.0 / ranks).mean())
    res['n']    = len(ranks)
    return res


def fmt_metrics(m):
    return (f"R@1={m['R@1']:.3f}  R@5={m.get('R@5',0):.3f}  "
            f"R@10={m['R@10']:.3f}  MRR={m['MRR']:.4f}  (n={m['n']})")


# ── FDR / ROC / Recall-FDR curve computation ──────────────────────────────────
def compute_scores(q_emb, l_emb, q_smis, l_smis, chunk=512):
    l_n = l_emb / (np.linalg.norm(l_emb, axis=1, keepdims=True) + 1e-8)
    q_n = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
    l_arr = np.array(l_smis)
    n_q   = len(q_smis)
    top1_sim     = np.empty(n_q, dtype=np.float32)
    top1_correct = np.zeros(n_q, dtype=bool)
    top1_pos_sim = np.full(n_q, -1.0, dtype=np.float32)

    for start in range(0, n_q, chunk):
        end  = min(start + chunk, n_q)
        sims = q_n[start:end] @ l_n.T
        idx  = np.argmax(sims, axis=1)
        top1_sim[start:end]     = sims[np.arange(end - start), idx]
        top1_correct[start:end] = (l_arr[idx] == np.array(q_smis[start:end]))
        for bi, qi in enumerate(range(start, end)):
            pos_mask = l_arr == q_smis[qi]
            if pos_mask.any():
                top1_pos_sim[qi] = float(sims[bi, pos_mask].max())

    return top1_sim, top1_correct, top1_pos_sim


def fdr_recall_curve(top1_sim, top1_correct, top1_pos_sim, q_smis, l_smis,
                     n_thresh=200):
    l_cnt = {}
    for s in l_smis:
        l_cnt[s] = l_cnt.get(s, 0) + 1
    has_pos    = np.array([l_cnt.get(s, 0) > 0 for s in q_smis])
    n_pos_q    = int(has_pos.sum())
    MIN_HITS   = 10
    thresholds = np.linspace(0.0, 1.0, n_thresh)
    fdrs_t1, recs_t1, fdrs_any, recs_any = [], [], [], []

    for tau in thresholds:
        hit = top1_sim >= tau
        n   = int(hit.sum())
        # top-1
        nc_t1  = int((hit & top1_correct).sum())
        fdr_t1 = (n - nc_t1) / n if n >= MIN_HITS else float('nan')
        rec_t1 = nc_t1 / n_pos_q if n_pos_q > 0 else 0.0
        # any-hit
        any_ok  = top1_pos_sim >= tau
        nc_any  = int((hit & any_ok).sum())
        nw_any  = int((hit & ~any_ok).sum())
        fdr_any = nw_any / n if n >= MIN_HITS else float('nan')
        rec_any = nc_any / n_pos_q if n_pos_q > 0 else 0.0

        fdrs_t1.append(fdr_t1);   recs_t1.append(rec_t1)
        fdrs_any.append(fdr_any); recs_any.append(rec_any)

    return (thresholds,
            np.array(fdrs_t1),  np.array(recs_t1),
            np.array(fdrs_any), np.array(recs_any))


def envelope_curve(fdrs, recalls, n_pts=500):
    valid    = ~np.isnan(fdrs)
    fdr_v    = fdrs[valid];   rec_v = recalls[valid]
    fdr_grid = np.linspace(0.0, 1.0, n_pts)
    env      = np.zeros(n_pts)
    for i, f in enumerate(fdr_grid):
        mask   = fdr_v <= f
        env[i] = rec_v[mask].max() if mask.any() else 0.0
    return fdr_grid, env


def find_wp(thresholds, fdrs, recalls, fdr_target=0.05):
    valid = ~np.isnan(fdrs)
    mask  = valid & (fdrs <= fdr_target)
    if not mask.any():
        return None
    best = np.where(mask)[0][np.argmax(recalls[mask])]
    return dict(tau=float(thresholds[best]), fdr=float(fdrs[best]),
                recall=float(recalls[best]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone',    choices=('ultra', 'dreams'), required=True)
    ap.add_argument('--device',      default='cuda:0')
    ap.add_argument('--epochs',      type=int,   default=20)
    ap.add_argument('--batch',       type=int,   default=256)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--n-unfreeze',  type=int,   default=4)
    ap.add_argument('--warmup-ratio',type=float, default=0.1)
    ap.add_argument('--embed-only',  action='store_true',
                    help='skip training, load saved checkpoint and embed')
    ap.add_argument('--train-only', action='store_true',
                    help='train the strict-adduct encoder; embedding uses the complete strict-adduct manifest')
    ap.add_argument('--adduct-scope', choices=('strict', 'composite'), default='composite')
    ap.add_argument('--project-root', type=Path, default=Path.cwd())
    ap.add_argument('--source-model-root', type=Path, default=_PUBLIC_MODEL_ROOT)
    ap.add_argument('--csv', type=Path)
    ap.add_argument('--ultra-checkpoint', type=Path)
    ap.add_argument('--dreams-loader', type=Path)
    ap.add_argument('--output-dir', type=Path)
    args = ap.parse_args()

    global _PROJECT_ROOT, _TRAIN_DIR, _COMP_DIR, MSNLIB_CSV, ULTRA_CKPT, DREAMS_LOADER, OUT_DIR, ADDUCT_SCOPE
    _PROJECT_ROOT = args.project_root.resolve()
    source_model_root = args.source_model_root.resolve()
    _TRAIN_DIR = str(source_model_root / 'train')
    _COMP_DIR = str(_PROJECT_ROOT / 'train' / 'comparison')
    MSNLIB_CSV = str((args.csv or _PROJECT_ROOT / 'datasets' / 'MSnLib' / 'MSnLib.csv').resolve())
    ULTRA_CKPT = str((args.ultra_checkpoint or _PROJECT_ROOT / 'train' / 'output' / 'phase2_rt_only' / 'stage_d_epoch_11.pt').resolve())
    DREAMS_LOADER = str((args.dreams_loader or Path(__file__).resolve().parent / 'original' / 'dreams_loader.py').resolve())
    OUT_DIR = str((args.output_dir or _PROJECT_ROOT / 'train' / 'comparison' / 'output').resolve())
    ADDUCT_SCOPE = args.adduct_scope
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, _TRAIN_DIR)
    sys.path.insert(0, str(source_model_root))

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device  = torch.device(args.device)
    bb      = args.backbone
    tag     = bb
    ckpt_path = os.path.join(OUT_DIR, f'lib_search_{bb}_ul2_best.pt')

    # ── load data (both adducts) ──────────────────────────────────────────────
    print('\n=== Loading MSnLib (any CE) ===')
    data, splits = {}, {}
    for adduct in ('pos', 'neg'):
        label = '[M+H]+' if adduct == 'pos' else '[M-H]-'
        print(f'Loading {MSNLIB_CSV}  adduct={label} ...')
        data[adduct]   = load_adduct(MSNLIB_CSV, adduct)
        splits[adduct] = split_compounds(data[adduct])

    train_datasets, train_sizes = {}, {}
    for adduct in ('pos', 'neg'):
        train_datasets[adduct] = LibSearchDataset(
            data[adduct], splits[adduct]['train'])
        train_sizes[adduct] = len(splits[adduct]['train'])

    n_train_total = sum(train_sizes.values())
    train_dl_pos  = DataLoader(train_datasets['pos'], batch_size=args.batch,
                               shuffle=True, collate_fn=collate_pairs,
                               num_workers=2, pin_memory=True)
    train_dl_neg  = DataLoader(train_datasets['neg'], batch_size=args.batch,
                               shuffle=True, collate_fn=collate_pairs,
                               num_workers=2, pin_memory=True)
    # interleave both adduct loaders
    n_batches = len(train_dl_pos) + len(train_dl_neg)
    print(f'\n[M+H]+ train={train_sizes["pos"]:,}  [M-H]- train={train_sizes["neg"]:,}  total={n_train_total:,}')
    print(f'Train batches/epoch: {n_batches}')

    # small test eval: 1 query × 1 lib per compound (same adduct)
    rng_test = random.Random(SEED + 99)
    test_sets = {}
    for adduct in ('pos', 'neg'):
        qrys, libs = [], []
        for smi in sorted(splits[adduct]['test']):
            specs = data[adduct][smi]
            a, b  = rng_test.sample(specs, 2)
            qrys.append((smi, a)); libs.append((smi, b))
        test_sets[adduct] = {
            'q_smis': [x[0] for x in qrys],
            'l_smis': [x[0] for x in libs],
            'q_entries': [x[1] for x in qrys],
            'l_entries': [x[1] for x in libs],
        }
    for adduct, lbl in [('pos', '[M+H]+'), ('neg', '[M-H]-')]:
        n = len(test_sets[adduct]['q_smis'])
        print(f'{lbl} test: {n:,} queries × {n:,} library')

    # ── load backbone ─────────────────────────────────────────────────────────
    print(f'\nLoading {bb.upper()} ...')
    if bb == 'ultra':
        model, d_model = load_ultra_backbone(device, args.n_unfreeze)
        encode_fn = encode_ultra
    else:
        model, d_model = load_dreams_backbone(device, args.n_unfreeze)
        encode_fn = encode_dreams

    proj = UL2Proj(d_model, D_PROJ).to(device)

    # ── zero-shot baseline ────────────────────────────────────────────────────
    if not args.embed_only:
        print(f'\n[{bb}/lib_search] === Zero-shot baseline ===')
        for adduct, lbl in [('pos', '[M+H]+'), ('neg', '[M-H]-')]:
            ts = test_sets[adduct]
            q_emb = extract_embeddings(model, proj, ts['q_entries'], device, bb)
            l_emb = extract_embeddings(model, proj, ts['l_entries'], device, bb)
            m = retrieval_metrics(q_emb, l_emb, ts['q_smis'], ts['l_smis'])
            print(f'[{bb}/lib_search] zero-shot {lbl}: {fmt_metrics(m)}')

        # ── optimiser + scheduler ─────────────────────────────────────────────
        enc_params   = [p for p in model.parameters() if p.requires_grad]
        proj_params  = list(proj.parameters())
        param_groups = [{'params': proj_params, 'lr': args.lr}]
        if enc_params:
            param_groups.append({'params': enc_params, 'lr': args.lr * 0.1})
        optimizer    = torch.optim.AdamW(param_groups, weight_decay=0.01)
        total_steps  = n_batches * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_r1 = 0.0
        zero_shot_results = {}
        history = []
        gs = 0

        # ── training loop ─────────────────────────────────────────────────────
        for ep in range(1, args.epochs + 1):
            model.train(); proj.train()
            ep_loss, ep_acc, ep_steps = 0.0, 0.0, 0

            # iterate positive and negative adduct batches in round-robin
            iter_pos = iter(train_dl_pos)
            iter_neg = iter(train_dl_neg)
            remaining = {'pos': len(train_dl_pos), 'neg': len(train_dl_neg)}
            bi = 0
            while remaining['pos'] > 0 or remaining['neg'] > 0:
                # pick next adduct
                if remaining['pos'] > 0 and remaining['neg'] > 0:
                    adduct = 'pos' if bi % 2 == 0 else 'neg'
                elif remaining['pos'] > 0:
                    adduct = 'pos'
                else:
                    adduct = 'neg'

                try:
                    samples_a, samples_b = (next(iter_pos) if adduct == 'pos'
                                            else next(iter_neg))
                    remaining[adduct] -= 1
                except StopIteration:
                    remaining[adduct] = 0
                    continue

                h_a = encode_fn(model, samples_a, device, no_grad=False)
                h_b = encode_fn(model, samples_b, device, no_grad=False)
                z_a = proj(h_a); z_b = proj(h_b)
                loss, acc = info_nce(z_a, z_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(proj.parameters()), 1.0)
                optimizer.step()
                scheduler.step()

                ep_loss  += loss.item()
                ep_acc   += acc
                ep_steps += 1
                gs += 1
                bi += 1

                if bi % 30 == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f'[{bb}/lib_search] ep{ep}/{args.epochs} '
                          f'b{bi}/{n_batches} loss={loss.item():.4f} '
                          f'acc={acc:.3f} lr={lr_now:.2e}', flush=True)

            # ── per-epoch eval ────────────────────────────────────────────────
            metrics_ep = {}
            with torch.no_grad():
                for adduct, lbl in [('pos', '[M+H]+'), ('neg', '[M-H]-')]:
                    ts    = test_sets[adduct]
                    q_emb = extract_embeddings(model, proj, ts['q_entries'], device, bb)
                    l_emb = extract_embeddings(model, proj, ts['l_entries'], device, bb)
                    metrics_ep[adduct] = retrieval_metrics(
                        q_emb, l_emb, ts['q_smis'], ts['l_smis'])

            avg_loss = ep_loss / max(ep_steps, 1)
            r1_pos   = metrics_ep['pos']['R@1']
            r1_neg   = metrics_ep['neg']['R@1']
            r1_avg   = (r1_pos + r1_neg) / 2
            mp, mn   = metrics_ep['pos'], metrics_ep['neg']
            print(f'[{bb}/lib_search] ep{ep}  loss={avg_loss:.4f}  '
                  f'acc={ep_acc/max(ep_steps,1):.3f}  '
                  f'[M+H]+ R@1={mp["R@1"]:.3f}  R@5={mp.get("R@5",0):.3f}  '
                  f'R@10={mp["R@10"]:.3f}  MRR={mp["MRR"]:.4f}  (n={mp["n"]})  |  '
                  f'[M-H]- R@1={mn["R@1"]:.3f}  R@5={mn.get("R@5",0):.3f}  '
                  f'R@10={mn["R@10"]:.3f}  MRR={mn["MRR"]:.4f}  (n={mn["n"]})')
            history.append({'ep': ep, 'loss': avg_loss,
                            'pos': mp, 'neg': mn})

            if r1_avg > best_r1:
                best_r1 = r1_avg
                torch.save({'epoch': ep,
                            'model_state_dict': model.state_dict(),
                            'proj_state_dict':  proj.state_dict(),
                            'best_r1': best_r1},
                           ckpt_path)
                print(f'  ★ best R@1={best_r1:.4f}')

        # ── final summary ─────────────────────────────────────────────────────
        print(f'\n[{bb}/lib_search] === FINAL RESULTS ===')
        h0 = history[0]  # approx zero-shot after warmup; print first ep
        hf = history[-1]
        for adduct, lbl in [('pos', '[M+H]+'), ('neg', '[M-H]-')]:
            print(f'[{bb}/lib_search] fine-tuned {lbl}: '
                  f'{fmt_metrics(hf[adduct])}')
        print(f'[{bb}/lib_search] best R@1 ([M+H]+): {history[np.argmax([h["pos"]["R@1"] for h in history])]["pos"]["R@1"]:.4f}')

        # save JSON
        out_json = os.path.join(OUT_DIR, 'lib_search_finetune.json')
        existing = {}
        if os.path.exists(out_json):
            with open(out_json) as f:
                existing = json.load(f)
        existing[bb] = {'history': history, 'best_r1': float(best_r1)}
        with open(out_json, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f'Saved → {out_json}')

    if args.train_only:
        return

    # ── full-library embedding extraction ─────────────────────────────────────
    print(f'\n=== Full-library embedding extraction ({bb}) ===')
    # load best checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        proj.load_state_dict(ckpt['proj_state_dict'])
        print(f'  Loaded checkpoint: ep={ckpt.get("epoch","?")}, '
              f'best_r1={ckpt.get("best_r1",float("nan")):.4f}')
    else:
        print(f'  No checkpoint found at {ckpt_path}, using current weights.')
    model.eval(); proj.eval()

    # For each adduct, build:
    #   query  = one spectrum per test compound
    #   library = all spectra from train+val compounds + remaining test spectra
    fdr_curves_pos = {}
    fdr_curves_neg = {}
    roc_curves     = {}

    for adduct, lbl_str in [('pos', 'pos'), ('neg', 'neg')]:
        adduct_label = '[M+H]+' if adduct == 'pos' else '[M-H]-'
        print(f'\n=== {adduct_label} ===')
        compound = data[adduct]
        sp       = splits[adduct]

        # build query set (1 spectrum per test compound)
        rng_q = random.Random(SEED + 77)
        query_samples, query_smis = [], []
        for smi in sorted(sp['test']):
            query_samples.append(rng_q.choice(compound[smi]))
            query_smis.append(smi)

        # build library = all spectra except chosen query spectra
        # (train + val + remaining test spectra)
        lib_samples, lib_smis = [], []
        chosen_ids = {id(s): True for s in query_samples}
        for smi in sorted(compound.keys()):
            for s in compound[smi]:
                if id(s) not in chosen_ids:
                    lib_samples.append(s)
                    lib_smis.append(smi)

        n_q = len(query_smis); n_l = len(lib_smis)
        pos_per_q = np.mean([lib_smis.count(s) for s in query_smis[:100]])
        zero_pos  = sum(1 for s in query_smis if s not in set(lib_smis))
        print(f'  queries: {n_q:,}  library: {n_l:,}  '
              f'(avg {pos_per_q:.1f} pos/query, {zero_pos} zero-pos)')

        # extract embeddings
        print(f'  Extracting {bb} embeddings ...')
        q_emb = extract_embeddings(model, proj, query_samples, device, bb,
                                   bs=CHUNK)
        l_emb = extract_embeddings(model, proj, lib_samples,   device, bb,
                                   bs=CHUNK)

        # save .npy
        np.save(os.path.join(OUT_DIR, f'emb_msnlib_{lbl_str}_query_{bb}.npy'), q_emb)
        np.save(os.path.join(OUT_DIR, f'emb_msnlib_{lbl_str}_lib_{bb}.npy'),   l_emb)
        with open(os.path.join(OUT_DIR, f'emb_msnlib_{lbl_str}_query_smis.json'), 'w') as f:
            json.dump(query_smis, f)
        with open(os.path.join(OUT_DIR, f'emb_msnlib_{lbl_str}_lib_smis.json'), 'w') as f:
            json.dump(lib_smis, f)
        print(f'  Saved embeddings → emb_msnlib_{lbl_str}_*_{bb}.npy')

        # full metrics
        print(f'  Computing retrieval metrics ...')
        m = retrieval_metrics(q_emb, l_emb, query_smis, lib_smis,
                              top_k=(1, 5, 10))
        print(f'  [{bb}] Top@1={m["R@1"]:.3f}  Top@5={m.get("R@5",0):.3f}  '
              f'Top@10={m["R@10"]:.3f}  MRR={m["MRR"]:.4f}')

        # FDR/Recall/ROC scores
        print(f'  Computing FDR/ROC curves ...')
        t1_sim, t1_cor, t1_pos_sim = compute_scores(
            q_emb, l_emb, query_smis, lib_smis)
        tau, fdr_t1, rec_t1, fdr_any, rec_any = fdr_recall_curve(
            t1_sim, t1_cor, t1_pos_sim, query_smis, lib_smis)

        wp_t1  = find_wp(tau, fdr_t1,  rec_t1)
        wp_any = find_wp(tau, fdr_any, rec_any)
        if wp_t1:
            print(f'  [{bb}][top-1] @FDR≤5%: τ={wp_t1["tau"]:.4f}  '
                  f'Recall={wp_t1["recall"]:.3f}')
        else:
            print(f'  [{bb}][top-1] FDR never reaches ≤5%')

        cr = dict(fdr_t1=fdr_t1, rec_t1=rec_t1,
                  fdr_any=fdr_any, rec_any=rec_any,
                  tau=tau, wp_t1=wp_t1, wp_any=wp_any,
                  top1_sim=t1_sim, top1_correct=t1_cor)
        if adduct == 'pos':
            fdr_curves_pos[bb] = cr
            roc_curves[lbl_str] = {bb: cr}
        else:
            fdr_curves_neg[bb] = cr
            roc_curves[lbl_str] = {bb: cr}

    # save per-backbone FDR summary
    fdr_summary = {}
    for adduct, curves_dict in [('pos', fdr_curves_pos), ('neg', fdr_curves_neg)]:
        for m, cr in curves_dict.items():
            key = f'{m}_{adduct}'
            fdr_summary[key] = {
                'wp_t1':  cr['wp_t1'],
                'wp_any': cr['wp_any'],
            }
    out_json2 = os.path.join(OUT_DIR, f'lib_search_eval_{bb}.json')
    with open(out_json2, 'w') as f:
        json.dump(fdr_summary, f, indent=2)
    print(f'Saved → {out_json2}')



if __name__ == '__main__':
    main()
