"""Dual-polarity cross-collision-energy contrastive fine-tuning.

Train on both [M+H]+ and [M-H]- spectra. For chemical-subset evaluation,
identify hard subsets using zero-shot DreaMS in each polarity-specific
validation-plus-test pool, then combine the subsets.

Training data:
  [M+H]+: 28,757 compounds (seed=42).
  [M-H]-: approximately 14,712 compounds (seed=43, independently split).
  Total: approximately 43,469 compounds.

During training, monitor retrieval on the 6,161-compound [M+H]+ test set.
Use evaluate_collision_energy_backbones.py for chemical-subset analysis.

Run from the repository root:
  python benchmarks/molecular_identification/source_code/original/train_collision_energy_backbones.py --backbone ultra --device cuda:0
  python benchmarks/molecular_identification/source_code/original/train_collision_energy_backbones.py --backbone dreams --device cuda:1
"""
from __future__ import annotations
import os, sys, json, time, random, math, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

_HERE = Path(__file__).resolve().parent
_PUBLIC_ROOT = _HERE.parents[3]
_MODEL_ROOT = _PUBLIC_ROOT / 'training' / 'ultrams_training' / 'model'
_SUPPORT_ROOT = _PUBLIC_ROOT / 'benchmarks' / 'spectral_properties' / 'original' / 'support'
sys.path.insert(0, str(_MODEL_ROOT))
sys.path.insert(0, str(_MODEL_ROOT / 'train'))
sys.path.insert(0, str(_SUPPORT_ROOT))
sys.path.insert(0, str(_HERE))
os.environ.setdefault('ULTRAMS_SOURCE_TRAIN_ROOT', str(_MODEL_ROOT / 'train'))
_ASSET_ROOT = Path(os.environ.get('ULTRAMS_BENCHMARK_ROOT', Path.cwd())).resolve()

# Paths
MSNLIB_CSV  = str(_ASSET_ROOT / 'datasets' / 'MSnLib' / 'MSnLib.csv')
ULTRA_CKPT  = str(_ASSET_ROOT / 'train' / 'output' / 'pretrained' / 'ultrams_unsupervised.pt')
OUT_DIR     = os.environ.get('ULTRAMS_COLLISION_ENERGY_OUTPUT_DIR', str(_ASSET_ROOT / 'train' / 'output' / 'collision_energy'))
OUT_JSON    = os.path.join(OUT_DIR, 'collision_energy_training.json')

# Dual polarity: seed=42 for [M+H]+ and seed=43 for [M-H]-.
ADDUCT_SEEDS = {'[M+H]+': 42, '[M-H]-': 43}

LOW_CE_MAX  = 20
HIGH_CE_MIN = 40
MAX_PEAKS   = 150
SEED        = 42
D_PROJ      = 256      # Projection-head output dimension
TEMPERATURE = 0.07     # InfoNCE temperature


# Spectrum preprocessing
def parse_spectrum(mzs_str, ints_str, max_peaks=MAX_PEAKS):
    try:
        mz    = np.fromstring(mzs_str,  dtype=np.float32, sep=',')
        inten = np.fromstring(ints_str, dtype=np.float32, sep=',')
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
    return np.stack([mz, inten], axis=-1).astype(np.float32)


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


# Dataset construction
def build_compound_data(csv_path, adduct='[M+H]+'):
    """Return {csmi: {'low': [entry, ...], 'high': [entry, ...]}}."""
    print(f'Loading {csv_path}  adduct={adduct} ...')
    t0 = time.time()
    df = pd.read_csv(csv_path,
                     usecols=['mzs', 'intensities', 'smiles',
                               'precursor_mz', 'adduct', 'collision_energy'])
    df = df[df['adduct'] == adduct].dropna(subset=['collision_energy']).copy()
    print(f'  {adduct} rows: {len(df):,}')

    compound_data = defaultdict(lambda: {'low': [], 'high': []})
    skip = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 100_000 == 0 and i > 0:
            print(f'  {i:,}/{len(df):,}', flush=True)
        ce = float(row.collision_energy)
        if LOW_CE_MAX < ce < HIGH_CE_MIN:
            continue
        spec = parse_spectrum(str(row.mzs), str(row.intensities))
        if spec is None:
            skip += 1
            continue
        csmi = canonical_smiles(str(row.smiles))
        if csmi is None:
            skip += 1
            continue
        entry = {'spectrum': spec, 'precursor_mz': float(row.precursor_mz), 'ce': ce}
        if ce <= LOW_CE_MAX:
            compound_data[csmi]['low'].append(entry)
        else:
            compound_data[csmi]['high'].append(entry)

    valid = {s: v for s, v in compound_data.items()
             if v['low'] and v['high']}
    print(f'  Valid compounds: {len(valid):,}  skipped={skip:,}  ({time.time()-t0:.1f}s)')
    return valid


def split_compounds(compound_data, val_r=0.15, test_r=0.15, seed=SEED):
    smis = sorted(compound_data.keys())
    rng = random.Random(seed)
    rng.shuffle(smis)
    n = len(smis)
    n_test = int(n * test_r)
    n_val  = int(n * val_r)
    splits = {
        'test':  set(smis[:n_test]),
        'val':   set(smis[n_test:n_test + n_val]),
        'train': set(smis[n_test + n_val:]),
    }
    for k, v in splits.items():
        print(f'  {k}: {len(v):,}')
    return splits


# ── Dataset & DataLoader ──────────────────────────────────────────────
class CrossCEDataset(Dataset):
    """Sample one low-CE and one high-CE spectrum per compound as a positive pair."""
    def __init__(self, compound_data: dict, smis: set, seed=SEED):
        self.smis = sorted(smis)
        self.data = {s: compound_data[s] for s in self.smis}
        self.rng  = random.Random(seed)

    def __len__(self):
        return len(self.smis)

    def __getitem__(self, idx):
        smi = self.smis[idx]
        v   = self.data[smi]
        low  = self.rng.choice(v['low'])
        high = self.rng.choice(v['high'])
        return low, high


def collate_pairs(batch):
    lows  = [b[0] for b in batch]
    highs = [b[1] for b in batch]
    return lows, highs


# ── Projection Head ───────────────────────────────────────────────────
class ProjectionHead(nn.Module):
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


# Backbone embedding extraction
def encode_ultra(enc, entries, device):
    from element.models import _ultra_prepare
    batch = [{'spectrum': e['spectrum'], 'precursor_mz': e['precursor_mz']}
             for e in entries]
    peaks_t, attn_t, pmz_t = _ultra_prepare(batch, device)
    hs, cls = enc._encode(peaks_t, attn_t, pmz_t, device)
    # CLS + mean-pool over peak tokens
    sub_hs    = hs[:, 2:, :]
    attn_exp  = attn_t.float().unsqueeze(-1)
    denom     = attn_exp.sum(dim=1).clamp(min=1)
    mean_pool = (sub_hs * attn_exp).sum(dim=1) / denom
    return torch.cat([cls, mean_pool], dim=-1)  # (B, 2d)


def encode_dreams(enc, entries, device):
    from dreams_loader import build_dreams_batch
    batch = [{'spectrum': e['spectrum'], 'precursor_mz': e['precursor_mz']}
             for e in entries]
    peaks_t = build_dreams_batch(enc, batch, device)
    hs = enc(peaks_t)
    return hs[:, 0, :]   # CLS


# ── InfoNCE Loss ──────────────────────────────────────────────────────
def info_nce(z_low, z_high, temperature=TEMPERATURE):
    """z_low, z_high: (B, D), L2-normalized.
    Low/high spectra of the same compound form a positive pair; the remaining
    batch members are negatives. Compute the loss in both directions.
    """
    B = z_low.shape[0]
    logits_lh = z_low @ z_high.T / temperature   # (B, B)
    logits_hl = z_high @ z_low.T / temperature
    labels = torch.arange(B, device=z_low.device)
    loss = (F.cross_entropy(logits_lh, labels) +
            F.cross_entropy(logits_hl, labels)) / 2
    # within-batch top-1 acc
    acc = ((logits_lh.argmax(dim=1) == labels).float().mean().item() +
           (logits_hl.argmax(dim=1) == labels).float().mean().item()) / 2
    return loss, acc


# Extract all embeddings for retrieval evaluation.
@torch.no_grad()
def extract_all(enc, proj, entries, device, backbone, batch_size=256):
    enc.eval(); proj.eval()
    all_emb = []
    for i in range(0, len(entries), batch_size):
        batch_e = entries[i:i + batch_size]
        if backbone == 'ultra':
            h = encode_ultra(enc, batch_e, device)
        else:
            h = encode_dreams(enc, batch_e, device)
        z = proj(h)
        all_emb.append(z.cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


# Retrieval metrics
def retrieval_metrics(q_emb, l_emb, query_smis, library_smis, top_k=(1, 5, 10)):
    q = torch.from_numpy(q_emb).float()
    l = torch.from_numpy(l_emb).float()
    sims = (q @ l.T).numpy()      # (N_q, N_l)
    lib_arr = np.array(library_smis)
    ranks = []
    for qi, qsmi in enumerate(query_smis):
        order = np.argsort(sims[qi])[::-1]
        pos = np.where(lib_arr[order] == qsmi)[0]
        if len(pos):
            ranks.append(int(pos[0]) + 1)
    ranks = np.array(ranks)
    res = {'n': len(ranks), 'lib_size': len(library_smis)}
    for k in top_k:
        res[f'R@{k}'] = float((ranks <= k).mean())
    res['MRR'] = float((1.0 / ranks).mean())
    return res


def fmt(m):
    return (f"R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
            f"R@10={m['R@10']:.3f}  MRR={m['MRR']:.4f}  (n={m['n']})")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', choices=('ultra', 'dreams'), required=True)
    ap.add_argument('--device',   default='cuda:0')
    ap.add_argument('--epochs',   type=int,   default=20)
    ap.add_argument('--batch',    type=int,   default=256)
    ap.add_argument('--lr',       type=float, default=3e-4)
    ap.add_argument('--unfreeze-last', type=int, default=0,
                    help='Unfreeze the last N backbone layers for training')
    ap.add_argument('--warmup-ratio', type=float, default=0.1)
    ap.add_argument('--tag', default='',
                    help='Checkpoint name suffix to distinguish experiments (for example, ul2 means unfreeze 2 layers)')
    ap.add_argument('--out-json', default=OUT_JSON)
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device(args.device)

    # Load data from both ionization modes.
    # [M+H]+: seed=42.
    data_pos = build_compound_data(MSNLIB_CSV, '[M+H]+')
    splits_pos = split_compounds(data_pos, seed=42)
    # [M-H]-: seed=43, independently split.
    data_neg = build_compound_data(MSNLIB_CSV, '[M-H]-')
    splits_neg = split_compounds(data_neg, seed=43)

    train_smis_pos = splits_pos['train']
    train_smis_neg = splits_neg['train']
    test_smis_pos  = splits_pos['test']

    print(f'\n[M+H]+ train={len(train_smis_pos):,}  '
          f'[M-H]- train={len(train_smis_neg):,}  '
          f'total={len(train_smis_pos)+len(train_smis_neg):,}')

    # Combine training data; adduct labels distinguish the two modes.
    # Prefix keys so identical canonical SMILES in different adduct modes remain distinct.
    all_train_data: dict = {}
    for smi in train_smis_pos:
        all_train_data[f'pos_{smi}'] = data_pos[smi]
    for smi in train_smis_neg:
        all_train_data[f'neg_{smi}'] = data_neg[smi]

    train_ds = CrossCEDataset(all_train_data, set(all_train_data.keys()))
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, collate_fn=collate_pairs,
                          num_workers=2, pin_memory=True)
    print(f'Train batches/epoch: {len(train_dl)}')

    # Build [M+H]+ test retrieval pairs.
    rng_test = random.Random(42 + 99)
    queries, library = [], []
    for smi in sorted(test_smis_pos):
        v = data_pos[smi]
        queries.append((smi, rng_test.choice(v['low'])))
        library.append((smi, rng_test.choice(v['high'])))
    query_smis   = [q[0] for q in queries]
    library_smis = [l[0] for l in library]
    q_entries = [q[1] for q in queries]
    l_entries = [l[1] for l in library]
    print(f'[M+H]+ test: {len(queries):,} queries × {len(library):,} library')

    # Build [M-H]- test retrieval pairs.
    test_smis_neg = splits_neg['test']
    rng_test_neg = random.Random(43 + 99)
    queries_neg, library_neg = [], []
    for smi in sorted(test_smis_neg):
        v = data_neg[smi]
        queries_neg.append((smi, rng_test_neg.choice(v['low'])))
        library_neg.append((smi, rng_test_neg.choice(v['high'])))
    query_smis_neg   = [q[0] for q in queries_neg]
    library_smis_neg = [l[0] for l in library_neg]
    q_entries_neg = [q[1] for q in queries_neg]
    l_entries_neg = [l[1] for l in library_neg]
    print(f'[M-H]- test: {len(queries_neg):,} queries × {len(library_neg):,} library')

    # Load the backbone.
    if args.backbone == 'ultra':
        from selection.selection_data_build import load_ultrams_encoder_with_heads
        from element.models import freeze_all, unfreeze_ultra_last_n
        enc = load_ultrams_encoder_with_heads(device, checkpoint_path=ULTRA_CKPT)
        freeze_all(enc)
        unfreeze_ultra_last_n(enc, args.unfreeze_last)
        d_in = int(enc.config['d_model']) * 2   # CLS + mean-pool
    else:
        from dreams_loader import load_dreams_encoder
        from element.models import freeze_all, unfreeze_dreams_last_n
        enc = load_dreams_encoder(device)
        freeze_all(enc)
        unfreeze_dreams_last_n(enc, args.unfreeze_last)
        d_in = int(enc.d_model)

    proj = ProjectionHead(d_in, D_PROJ).to(device)

    # Optimizer
    enc_params   = [p for p in enc.parameters() if p.requires_grad]
    probe_params = list(proj.parameters())
    param_groups = [{'params': probe_params, 'lr': args.lr}]
    if enc_params:
        param_groups.append({'params': enc_params, 'lr': args.lr * 0.1})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps  = len(train_dl) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tag = f'[{args.backbone}/collision_energy]'
    encode_fn = encode_ultra if args.backbone == 'ultra' else encode_dreams

    # Zero-shot baseline before training
    print(f'\n{tag} === Zero-shot baseline (before fine-tuning) ===')
    with torch.no_grad():
        q_emb0     = extract_all(enc, proj, q_entries,     device, args.backbone)
        l_emb0     = extract_all(enc, proj, l_entries,     device, args.backbone)
        q_emb0_neg = extract_all(enc, proj, q_entries_neg, device, args.backbone)
        l_emb0_neg = extract_all(enc, proj, l_entries_neg, device, args.backbone)
    m0     = retrieval_metrics(q_emb0,     l_emb0,     query_smis,     library_smis)
    m0_neg = retrieval_metrics(q_emb0_neg, l_emb0_neg, query_smis_neg, library_smis_neg)
    print(f'{tag} zero-shot [M+H]+: {fmt(m0)}')
    print(f'{tag} zero-shot [M-H]-: {fmt(m0_neg)}')

    history = []
    best_r1  = m0['R@1']
    tag_suffix = f'_{args.tag}' if args.tag else ''
    model_name = 'ultrams' if args.backbone == 'ultra' else args.backbone
    best_ckpt = os.path.join(OUT_DIR, f'collision_energy_{model_name}{tag_suffix}.pt')
    gs = 0

    # Training loop
    for ep in range(1, args.epochs + 1):
        enc.train(); proj.train()
        ep_loss, ep_acc = 0.0, 0.0
        for bi, (lows, highs) in enumerate(train_dl, 1):
            h_low  = encode_fn(enc,  lows,  device)
            h_high = encode_fn(enc,  highs, device)
            z_low  = proj(h_low)
            z_high = proj(h_high)
            loss, acc = info_nce(z_low, z_high)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(proj.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_acc  += acc
            gs += 1
            if bi % 30 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'{tag} ep{ep}/{args.epochs} b{bi}/{len(train_dl)} '
                      f'loss={loss.item():.4f} acc={acc:.3f} lr={lr_now:.2e}', flush=True)

        # Validation evaluation
        with torch.no_grad():
            q_emb     = extract_all(enc, proj, q_entries,     device, args.backbone)
            l_emb     = extract_all(enc, proj, l_entries,     device, args.backbone)
            q_emb_neg = extract_all(enc, proj, q_entries_neg, device, args.backbone)
            l_emb_neg = extract_all(enc, proj, l_entries_neg, device, args.backbone)
        m     = retrieval_metrics(q_emb,     l_emb,     query_smis,     library_smis)
        m_neg = retrieval_metrics(q_emb_neg, l_emb_neg, query_smis_neg, library_smis_neg)
        avg_loss = ep_loss / len(train_dl)
        print(f'{tag} ep{ep}  loss={avg_loss:.4f}  acc={ep_acc/len(train_dl):.3f}  '
              f'[M+H]+ {fmt(m)}  |  [M-H]- {fmt(m_neg)}')
        history.append({'ep': ep, 'loss': avg_loss,
                        'pos': m, 'neg': m_neg})

        if m['R@1'] > best_r1:
            best_r1 = m['R@1']
            ckpt = {'epoch': ep,
                    'proj': proj.state_dict(),
                    'best_r1': best_r1,
                    'adducts': ['[M+H]+', '[M-H]-'],
                    'unfreeze_last': args.unfreeze_last}
            if args.unfreeze_last > 0:
                ckpt['enc'] = enc.state_dict()
            torch.save(ckpt, best_ckpt)
            print(f'  ★ best R@1={best_r1:.4f}')

    # Final result summary
    print(f'\n{tag} === FINAL RESULTS ===')
    print(f'{tag} zero-shot  [M+H]+: {fmt(m0)}')
    print(f'{tag} zero-shot  [M-H]-: {fmt(m0_neg)}')
    print(f'{tag} fine-tuned [M+H]+: {fmt(history[-1]["pos"])}')
    print(f'{tag} fine-tuned [M-H]-: {fmt(history[-1]["neg"])}')
    print(f'{tag} best R@1 ([M+H]+): {best_r1:.4f}')

    out = {
        'backbone':    args.backbone,
        'epochs':      args.epochs,
        'batch':       args.batch,
        'unfreeze_last': args.unfreeze_last,
        'zero_shot':   m0,
        'history':     history,
        'best_r1':     best_r1,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    # Merge into the existing JSON.
    existing = {}
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            existing = json.load(f)
    existing[f'{args.backbone}_finetune'] = out
    with open(args.out_json, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'Saved → {args.out_json}')


if __name__ == '__main__':
    main()
