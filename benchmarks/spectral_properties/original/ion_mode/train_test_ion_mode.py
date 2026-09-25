"""
Build compact polarity classification benchmark from MSnLib (and optional Spectraverse).
=======================================================================================
Selection criteria:
  - Only canonical adducts: [M+H]+ (pos) and [M-H]- (neg)
  - Balanced splits (50/50 POS/NEG)
  - Train: 100K per class (for linear/MLP probe)
  - Val: 3K per class
  - Test: 5K per class

Protocols (--protocol):
  - in_domain_msnlib: train/val/test all from MSnLib.
  - app_spectraverse: train probe on MSnLib only; val+test on Spectraverse.
    This measures transfer from MSnLib training spectra to SpectraVerse spectra.
    Use --fit-pol-head-msnlib-both for: MSnLib-train / SV-eval. DreaMS: random-init
    PolarityHead on frozen emb. Ultra: built-in Stage-D PolarityHead weights, then
    fine-tune that head on frozen Phase2 CLS; eval on Spectraverse only.
  - spectraverse_train_probe: train/val/test all Spectraverse. Non-rt_only_d
    models fit LogReg and a PolarityHead-matched head on SV train; rt_only_d
    skips embedding LogReg/MLP/fusion and uses the pretrained PolarityHead.

Evaluation (frozen spectrum encoder):
  - logreg: LogisticRegressionCV fitted on frozen spectrum embeddings.
  - mlp: multilayer perceptron fitted on frozen spectrum embeddings.
  - pol_head: rt_only_d = Stage-D head (fixed; val threshold). Other backbones =
    same head architecture trained on benchmark train (val threshold).
  - rt_only_d + fused: optional val-tuned mix of z-scored pol_head logit + LogReg
    decision_function (train marginals for z-score only).

Usage:
  Run through `benchmarks/spectral_properties/run_experiment.py ion_mode`.
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.environ.get('ULTRAMS_SOURCE_SUPPLEMENT_ROOT'):
    sys.path.insert(0, os.environ['ULTRAMS_SOURCE_SUPPLEMENT_ROOT'])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'support'))

POSITIVE_ADDUCTS = {'[M+H]+'}
NEGATIVE_ADDUCTS = {'[M-H]-'}


MSNLIB_CSV = os.environ.get('ULTRAMS_MSNLIB_CSV', 'datasets/MSnLib/MSnLib.csv')
SPECTRAVERSE_CSV = os.environ.get('ULTRAMS_SPECTRAVERSE_CSV', 'datasets/Spectraverse/Spectraverse.csv')

DATASETS = {
    'MSnLib': MSNLIB_CSV,
}

# Default subsampling seed; override with --seed (same CSV + same seed => same indices).
DEFAULT_RNG_SEED = 42
TRAIN_PER_CLASS = 100000
VAL_PER_CLASS = 3000
TEST_PER_CLASS_PER_SOURCE = 5000

# GPU-oriented defaults (override: --encode-batch-size / --probe-*).
DEFAULT_ENCODE_BATCH_SIZE = 512
DEFAULT_PROBE_BATCH_SIZE = 4096
DEFAULT_PROBE_FORWARD_BATCH_SIZE = 16384


def adduct_to_polarity(adduct):
    if adduct in POSITIVE_ADDUCTS:
        return 1
    if adduct in NEGATIVE_ADDUCTS:
        return 0
    return -1


def load_dataset_fold(csv_path, fold_name, max_rows=None, rng_seed=DEFAULT_RNG_SEED):
    """Load one fold from a CSV, return samples with valid polarity only."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[df['fold'] == fold_name].reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=rng_seed).reset_index(drop=True)

    samples, pols = [], []
    for _, row in df.iterrows():
        adduct = str(row.get('adduct', ''))
        pol = adduct_to_polarity(adduct)
        if pol < 0:
            continue
        mzs = np.array([float(x) for x in str(row['mzs']).split(',')], dtype=np.float32)
        ints = np.array([float(x) for x in str(row['intensities']).split(',')], dtype=np.float32)
        samples.append({
            'spectrum': np.stack([mzs, ints], axis=-1),
            'precursor_mz': float(row['precursor_mz']),
            'adduct': adduct,
        })
        pols.append(pol)
    return samples, np.array(pols)


def balanced_sample(samples, pols, n_per_class, rng):
    """Sample n_per_class from each polarity, return indices."""
    pos_idx = np.where(pols == 1)[0]
    neg_idx = np.where(pols == 0)[0]
    n = min(n_per_class, len(pos_idx), len(neg_idx))
    sel_pos = rng.choice(pos_idx, n, replace=False)
    sel_neg = rng.choice(neg_idx, n, replace=False)
    return np.sort(np.concatenate([sel_pos, sel_neg]))


def build_benchmark(protocol='in_domain_msnlib', rng_seed=DEFAULT_RNG_SEED):
    """Build benchmark splits. See module docstring for protocol definitions."""
    rng = np.random.RandomState(int(rng_seed))

    if protocol == 'spectraverse_train_probe':
        print("Loading train data (Spectraverse train — DreaMS LogReg fits here)...")
        train_s, train_p = load_dataset_fold(SPECTRAVERSE_CSV, 'train', rng_seed=rng_seed)
        print(f"  Spectraverse train: {len(train_s)} (pos={(train_p==1).sum()}, neg={(train_p==0).sum()})")
        train_p = np.array(train_p)
        idx = balanced_sample(train_s, train_p, TRAIN_PER_CLASS, rng)
        train_samples = [train_s[i] for i in idx]
        train_pols = train_p[idx]
        print(f"  Selected train: {len(train_samples)} (pos={(train_pols==1).sum()}, neg={(train_pols==0).sum()})")
        del train_s, train_p
    elif protocol in ('in_domain_msnlib', 'app_spectraverse'):
        print("Loading train data (MSnLib train — probe training)...")
        train_s, train_p = load_dataset_fold(MSNLIB_CSV, 'train', rng_seed=rng_seed)
        print(f"  MSnLib train: {len(train_s)} (pos={(train_p==1).sum()}, neg={(train_p==0).sum()})")
        train_p = np.array(train_p)
        idx = balanced_sample(train_s, train_p, TRAIN_PER_CLASS, rng)
        train_samples = [train_s[i] for i in idx]
        train_pols = train_p[idx]
        print(f"  Selected train: {len(train_samples)} (pos={(train_pols==1).sum()}, neg={(train_pols==0).sum()})")
        del train_s, train_p
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    if protocol == 'in_domain_msnlib':
        print("\nLoading val data (MSnLib)...")
        val_s, val_p = load_dataset_fold(MSNLIB_CSV, 'val', rng_seed=rng_seed)
        print(f"  MSnLib val: {len(val_s)} (pos={(val_p==1).sum()}, neg={(val_p==0).sum()})")
        val_p = np.array(val_p)
        idx = balanced_sample(val_s, val_p, VAL_PER_CLASS, rng)
        val_samples = [val_s[i] for i in idx]
        val_pols = val_p[idx]
        print(f"  Selected val: {len(val_samples)} (pos={(val_pols==1).sum()}, neg={(val_pols==0).sum()})")
        del val_s, val_p

        print("\nLoading test data (MSnLib)...")
        test_s, test_p = load_dataset_fold(MSNLIB_CSV, 'test', rng_seed=rng_seed)
        print(f"  MSnLib test: {len(test_s)} (pos={(test_p==1).sum()}, neg={(test_p==0).sum()})")
        test_p = np.array(test_p)
        idx = balanced_sample(test_s, test_p, TEST_PER_CLASS_PER_SOURCE, rng)
        test_samples = [test_s[i] for i in idx]
        test_pols = test_p[idx]
        test_sources = np.array(['MSnLib'] * len(test_samples))
        del test_s, test_p
    elif protocol in ('app_spectraverse', 'spectraverse_train_probe'):
        print("\nLoading val data (Spectraverse)...")
        val_s, val_p = load_dataset_fold(SPECTRAVERSE_CSV, 'val', rng_seed=rng_seed)
        print(f"  Spectraverse val: {len(val_s)} (pos={(val_p==1).sum()}, neg={(val_p==0).sum()})")
        val_p = np.array(val_p)
        idx = balanced_sample(val_s, val_p, VAL_PER_CLASS, rng)
        val_samples = [val_s[i] for i in idx]
        val_pols = val_p[idx]
        print(f"  Selected val: {len(val_samples)} (pos={(val_pols==1).sum()}, neg={(val_pols==0).sum()})")
        del val_s, val_p

        print("\nLoading test data (Spectraverse)...")
        test_s, test_p = load_dataset_fold(SPECTRAVERSE_CSV, 'test', rng_seed=rng_seed)
        print(f"  Spectraverse test: {len(test_s)} (pos={(test_p==1).sum()}, neg={(test_p==0).sum()})")
        test_p = np.array(test_p)
        idx = balanced_sample(test_s, test_p, TEST_PER_CLASS_PER_SOURCE, rng)
        test_samples = [test_s[i] for i in idx]
        test_pols = test_p[idx]
        test_sources = np.array(['Spectraverse'] * len(test_samples))
        del test_s, test_p
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    print(f"  Selected test: {len(test_samples)} (pos={(test_pols==1).sum()}, neg={(test_pols==0).sum()})")
    for name in np.unique(test_sources):
        mask = test_sources == name
        print(f"    {name}: {mask.sum()} (pos={(test_pols[mask]==1).sum()}, neg={(test_pols[mask]==0).sum()})")

    all_samples = train_samples + val_samples + test_samples
    all_pols = np.concatenate([train_pols, val_pols, test_pols])
    train_range = np.arange(len(train_samples))
    val_range = np.arange(len(train_samples), len(train_samples) + len(val_samples))
    test_range = np.arange(len(train_samples) + len(val_samples), len(all_samples))

    return all_samples, all_pols, train_range, val_range, test_range, test_sources


def prep_spectrum(spec_np, max_peaks=150):
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    mx = inten.max()
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0, 1)
    if len(mz) > max_peaks:
        idx = np.argsort(inten)[-max_peaks:]
        idx = np.sort(idx)
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


@torch.no_grad()
def extract_embeddings(model, samples, device, model_type, bs=DEFAULT_ENCODE_BATCH_SIZE):
    model.eval()
    all_emb = []
    for s in range(0, len(samples), bs):
        batch = samples[s:s + bs]
        B = len(batch)
        if model_type == 'dreams':
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparison'))
            from dreams_loader import build_dreams_batch
            peaks = build_dreams_batch(model, batch, device)
            out = model(peaks)
            emb = out[:, 0, :]
        else:
            mp = 150
            spectra = [prep_spectrum(x['spectrum'], mp) for x in batch]
            pmzs = torch.tensor([x['precursor_mz'] for x in batch], dtype=torch.float32, device=device)
            padded = np.zeros((B, mp, 2), dtype=np.float32)
            attn = np.zeros((B, mp), dtype=np.int64)
            for i, sp in enumerate(spectra):
                k = min(len(sp), mp)
                if k > 0:
                    padded[i, :k] = sp[:k]
                    attn[i, :k] = 1
            peaks_t = torch.from_numpy(padded).to(device)
            attn_t = torch.from_numpy(attn).to(device)
            ms_emb = model.peak_encoder(peaks_t)
            cls_emb = model.cls_emb.expand(B, 1, -1)
            pi = torch.stack([pmzs, torch.full_like(pmzs, 1.1)], dim=-1).unsqueeze(1)
            prec_emb = model.peak_encoder(pi) + model.precursor_type_emb
            full = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
            S = full.shape[1]
            pos = torch.arange(S, device=device).unsqueeze(0)
            full = model.dropout(full + model.pos_emb(pos))
            fa = torch.cat([torch.ones(B, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1)
            out = model.encoder(inputs_embeds=full, attention_mask=fa)
            emb = out.last_hidden_state[:, 0, :]
        all_emb.append(emb.cpu().numpy())
        if s > 0 and (s // bs) % 100 == 0:
            print(f"    {s}/{len(samples)}", flush=True)
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_cls_and_pol_head_logits(phase2_model, samples, device,
                                   bs=DEFAULT_ENCODE_BATCH_SIZE):
    """Same CLS as Phase2 forward; polarity logits from trained PolarityHead."""
    phase2_model.eval()
    cls_list, pol_list = [], []
    for s in range(0, len(samples), bs):
        batch = samples[s:s + bs]
        B = len(batch)
        mp = 150
        spectra = [prep_spectrum(x['spectrum'], mp) for x in batch]
        pmzs = torch.tensor([x['precursor_mz'] for x in batch], dtype=torch.float32, device=device)
        padded = np.zeros((B, mp, 2), dtype=np.float32)
        attn = np.zeros((B, mp), dtype=np.int64)
        for i, sp in enumerate(spectra):
            k = min(len(sp), mp)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i, :k] = 1
        peaks_t = torch.from_numpy(padded).to(device)
        attn_t = torch.from_numpy(attn).to(device)
        _, cls = phase2_model._encode(peaks_t, attn_t, pmzs, device)
        pol_logit = phase2_model.pol_head(cls)
        cls_list.append(cls.cpu().numpy())
        pol_list.append(pol_logit.cpu().numpy())
        if s > 0 and (s // bs) % 100 == 0:
            print(f"    {s}/{len(samples)}", flush=True)
    return np.concatenate(cls_list, axis=0), np.concatenate(pol_list, axis=0)


@torch.no_grad()
def extract_phase2_cls_embeddings(phase2_model, samples, device,
                                  bs=DEFAULT_ENCODE_BATCH_SIZE):
    """Phase2 CLS only (no pretrained PolarityHead forward); for trainable PolHead on MSnLib."""
    phase2_model.eval()
    cls_list = []
    for s in range(0, len(samples), bs):
        batch = samples[s:s + bs]
        B = len(batch)
        mp = 150
        spectra = [prep_spectrum(x['spectrum'], mp) for x in batch]
        pmzs = torch.tensor([x['precursor_mz'] for x in batch], dtype=torch.float32, device=device)
        padded = np.zeros((B, mp, 2), dtype=np.float32)
        attn = np.zeros((B, mp), dtype=np.int64)
        for i, sp in enumerate(spectra):
            k = min(len(sp), mp)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i, :k] = 1
        peaks_t = torch.from_numpy(padded).to(device)
        attn_t = torch.from_numpy(attn).to(device)
        _, cls = phase2_model._encode(peaks_t, attn_t, pmzs, device)
        cls_list.append(cls.cpu().numpy())
        if s > 0 and (s // bs) % 100 == 0:
            print(f"    {s}/{len(samples)}", flush=True)
    return np.concatenate(cls_list, axis=0)


def load_phase2_stage_d_for_fusion(device, checkpoint_path=None):
    """
    Phase2Model from UltraMS ``phase2/train_phase2_rt_only.py`` **Stage D**
    (joint MLM + RT + polarity). Uses the same encoder/CLS path as polarity probes.

    ``checkpoint_path``: optional ``.pt`` from ``train_stage_d`` (keys include
    ``model_state_dict``). Default: ``<train_root>/output/phase2_rt_only/stage_d_epoch_1.pt``.
    Relative paths are resolved under the ``train/`` package root.
    """
    from phase2.train_phase2_rt_only import Phase2Model, PHASE2_CONFIG
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG

    train_root = os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if checkpoint_path:
        p = checkpoint_path
        path = p if os.path.isabs(p) else os.path.join(train_root, p)
    else:
        path = os.environ.get('ULTRAMS_STAGE_D_CKPT', os.path.join(train_root, 'output', 'phase2_rt_only', 'stage_d_epoch_1.pt'))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Ultra Stage-D checkpoint not found: {path}\n'
            f'Train Stage D (rt_only) or pass checkpoint_path= to load_phase2_stage_d_for_fusion.')

    base = UltraExplorerMLM(CONFIG)
    model = Phase2Model(base, PHASE2_CONFIG)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  [Ultra Phase2 Stage-D] {path}\n"
          f"    ep={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')}")
    return model.to(device).eval()


def load_phase2_model_rt_only_d(device):
    """Backward-compatible alias: Stage-D weights under default rt_only output path."""
    return load_phase2_stage_d_for_fusion(device, checkpoint_path=None)


def eval_pol_head_with_val_threshold(pol_logits_np, pols, val_idx, test_idx):
    """
    PolarityHead: tune scalar threshold on val only, then report val/test.
    (Head weights fixed; no training on MSnLib train for the head.)
    """
    va_logits = pol_logits_np[val_idx]
    y_va = pols[val_idx]
    thr, _ = _best_threshold_binary(va_logits, y_va)
    results = {
        'pol_head_meta': {
            'threshold_from_val': float(thr),
            'trained_on': 'pretrained_joint_stage_d',
            'note': 'Phase2 PolarityHead (fixed); threshold maximizes val accuracy (binary).',
        },
    }
    for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
        logits_np = pol_logits_np[split_idx]
        y_true = pols[split_idx]
        y_pred = (logits_np > thr).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        print(f"    {split_name}: Acc={acc:.4f}  F1={f1_mac:.4f}  (pol_head, thr={thr:.4f})")
        results[f'pol_head_{split_name}'] = {
            'accuracy': float(acc), 'f1_macro': float(f1_mac),
            'f1_neg': float(f1_neg), 'f1_pos': float(f1_pos),
            'n': len(split_idx),
        }
    return results


def fuse_pol_head_and_logreg(pol_logits_np, z_logreg_np, pols, train_idx, val_idx,
                             test_idx, n_weights=101):
    """
    Late-fuse PolarityHead logit and LogReg decision_function.
    Z-score both using train split only; grid-search mix weight w and threshold
    on val only; evaluate val/test with chosen (w, thr).
    """
    tr = train_idx
    mu_p = float(np.mean(pol_logits_np[tr]))
    sig_p = float(np.std(pol_logits_np[tr]) + 1e-8)
    mu_z = float(np.mean(z_logreg_np[tr]))
    sig_z = float(np.std(z_logreg_np[tr]) + 1e-8)
    pn = (pol_logits_np - mu_p) / sig_p
    zn = (z_logreg_np - mu_z) / sig_z

    y_va = pols[val_idx]
    best_w, best_thr, best_val_acc = 0.5, 0.0, -1.0
    for w in np.linspace(0.0, 1.0, n_weights):
        comb_va = w * pn[val_idx] + (1.0 - w) * zn[val_idx]
        thr, vac = _best_threshold_binary(comb_va, y_va)
        if vac > best_val_acc:
            best_val_acc = vac
            best_w = float(w)
            best_thr = float(thr)

    results = {
        'fused_meta': {
            'weight_pol_head': best_w,
            'weight_logreg': float(1.0 - best_w),
            'threshold_on_combined': best_thr,
            'val_accuracy_for_selection': float(best_val_acc),
            'zscore_from': 'train split only',
            'n_weights_grid': int(n_weights),
        },
    }
    for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
        comb = best_w * pn[split_idx] + (1.0 - best_w) * zn[split_idx]
        y_true = pols[split_idx]
        y_pred = (comb > best_thr).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        print(f"    {split_name}: Acc={acc:.4f}  F1={f1_mac:.4f}  "
              f"(fused w_pol={best_w:.3f}, thr={best_thr:.4f})")
        results[f'fused_{split_name}'] = {
            'accuracy': float(acc), 'f1_macro': float(f1_mac),
            'f1_neg': float(f1_neg), 'f1_pos': float(f1_pos),
            'n': len(split_idx),
        }
    return results


def _best_threshold_binary(logits_np, y_np, n_grid=401):
    """Pick scalar threshold t maximizing accuracy on balanced binary labels."""
    if logits_np.size == 0:
        return 0.0, 0.0
    lo, hi = float(np.min(logits_np)), float(np.max(logits_np))
    if hi <= lo:
        ts = np.array([0.0, lo], dtype=np.float64)
    else:
        ts = np.linspace(lo, hi, n_grid, dtype=np.float64)
    best_t, best_acc = 0.0, -1.0
    for t in ts:
        pred = (logits_np > t).astype(np.int64)
        acc = (pred == y_np).mean()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, float(best_acc)


@torch.no_grad()
def _mlp_forward_logits(mlp, X_np, device, bs=DEFAULT_PROBE_FORWARD_BATCH_SIZE):
    X = torch.from_numpy(X_np).float().to(device)
    out = []
    for s in range(0, len(X), bs):
        out.append(mlp(X[s:s + bs]).squeeze(-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_mlp_probe(emb, pols, train_idx, val_idx, test_idx, device,
                    hidden1=512, hidden2=256, max_epochs=120, lr=3e-3,
                    min_lr=1e-5, bs=DEFAULT_PROBE_BATCH_SIZE,
                    forward_bs=DEFAULT_PROBE_FORWARD_BATCH_SIZE,
                    patience=25, wd=2e-4, dropout=0.15,
                    rng_seed=DEFAULT_RNG_SEED):
    """
    Deeper MLP on frozen embeddings: cosine LR, early stopping on val acc,
    threshold tuned on val (same protocol for every backbone).
    """
    import torch.nn.functional as F
    dim = emb.shape[1]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(dim, hidden1), torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden1, hidden2), torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden2, 1),
    ).to(device)

    X_tr = torch.from_numpy(emb[train_idx]).float().to(device)
    y_tr = torch.from_numpy(pols[train_idx]).float().to(device)
    X_va_np = emb[val_idx]
    y_va = pols[val_idx]
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=wd)

    torch.manual_seed(int(rng_seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(rng_seed))
    n = len(train_idx)
    best_state, best_thr, best_val_acc = None, 0.0, -1.0
    bad = 0
    last_ep = 0

    for ep in range(1, max_epochs + 1):
        last_ep = ep
        mlp.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            logits = mlp(X_tr[idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        cos = 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / max(1, max_epochs - 1)))
        lr_now = min_lr + (lr - min_lr) * cos
        for g in opt.param_groups:
            g['lr'] = lr_now

        mlp.eval()
        va_logits = _mlp_forward_logits(mlp, X_va_np, device, bs=forward_bs)
        thr, _ = _best_threshold_binary(va_logits, y_va)
        y_pred_va = (va_logits > thr).astype(np.int64)
        val_acc = float(accuracy_score(y_va, y_pred_va))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_thr = thr
            best_state = copy.deepcopy(mlp.state_dict())
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            print(f"    epoch {ep}: loss={total_loss/n:.4f}  lr={lr_now:.2e}  "
                  f"val_acc@{thr:.4f}={val_acc:.4f}  best_val={best_val_acc:.4f}")
        if bad >= patience:
            print(f"    early stop at epoch {ep} (patience={patience})")
            break

    if best_state is None:
        best_state = mlp.state_dict()
    mlp.load_state_dict(best_state)
    va_logits_final = _mlp_forward_logits(mlp, X_va_np, device, bs=forward_bs)
    best_thr, val_acc_at_thr = _best_threshold_binary(va_logits_final, y_va)
    best_val_acc = float(max(best_val_acc, val_acc_at_thr))

    results = {'mlp_meta': {
        'hidden1': hidden1, 'hidden2': hidden2, 'max_epochs': max_epochs,
        'epochs_run': last_ep, 'best_val_accuracy': best_val_acc,
        'threshold_from_val': float(best_thr),
        'patience': patience,
        'train_batch_size': int(bs),
        'forward_batch_size': int(forward_bs),
    }}

    with torch.no_grad():
        for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
            X_np = emb[split_idx]
            logits_np = _mlp_forward_logits(mlp, X_np, device, bs=forward_bs)
            y_true = pols[split_idx]
            y_pred = (logits_np > best_thr).astype(np.int64)
            acc = accuracy_score(y_true, y_pred)
            f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            print(f"    {split_name}: Acc={acc:.4f}  F1={f1_mac:.4f}  "
                  f"F1-neg={f1_neg:.4f}  F1-pos={f1_pos:.4f}  (thr={best_thr:.4f})")
            results[f'mlp_{split_name}'] = {
                'accuracy': float(acc), 'f1_macro': float(f1_mac),
                'f1_neg': float(f1_neg), 'f1_pos': float(f1_pos),
                'n': len(split_idx),
            }
    return results


def train_polarity_head_probe(emb, pols, train_idx, val_idx, test_idx, device,
                              max_epochs=80, lr=3e-3, min_lr=1e-5,
                              bs=DEFAULT_PROBE_BATCH_SIZE,
                              forward_bs=DEFAULT_PROBE_FORWARD_BATCH_SIZE,
                              patience=20, wd=2e-4, rng_seed=DEFAULT_RNG_SEED,
                              init_state_dict=None):
    """
    PolarityHead (same arch as Ultra Phase2 PolarityHead) on frozen embeddings.
    Trains on train_idx only; checkpoint by val acc with per-epoch val threshold;
    final val/test use threshold from val (matches rt_only_d pol_head protocol).
    init_state_dict: if set, load before training (e.g. Stage-D pol_head for Ultra fine-tune).
    """
    import torch.nn.functional as F
    from phase2.train_phase2_rt_only import PolarityHead

    dim = emb.shape[1]
    head = PolarityHead(dim).to(device)
    if init_state_dict is not None:
        head.load_state_dict(init_state_dict)
    X_tr = torch.from_numpy(emb[train_idx]).float().to(device)
    y_tr = torch.from_numpy(pols[train_idx]).float().to(device)
    X_va_np = emb[val_idx]
    y_va = pols[val_idx]
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    torch.manual_seed(int(rng_seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(rng_seed))
    n = len(train_idx)
    best_state, best_thr, best_val_acc = None, 0.0, -1.0
    bad = 0
    last_ep = 0

    for ep in range(1, max_epochs + 1):
        last_ep = ep
        head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            logits = head(X_tr[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        cos = 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / max(1, max_epochs - 1)))
        lr_now = min_lr + (lr - min_lr) * cos
        for g in opt.param_groups:
            g['lr'] = lr_now

        head.eval()
        va_logits = _pol_head_forward_logits(head, X_va_np, device, bs=forward_bs)
        thr, _ = _best_threshold_binary(va_logits, y_va)
        y_pred_va = (va_logits > thr).astype(np.int64)
        val_acc = float(accuracy_score(y_va, y_pred_va))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_thr = thr
            best_state = copy.deepcopy(head.state_dict())
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            print(f"    epoch {ep}: loss={total_loss/n:.4f}  lr={lr_now:.2e}  "
                  f"val_acc@{thr:.4f}={val_acc:.4f}  best_val={best_val_acc:.4f}")
        if bad >= patience:
            print(f"    early stop at epoch {ep} (patience={patience})")
            break

    if best_state is None:
        best_state = head.state_dict()
    head.load_state_dict(best_state)
    va_logits_final = _pol_head_forward_logits(head, X_va_np, device, bs=forward_bs)
    best_thr, val_acc_at_thr = _best_threshold_binary(va_logits_final, y_va)
    best_val_acc = float(max(best_val_acc, val_acc_at_thr))

    _init_note = (
        'stage_d_pol_head_finetune' if init_state_dict is not None else 'random_init')
    results = {
        'pol_head_meta': {
            'architecture': (
                'Ultra PolarityHead: Linear(d,d/2)→GELU→LayerNorm(d/2)→Linear(d/2,1)'
            ),
            'weight_init': _init_note,
            'trained_on': 'benchmark_train_split',
            'max_epochs': max_epochs,
            'epochs_run': last_ep,
            'best_val_accuracy': best_val_acc,
            'threshold_from_val': float(best_thr),
            'patience': patience,
            'train_batch_size': int(bs),
            'forward_batch_size': int(forward_bs),
            'note': (
                'Frozen encoder; head trained on probe train; threshold maximizes val accuracy.'
                + (' Stage-D pol_head weights then MSnLib-train fine-tune.'
                   if init_state_dict is not None else '')),
        },
    }

    head.eval()
    with torch.no_grad():
        for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
            logits_np = _pol_head_forward_logits(head, emb[split_idx], device,
                                                  bs=forward_bs)
            y_true = pols[split_idx]
            y_pred = (logits_np > best_thr).astype(np.int64)
            acc = accuracy_score(y_true, y_pred)
            f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            print(f"    {split_name}: Acc={acc:.4f}  F1={f1_mac:.4f}  "
                  f"(pol_head-fit, thr={best_thr:.4f})")
            results[f'pol_head_{split_name}'] = {
                'accuracy': float(acc), 'f1_macro': float(f1_mac),
                'f1_neg': float(f1_neg), 'f1_pos': float(f1_pos),
                'n': len(split_idx),
            }
    return results


@torch.no_grad()
def _pol_head_forward_logits(head_module, X_np, device,
                             bs=DEFAULT_PROBE_FORWARD_BATCH_SIZE):
    X = torch.from_numpy(X_np).float().to(device)
    out = []
    for s in range(0, len(X), bs):
        out.append(head_module(X[s:s + bs]).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_model(name, device):
    PHASE2_CKPTS = {
        'rt_only_c': ('./output/phase2_rt_only/stage_c_epoch_1.pt', 'base_state_dict'),
        'rt_only_d': (os.environ.get('ULTRAMS_STAGE_D_CKPT', './output/phase2_rt_only/stage_d_epoch_1.pt'), 'base_state_dict'),
    }
    if name in ('v9', 'rt_only_c', 'rt_only_d'):
        from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG
        model = UltraExplorerMLM(CONFIG)
        if name in PHASE2_CKPTS:
            path, key = PHASE2_CKPTS[name]
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt[key])
            print(f"  [{name}] ep={ckpt.get('epoch','?')}, step={ckpt.get('global_step','?')}")
        else:
            ckpt = torch.load('./output/ue_multiscale_v9/checkpoint_epoch_3.pt',
                              map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  [v9] ep={ckpt.get('epoch','?')}")
        return model.to(device).eval(), 'v9'
    elif name == 'dreams':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'comparison'))
        from dreams_loader import load_dreams_encoder
        model = load_dreams_encoder(device)
        return model, 'dreams'
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['dreams', 'rt_only_d'],
                        choices=['v9', 'dreams', 'rt_only_c', 'rt_only_d'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--probes', default='logreg',
                        choices=('none', 'logreg', 'mlp', 'both'),
                        help='none=no LogReg/MLP (pol_head only if enabled); logreg=default; mlp; both')
    parser.add_argument('--protocol', default='in_domain_msnlib',
                        choices=('in_domain_msnlib', 'app_spectraverse', 'spectraverse_train_probe'),
                        help='spectraverse_train_probe: SV train/val/test; DreaMS LogReg on SV train; rt_only_d PolarityHead only.')
    parser.add_argument('--no-pol-head', action='store_true',
                        help='Skip Phase2 PolarityHead eval for rt_only_d (LogReg/MLP only).')
    parser.add_argument('--no-pol-head-fit', action='store_true',
                        help='Do not train PolarityHead on DreaMS/other baselines (only MLP/LogReg as configured).')
    parser.add_argument('--ultra-pol-vs-dreams-mlp', action='store_true',
                        help='SV probe; rt_only_d PolarityHead vs DreaMS MLP only (no LogReg, no PolHead-fit).')
    parser.add_argument('--ultra-pol-vs-dreams-polhead', action='store_true',
                        help='SV probe; same-arch PolarityHead: rt_only_d pretrained vs DreaMS PolHead-fit (no LogReg/MLP).')
    parser.add_argument('--fit-pol-head-msnlib-both', action='store_true',
                        help='app_spectraverse: Ultra = Stage-D PolarityHead fine-tuned on MSnLib (frozen CLS); '
                             'DreaMS = PolHead-fit from scratch; SV val/test. Sets --probes none, models, protocol.')
    parser.add_argument('--no-fuse', action='store_true',
                        help='Skip pol_head + LogReg late fusion (rt_only_d).')
    parser.add_argument('--seed', type=int, default=DEFAULT_RNG_SEED,
                        help='RNG seed for balanced subsampling from each fold + sklearn/torch probes (default 42).')
    parser.add_argument('--encode-batch-size', type=int, default=DEFAULT_ENCODE_BATCH_SIZE,
                        help='Batch for spectrum→embedding (DreaMS / Ultra CLS). Increase on H100.')
    parser.add_argument('--probe-batch-size', type=int, default=DEFAULT_PROBE_BATCH_SIZE,
                        help='Mini-batch for MLP / PolHead-fit training.')
    parser.add_argument('--probe-forward-batch-size', type=int, default=DEFAULT_PROBE_FORWARD_BATCH_SIZE,
                        help='Chunk size for probe forward-only passes (val/test logits).')
    args = parser.parse_args()
    _excl = (args.ultra_pol_vs_dreams_mlp, args.ultra_pol_vs_dreams_polhead, args.fit_pol_head_msnlib_both)
    if sum(bool(x) for x in _excl) > 1:
        parser.error('use at most one of: --ultra-pol-vs-dreams-mlp, --ultra-pol-vs-dreams-polhead, '
                     '--fit-pol-head-msnlib-both')
    if args.ultra_pol_vs_dreams_mlp:
        args.protocol = 'spectraverse_train_probe'
        args.probes = 'mlp'
        args.no_pol_head_fit = True
        args.models = ['dreams', 'rt_only_d']
    elif args.ultra_pol_vs_dreams_polhead:
        args.protocol = 'spectraverse_train_probe'
        args.probes = 'none'
        args.no_pol_head_fit = False
        args.models = ['dreams', 'rt_only_d']
    elif args.fit_pol_head_msnlib_both:
        args.protocol = 'app_spectraverse'
        args.probes = 'none'
        args.no_pol_head_fit = False
        args.models = ['dreams', 'rt_only_d']
    if min(args.encode_batch_size, args.probe_batch_size, args.probe_forward_batch_size) < 1:
        parser.error('encode/probe batch sizes must be >= 1')
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    device = torch.device(device_name)

    print("=" * 70)
    print(f"Polarity Benchmark  protocol={args.protocol}  rng_seed={args.seed}")
    if args.ultra_pol_vs_dreams_mlp:
        print("  mode: Ultra PolarityHead vs DreaMS MLP only (--ultra-pol-vs-dreams-mlp)")
    elif args.ultra_pol_vs_dreams_polhead:
        print("  mode: same-arch PolarityHead — Ultra pretrained vs DreaMS PolHead-fit "
              "(--ultra-pol-vs-dreams-polhead)")
    elif args.fit_pol_head_msnlib_both:
        _u = "Ultra PolHead random init" if args.msnlib_ultra_pol_scratch else "Ultra Stage-D pol_head fine-tune"
        print(f"  mode: MSnLib train → {_u} + DreaMS PolHead scratch; "
              f"SV val/test (--fit-pol-head-msnlib-both)")
    print(f"  batches: encode={args.encode_batch_size}  probe_train={args.probe_batch_size}  "
          f"probe_forward={args.probe_forward_batch_size}")
    print("  [M+H]+ / [M-H]- only; balanced train/val/test per protocol")
    print("  Same CSV + same --seed => same train/val/test subsample indices.")
    print("=" * 70)

    samples, pols, train_idx, val_idx, test_idx, test_sources = build_benchmark(
        args.protocol, rng_seed=args.seed)
    print(f"\nBenchmark size: {len(samples):,}")

    all_results = {}
    want_pol_head = not args.no_pol_head
    ultra_pol_head_only = args.protocol == 'spectraverse_train_probe'
    want_fuse = (
        want_pol_head and (not args.no_fuse)
        and args.probes in ('logreg', 'both')
        and (not ultra_pol_head_only))

    for mn in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {mn}")
        print(f"{'='*60}")

        pol_logits = None
        ultra_pol_head_init_sd = None
        if mn == 'rt_only_d' and want_pol_head:
            model = load_phase2_model_rt_only_d(device)
            if args.fit_pol_head_msnlib_both:
                if args.msnlib_ultra_pol_scratch:
                    ultra_pol_head_init_sd = None
                    print(f"  Extracting Phase2 CLS only ({len(samples)} samples; "
                          f"random PolarityHead → train on MSnLib train, --msnlib-ultra-pol-scratch)...")
                else:
                    ultra_pol_head_init_sd = copy.deepcopy(model.pol_head.state_dict())
                    print(f"  Extracting Phase2 CLS only ({len(samples)} samples; "
                          f"Stage-D PolarityHead → fine-tune on MSnLib train)...")
                emb = extract_phase2_cls_embeddings(
                    model, samples, device, bs=args.encode_batch_size)
                print(f"  Shape: emb={emb.shape}")
            else:
                print(f"  Extracting CLS + pretrained PolarityHead ({len(samples)} samples)...")
                emb, pol_logits = extract_cls_and_pol_head_logits(
                    model, samples, device, bs=args.encode_batch_size)
                print(f"  Shape: emb={emb.shape}, pol_logits={pol_logits.shape}")
            del model; torch.cuda.empty_cache()
        else:
            model, mtype = load_model(mn, device)
            print(f"  Extracting embeddings ({len(samples)} samples)...")
            emb = extract_embeddings(
                model, samples, device, mtype, bs=args.encode_batch_size)
            del model; torch.cuda.empty_cache()
            print(f"  Shape: {emb.shape}")

        model_results = {}
        logreg_z = None

        skip_probe_for_ultra = ultra_pol_head_only and mn == 'rt_only_d'

        if args.probes in ('logreg', 'both') and not skip_probe_for_ultra:
            print(f"  [LogReg] Training linear probe...")
            clf = LogisticRegressionCV(max_iter=2000, cv=3, n_jobs=-1, random_state=args.seed)
            clf.fit(emb[train_idx], pols[train_idx])
            logreg_z = clf.decision_function(emb)
            for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
                y_pred = clf.predict(emb[split_idx])
                y_true = pols[split_idx]
                acc = accuracy_score(y_true, y_pred)
                f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
                f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                print(f"    {split_name}: Acc={acc:.4f}  F1={f1_mac:.4f}")
                model_results[f'logreg_{split_name}'] = {
                    'accuracy': float(acc), 'f1_macro': float(f1_mac),
                    'f1_neg': float(f1_neg), 'f1_pos': float(f1_pos),
                    'n': len(split_idx),
                }
        elif skip_probe_for_ultra and args.probes in ('logreg', 'both'):
            print(f"  [LogReg] skipped for rt_only_d (protocol={args.protocol}; use pol_head only).")

        if args.probes in ('mlp', 'both') and not skip_probe_for_ultra:
            print(f"  [MLP] Training MLP probe...")
            mlp_results = train_mlp_probe(
                emb, pols, train_idx, val_idx, test_idx, device,
                bs=args.probe_batch_size, forward_bs=args.probe_forward_batch_size,
                rng_seed=args.seed)
            model_results.update(mlp_results)
        elif skip_probe_for_ultra and args.probes in ('mlp', 'both'):
            print(f"  [MLP] skipped for rt_only_d (pol_head-only protocol).")

        fit_pol_here = (
            want_pol_head and (not args.no_pol_head_fit)
            and ((mn != 'rt_only_d') or args.fit_pol_head_msnlib_both))
        if fit_pol_here:
            if mn == 'rt_only_d' and args.fit_pol_head_msnlib_both:
                if args.msnlib_ultra_pol_scratch:
                    print(f"  [PolHead-fit] random-init PolarityHead on Phase2 CLS, MSnLib train...")
                else:
                    print(f"  [PolHead-fit] built-in PolarityHead (Stage-D init) on Phase2 CLS, MSnLib train...")
            else:
                print(f"  [PolHead-fit] Ultra-matched PolarityHead on embeddings (train split)...")
            phfit = train_polarity_head_probe(
                emb, pols, train_idx, val_idx, test_idx, device,
                bs=args.probe_batch_size, forward_bs=args.probe_forward_batch_size,
                rng_seed=args.seed,
                init_state_dict=ultra_pol_head_init_sd)
            model_results.update(phfit)
        elif want_pol_head and mn != 'rt_only_d' and args.no_pol_head_fit:
            print(f"  [PolHead-fit] skipped (--no-pol-head-fit).")

        if mn == 'rt_only_d' and want_pol_head and (not args.fit_pol_head_msnlib_both):
            print(f"  [PolHead] Threshold on val, eval val/test (pretrained head fixed)...")
            ph = eval_pol_head_with_val_threshold(pol_logits, pols, val_idx, test_idx)
            model_results.update(ph)
            if want_fuse and logreg_z is not None:
                print(f"  [Fused] pol_head + LogReg (w, thr from val; z-score from train)...")
                fz = fuse_pol_head_and_logreg(
                    pol_logits, logreg_z, pols, train_idx, val_idx, test_idx)
                model_results.update(fz)

        all_results[mn] = model_results
        del emb
        if mn == 'rt_only_d' and want_pol_head and pol_logits is not None:
            del pol_logits

    methods_print = []
    if args.probes == 'none':
        methods_print = []
    elif args.probes == 'logreg':
        methods_print = ['logreg']
    elif args.probes == 'mlp':
        methods_print = ['mlp']
    else:
        methods_print = ['logreg', 'mlp']

    pol_head_in_results = any(
        'pol_head_val' in all_results.get(m, {}) for m in args.models)
    fused_in_results = any(
        'fused_val' in all_results.get(m, {}) for m in args.models)

    print(f"\n{'='*95}")
    print(f"POLARITY PROBE SUMMARY  (probes={args.probes}"
          f"{', pol_head=on' if (want_pol_head and pol_head_in_results) else ''}"
          f"{', fused=on' if fused_in_results else ''})")
    print(f"{'='*95}")
    print(f"{'Model':<15} {'Method':<10} {'Split':<6} {'N':>7} {'Acc':>8} {'F1-macro':>10} {'F1-neg':>8} {'F1-pos':>8}")
    print("-" * 95)
    for mn in args.models:
        if mn not in all_results:
            continue
        for method in methods_print:
            for sn in ['val', 'test']:
                key = f'{method}_{sn}'
                r = all_results[mn].get(key, {})
                if not r:
                    continue
                print(f"{mn:<15} {method:<10} {sn:<6} {r['n']:>7} "
                      f"{r['accuracy']:>7.4f} {r.get('f1_macro',0):>9.4f} "
                      f"{r.get('f1_neg',0):>7.4f} {r.get('f1_pos',0):>7.4f}")
        if pol_head_in_results:
            for sn in ['val', 'test']:
                key = f'pol_head_{sn}'
                r = all_results[mn].get(key, {})
                if not r:
                    continue
                print(f"{mn:<15} {'pol_head':<10} {sn:<6} {r['n']:>7} "
                      f"{r['accuracy']:>7.4f} {r.get('f1_macro',0):>9.4f} "
                      f"{r.get('f1_neg',0):>7.4f} {r.get('f1_pos',0):>7.4f}")
        if fused_in_results:
            for sn in ['val', 'test']:
                key = f'fused_{sn}'
                r = all_results[mn].get(key, {})
                if not r:
                    continue
                print(f"{mn:<15} {'fused':<10} {sn:<6} {r['n']:>7} "
                      f"{r['accuracy']:>7.4f} {r.get('f1_macro',0):>9.4f} "
                      f"{r.get('f1_neg',0):>7.4f} {r.get('f1_pos',0):>7.4f}")
        print()

    ref_model = next(
        (m for m in args.models if m != 'rt_only_d' and m in all_results), None)
    head_vs_mlp = (
        ultra_pol_head_only and ref_model and ('rt_only_d' in all_results)
        and args.probes == 'mlp' and args.no_pol_head_fit)
    head_vs_same_pol = (
        ultra_pol_head_only and ref_model and ('rt_only_d' in all_results)
        and args.probes == 'none' and (not args.no_pol_head_fit))
    head_msnlib_both = (
        (args.protocol == 'app_spectraverse') and ref_model and ('rt_only_d' in all_results)
        and bool(getattr(args, 'fit_pol_head_msnlib_both', False))
        and args.probes == 'none')
    if ref_model and 'rt_only_d' in all_results:
        print(f"{'='*95}")
        print(f"DELTA: rt_only_d minus {ref_model} (percentage points, accuracy)")
        if head_vs_mlp:
            print("  PRIMARY: UltraMs PolarityHead vs DreaMS MLP (frozen encoder, same SV splits).")
        elif head_msnlib_both:
            print(
                "  PRIMARY: both PolarityHeads trained on MSnLib train; val/test on Spectraverse.")
        elif head_vs_same_pol:
            print(
                "  PRIMARY: same-arch PolarityHead — Ultra pretrained vs DreaMS PolHead-fit "
                "(val threshold both).")
        elif ultra_pol_head_only:
            print(
                f"  (spectraverse_train_probe: pol_head vs pol_head when PolHead-fit on; "
                f"else see per-method rows)")
        print(f"{'='*95}")
        if (head_vs_same_pol or head_msnlib_both) and pol_head_in_results:
            for split in ('val', 'test'):
                rd = all_results['rt_only_d'].get(f'pol_head_{split}', {})
                rm = all_results[ref_model].get(f'pol_head_{split}', {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  pol_head vs pol_head   {split:<5}:  {dpp:+.2f} pp")
        if head_vs_mlp and pol_head_in_results:
            for split in ('val', 'test'):
                rd = all_results['rt_only_d'].get(f'pol_head_{split}', {})
                rm = all_results[ref_model].get(f'mlp_{split}', {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  pol_head vs mlp       {split:<5}:  {dpp:+.2f} pp")
        for split in ('val', 'test'):
            for method in methods_print:
                kd = f'{method}_{split}'
                rd = all_results['rt_only_d'].get(kd, {})
                rm = all_results[ref_model].get(kd, {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  {method:<8} {split:<5}:  {dpp:+.2f} pp")
        if pol_head_in_results and (not head_vs_mlp) and (not head_vs_same_pol) and (not head_msnlib_both):
            print(
                f"  — PolarityHead vs PolarityHead (Ultra pretrained vs {ref_model} fitted, "
                f"same arch & thr protocol):")
            for split in ('val', 'test'):
                rd = all_results['rt_only_d'].get(f'pol_head_{split}', {})
                rm = all_results[ref_model].get(f'pol_head_{split}', {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  pol_head vs pol_head  {split:<5}:  {dpp:+.2f} pp")
            print(f"  — (secondary) Ultra pol_head vs {ref_model} LogReg:")
            for split in ('val', 'test'):
                rd = all_results['rt_only_d'].get(f'pol_head_{split}', {})
                rm = all_results[ref_model].get(f'logreg_{split}', {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  pol_head vs {ref_model}_logreg {split:<5}:  {dpp:+.2f} pp")
        if fused_in_results:
            print(f"  — Fused (pol+logreg, val-tuned w) vs {ref_model} logreg:")
            for split in ('val', 'test'):
                rd = all_results['rt_only_d'].get(f'fused_{split}', {})
                rm = all_results[ref_model].get(f'logreg_{split}', {})
                if rd.get('accuracy') is not None and rm.get('accuracy') is not None:
                    dpp = 100.0 * (float(rd['accuracy']) - float(rm['accuracy']))
                    print(f"  fused    vs {ref_model}_logreg {split:<5}:  {dpp:+.2f} pp")

    if args.fit_pol_head_msnlib_both:
        out = './output/comparison/polarity_probe_app_spectraverse_trainable_polhead.json'
    else:
        out = f'./output/comparison/polarity_probe_{args.protocol}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _methods = {
        'none': 'No embedding probes; PolarityHead rows only (unless --no-pol-head)',
        'logreg': 'LogisticRegressionCV(cv=3) on frozen embeddings (primary default)',
        'mlp': (
            'Deep MLP (GELU 512→256, dropout, cosine LR, early stop, val logit threshold)'
        ),
        'both': 'LogReg + MLP as above',
    }
    if args.protocol == 'in_domain_msnlib':
        _src_story = 'All splits: MSnLib (in-domain linear probe).'
    elif args.protocol == 'spectraverse_train_probe':
        if getattr(args, 'ultra_pol_vs_dreams_polhead', False):
            _src_story = (
                'Spectraverse train/val/test. rt_only_d: pretrained PolarityHead. DreaMS: PolHead-fit '
                '(same Phase2 architecture on frozen embeddings; val threshold). No LogReg/MLP.'
            )
        elif args.no_pol_head_fit and args.probes == 'mlp':
            _src_story = (
                'Spectraverse train/val/test. rt_only_d: pretrained PolarityHead only. '
                'Baselines: MLP on frozen embeddings only (no LogReg / no PolHead-fit).'
            )
        else:
            _src_story = (
                'All splits: Spectraverse. Non-rt_only_d: LogReg/MLP and optional PolHead-fit on SV train; '
                'rt_only_d: pretrained PolarityHead only when probe skips embedding classifiers.'
            )
    else:
        if getattr(args, 'fit_pol_head_msnlib_both', False):
            _src_story = (
                'MSnLib train: Ultra PolarityHead on frozen Phase2 CLS '
                + ('(random init; --msnlib-ultra-pol-scratch). '
                   if getattr(args, 'msnlib_ultra_pol_scratch', False) else
                   '(Stage-D weights then fine-tune). ')
                + 'DreaMS: random-init PolarityHead on frozen emb. Spectraverse val/test.'
            )
        else:
            _src_story = (
                'Train probe: MSnLib train only; val/test: Spectraverse (application-domain OOD). '
                'Motivation: deploy polarity scoring trained on one curated library on a public benchmark.'
            )
    _train_sel = (
        f'{TRAIN_PER_CLASS} per class (Spectraverse train)'
        if args.protocol == 'spectraverse_train_probe'
        else f'{TRAIN_PER_CLASS} per class (MSnLib train)')
    _val_note = (
        'Spectraverse val'
        if args.protocol in ('app_spectraverse', 'spectraverse_train_probe')
        else 'MSnLib val')
    _test_note = (
        'Spectraverse test'
        if args.protocol in ('app_spectraverse', 'spectraverse_train_probe')
        else 'MSnLib test')
    meta = {
        'description': 'Polarity probe; default models dreams vs rt_only_d, logreg; protocol defines splits',
        'protocol': args.protocol,
        'protocol_note': _src_story,
        'probes': args.probes,
        'method': _methods[args.probes],
        'selection': {
            'adducts': 'canonical POSITIVE_ADDUCTS + NEGATIVE_ADDUCTS only',
            'train': _train_sel,
            'val': f'{VAL_PER_CLASS} per class (balanced, {_val_note})',
            'test': f'{TEST_PER_CLASS_PER_SOURCE} per class (balanced, {_test_note})',
        },
        'rt_only_d_probe_policy': (
            'pol_head_only_skips_train_logreg'
            if ultra_pol_head_only else 'same_as_other_models'),
        'batch_sizes': {
            'encode': int(args.encode_batch_size),
            'probe_train': int(args.probe_batch_size),
            'probe_forward': int(args.probe_forward_batch_size),
        },
        'ultra_pol_vs_dreams_mlp': bool(getattr(args, 'ultra_pol_vs_dreams_mlp', False)),
        'ultra_pol_vs_dreams_polhead': bool(getattr(args, 'ultra_pol_vs_dreams_polhead', False)),
        'fit_pol_head_msnlib_both': bool(getattr(args, 'fit_pol_head_msnlib_both', False)),
        'msnlib_ultra_pol_scratch': bool(getattr(args, 'msnlib_ultra_pol_scratch', False)),
        'no_pol_head_fit': bool(args.no_pol_head_fit),
        'models': list(args.models),
        'rng_seed': int(args.seed),
        'repro_note': 'Balanced subsamples use np.random.RandomState(seed); CSV row order must be unchanged.',
        'pol_head_eval': bool(want_pol_head),
        'pol_head_note': (
            (
                'app_spectraverse + --fit-pol-head-msnlib-both: Ultra = PolHead on frozen CLS, MSnLib train '
                '(Stage-D init by default; --msnlib-ultra-pol-scratch for random init). '
                'DreaMS = random-init PolHead, MSnLib train. SV eval.'
            )
            if getattr(args, 'fit_pol_head_msnlib_both', False) else (
                'rt_only_d: Phase2 PolarityHead (pretrained), val-tuned threshold. '
                'Other models: same PolarityHead architecture trained on benchmark train, val threshold. '
                'Primary fair compare: pol_head row vs pol_head row.'
                + (
                    ' spectraverse_train_probe: pol_head vs pol_head vs logreg secondary.'
                    if ultra_pol_head_only else '')
            )
        ) if want_pol_head else None,
        'fused_pol_logreg': bool(fused_in_results),
        'fused_note': (
            'rt_only_d: z-score pol_head logit + LogReg decision_function (train marginals); '
            'mix weight w and threshold chosen on val grid; no backbone/head retraining.'
        ) if fused_in_results else None,
    }
    with open(out, 'w') as f:
        json.dump({'meta': meta, 'results': all_results}, f, indent=2)
    print(f"Results saved to: {out}")


if __name__ == '__main__':
    main()
