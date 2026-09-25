"""Spectrum-to-molecule candidate ranking with residual projection.

The frozen test-query JSONL writer preserves the original experiment's
per-spectrum candidate order and scores.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

_COMPARISON_DIR = os.path.dirname(os.path.abspath(__file__))
_LIGHT_ULTRA_DIR = os.environ.get('LIGHT_ULTRA_ROOT', os.path.dirname(os.path.dirname(_COMPARISON_DIR)))
_TRAIN_DIR = os.path.join(_LIGHT_ULTRA_DIR, 'train')
_PUBLIC_MODEL_ROOT = os.path.abspath(os.path.join(_COMPARISON_DIR, '..', '..', '..', '..', 'training', 'ultrams_training', 'model'))
sys.path.insert(0, os.path.join(_PUBLIC_MODEL_ROOT, 'train'))
sys.path.insert(0, _PUBLIC_MODEL_ROOT)

CHEMBERTA_PATH = os.path.join(_LIGHT_ULTRA_DIR, 'model/feature/ChemBERTa-100M-MLM')

DATASET_CONFIGS = {
    'masspecgym': {
        'csv': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/MassSpecGym/MassSpecGym.csv'),
        'candidates_formula': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/MassSpecGym/MassSpecGym_candidates_formula.json'),
        'candidates_mass': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/MassSpecGym/MassSpecGym_candidates_mass.json'),
        'tasks': ['formula', 'mass'],
    },
    'msnlib': {
        'csv': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/MSnLib/MSnLib.csv'),
        'candidates_mass': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/MSnLib/MSnLib_candidates.json'),
        'tasks': ['mass'],
    },
    'spectraverse': {
        'csv': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/Spectraverse/Spectraverse.csv'),
        'candidates_mass': os.path.join(_LIGHT_ULTRA_DIR, 'datasets/Spectraverse/Spectraverse_candidates.json'),
        'tasks': ['mass'],
    },
}


# ── ChemBERTa molecule encoder ─────────────────────────────

class MolEncoder(nn.Module):
    def __init__(self, pretrained_path=CHEMBERTA_PATH):
        super().__init__()
        from transformers import RobertaModel, AutoTokenizer
        self.bert = RobertaModel.from_pretrained(pretrained_path, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.hidden_dim = self.bert.config.hidden_size

    @torch.no_grad()
    def encode_batch(self, smiles_list, device):
        encoded = self.tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        ).to(device)
        out = self.bert(**encoded, return_dict=True)
        return out.last_hidden_state[:, 0, :]


MOL_EMB_CACHE_DIR = os.path.join(_TRAIN_DIR, 'output/comparison/mol_emb_cache')


@torch.no_grad()
def precompute_mol_embeddings(mol_encoder, smiles_list, device, batch_size=256,
                              cache_tag=None):
    """Encode SMILES with ChemBERTa; results are cached to disk per cache_tag."""
    if cache_tag:
        os.makedirs(MOL_EMB_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(MOL_EMB_CACHE_DIR, f'{cache_tag}.pt')
        if os.path.exists(cache_path):
            print(f"    Loading cached mol embeddings: {cache_path}")
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            hit = {sm: cached[sm] for sm in smiles_list if sm in cached}
            miss = [sm for sm in smiles_list if sm not in cached]
            print(f"    Cache hit: {len(hit)}, miss: {len(miss)}")
            if not miss:
                return hit
            new_emb = _encode_smiles(mol_encoder, miss, device, batch_size)
            hit.update(new_emb)
            merged = {**cached, **new_emb}
            torch.save(merged, cache_path)
            print(f"    Updated cache: {len(merged)} total")
            return hit

    embeddings = _encode_smiles(mol_encoder, smiles_list, device, batch_size)

    if cache_tag:
        os.makedirs(MOL_EMB_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(MOL_EMB_CACHE_DIR, f'{cache_tag}.pt')
        torch.save(embeddings, cache_path)
        print(f"    Saved mol embeddings to: {cache_path} ({len(embeddings)} entries)")

    return embeddings


@torch.no_grad()
def _encode_smiles(mol_encoder, smiles_list, device, batch_size=256):
    mol_encoder.eval()
    embeddings = {}
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        emb = mol_encoder.encode_batch(batch, device)
        for j, sm in enumerate(batch):
            embeddings[sm] = emb[j].cpu()
        if (i // batch_size) % 50 == 0:
            print(f"    mol embed: {i}/{len(smiles_list)}", flush=True)
    return embeddings


# ── Data loading ───────────────────────────────────────────

def load_dataset(name):
    import pandas as pd
    cfg = DATASET_CONFIGS[name]
    df = pd.read_csv(cfg['csv'])

    if 'fold' not in df.columns:
        df['fold'] = 'test'

    samples, folds = [], []
    optional_cols = [
        'adduct', 'precursor_type', 'ionmode', 'collision_energy',
        'instrument_type', 'spectrum_id', 'identifier'
    ]

    def _clean_meta(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    for i, row in df.iterrows():
        mzs = np.array([float(x) for x in str(row['mzs']).split(',')], dtype=np.float32)
        ints = np.array([float(x) for x in str(row['intensities']).split(',')], dtype=np.float32)
        spectrum = np.stack([mzs, ints], axis=-1)
        sm = str(row.get('smiles', ''))
        sample = {
            'sample_idx': int(i),
            'spectrum': spectrum,
            'precursor_mz': float(row['precursor_mz']),
            'smiles': sm if sm != 'nan' else '',
            'fold': row.get('fold', 'test'),
        }
        for col in optional_cols:
            if col in df.columns:
                sample[col] = _clean_meta(row.get(col))
        samples.append(sample)
        folds.append(row.get('fold', 'test'))

    folds = np.array(folds)
    train_idx = np.where(folds == 'train')[0]
    val_idx = np.where(folds == 'val')[0]
    test_idx = np.where(folds == 'test')[0]
    print(f"[{name}] {len(samples)} total (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
    return samples, train_idx, val_idx, test_idx, cfg


# ── Spectrum preprocessing ─────────────────────────────────

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


# ── Spectrum embedding extraction ──────────────────────────

@torch.no_grad()
def extract_v9(model, samples, device, bs=64):
    model.eval()
    mp = model.max_peaks
    all_emb = []
    for s in range(0, len(samples), bs):
        batch = samples[s:s + bs]
        B = len(batch)
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
        all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def _load_dreams_loader():
    import importlib.util
    _comp = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(_comp, 'dreams_loader.py')
    spec = importlib.util.spec_from_file_location('dreams_loader', src)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('dreams_loader', mod)
    spec.loader.exec_module(mod)
    return mod

@torch.no_grad()
def extract_dreams(model, samples, device, bs=128):
    dl = _load_dreams_loader()
    return dl.extract_embeddings(model, samples, device, batch_size=bs)


# ── Model loading ──────────────────────────────────────────

PHASE2_CKPTS = {
    'phase2': (os.path.join(_TRAIN_DIR, 'output/phase2_vicreg_rt/stage_c_step_68000.pt'), 'base_state_dict'),
    'phase2_ep1': (os.path.join(_TRAIN_DIR, 'output/phase2_vicreg_rt/stage_c_step_138000.pt'), 'base_state_dict'),
    'stage_a': (os.path.join(_TRAIN_DIR, 'output/phase2_vicreg_rt/stage_a_epoch_5.pt'), 'base_state_dict'),
    'stage_d': (os.path.join(_TRAIN_DIR, 'output/phase2_staged/stage_d_step_158000.pt'), 'base_state_dict'),
    'stage_ae': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_ae_epoch_5.pt'), 'base_state_dict'),
    'rt_only_c': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_c_epoch_1.pt'), 'base_state_dict'),
    'rt_only_c2': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_c_epoch_2.pt'), 'base_state_dict'),
    'rt_only_d': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_d_epoch_1.pt'), 'base_state_dict'),
    'rt_only_d11': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_d_epoch_11.pt'), 'base_state_dict'),
    'stage_o': (os.path.join(_TRAIN_DIR, 'output/phase2_rt_only/stage_o_epoch_3.pt'), 'base_state_dict'),
}

def load_model(name, device):
    if name in ('v9', 'phase2', 'phase2_ep1', 'stage_a', 'stage_d', 'stage_ae', 'rt_only_c', 'rt_only_c2', 'rt_only_d', 'rt_only_d11', 'stage_o'):
        from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG
        model = UltraExplorerMLM(CONFIG)
        if name in PHASE2_CKPTS:
            path, key = PHASE2_CKPTS[name]
            ckpt = torch.load(path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
            if missing or unexpected:
                print(f"  [{name}] state_dict: missing={missing}, unexpected={unexpected}")
            print(f"  [{name}] ep={ckpt.get('epoch','?')}, step={ckpt.get('global_step','?')}")
        else:
            ckpt = torch.load(os.path.join(_TRAIN_DIR, 'output/ue_multiscale_v9/checkpoint_epoch_3.pt'),
                              map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  [v9] ep={ckpt.get('epoch', '?')}")
        return model.to(device).eval(), 1024
    else:
        dl = _load_dreams_loader()
        model = dl.load_dreams_encoder(device)
        return model, 1024

EXTRACTORS = {'v9': extract_v9, 'dreams': extract_dreams, 'phase2': extract_v9, 'phase2_ep1': extract_v9, 'stage_a': extract_v9, 'stage_d': extract_v9, 'stage_ae': extract_v9, 'rt_only_c': extract_v9, 'rt_only_c2': extract_v9, 'rt_only_d': extract_v9, 'rt_only_d11': extract_v9, 'stage_o': extract_v9}


# ── Deep residual projection ───────────────────────────────

class ResidualProjection(nn.Module):
    """Spectrum → molecule space with residual connection.

    Main path:  in_dim → in_dim (GELU + LN + Dropout) → out_dim
    Skip path:  in_dim → out_dim  (linear, no bias)
    Output = main + skip   (L2-normalize applied externally during training/eval)

    Residual keeps the projection well-conditioned even though molecule
    embeddings are direct ChemBERTa outputs with no paired projection.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.main(x) + self.skip(x)


# ── Projection training ────────────────────────────────────

def train_projection(spec_emb, mol_emb_dict, samples, train_idx, spec_dim, mol_dim,
                     device, epochs=100, lr=5e-4, batch_size=512, seed=42,
                     temperature=0.07, val_idx=None):
    """Train ResidualProjection with optional val-based early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    valid_pairs = [(i, samples[i]['smiles']) for i in train_idx
                   if samples[i]['smiles'] and samples[i]['smiles'] != 'nan'
                   and samples[i]['smiles'] in mol_emb_dict]

    if len(valid_pairs) < 100:
        print(f"    Only {len(valid_pairs)} valid pairs, using untrained projection")
        proj = ResidualProjection(spec_dim, mol_dim).to(device)
        proj.eval()
        return proj

    print(f"    Training projection on {len(valid_pairs)} pairs "
          f"({spec_dim}→{mol_dim}, ResidualProj, T={temperature:.3f})")

    proj = ResidualProjection(spec_dim, mol_dim).to(device)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    spec_arr = np.array([spec_emb[i] for i, _ in valid_pairs])
    mol_arr = np.array([mol_emb_dict[sm].numpy() for _, sm in valid_pairs])
    spec_t = torch.from_numpy(spec_arr).float().to(device)
    mol_t = F.normalize(torch.from_numpy(mol_arr).float().to(device), dim=-1)

    val_spec_t, val_mol_t = None, None
    if val_idx is not None:
        vp = [(i, samples[i]['smiles']) for i in val_idx
              if samples[i]['smiles'] and samples[i]['smiles'] != 'nan'
              and samples[i]['smiles'] in mol_emb_dict]
        if len(vp) > 2048:
            rng = np.random.RandomState(seed)
            vp = [vp[j] for j in rng.choice(len(vp), 2048, replace=False)]
        val_spec_t = torch.from_numpy(np.array([spec_emb[i] for i, _ in vp])).float().to(device)
        val_mol_t = F.normalize(torch.from_numpy(
            np.array([mol_emb_dict[sm].numpy() for _, sm in vp])).float().to(device), dim=-1)

    n = len(valid_pairs)
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        proj.train()
        perm = torch.randperm(n)
        total_loss = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            s_proj = F.normalize(proj(spec_t[idx]), dim=-1)
            logits = s_proj @ mol_t[idx].T / temperature
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        scheduler.step()
        train_loss = total_loss / n

        if val_spec_t is not None and (epoch + 1) % 1 == 0:
            proj.eval()
            with torch.no_grad():
                vp = F.normalize(proj(val_spec_t), dim=-1)
                vl = vp @ val_mol_t.T / temperature
                val_loss = F.cross_entropy(vl, torch.arange(len(val_spec_t), device=device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in proj.state_dict().items()}
                best_epoch = epoch + 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}"
                      f"  lr={scheduler.get_last_lr()[0]:.2e}"
                      f"{'  *best' if epoch + 1 == best_epoch else ''}")
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      epoch {epoch+1}: loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state is not None:
        proj.load_state_dict(best_state)
        print(f"      → restored best epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    proj.eval()
    return proj


# ── Retrieval evaluation ───────────────────────────────────

CAND_BINS = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, float('inf'))]

N_RAW_RANKS = 20   # how many rank positions to record for similarity curves

def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float):
        return None if np.isnan(v) else v
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), separators=(',', ':')) + '\n')


def evaluate_retrieval(spec_emb, proj, mol_emb_dict, samples, test_idx,
                       candidates, device, top_k=(1, 5, 10, 20), stratified=False,
                       collect_raw=False, collect_details=False, detail_topk=50):
    """Rank candidates per query spectrum using projected cosine similarity."""
    ranks, n_cands_list = [], []
    gt_sims, top1_sims, topn_sims = [], [], []
    query_indices, details = [], []
    skipped = 0

    for qi in test_idx:
        sm = samples[qi]['smiles']
        if not sm or sm == 'nan' or sm not in candidates:
            skipped += 1
            continue
        cand_list = list(candidates[sm])
        if sm not in cand_list:
            cand_list.append(sm)

        available_cands = [c for c in cand_list if c in mol_emb_dict]
        if sm not in available_cands:
            skipped += 1
            continue

        q_emb = torch.from_numpy(spec_emb[qi:qi+1]).float().to(device)
        if proj is not None:
            q_proj = F.normalize(proj(q_emb), dim=-1)
        else:
            q_proj = F.normalize(q_emb, dim=-1)

        c_embs = torch.stack([mol_emb_dict[c] for c in available_cands]).to(device)
        c_embs = F.normalize(c_embs, dim=-1)

        sims = (q_proj @ c_embs.T).squeeze(0)
        sorted_idx = torch.argsort(sims, descending=True)

        gt_pos = available_cands.index(sm)
        rank = (sorted_idx == gt_pos).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        n_cands_list.append(len(available_cands))
        query_indices.append(int(qi))

        if collect_raw or collect_details:
            sims_np = sims.detach().cpu().float().numpy()
            sorted_np = sorted_idx.detach().cpu().numpy()
            sorted_sims = sims_np[sorted_np]

        if collect_raw:
            gt_sims.append(float(sims_np[gt_pos]))
            top1_sims.append(float(sorted_sims[0]))
            n_take = min(N_RAW_RANKS, len(sorted_sims))
            row = np.full(N_RAW_RANKS, np.nan, dtype=np.float32)
            row[:n_take] = sorted_sims[:n_take]
            topn_sims.append(row)

        if collect_details:
            k = min(detail_topk, len(sorted_np))
            top_idx = sorted_np[:k]
            top_candidates = [available_cands[int(j)] for j in top_idx]
            top_scores = [float(x) for x in sorted_sims[:k]]
            sample = samples[qi]
            details.append({
                'sample_idx': int(qi),
                'fold': sample.get('fold'),
                'smiles': sm,
                'adduct': sample.get('adduct', sample.get('precursor_type')),
                'precursor_type': sample.get('precursor_type'),
                'ionmode': sample.get('ionmode'),
                'precursor_mz': sample.get('precursor_mz'),
                'collision_energy': sample.get('collision_energy'),
                'instrument_type': sample.get('instrument_type'),
                'spectrum_id': sample.get('spectrum_id', sample.get('identifier')),
                'rank': int(rank),
                'n_candidates': int(len(available_cands)),
                'gt_score': float(sims_np[gt_pos]),
                'top1_smiles': top_candidates[0] if top_candidates else None,
                'top1_score': top_scores[0] if top_scores else None,
                'top_candidates': top_candidates,
                'top_scores': top_scores,
            })

    if len(ranks) < 10:
        print(f"    Only {len(ranks)} evaluable (skipped {skipped})")
        return {}

    ranks = np.array(ranks)
    n_cands_arr = np.array(n_cands_list)
    results = {'n_queries': len(ranks), 'avg_candidates': float(np.mean(n_cands_arr))}
    for k in top_k:
        results[f'top{k}'] = float((ranks <= k).mean() * 100)
    results['mrr'] = float((1.0 / ranks).mean())
    results['median_rank'] = int(np.median(ranks))

    print(f"    n={len(ranks)}, avg_cands={np.mean(n_cands_arr):.0f}: "
          f"Top1={results['top1']:.2f}%  Top5={results['top5']:.2f}%  "
          f"Top10={results['top10']:.2f}%  Top20={results['top20']:.2f}%  "
          f"MRR={results['mrr']:.4f}")

    if stratified:
        results['stratified'] = {}
        for lo, hi in CAND_BINS:
            mask = (n_cands_arr >= lo) & (n_cands_arr < hi)
            cnt = mask.sum()
            if cnt < 5:
                continue
            bin_ranks = ranks[mask]
            label = f"{lo}-{int(hi) if hi != float('inf') else 'inf'}"
            bin_r = {'n': int(cnt), 'avg_cands': float(n_cands_arr[mask].mean())}
            for k in top_k:
                bin_r[f'top{k}'] = float((bin_ranks <= k).mean() * 100)
            bin_r['mrr'] = float((1.0 / bin_ranks).mean())
            results['stratified'][label] = bin_r
            print(f"      [{label}] n={cnt}, top1={bin_r['top1']:.2f}%  mrr={bin_r['mrr']:.4f}")

    if collect_raw:
        results['_raw'] = {
            'gt_sims':   np.array(gt_sims,   dtype=np.float32),
            'top1_sims': np.array(top1_sims, dtype=np.float32),
            'topn_sims': np.array(topn_sims, dtype=np.float32),  # (n, N_RAW_RANKS)
            'ranks':     ranks,
            'query_indices': np.array(query_indices, dtype=np.int64),
        }

    if collect_details:
        results['_details'] = details

    return results


# ── Main ───────────────────────────────────────────────────

TEMP_GRID = [0.03, 0.05, 0.07, 0.1, 0.15]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['rt_only_d11', 'dreams'],
                        choices=['v9', 'dreams', 'phase2', 'phase2_ep1', 'stage_a', 'stage_d',
                                 'stage_ae', 'rt_only_c', 'rt_only_c2', 'rt_only_d', 'rt_only_d11', 'stage_o'])
    parser.add_argument('--datasets', nargs='+', default=['msnlib'],
                        choices=['masspecgym', 'msnlib', 'spectraverse'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--proj-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi-seed', type=int, default=0)
    parser.add_argument('--seed-start', type=int, default=1,
                        help='Start seed for multi-seed (e.g. 6 to run seeds 6..N)')
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--early-stop', action='store_true',
                        help='Val-based early stopping for projection')
    parser.add_argument('--save-raw-scores', action='store_true',
                        help='Save per-query similarity scores (gt_sim, topN_sims) to npz')
    parser.add_argument('--save-per-spectrum', action='store_true',
                        help='Save per-spectrum retrieval details to JSONL')
    parser.add_argument('--per-spectrum-topk', type=int, default=50,
                        help='Number of top candidates/scores to save per spectrum')
    parser.add_argument('--save-proj', action='store_true',
                        help='Save trained projection weights to disk')
    parser.add_argument('--load-proj', action='store_true',
                        help='Load projection from disk (skip training)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    seeds = list(range(args.seed_start, args.multi_seed + 1)) if args.multi_seed > 0 else [args.seed]
    temps = [args.temperature]

    mol_encoder = None  # lazy-loaded only when cache misses occur

    all_results = {}
    multi_seed_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ds_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        samples, train_idx, val_idx, test_idx, cfg = load_dataset(ds_name)

        for task in cfg['tasks']:
            cands_key = f'candidates_{task}'
            cands_path = cfg.get(cands_key)
            if not cands_path or not os.path.exists(cands_path):
                print(f"  Skipping task={task}: candidates not found")
                continue

            print(f"\n  Task: {task}")
            print(f"  Loading candidates from {cands_path}...")
            with open(cands_path) as f:
                candidates = json.load(f)
            print(f"  {len(candidates)} queries in candidate file")

            eval_smiles = set()
            for idx_set in (val_idx, test_idx):
                for i in idx_set:
                    sm = samples[i]['smiles']
                    if sm and sm != 'nan' and sm in candidates:
                        eval_smiles.update(candidates[sm])
                        eval_smiles.add(sm)

            train_gt_smiles = set()
            for i in train_idx:
                sm = samples[i]['smiles']
                if sm and sm != 'nan':
                    train_gt_smiles.add(sm)

            all_smiles = sorted(eval_smiles | train_gt_smiles)
            print(f"  {len(eval_smiles)} eval candidate SMILES + "
                  f"{len(train_gt_smiles)} train GT SMILES = "
                  f"{len(all_smiles)} unique total")

            cache_tag = f"{ds_name}_{task}_chemberta_full"
            print(f"  Encoding molecules with ChemBERTa (cache: {cache_tag})...")
            # lazy-load ChemBERTa only on cache miss
            cache_path = os.path.join(MOL_EMB_CACHE_DIR, f'{cache_tag}.pt')
            if not os.path.exists(cache_path):
                if mol_encoder is None:
                    print("  Loading ChemBERTa molecule encoder (cache miss)...")
                    mol_encoder = MolEncoder(CHEMBERTA_PATH).to(device).eval()
            else:
                cached_keys = set(torch.load(cache_path, map_location='cpu', weights_only=True).keys())
                if not all(sm in cached_keys for sm in all_smiles):
                    if mol_encoder is None:
                        print("  Loading ChemBERTa molecule encoder (partial cache miss)...")
                        mol_encoder = MolEncoder(CHEMBERTA_PATH).to(device).eval()
            mol_emb_dict = precompute_mol_embeddings(
                mol_encoder, all_smiles, device, cache_tag=cache_tag
            )
            mol_dim = 768  # ChemBERTa-100M hidden dim
            if mol_encoder is not None:
                mol_dim = mol_encoder.hidden_dim

            task_key = f"{ds_name}_{task}"
            all_results[task_key] = {}

            emb_cache = {}
            for mn in args.models:
                print(f"\n  --- Model: {mn} ---")
                model, spec_dim = load_model(mn, device)
                print(f"  Extracting spectrum embeddings...")
                spec_emb = EXTRACTORS[mn](model, samples, device)
                del model; torch.cuda.empty_cache()
                print(f"  Embeddings: {spec_emb.shape}")
                emb_cache[mn] = (spec_emb, spec_dim)

            for mn in args.models:
                spec_emb, spec_dim = emb_cache[mn]
                best_temp = args.temperature

                for sd in seeds:
                    if len(seeds) > 1:
                        print(f"\n  ── Seed {sd} ──")
                    proj_ckpt_path = os.path.join(
                        _TRAIN_DIR,
                        f'output/comparison/proj_{ds_name}_{task}_{mn}_seed{sd}.pt'
                    )
                    if args.load_proj and os.path.exists(proj_ckpt_path):
                        print(f"\n  [{mn}] Loading projection from {proj_ckpt_path} ...")
                        proj = ResidualProjection(spec_dim, mol_dim).to(device)
                        proj.load_state_dict(torch.load(proj_ckpt_path, map_location=device,
                                                         weights_only=True))
                        proj.eval()
                    else:
                        print(f"\n  [{mn}] Training (T={best_temp:.3f}, seed={sd})...")
                        proj = train_projection(
                            spec_emb, mol_emb_dict, samples, train_idx,
                            spec_dim, mol_dim, device, epochs=args.proj_epochs,
                            seed=sd, temperature=best_temp,
                            val_idx=val_idx if args.early_stop else None
                        )
                        if args.save_proj:
                            os.makedirs(os.path.dirname(proj_ckpt_path), exist_ok=True)
                            torch.save(proj.state_dict(), proj_ckpt_path)
                            print(f"  [{mn}] Projection saved to {proj_ckpt_path}")

                    model_results = {}
                    for split_name, split_idx in [('val', val_idx), ('test', test_idx)]:
                        print(f"  [{mn}] Evaluating {split_name} ({task}):")
                        r = evaluate_retrieval(
                            spec_emb, proj, mol_emb_dict, samples, split_idx,
                            candidates, device, stratified=True,
                            collect_raw=args.save_raw_scores,
                            collect_details=args.save_per_spectrum,
                            detail_topk=args.per_spectrum_topk,
                        )
                        r['temperature'] = best_temp

                        if args.save_raw_scores and '_raw' in r:
                            raw = r.pop('_raw')
                            raw_out = os.path.join(
                                _TRAIN_DIR,
                                f'output/comparison/10_raw_scores_{ds_name}_{task}_{mn}_{split_name}.npz'
                            )
                            np.savez_compressed(raw_out, **raw)
                            print(f"    Raw scores saved: {raw_out}")

                        if args.save_per_spectrum and '_details' in r:
                            details = r.pop('_details')
                            details_out = os.path.join(
                                _TRAIN_DIR,
                                f'output/comparison/10_per_spectrum_{ds_name}_{task}_{mn}_{split_name}.jsonl'
                            )
                            write_jsonl(details_out, details)
                            print(f"    Per-spectrum details saved: {details_out}")

                        model_results[split_name] = r
                        if len(seeds) > 1:
                            multi_seed_results[f"{task_key}_{split_name}"][mn][sd] = r

                    if len(seeds) == 1:
                        all_results[task_key][mn] = model_results

                    del proj; torch.cuda.empty_cache()

            for mn in args.models:
                del emb_cache[mn]
            del emb_cache

    # Summary
    if len(seeds) == 1:
        print(f"\n{'='*100}")
        print("CROSS-MODAL RETRIEVAL SUMMARY (ResidualProjection + Enhanced)")
        print(f"{'='*100}")
        print(f"{'Task':<25} {'Model':<14} {'T':>5} {'Split':<6} "
              f"{'Top1':>8} {'Top5':>8} {'Top10':>8} {'Top20':>8} {'MRR':>8}")
        print("-" * 100)
        for task_key, models in all_results.items():
            for mn, splits in models.items():
                for split_name, r in splits.items():
                    if not r:
                        continue
                    print(f"{task_key:<25} {mn:<14} {r.get('temperature', 0.07):>5.3f} {split_name:<6} "
                          f"{r.get('top1', 0):>7.2f}% {r.get('top5', 0):>7.2f}% "
                          f"{r.get('top10', 0):>7.2f}% {r.get('top20', 0):>7.2f}% "
                          f"{r.get('mrr', 0):>8.4f}")
            print()

        print(f"\n{'='*100}")
        print("STRATIFIED ANALYSIS")
        print(f"{'='*100}")
        for task_key, models in all_results.items():
            for mn, splits in models.items():
                r = splits.get('test', {})
                if not r or 'stratified' not in r:
                    continue
                print(f"\n  {task_key} / {mn} (T={r.get('temperature', 0.07):.3f}):")
                for bin_label, br in r['stratified'].items():
                    print(f"    [{bin_label:>10}] n={br['n']:>5}  "
                          f"Top1={br['top1']:>6.2f}%  MRR={br['mrr']:.4f}")

        out = os.path.join(_TRAIN_DIR, 'output/comparison/10_plot_results.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {out}")
    else:
        print(f"\n{'='*100}")
        print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
        print(f"{'='*100}")
        print(f"{'Task+Split':<35} {'Model':<14} {'Top1 mean±std':>16} {'Top1 best':>10} {'best_seed':>10} {'MRR mean':>10}")
        print("-" * 100)
        for tk_split, models in sorted(multi_seed_results.items()):
            for mn in args.models:
                if mn not in models:
                    continue
                seed_data = models[mn]
                top1s = [seed_data[s].get('top1', 0) for s in seeds if s in seed_data]
                mrrs = [seed_data[s].get('mrr', 0) for s in seeds if s in seed_data]
                if not top1s:
                    continue
                best_idx = int(np.argmax(top1s))
                best_seed = [s for s in seeds if s in seed_data][best_idx]
                print(f"{tk_split:<35} {mn:<14} "
                      f"{np.mean(top1s):>6.2f}±{np.std(top1s):>5.2f}% "
                      f"{np.max(top1s):>8.2f}% "
                      f"{best_seed:>10} "
                      f"{np.mean(mrrs):>10.4f}")
            print()
        out = os.path.join(_TRAIN_DIR, 'output/comparison/10_plot_multi_seed.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        seed_export = {}
        for tk_split, models in multi_seed_results.items():
            seed_export[tk_split] = {}
            for mn, sd_data in models.items():
                seed_export[tk_split][mn] = {
                    str(s): r for s, r in sd_data.items()
                }
        with open(out, 'w') as f:
            json.dump(seed_export, f, indent=2)
        print(f"Results saved to: {out}")


if __name__ == '__main__':
    main()
