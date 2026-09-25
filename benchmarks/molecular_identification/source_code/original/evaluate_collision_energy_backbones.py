"""
Cross-CE per-sample retrieval for chemical subset analysis
==========================================================
输出两个 fine-tuned 模型对每个化合物的 per-sample rank，
以便按化学特性（脂肪族/高手性/柔性）划定子集后比较性能。

用法 (从 train/ 目录):
  nohup conda run -n msa python -u showcase/cross_ce_per_sample_chem.py \
    --device cuda:0 \
    > output/showcase/log_per_sample_chem.txt 2>&1 &
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

_HERE = Path(__file__).resolve().parent
_PUBLIC_ROOT = _HERE.parents[3]
_MODEL_ROOT = _PUBLIC_ROOT / 'training' / 'ultrams_training' / 'model'
_SUPPORT_ROOT = _PUBLIC_ROOT / 'benchmarks' / 'spectral_properties' / 'original' / 'support'
sys.path.insert(0, str(_MODEL_ROOT))
sys.path.insert(0, str(_MODEL_ROOT / 'train'))
sys.path.insert(0, str(_SUPPORT_ROOT))
sys.path.insert(0, str(_HERE))
os.environ.setdefault('ULTRAMS_SOURCE_TRAIN_ROOT', str(_MODEL_ROOT / 'train'))
_ASSET_ROOT = Path(os.environ.get('LIGHT_ULTRA_ROOT', Path.cwd())).resolve()

def load_ultra(device, checkpoint_path):
    """Load UltraExplorerMLM from base_state_dict (correct way per 06_cross_modal_retrieval.py)."""
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG as ULTRA_CONFIG
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', ULTRA_CONFIG)
    model = UltraExplorerMLM(config)
    if 'base_state_dict' in ckpt:
        model.load_state_dict(ckpt['base_state_dict'], strict=True)
        print(f"  [UltraMs] Loaded base_state_dict  ep={ckpt.get('epoch','?')}")
    else:
        sd = {k[5:]: v for k, v in ckpt['model_state_dict'].items() if k.startswith('base.')}
        model.load_state_dict(sd, strict=True)
        print(f"  [UltraMs] Loaded model_state_dict (base.*)  ep={ckpt.get('epoch','?')}")
    return model.to(device).eval(), config['d_model']

MSNLIB_CSV = str(_ASSET_ROOT / 'datasets' / 'MSnLib' / 'MSnLib.csv')
ULTRA_BACKBONE = str(_ASSET_ROOT / 'train' / 'output' / 'phase2_rt_only' / 'stage_d_epoch_11.pt')
CKPT_U = os.environ.get('ULTRAMS_CROSS_CE_ULTRA_CHECKPOINT', str(_ASSET_ROOT / 'train' / 'output' / 'showcase' / 'cross_ce_ultra_v2_ul2_best.pt'))
CKPT_D = os.environ.get('ULTRAMS_CROSS_CE_DREAMS_CHECKPOINT', str(_ASSET_ROOT / 'train' / 'output' / 'showcase' / 'cross_ce_dreams_v2_ul2_best.pt'))
OUT_DIR = os.environ.get('ULTRAMS_COLLISION_ENERGY_OUTPUT_DIR', str(_ASSET_ROOT / 'train' / 'output' / 'showcase'))
OUT_JSON = os.path.join(OUT_DIR, 'cross_ce_per_sample_chem.json')

LOW_CE_MAX  = 20
HIGH_CE_MIN = 40
MAX_PEAKS   = 150
D_PROJ      = 256


# ── 工具 ──────────────────────────────────────────────────────────────────
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
    return np.stack([mz, inten], axis=-1).astype(np.float32)


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def mol_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    n_arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_ster  = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mw      = Descriptors.MolWt(mol)
    atoms   = {a.GetSymbol() for a in mol.GetAtoms()}
    return dict(
        mw=round(mw, 2),
        n_arom_rings=n_arom,
        n_stereocentres=n_ster,
        n_rot_bonds=n_rot,
        aliphatic=(n_arom == 0),
        hi_stereo=(n_ster >= 4),
        flexible=(n_rot >= 10),
        has_N=('N' in atoms),
        has_S=('S' in atoms),
    )


def build_compound_data(csv_path, adduct='[M+H]+'):
    import pandas as pd
    print(f'Loading {adduct} ...', flush=True)
    df = pd.read_csv(csv_path, usecols=['mzs', 'intensities', 'smiles',
                                        'precursor_mz', 'adduct', 'collision_energy'])
    df = df[df['adduct'] == adduct].dropna(subset=['collision_energy']).copy()
    compound_data = defaultdict(lambda: {'low': [], 'high': []})
    skip = 0
    for i, row in enumerate(df.itertuples(index=False)):
        if i % 200_000 == 0 and i > 0:
            print(f'  {i:,}/{len(df):,}', flush=True)
        ce = float(row.collision_energy)
        if LOW_CE_MAX < ce < HIGH_CE_MIN:
            continue
        spec = parse_spectrum(row.mzs, row.intensities)
        if spec is None:
            skip += 1
            continue
        csmi = canonical_smiles(str(row.smiles))
        if csmi is None:
            skip += 1
            continue
        entry = {'spectrum': spec, 'precursor_mz': float(row.precursor_mz), 'ce': ce}
        compound_data[csmi]['low' if ce <= LOW_CE_MAX else 'high'].append(entry)
    valid = {s: v for s, v in compound_data.items() if v['low'] and v['high']}
    print(f'  Valid compounds: {len(valid):,}  skipped={skip:,}', flush=True)
    return valid


def split_nontraining(compound_data, val_r=0.15, test_r=0.15, seed=42):
    smis = sorted(compound_data.keys())
    rng = random.Random(seed)
    rng.shuffle(smis)
    n = len(smis)
    return set(smis[:int(n * (val_r + test_r))])


def build_pool(compound_data, nontr_smis, seed_offset=99):
    rng = random.Random(42 + seed_offset)
    queries, library = [], []
    for smi in sorted(nontr_smis):
        v = compound_data[smi]
        queries.append((smi, rng.choice(v['low'])))
        library.append((smi, rng.choice(v['high'])))
    return queries, library


class ProjectionHead(torch.nn.Module):
    def __init__(self, d_in, d_out=D_PROJ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d_in),
            torch.nn.Linear(d_in, d_in),
            torch.nn.GELU(),
            torch.nn.Linear(d_in, d_out),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


@torch.no_grad()
def encode_ultra(model, entries, device, batch_size=64):
    """Extract CLS+mean_pool embeddings (2048-dim) from UltraExplorerMLM."""
    model.eval()
    mp = model.max_peaks
    all_emb = []
    for s in range(0, len(entries), batch_size):
        batch = entries[s:s + batch_size]
        B = len(batch)
        padded = np.zeros((B, mp, 2), dtype=np.float32)
        attn   = np.zeros((B, mp),    dtype=np.int64)
        pmzs   = []
        for i, e in enumerate(batch):
            sp = np.asarray(e['spectrum'], dtype=np.float32)
            k  = min(len(sp), mp)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i, :k]   = 1
            pmzs.append(float(e['precursor_mz']))
        peaks_t = torch.from_numpy(padded).to(device)
        attn_t  = torch.from_numpy(attn).to(device)
        pmz_t   = torch.tensor(pmzs, dtype=torch.float32, device=device)

        ms_emb   = model.peak_encoder(peaks_t)
        cls_e    = model.cls_emb.expand(B, 1, -1)
        pi       = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], -1).unsqueeze(1)
        prec_emb = model.peak_encoder(pi) + model.precursor_type_emb
        full     = torch.cat([cls_e, prec_emb, ms_emb], 1)
        S        = full.shape[1]
        pos      = torch.arange(S, device=device).unsqueeze(0)
        full     = model.dropout(full + model.pos_emb(pos))
        fa       = torch.cat([torch.ones(B, 2, dtype=attn_t.dtype, device=device), attn_t], 1)
        hs       = model.encoder(inputs_embeds=full, attention_mask=fa).last_hidden_state
        cls      = hs[:, 0, :]
        sub_hs   = hs[:, 2:, :]
        attn_exp = attn_t.float().unsqueeze(-1)
        mean_pool = (sub_hs * attn_exp).sum(1) / attn_exp.sum(1).clamp(min=1)
        all_emb.append(torch.cat([cls, mean_pool], dim=-1).cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


def encode_dreams(enc, entries, device, batch_size=256):
    from dreams_loader import build_dreams_batch
    all_emb = []
    for i in range(0, len(entries), batch_size):
        batch = [{'spectrum': e['spectrum'], 'precursor_mz': e['precursor_mz']}
                 for e in entries[i:i + batch_size]]
        peaks_t = build_dreams_batch(enc, batch, device)
        with torch.no_grad():
            hs = enc(peaks_t)
        all_emb.append(hs[:, 0, :].cpu().float().numpy())
    return np.concatenate(all_emb, axis=0)


def compute_ranks(q_emb, l_emb, query_smis, lib_smis):
    """Returns (ranks, top1_smis): rank of GT and top-1 retrieved SMILES for each query."""
    lib_arr = np.array(lib_smis)
    q = torch.from_numpy(q_emb).float()
    l = torch.from_numpy(l_emb).float()
    sims = (q @ l.T).numpy()
    ranks, top1_smis = [], []
    for qi, qsmi in enumerate(query_smis):
        order = np.argsort(sims[qi])[::-1]
        top1_smis.append(lib_arr[order[0]])
        pos = np.where(lib_arr[order] == qsmi)[0]
        ranks.append(int(pos[0]) + 1 if len(pos) else len(lib_smis) + 1)
    return np.array(ranks), top1_smis


def r1_on_mask(ranks, mask):
    sub = ranks[mask]
    return float((sub <= 1).mean()) if len(sub) > 0 else 0.0


def chem_subset_analysis(records, label):
    """按化学属性划定子集并统计 R@1"""
    subsets = {
        'All':             lambda p: True,
        'Aliphatic':       lambda p: p.get('aliphatic', False),
        'Hi-Stereo (≥4)':  lambda p: p.get('hi_stereo', False),
        'Flexible (≥10)':  lambda p: p.get('flexible', False),
        'Aliphatic+HiStereo': lambda p: p.get('aliphatic') and p.get('hi_stereo'),
        'Polycyclic-Arom': lambda p: p.get('n_arom_rings', 0) >= 3,
        'Has-S':           lambda p: p.get('has_S', False),
    }
    print(f'\n=== Chemical Subset Analysis: {label} ===')
    print(f"{'Subset':<22} {'n':>5} {'R@1 Ultra':>10} {'R@1 DreaMS':>11} {'Δ (pp)':>8}")
    print('-' * 60)
    results = {}
    for name, fn in subsets.items():
        mask = [i for i, r in enumerate(records) if fn(r['props'])]
        if len(mask) < 10:
            continue
        ru = np.mean([records[i]['rank_ultra_ft'] <= 1 for i in mask])
        rd = np.mean([records[i]['rank_dream_ft'] <= 1 for i in mask])
        delta = (ru - rd) * 100
        print(f"{name:<22} {len(mask):>5} {ru:>10.3f} {rd:>11.3f} {delta:>+8.1f}pp")
        results[name] = {'n': len(mask), 'r1_ultra': ru, 'r1_dream': rd, 'delta_pp': delta}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default=OUT_JSON)
    args = ap.parse_args()
    device = torch.device(args.device)
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}

    # seeds aligned with cross_ce_finetune_v2.py
    for adduct, tag, split_seed, pool_seed in [
        ('[M+H]+', 'pos', 42,  99),
        ('[M-H]-', 'neg', 43, 199),
    ]:
        print(f'\n{"="*70}')
        print(f'Processing {adduct}  (split_seed={split_seed})')
        print('='*70)

        cdata = build_compound_data(MSNLIB_CSV, adduct)
        nontr = split_nontraining(cdata, seed=split_seed)
        queries, library = build_pool(cdata, nontr, seed_offset=pool_seed)
        q_smis = [s for s, _ in queries]
        l_smis = [s for s, _ in library]
        q_ent  = [e for _, e in queries]
        l_ent  = [e for _, e in library]
        N = len(q_smis)
        print(f'Pool size: {N:,}', flush=True)

        # ── DreaMS encoder ──────────────────────────────────────────────
        print('\nLoading DreaMS encoder...')
        from dreams_loader import load_dreams_encoder
        enc_d = load_dreams_encoder(device)
        enc_d.eval()
        d_in_d = enc_d.d_model

        print('Encoding with DreaMS zero-shot...')
        q_d0 = encode_dreams(enc_d, q_ent, device)
        l_d0 = encode_dreams(enc_d, l_ent, device)
        qn_d0 = F.normalize(torch.from_numpy(q_d0), dim=-1).numpy()
        ln_d0 = F.normalize(torch.from_numpy(l_d0), dim=-1).numpy()
        ranks_d_zs, _ = compute_ranks(qn_d0, ln_d0, q_smis, l_smis)
        print(f'DreaMS zero-shot R@1={( ranks_d_zs==1).mean():.3f}')

        proj_d = ProjectionHead(d_in_d, D_PROJ).to(device)
        ck_d = torch.load(CKPT_D, map_location='cpu', weights_only=False)
        proj_d.load_state_dict(ck_d['proj'])
        if 'enc' in ck_d:
            enc_d.load_state_dict(ck_d['enc'])
            enc_d.eval()
            print('Re-encoding DreaMS with fine-tuned backbone...')
            q_d0 = encode_dreams(enc_d, q_ent, device)
            l_d0 = encode_dreams(enc_d, l_ent, device)
        proj_d.eval()
        with torch.no_grad():
            q_d_ft = proj_d(torch.from_numpy(q_d0).float().to(device)).cpu().numpy()
            l_d_ft = proj_d(torch.from_numpy(l_d0).float().to(device)).cpu().numpy()
        ranks_d_ft, top1_smis_d = compute_ranks(q_d_ft, l_d_ft, q_smis, l_smis)
        print(f'DreaMS fine-tuned R@1={(ranks_d_ft==1).mean():.3f}')
        del enc_d, proj_d; torch.cuda.empty_cache()

        # ── UltraMs encoder ─────────────────────────────────────────────
        print('\nLoading UltraMs encoder...')
        enc_u, d_model_u = load_ultra(device, ULTRA_BACKBONE)
        d_in_u = d_model_u * 2   # cls + mean_pool

        print('Encoding with UltraMs...')
        q_u0 = encode_ultra(enc_u, q_ent, device)
        l_u0 = encode_ultra(enc_u, l_ent, device)

        proj_u = ProjectionHead(d_in_u, D_PROJ).to(device)
        ck_u = torch.load(CKPT_U, map_location='cpu', weights_only=False)
        proj_u.load_state_dict(ck_u['proj'])
        if 'enc' in ck_u:
            base_sd = {k[5:]: v for k, v in ck_u['enc'].items() if k.startswith('base.')}
            enc_u.load_state_dict(base_sd, strict=True)
            enc_u.eval()
            print('Re-encoding UltraMs with fine-tuned backbone...')
            q_u0 = encode_ultra(enc_u, q_ent, device)
            l_u0 = encode_ultra(enc_u, l_ent, device)
        proj_u.eval()
        with torch.no_grad():
            q_u_ft = proj_u(torch.from_numpy(q_u0).float().to(device)).cpu().numpy()
            l_u_ft = proj_u(torch.from_numpy(l_u0).float().to(device)).cpu().numpy()
        ranks_u_ft, top1_smis_u = compute_ranks(q_u_ft, l_u_ft, q_smis, l_smis)
        print(f'UltraMs fine-tuned R@1={(ranks_u_ft==1).mean():.3f}')
        del enc_u, proj_u; torch.cuda.empty_cache()

        # ── Build per-sample records with molecular props ────────────────
        print('\nComputing molecular properties...', flush=True)
        records = []
        for i, smi in enumerate(q_smis):
            records.append({
                'smiles':           smi,
                'rank_dream_zs':    int(ranks_d_zs[i]),
                'rank_dream_ft':    int(ranks_d_ft[i]),
                'rank_ultra_ft':    int(ranks_u_ft[i]),
                'top1_smi_dream':   top1_smis_d[i],
                'top1_smi_ultra':   top1_smis_u[i],
                'props':            mol_props(smi),
            })

        # ── Chemical subset analysis ─────────────────────────────────────
        subset_res = chem_subset_analysis(records, adduct)
        results[adduct] = {'n': N, 'subsets': subset_res}

        # Save per-sample records (without large arrays)
        per_sample_path = os.path.join(OUT_DIR, f'per_sample_chem_{tag}.json')
        with open(per_sample_path, 'w') as f:
            json.dump(records, f)
        print(f'Saved per-sample → {per_sample_path}')

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved summary → {args.out}')


if __name__ == '__main__':
    main()
