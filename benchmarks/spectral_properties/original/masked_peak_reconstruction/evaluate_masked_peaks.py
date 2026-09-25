"""
Masked peak reconstruction
==========================
Uses DreaMS's own SpectrumPreprocessor and masking semantics (mask_mz_hot:
only m/z is set to mask_val=-1, intensity preserved; precursor prepended).

Data sources:
  - MassSpecGym test set (default)
  - GeMS_A10 (DreaMS's own training data) via --data gems

Run through `benchmarks/spectral_properties/run_experiment.py masked_peak_reconstruction`.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.environ.get('ULTRAMS_DREAMS_ROOT'):
    sys.path.insert(0, os.environ['ULTRAMS_DREAMS_ROOT'])

MASSGYM_CSV = os.environ.get('ULTRAMS_MASSGYM_CSV', 'datasets/MassSpecGym/MassSpecGym.csv')
GEMS_A10 = os.environ.get('ULTRAMS_GEMS_HDF5', 'datasets/GeMS_A10.hdf5')


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_massgym_spectra(n_samples=5000, seed=42):
    import pandas as pd
    df = pd.read_csv(MASSGYM_CSV)
    df = df[df['fold'] == 'test'].reset_index(drop=True)
    if n_samples and n_samples < len(df):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), size=n_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    spectra = []
    for _, row in df.iterrows():
        mzs = np.array([float(x) for x in str(row['mzs']).split(',')], dtype=np.float64)
        ints = np.array([float(x) for x in str(row['intensities']).split(',')], dtype=np.float64)
        valid = mzs > 0
        mzs, ints = mzs[valid], ints[valid]
        if len(mzs) < 5:
            continue
        spec = np.stack([mzs, ints], axis=-1)  # (n_peaks, 2)
        spectra.append({
            'spectrum_raw': spec,
            'precursor_mz': float(row['precursor_mz']),
        })
    print(f"[Data] MassSpecGym test: {len(spectra)} spectra")
    return spectra


def load_gems_spectra(n_samples=10000, seed=42):
    import h5py
    f = h5py.File(GEMS_A10, 'r')
    total = len(f['spectrum'])
    rng = np.random.RandomState(seed)

    # Use a contiguous block starting at a random offset for fast HDF5 reads
    start_idx = rng.randint(0, max(1, total - n_samples))
    end_idx = min(start_idx + n_samples, total)
    print(f"[Data] GeMS_A10: reading contiguous block [{start_idx}:{end_idx}] from {total} total")

    raw_block = f['spectrum'][start_idx:end_idx]     # (N, 2, 128)
    pmz_block = f['precursor_mz'][start_idx:end_idx]  # (N,)

    spectra = []
    for i in range(len(raw_block)):
        mzs = raw_block[i, 0].astype(np.float64)
        ints = raw_block[i, 1].astype(np.float64)
        valid = mzs > 0
        if valid.sum() < 5:
            continue
        spec = np.stack([mzs[valid], ints[valid]], axis=-1)
        spectra.append({
            'spectrum_raw': spec,
            'precursor_mz': float(pmz_block[i]),
        })
    f.close()
    print(f"[Data] GeMS_A10: {len(spectra)} spectra loaded")
    return spectra


# ── DreaMS Reconstruction (using its own pipeline) ───────────────────────────

@torch.no_grad()
def reconstruct_dreams(spectra, device, mask_ratio=0.15, seed=42):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from dreams_loader import load_dreams_model

    model = load_dreams_model(device)
    spec_preproc = model.spec_preproc
    bin_size = model.hot_mz_bin_size
    mask_val = model.mask_val
    n_peaks_total = spec_preproc.n_highest_peaks + 1  # +1 for prepended precursor

    rng = np.random.RandomState(seed)
    all_mz_gt, all_mz_pred, all_pmz = [], [], []
    batch_size = 64

    for start in range(0, len(spectra), batch_size):
        batch_specs = spectra[start:start + batch_size]
        B = len(batch_specs)

        spec_real = np.zeros((B, n_peaks_total, 2), dtype=np.float32)
        spec_mask = np.zeros((B, n_peaks_total, 2), dtype=np.float32)
        mask_flags = np.zeros((B, n_peaks_total), dtype=np.bool_)

        for i, sp in enumerate(batch_specs):
            processed = spec_preproc(sp['spectrum_raw'], prec_mz=sp['precursor_mz'])
            spec_real[i] = processed
            spec_mask[i] = processed.copy()

            valid_fragment = (processed[:, 0] > 0) & (processed[:, 1] < 1.05)
            frag_idx = np.where(valid_fragment)[0]
            if len(frag_idx) < 2:
                continue
            nm = max(1, int(len(frag_idx) * mask_ratio))
            chosen = rng.choice(frag_idx, size=nm, replace=False)
            mask_flags[i, chosen] = True
            spec_mask[i, chosen, 0] = mask_val  # only mask m/z, keep intensity

        spec_real_t = torch.tensor(spec_real, dtype=torch.float32, device=device)
        spec_mask_t = torch.tensor(spec_mask, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_flags, device=device)

        embs = model(spec_mask_t)
        if mask_t.sum() == 0:
            continue

        pred_logits = model.ff_out(embs[mask_t])
        pred_bins = pred_logits.argmax(-1).cpu().numpy()
        pred_mz = pred_bins.astype(np.float64) * bin_size

        gt_mz = spec_real_t[mask_t][:, 0].cpu().numpy().astype(np.float64)

        all_mz_gt.append(gt_mz)
        all_mz_pred.append(pred_mz)

        pmz_batch = np.array([sp['precursor_mz'] for sp in batch_specs])
        batch_idx_np = np.where(mask_flags)[0]
        all_pmz.append(pmz_batch[batch_idx_np])

        if (start // batch_size) % 20 == 0:
            print(f"  DreaMS: {start}/{len(spectra)}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        'mz_gt': np.concatenate(all_mz_gt),
        'mz_pred': np.concatenate(all_mz_pred),
        'pmz': np.concatenate(all_pmz),
    }


# ── UltraMS reconstruction ──────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_v9(spectra, device, mask_ratio=0.15, seed=42):
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG
    model = UltraExplorerMLM(CONFIG)
    ckpt = torch.load(os.environ.get('ULTRAMS_RECONSTRUCTION_CKPT', './output/ue_multiscale_v9/checkpoint_epoch_3.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  [v9] ep={ckpt.get('epoch','?')}, step={ckpt.get('global_step','?')}")
    del ckpt
    model.to(device).eval()

    rng = np.random.RandomState(seed)
    max_peaks = model.max_peaks
    ms_start = model.ms_start_idx
    all_mz_gt, all_mz_pred, all_int_gt, all_int_pred, all_pmz = [], [], [], [], []
    all_int_gt_raw, all_int_pred_raw = [], []
    batch_size = 64

    for start in range(0, len(spectra), batch_size):
        batch_specs = spectra[start:start + batch_size]
        B = len(batch_specs)

        masked = np.zeros((B, max_peaks, 2), dtype=np.float32)
        orig = np.zeros((B, max_peaks, 2), dtype=np.float32)
        flags = np.zeros((B, max_peaks), dtype=np.bool_)
        attn = np.zeros((B, max_peaks), dtype=np.int64)
        pmzs = np.zeros(B, dtype=np.float32)

        for i, sp in enumerate(batch_specs):
            raw = sp['spectrum_raw']
            mz, it = raw[:, 0].astype(np.float32), raw[:, 1].astype(np.float32)
            valid = mz > 0
            mz, it = mz[valid], it[valid]
            mx = it.max()
            if mx > 0:
                it = it / mx
            it = np.clip(it, 0, 1)
            order = np.argsort(mz)
            mz, it = mz[order], it[order]
            if len(mz) > max_peaks:
                top = np.argsort(it)[-max_peaks:]
                top = np.sort(top)
                mz, it = mz[top], it[top]

            n = len(mz)
            orig[i, :n, 0] = mz
            orig[i, :n, 1] = it
            masked[i, :n, 0] = mz
            masked[i, :n, 1] = it
            attn[i, :n] = 1
            pmzs[i] = sp['precursor_mz']

            nm = max(1, int(n * mask_ratio))
            mi = rng.choice(n, size=nm, replace=False)
            flags[i, mi] = True
            masked[i, mi] = 0.0

        masked_t = torch.from_numpy(masked).to(device)
        orig_t = torch.from_numpy(orig).to(device)
        flags_t = torch.from_numpy(flags).to(device)
        attn_t = torch.from_numpy(attn).to(device)
        pmz_t = torch.from_numpy(pmzs).to(device)

        ms_emb = model.peak_encoder(masked_t)
        cls_emb = model.cls_emb.expand(B, 1, -1)
        prec_input = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], dim=-1).unsqueeze(1)
        prec_emb = model.peak_encoder(prec_input) + model.precursor_type_emb
        full_emb = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
        S = full_emb.shape[1]
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        full_emb = full_emb + model.pos_emb(pos_ids)

        prefix_attn = torch.ones(B, 2, dtype=attn_t.dtype, device=device)
        full_attn = torch.cat([prefix_attn, attn_t], dim=1)

        enc_out = model.encoder(inputs_embeds=full_emb, attention_mask=full_attn)
        hs = enc_out.last_hidden_state

        batch_idx, peak_idx = torch.where(flags_t)
        if batch_idx.numel() == 0:
            continue
        hidden_idx = peak_idx + ms_start
        valid_mask = hidden_idx < hs.shape[1]
        batch_idx = batch_idx[valid_mask]
        peak_idx = peak_idx[valid_mask]
        hidden_idx = hidden_idx[valid_mask]

        hc = hs[batch_idx, hidden_idx]
        mzv = orig_t[batch_idx, peak_idx, 0]
        intv = orig_t[batch_idx, peak_idx, 1]

        l0_logits = model.mz_level0_head(hc)
        l0_probs = F.softmax(l0_logits.float(), dim=-1)
        soft_mz_0 = (l0_probs * model.level0_centers).sum(-1)
        cb0 = model.peak_encoder.mz_codebook(soft_mz_0)

        l1_logits = model.mz_level1_head(torch.cat([hc, cb0], dim=-1))
        l1_probs = F.softmax(l1_logits.float(), dim=-1)
        snapped = (soft_mz_0 / 10.0).floor() * 10.0
        soft_mz_1 = snapped + (l1_probs * model.level1_offsets).sum(-1)
        cb1 = model.peak_encoder.mz_codebook(soft_mz_1)

        l2_logits = model.mz_level2_head(torch.cat([hc, cb1], dim=-1))
        int_logits = model.int_predictor(hc)

        l0p = l0_logits.argmax(-1)
        l1p = l1_logits.argmax(-1)
        l2p = l2_logits.argmax(-1)
        pred_mz = l0p.float() * 10.0 + l1p.float() + l2p.float() * 0.1
        gt_mz = mzv

        int_gt_bins = (intv.clamp(0, 1) / 0.1).round().long().clamp(0, 9)
        int_p = int_logits.argmax(-1)

        all_mz_gt.append(gt_mz.cpu().numpy())
        all_mz_pred.append(pred_mz.cpu().numpy())
        all_int_gt.append(int_gt_bins.cpu().numpy())
        all_int_pred.append(int_p.cpu().numpy())
        all_pmz.append(pmz_t[batch_idx].cpu().numpy())
        all_int_gt_raw.append(intv.cpu().numpy())
        all_int_pred_raw.append((int_p.float() * 0.1).cpu().numpy())

        if (start // batch_size) % 20 == 0:
            print(f"  v9: {start}/{len(spectra)}", flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        'mz_gt': np.concatenate(all_mz_gt),
        'mz_pred': np.concatenate(all_mz_pred),
        'int_gt': np.concatenate(all_int_gt),
        'int_pred': np.concatenate(all_int_pred),
        'int_gt_raw': np.concatenate(all_int_gt_raw),
        'int_pred_raw': np.concatenate(all_int_pred_raw),
        'pmz': np.concatenate(all_pmz),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(data, model_name):
    from scipy import stats as sp_stats

    mz_gt = data['mz_gt'].astype(np.float64)
    mz_pred = data['mz_pred'].astype(np.float64)
    n = len(mz_gt)
    mz_err = np.abs(mz_pred - mz_gt)

    ss_res = np.sum((mz_gt - mz_pred) ** 2)
    ss_tot = np.sum((mz_gt - mz_gt.mean()) ** 2)
    mz_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pearson_r, _ = sp_stats.pearsonr(mz_gt, mz_pred) if n > 2 else (0, 0)
    spearman_r, _ = sp_stats.spearmanr(mz_gt, mz_pred) if n > 2 else (0, 0)

    results = {
        'n_masked_peaks': n,
        'mz_mae': float(np.mean(mz_err)),
        'mz_rmse': float(np.sqrt(np.mean(mz_err ** 2))),
        'mz_median_ae': float(np.median(mz_err)),
        'mz_acc_10da': float((mz_err <= 5.0).mean()),
        'mz_acc_1da': float((mz_err <= 0.5).mean()),
        'mz_acc_01da': float((mz_err <= 0.05).mean()),
        'mz_r2': float(mz_r2),
        'mz_pearson_r': float(pearson_r),
        'mz_spearman_r': float(spearman_r),
        'mz_err_p25': float(np.percentile(mz_err, 25)),
        'mz_err_p50': float(np.percentile(mz_err, 50)),
        'mz_err_p75': float(np.percentile(mz_err, 75)),
        'mz_err_p90': float(np.percentile(mz_err, 90)),
        'mz_err_p95': float(np.percentile(mz_err, 95)),
    }

    if 'int_gt_raw' in data:
        ig = data['int_gt_raw'].astype(np.float64)
        ip = data['int_pred_raw'].astype(np.float64)
        int_err = np.abs(ip - ig)
        ss_res_i = np.sum((ig - ip) ** 2)
        ss_tot_i = np.sum((ig - ig.mean()) ** 2)
        int_r2 = 1.0 - ss_res_i / ss_tot_i if ss_tot_i > 0 else 0.0
        int_pearson, _ = sp_stats.pearsonr(ig, ip) if n > 2 else (0, 0)
        results['int_mae'] = float(np.mean(int_err))
        results['int_r2'] = float(int_r2)
        results['int_pearson_r'] = float(int_pearson)

    if 'int_gt' in data:
        int_gt = data['int_gt']
        int_pred = data['int_pred']
        int_bin_err = np.abs(int_pred.astype(float) - int_gt.astype(float))
        results['int_bin_acc'] = float((int_pred == int_gt).mean())
        results['int_bin_acc_1off'] = float((int_bin_err <= 1).mean())

    print(f"\n  [{model_name}] n={n}")
    print(f"    mz MAE:       {results['mz_mae']:.4f} Da")
    print(f"    mz RMSE:      {results['mz_rmse']:.4f} Da")
    print(f"    mz median AE: {results['mz_median_ae']:.4f} Da")
    print(f"    mz R²:        {results['mz_r2']:.6f}")
    print(f"    mz Pearson r: {results['mz_pearson_r']:.6f}")
    print(f"    mz Spearman ρ:{results['mz_spearman_r']:.6f}")
    print(f"    mz@10Da:      {results['mz_acc_10da']*100:.2f}%")
    print(f"    mz@1Da:       {results['mz_acc_1da']*100:.2f}%")
    print(f"    mz@0.1Da:     {results['mz_acc_01da']*100:.2f}%")
    print(f"    mz err P25/P50/P75/P90: {results['mz_err_p25']:.3f}/{results['mz_err_p50']:.3f}/{results['mz_err_p75']:.3f}/{results['mz_err_p90']:.3f}")
    if 'int_r2' in results:
        print(f"    Int MAE:      {results['int_mae']:.4f}")
        print(f"    Int R²:       {results['int_r2']:.4f}")
        print(f"    Int Pearson r:{results['int_pearson_r']:.4f}")
    if 'int_bin_acc' in results:
        print(f"    Int bin acc:  {results['int_bin_acc']*100:.2f}%")
    return results


def compute_stratified(data):
    pmz = data.get('pmz')
    if pmz is None:
        return {}
    mz_err = np.abs(data['mz_pred'] - data['mz_gt'])
    strata = {
        'pmz<300': pmz < 300,
        '300<=pmz<600': (pmz >= 300) & (pmz < 600),
        'pmz>=600': pmz >= 600,
    }
    results = {}
    for name, m in strata.items():
        if m.sum() < 10:
            continue
        results[name] = {
            'n': int(m.sum()),
            'mz_mae': float(np.mean(mz_err[m])),
            'mz_acc_1da': float((mz_err[m] <= 0.5).mean()),
            'mz_acc_01da': float((mz_err[m] <= 0.05).mean()),
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='massgym', choices=['massgym', 'gems'])
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mask-ratio', type=float, default=0.15)
    parser.add_argument('--skip-dreams', action='store_true')
    parser.add_argument('--skip-v9', action='store_true')
    args = parser.parse_args()
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    device = torch.device(device_name)

    if args.data == 'gems':
        spectra = load_gems_spectra(n_samples=args.n_samples)
        data_name = 'GeMS_A10'
    else:
        spectra = load_massgym_spectra(n_samples=args.n_samples)
        data_name = 'MassSpecGym_test'

    all_results = {'data_source': data_name, 'n_spectra': len(spectra),
                   'mask_ratio': args.mask_ratio, 'protocol': 'fair'}

    if not args.skip_v9:
        print(f"\n{'='*60}\nUltraMS reconstruction ({data_name})\n{'='*60}")
        v9_data = reconstruct_v9(spectra, device, mask_ratio=args.mask_ratio)
        v9_metrics = compute_metrics(v9_data, 'UltraMS')
        v9_strat = compute_stratified(v9_data)
        all_results['v9'] = {'metrics': v9_metrics, 'stratified': v9_strat}

    if not args.skip_dreams:
        print(f"\n{'='*60}\nDreaMS Reconstruction ({data_name}) — Fair Protocol\n{'='*60}")
        print("  Using DreaMS's own SpectrumPreprocessor (prepend precursor, mask m/z only)")
        dreams_data = reconstruct_dreams(spectra, device, mask_ratio=args.mask_ratio)
        dreams_metrics = compute_metrics(dreams_data, 'DreaMS')
        dreams_strat = compute_stratified(dreams_data)
        all_results['dreams'] = {'metrics': dreams_metrics, 'stratified': dreams_strat}

    # Print comparison
    print(f"\n{'='*60}")
    print(f"COMPARISON ({data_name}, Fair Protocol)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'UltraMS':>12} {'DreaMS':>12}")
    print("-" * 48)
    v9m = all_results.get('v9', {}).get('metrics', {})
    drm = all_results.get('dreams', {}).get('metrics', {})
    for label, key in [('mz MAE', 'mz_mae'), ('mz@10Da', 'mz_acc_10da'),
                        ('mz@1Da', 'mz_acc_1da'), ('mz@0.1Da', 'mz_acc_01da'),
                        ('Int acc', 'int_bin_acc')]:
        v9v = v9m.get(key)
        drv = drm.get(key)
        v9s = f"{v9v:.4f}" if v9v is not None else "N/A"
        drs = f"{drv:.4f}" if drv is not None else "N/A"
        print(f"  {label:<18} {v9s:>12} {drs:>12}")

    suffix = 'gems' if args.data == 'gems' else 'massgym'
    out_dir = './output/comparison'
    os.makedirs(out_dir, exist_ok=True)

    out = f'{out_dir}/reconstruction_fair_{suffix}.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to: {out}")

    # Save raw arrays for detailed visualization
    npz_path = f'{out_dir}/reconstruction_fair_{suffix}_raw.npz'
    arrays = {}
    if not args.skip_v9:
        for k, v in v9_data.items():
            arrays[f'v9_{k}'] = v
    if not args.skip_dreams:
        for k, v in dreams_data.items():
            arrays[f'dreams_{k}'] = v
    np.savez_compressed(npz_path, **arrays)
    print(f"Saved raw arrays to: {npz_path}")


if __name__ == '__main__':
    main()
