"""
Standalone DreaMS model loader that bypasses the SpecBridge adapter.
Handles PyTorch 2.6 weights_only changes and matchms API updates.
"""

import os
import sys
import types
import argparse
import torch
import torch.nn as nn
import numpy as np

_COMPARISON_DIR = os.path.dirname(os.path.abspath(__file__))
_LIGHT_ULTRA_DIR = os.environ.get('LIGHT_ULTRA_ROOT', os.path.dirname(os.path.dirname(_COMPARISON_DIR)))
_TRAIN_DIR = os.path.join(_LIGHT_ULTRA_DIR, 'train')
_RESOURCE_DIR = os.path.join(_TRAIN_DIR, 'comparison', 'resources')

# Fix matchms API before DreaMS imports
try:
    import matchms.similarity
    if not hasattr(matchms.similarity, 'ModifiedCosine'):
        matchms.similarity.ModifiedCosine = matchms.similarity.ModifiedCosineGreedy
except Exception:
    pass

sys.path.insert(0, os.path.join(_RESOURCE_DIR, 'dreams', 'DreaMS'))

DREAMS_CKPT = os.path.join(
    _RESOURCE_DIR, 'DreaMS_Check/ssl_model.ckpt'
)
N_HIGHEST_PEAKS = 60


def _patch_msml():
    """Create stub for msml module required by legacy DreaMS checkpoint."""
    if 'msml' not in sys.modules:
        for name in ['msml', 'msml.utils', 'msml.utils.data', 'msml.utils.dformats']:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        from dreams.utils.dformats import DataFormatA
        from dreams.utils.data import SpectrumPreprocessor
        sys.modules['msml.utils.data'].DataFormatA = DataFormatA
        sys.modules['msml.utils.data'].SpectrumPreprocessor = SpectrumPreprocessor
        sys.modules['msml.utils.dformats'].DataFormatA = DataFormatA


def load_dreams_model(device='cpu', ckpt_path=DREAMS_CKPT):
    """Load the full DreaMS model with reconstruction head."""
    _patch_msml()

    from dreams.models.dreams.dreams import DreaMS
    import dreams.utils.data as du
    import dreams.utils.dformats as dformats

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([argparse.Namespace])

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['hyper_parameters']['args']

    dformat = dformats.DataFormatA()
    args.dformat = dformat
    args.gains_dir = '.'

    spec_preproc = du.SpectrumPreprocessor(dformat, n_highest_peaks=N_HIGHEST_PEAKS)
    model = DreaMS(args, spec_preproc)

    sd = ckpt['state_dict']
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    has_ff_out = model.ff_out is not None
    print(f"[DreaMS] Loaded full model: d_model={model.d_model}, "
          f"n_layers={args.n_layers}, has_ff_out={has_ff_out}")
    return model


def load_dreams_encoder(device='cpu', ckpt_path=DREAMS_CKPT):
    """Load DreaMS as encoder-only (for embedding extraction)."""
    model = load_dreams_model(device, ckpt_path)
    model.ff_out = None
    model.ro_out = None
    return model


def prep_peaks(spec_np, n=N_HIGHEST_PEAKS):
    """Manual peak pad (no precursor). Prefer build_dreams_batch + official spec_preproc for eval."""
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((n, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return np.zeros((n, 2), dtype=np.float32)
    mx = inten.max()
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0, 1)
    if len(mz) > n:
        idx = np.argsort(inten)[-n:]
        idx = np.sort(idx)
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    mz, inten = mz[order], inten[order]
    out = np.zeros((n, 2), dtype=np.float32)
    out[:len(mz), 0] = mz
    out[:len(mz), 1] = inten
    return out


def build_dreams_batch(model, batch_samples, device):
    """
    DreaMS training layout (SpectrumPreprocessor): trim/pad to n_highest_peaks,
    relative intensities, then prepend [prec_mz, prec_intens] as row 0.
    Encoder output[:, 0, :] is the precursor token representation.
    """
    tensors = []
    for s in batch_samples:
        spec = np.asarray(s['spectrum'], dtype=np.float32)
        if spec.ndim != 2 or spec.shape[1] != 2:
            spec = np.zeros((0, 2), dtype=np.float32)
        prec = s.get('precursor_mz', None)
        if prec is None:
            raise KeyError(
                "DreaMS input requires sample['precursor_mz'] (prepends precursor peak; "
                "see dreams.utils.data.SpectrumPreprocessor)."
            )
        prepped = model.spec_preproc(
            spec, prec_mz=float(prec), high_form=True, augment=False)
        tensors.append(prepped)
    return torch.from_numpy(np.stack(tensors, axis=0)).to(device)


@torch.no_grad()
def extract_embeddings(model, samples, device, batch_size=128):
    """Precursor-token embeddings: encoder output at index 0 after official preprocessing."""
    model.eval()
    all_emb = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        peaks = build_dreams_batch(model, batch, device)
        out = model(peaks)
        all_emb.append(out[:, 0, :].cpu().numpy())
        if (start // batch_size) % 50 == 0:
            print(f"    DreaMS embed: {start}/{len(samples)}", flush=True)
    return np.concatenate(all_emb, axis=0)


@torch.no_grad()
def extract_embeddings_no_precursor(model, samples, device, batch_size=128):
    """Extract embeddings using prep_peaks (no precursor prepending) - legacy method."""
    model.eval()
    all_emb = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        tensors = []
        for s in batch:
            spec = np.asarray(s['spectrum'], dtype=np.float32)
            prepped = prep_peaks(spec, n=N_HIGHEST_PEAKS)
            tensors.append(prepped)
        peaks = torch.from_numpy(np.stack(tensors, axis=0)).to(device)
        out = model(peaks)
        all_emb.append(out[:, 0, :].cpu().numpy())
    return np.concatenate(all_emb, axis=0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_dreams_encoder(device)
    print(f"Model type: {type(model).__name__}")
    samples = [
        {'spectrum': np.stack([np.arange(100., 600., 100), np.linspace(0.2, 1.0, 5)], axis=-1).astype(np.float32),
         'precursor_mz': 501.0},
        {'spectrum': np.stack([np.arange(110., 610., 100), np.linspace(0.3, 1.0, 5)], axis=-1).astype(np.float32),
         'precursor_mz': 601.0},
    ]
    peaks = build_dreams_batch(model, samples, device)
    out = model(peaks)
    print(f'Input shape (with prepended precursor): {tuple(peaks.shape)}')
    print(f'Output shape: {tuple(out.shape)}  (use out[:,0,:] for precursor token)')
