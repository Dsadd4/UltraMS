"""UltraProbe / DreamsProbe: backbone encoder to probe head.

probe_mode:
  cls_only: CLS/precursor token to MLP head (input dimension d).
  cls_agg: CLS plus CrossAttnAggregator over sub-tokens to MLP (dimension 2d).

Ultra sequence layout (UltraMSWithHeads._encode):
  Position 0: CLS; position 1: precursor_emb; positions 2 onward: peak tokens.

DreaMS sequence layout:
  Position 0: precursor token (CLS); positions 1 onward: peak tokens (60).
"""
from __future__ import annotations
import os, sys
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_PEAKS = 150      # Use 150 spectral peaks for both UltraMS and DreaMS.


# Freeze/unfreeze utilities
def freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_ultra_last_n(encoder: nn.Module, n: int):
    if n <= 0:
        return
    layers = encoder.base.encoder.encoder.layer
    L = len(layers)
    for i in range(max(0, L - n), L):
        for p in layers[i].parameters():
            p.requires_grad_(True)


def unfreeze_dreams_last_n(dreams: nn.Module, n: int):
    if n <= 0:
        return
    L = dreams.n_layers
    start = max(0, L - n)
    if dreams.vanilla_transformer:
        for i in range(start, L):
            for p in dreams.transformer_encoder.layers[i].parameters():
                p.requires_grad_(True)
        return
    te = dreams.transformer_encoder
    for i in range(start, L):
        for p in te.atts[i].parameters():
            p.requires_grad_(True)
        for p in te.ffs[i].parameters():
            p.requires_grad_(True)
    if hasattr(te, 'scales') and te.scales is not None:
        for i in range(start, L):
            for idx in (2 * i, 2 * i + 1):
                if idx < len(te.scales):
                    for p in te.scales[idx].parameters():
                        p.requires_grad_(True)
        for p in te.scales[-1].parameters():
            p.requires_grad_(True)


# ── CrossAttnAggregator ───────────────────────────────────────────────
class CrossAttnAggregator(nn.Module):
    """Apply scaled dot-product attention from a learned query to sub-token states.
    Only d_model parameters are introduced; gradients propagate upstream through
    the query vector.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.scale = d_model ** -0.5

    def forward(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """hidden: (B, L, d).
        mask: (B, L) boolean; True marks valid tokens. None means all tokens are valid.
        Return shape: (B, d).
        """
        scores = (hidden * self.q).sum(-1) * self.scale   # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)            # (B, L)
        return (weights.unsqueeze(-1) * hidden).sum(1)    # (B, d)


# ── MLP Head ──────────────────────────────────────────────────────────
def build_head(d_in: int, n_tasks: int, dropout: float = 0.1) -> nn.Sequential:
    d_hidden = d_in // 2
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Dropout(dropout),
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, n_tasks),
    )


# UltraMS input preprocessing
def _ultra_prepare(batch: List[Dict], device: torch.device, max_peaks: int = MAX_PEAKS):
    """Return (peaks_t, attn_t, pmz_t).
    The spectrum has already been processed by prep_spectrum; pad to max_peaks.
    """
    B = len(batch)
    padded = np.zeros((B, max_peaks, 2), dtype=np.float32)
    attn   = np.zeros((B, max_peaks),    dtype=np.int64)
    pmzs   = []
    for i, s in enumerate(batch):
        sp = np.asarray(s['spectrum'], dtype=np.float32)
        k  = min(len(sp), max_peaks)
        if k > 0:
            padded[i, :k] = sp[:k]
            attn[i, :k]   = 1
        pmzs.append(float(s['precursor_mz']))
    return (
        torch.from_numpy(padded).to(device),
        torch.from_numpy(attn).to(device),
        torch.tensor(pmzs, dtype=torch.float32, device=device),
    )




# ── Ultra Probe ───────────────────────────────────────────────────────
class UltraProbe(nn.Module):
    def __init__(self, encoder: nn.Module, n_tasks: int,
                 dropout: float = 0.1, probe_mode: str = 'cls_agg'):
        super().__init__()
        self.encoder = encoder
        self.probe_mode = probe_mode
        d = int(encoder.config['d_model'])
        if probe_mode == 'cls_agg':
            self.aggregator = CrossAttnAggregator(d)
            self.head = build_head(2 * d, n_tasks, dropout)
        else:  # cls_only
            self.aggregator = None
            self.head = build_head(d, n_tasks, dropout)

    def forward(self, batch: List[Dict], device: torch.device) -> torch.Tensor:
        peaks_t, attn_t, pmz_t = _ultra_prepare(batch, device)
        hs, cls = self.encoder._encode(peaks_t, attn_t, pmz_t, device)
        if self.probe_mode == 'cls_agg':
            B = hs.shape[0]
            sub_hs   = hs[:, 1:, :]
            pre_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
            sub_mask  = torch.cat([pre_valid, attn_t.bool()], dim=1)
            agg = self.aggregator(sub_hs, sub_mask)
            return self.head(torch.cat([cls, agg], dim=-1))
        return self.head(cls)


# ── DreaMS Probe ──────────────────────────────────────────────────────
class DreamsProbe(nn.Module):
    def __init__(self, dreams: nn.Module, n_tasks: int,
                 dropout: float = 0.1, probe_mode: str = 'cls_agg'):
        super().__init__()
        self.dreams = dreams
        self.probe_mode = probe_mode
        d = int(dreams.d_model)
        if probe_mode == 'cls_agg':
            self.aggregator = CrossAttnAggregator(d)
            self.head = build_head(2 * d, n_tasks, dropout)
        else:  # cls_only
            self.aggregator = None
            self.head = build_head(d, n_tasks, dropout)

    def forward(self, batch: List[Dict], device: torch.device) -> torch.Tensor:
        from comparison.dreams_loader import build_dreams_batch
        peaks_t = build_dreams_batch(self.dreams, batch, device)  # (B, 61, 2)
        hs = self.dreams(peaks_t)        # (B, 61, d)
        cls     = hs[:, 0, :]           # precursor token
        if self.probe_mode == 'cls_agg':
            peak_hs   = hs[:, 1:, :]
            peak_mask = (peaks_t[:, 1:, 0] != 0)
            agg = self.aggregator(peak_hs, peak_mask)
            return self.head(torch.cat([cls, agg], dim=-1))
        return self.head(cls)
