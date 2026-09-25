"""
Per-atom query pooling probe for interpretable heteroatom prediction.

Each heteroatom (N, O, S, Cl, F, Br) has a dedicated learnable query vector
that attends over peak token representations, producing per-peak attention
weights for case study visualization.

Ultra sequence:  pos0=CLS | pos1=precursor | pos2:=peaks
DreaMS sequence: pos0=precursor            | pos1:=peaks

Usage (return_attn=True for case study):
    logits, attn_maps = model(batch, device, return_attn=True)
    # attn_maps: (B, n_atoms, n_peaks)  — peak-level attribution per atom
"""
from __future__ import annotations
import os, sys
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

_HERE       = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT  = os.path.dirname(_TRAIN_ROOT)
sys.path.insert(0, _TRAIN_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from element.models import _ultra_prepare, freeze_all, unfreeze_ultra_last_n, unfreeze_dreams_last_n

MAX_PEAKS = 150
HETEROATOM_NAMES = ['N', 'O', 'S', 'Cl', 'F', 'Br']
HETEROATOM_COLS  = ['cnt_N', 'cnt_O', 'cnt_S', 'cnt_Cl', 'cnt_F', 'cnt_Br']
FOUR_ATOM_NAMES  = ['N', 'O', 'S', 'Cl']
FOUR_ATOM_COLS   = ['cnt_N', 'cnt_O', 'cnt_S', 'cnt_Cl']

# Mapping: heteroatom name → index in formula_count cols (C,H,N,O,S,Cl,F,Br)
FORMULA_IDX = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'S': 4, 'Cl': 5, 'F': 6, 'Br': 7}


def _build_head(d_in: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Dropout(dropout),
        nn.Linear(d_in, d_in // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_in // 2, 1),
    )


class UltraAtomQueryProbe(nn.Module):
    """
    Per-atom query pooling probe for Ultra backbone.

    For atom k:
        scores_k = q_k · K(peaks)^T / √d        → masked softmax → α_k  (B, L_peaks)
        agg_k    = α_k · V(peaks)                                         (B, d)
        count_k  = head_k([CLS; agg_k])                                   (B, 1)
    logits: (B, n_atoms)
    """

    def __init__(self, phase2: nn.Module, atom_names: List[str], dropout: float = 0.1):
        super().__init__()
        self.phase2     = phase2
        self.atom_names = atom_names
        n_atoms = len(atom_names)
        d = int(phase2.config['d_model'])
        self.d = d

        self.atom_queries = nn.Parameter(torch.empty(n_atoms, d))
        nn.init.trunc_normal_(self.atom_queries, std=0.02)

        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.scale  = d ** -0.5

        self.heads = nn.ModuleList([_build_head(2 * d, dropout) for _ in atom_names])
        # sentinel for checkpoint compatibility
        self.aggregator = None

    def forward(
        self,
        batch: List[Dict],
        device: torch.device,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        peaks_t, attn_t, pmz_t = _ultra_prepare(batch, device)
        hs, cls = self.phase2._encode(peaks_t, attn_t, pmz_t, device)

        # peak tokens only: skip pos0 (CLS) and pos1 (precursor)
        peak_hs   = hs[:, 2:, :]      # (B, max_peaks, d)
        peak_mask = attn_t.bool()     # (B, max_peaks)

        K = self.k_proj(peak_hs)      # (B, L, d)
        V = self.v_proj(peak_hs)      # (B, L, d)

        all_logits, all_attn = [], []
        for k in range(len(self.atom_names)):
            q      = self.atom_queries[k]                              # (d,)
            scores = (K * q).sum(-1) * self.scale                     # (B, L)
            scores = scores.masked_fill(~peak_mask, float('-inf'))
            attn_w = torch.softmax(scores, dim=-1)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)               # guard all-pad rows

            agg  = (attn_w.unsqueeze(-1) * V).sum(1)                 # (B, d)
            feat = torch.cat([cls, agg], dim=-1)                      # (B, 2d)
            all_logits.append(self.heads[k](feat))
            all_attn.append(attn_w)

        logits = torch.cat(all_logits, dim=-1)                        # (B, n_atoms)
        if return_attn:
            return logits, torch.stack(all_attn, dim=1)               # (B, n_atoms, L)
        return logits


class DreamsAtomQueryProbe(nn.Module):
    """
    Per-atom query pooling probe for DreaMS backbone.

    DreaMS sequence: pos0=precursor | pos1:=peaks (max 60)
    """

    def __init__(self, dreams: nn.Module, atom_names: List[str], dropout: float = 0.1):
        super().__init__()
        self.dreams     = dreams
        self.atom_names = atom_names
        n_atoms = len(atom_names)
        d = int(dreams.d_model)
        self.d = d

        self.atom_queries = nn.Parameter(torch.empty(n_atoms, d))
        nn.init.trunc_normal_(self.atom_queries, std=0.02)

        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.scale  = d ** -0.5

        self.heads = nn.ModuleList([_build_head(2 * d, dropout) for _ in atom_names])
        self.aggregator = None

    def forward(
        self,
        batch: List[Dict],
        device: torch.device,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        from comparison.dreams_loader import build_dreams_batch
        peaks_t  = build_dreams_batch(self.dreams, batch, device)   # (B, 61, 2)
        hs       = self.dreams(peaks_t)                              # (B, 61, d)
        cls      = hs[:, 0, :]                                       # (B, d)
        peak_hs  = hs[:, 1:, :]                                      # (B, 60, d)
        peak_mask = (peaks_t[:, 1:, 0] != 0)                        # (B, 60)

        K = self.k_proj(peak_hs)
        V = self.v_proj(peak_hs)

        all_logits, all_attn = [], []
        for k in range(len(self.atom_names)):
            q      = self.atom_queries[k]
            scores = (K * q).sum(-1) * self.scale
            scores = scores.masked_fill(~peak_mask, float('-inf'))
            attn_w = torch.softmax(scores, dim=-1)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)

            agg  = (attn_w.unsqueeze(-1) * V).sum(1)
            feat = torch.cat([cls, agg], dim=-1)
            all_logits.append(self.heads[k](feat))
            all_attn.append(attn_w)

        logits = torch.cat(all_logits, dim=-1)
        if return_attn:
            return logits, torch.stack(all_attn, dim=1)
        return logits


def _patch_phase2_import():
    """Pre-register phase2.train_phase2_rt_only from .pyc when source is missing."""
    import importlib.util
    key = 'phase2.train_phase2_rt_only'
    if key in sys.modules:
        return
    pyc = os.path.join(_TRAIN_ROOT, 'phase2', '__pycache__',
                       'train_phase2_rt_only.cpython-311.pyc')
    if not os.path.isfile(pyc):
        return
    spec = importlib.util.spec_from_file_location(key, pyc)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)


def load_probe_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    ultra_ckpt: Optional[str] = None,
) -> nn.Module:
    """Rebuild probe model and restore probe weights from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    atom_names = ckpt['atom_names']
    backbone   = ckpt['backbone']

    if backbone == 'ultra':
        _patch_phase2_import()
        from selection.selection_data_build import load_phase2_stage_d_for_fusion
        enc   = load_phase2_stage_d_for_fusion(device, checkpoint_path=ultra_ckpt)
        model = UltraAtomQueryProbe(enc, atom_names).to(device)
    else:
        from comparison.dreams_loader import load_dreams_encoder
        enc   = load_dreams_encoder(device)
        model = DreamsAtomQueryProbe(enc, atom_names).to(device)

    freeze_all(model)
    model.atom_queries.data = ckpt['atom_queries'].to(device)
    model.k_proj.load_state_dict(ckpt['k_proj_state'])
    model.v_proj.load_state_dict(ckpt['v_proj_state'])
    for head, state in zip(model.heads, ckpt['heads_state']):
        head.load_state_dict(state)

    norm_mean = np.array(ckpt['norm_mean'], dtype=np.float32)
    norm_std  = np.array(ckpt['norm_std'],  dtype=np.float32)
    model.eval()
    return model, atom_names, norm_mean, norm_std
