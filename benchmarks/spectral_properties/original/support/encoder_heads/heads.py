"""Benchmark checkpoint encoder and retention-time / polarity head definitions."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get('ULTRAMS_SOURCE_SUPPLEMENT_ROOT'):
    sys.path.insert(0, os.environ['ULTRAMS_SOURCE_SUPPLEMENT_ROOT'])

from ultrams_pretraining_model import CONFIG as ULTRAMS_CONFIG

ENCODER_HEAD_CONFIG = {**ULTRAMS_CONFIG, 'freeze_layers': 12}



class RTHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.LayerNorm(d_model // 2), nn.Linear(d_model // 2, 1))

    def forward(self, cls_emb):
        return self.head(cls_emb).squeeze(-1)


class PolarityHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.LayerNorm(d_model // 2), nn.Linear(d_model // 2, 1))

    def forward(self, cls_emb):
        return self.head(cls_emb).squeeze(-1)


class UltraMSWithHeads(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base = base_model
        self.config = config
        d = config['d_model']
        self.rt_head = RTHead(d)
        self.pol_head = PolarityHead(d)
        self._freeze(config['freeze_layers'])

    def _freeze(self, n):
        for p in self.base.peak_encoder.parameters():
            p.requires_grad = False
        layers = self.base.encoder.encoder.layer
        for i in range(min(n, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad = False
        nf = sum(1 for p in self.parameters() if not p.requires_grad)
        nt = sum(1 for p in self.parameters())
        print(f"  [Freeze] {nf}/{nt} frozen, {nt-nf} trainable")

    def _encode(self, spectra, attn_mask, pmz, device):
        B = spectra.shape[0]
        ms_emb = self.base.peak_encoder(spectra)
        cls_e = self.base.cls_emb.expand(B, 1, -1)
        pi = torch.stack([pmz, torch.full_like(pmz, 1.1)], -1).unsqueeze(1)
        pe = self.base.peak_encoder(pi) + self.base.precursor_type_emb
        fe = torch.cat([cls_e, pe, ms_emb], 1)
        S = fe.shape[1]
        pos = torch.arange(S, device=device).unsqueeze(0)
        fe = self.base.dropout(fe + self.base.pos_emb(pos))
        pa = torch.ones(B, 2, dtype=attn_mask.dtype, device=device)
        fa = torch.cat([pa, attn_mask], 1)
        hs = self.base.encoder(inputs_embeds=fe, attention_mask=fa).last_hidden_state
        return hs, hs[:, 0, :]
