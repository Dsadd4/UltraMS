"""Inference architecture with the state-dict names of the UltraMS v9 trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import BertConfig, BertModel


MODEL_DEFAULTS: dict[str, Any] = {
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 16,
    "dim_feedforward": 4096,
    "dropout": 0.1,
    "max_peaks": 150,
    "mz_range": (0, 2000),
    "mz_bin_size": 0.1,
    "int_bin_size": 0.1,
    "mask_ratio": 0.15,
    "mz_codebook_dim": 512,
    "int_codebook_dim": 512,
    "coarse_num": 2000,
    "teacher_forcing_ratio": 0.0,
    "difficulty_weight_alpha": 2.0,
    "int_loss_weight": 0.1,
}


def model_config(saved: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(MODEL_DEFAULTS)
    if isinstance(saved, Mapping):
        for source in (saved, saved.get("model")):
            if isinstance(source, Mapping):
                for key in result:
                    if key in source:
                        result[key] = source[key]
    result["mz_range"] = tuple(result["mz_range"])
    return result


class SoftCodebookLookup(nn.Module):
    def __init__(
        self,
        num_codes: int,
        d_out: int,
        val_min: float = 0.0,
        val_max: float = 1.0,
        alpha_init: float = 0.1,
        circular: bool = False,
    ) -> None:
        super().__init__()
        self.circular = circular
        self.val_range = val_max - val_min
        self.codebook = nn.Embedding(num_codes, d_out)
        self.register_buffer("centers", torch.linspace(val_min, val_max, num_codes))
        self.alpha = nn.Parameter(torch.full((num_codes,), alpha_init))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        diff = values.unsqueeze(-1) - self.centers
        if self.circular:
            abs_diff = diff.abs()
            diff = torch.min(abs_diff, self.val_range - abs_diff)
        weights = torch.softmax(-diff.pow(2) / (self.alpha.abs() + 1e-6), dim=-1)
        return weights @ self.codebook.weight


class MultiResolutionMzCodebook(nn.Module):
    def __init__(
        self,
        d_out: int,
        mz_max: float = 2000.0,
        l0_num: int = 200,
        l1_num: int = 2000,
        l2_num: int = 200,
        l3_num: int = 200,
    ) -> None:
        super().__init__()
        d_each = d_out // 4
        d_last = d_out - 3 * d_each
        self.level0 = SoftCodebookLookup(l0_num, d_each, 0.0, mz_max, 100.0)
        self.level1 = SoftCodebookLookup(l1_num, d_each, 0.0, mz_max, 1.0)
        self.level2 = SoftCodebookLookup(l2_num, d_each, 0.0, 1.0, 0.01, True)
        self.level3 = SoftCodebookLookup(l3_num, d_last, 0.0, 0.1, 0.001, True)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.level0(mz),
                self.level1(mz),
                self.level2(torch.fmod(mz.abs(), 1.0)),
                self.level3(torch.fmod(mz.abs(), 0.1)),
            ),
            dim=-1,
        )


class CodebookIntensityEmbedding(nn.Module):
    def __init__(self, d_out: int, num_codes: int = 100, alpha_init: float = 0.1):
        super().__init__()
        self.lookup = SoftCodebookLookup(num_codes, d_out, 0.0, 1.0, alpha_init)

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        return self.lookup(intensity)


class CodebookPeakEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        mz_dim: int = 512,
        int_dim: int = 512,
        int_num_codes: int = 100,
        mz_max: float = 2000.0,
        coarse_num: int = 2000,
    ) -> None:
        super().__init__()
        self.mz_dim = mz_dim
        self.int_dim = int_dim
        self.mz_codebook = MultiResolutionMzCodebook(
            mz_dim, mz_max=mz_max, l1_num=coarse_num
        )
        self.int_codebook = CodebookIntensityEmbedding(int_dim, int_num_codes)
        self.mz_type_emb = nn.Parameter(torch.zeros(mz_dim))
        self.int_type_emb = nn.Parameter(torch.zeros(int_dim))
        self.proj = nn.Linear(mz_dim + int_dim, d_model)

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        mz = self.mz_codebook(peaks[..., 0]) + self.mz_type_emb
        intensity = self.int_codebook(peaks[..., 1]) + self.int_type_emb
        return self.proj(torch.cat((mz, intensity), dim=-1))


class UltraMSBackbone(nn.Module):
    """The full pretraining model layout, with an inference-only forward path."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.max_peaks = int(config["max_peaks"])
        self.mz_range = tuple(config["mz_range"])
        self.mz_bin_size = float(config["mz_bin_size"])
        self.mz_num_bins = int((self.mz_range[1] - self.mz_range[0]) / self.mz_bin_size) + 1
        self.int_bin_size = float(config["int_bin_size"])
        self.int_num_bins = int(1.0 / self.int_bin_size)
        self.ms_start_idx = 2
        self.mask_ratio = float(config["mask_ratio"])
        self.int_loss_weight = float(config.get("int_loss_weight", 1.0))
        self.teacher_forcing_ratio = 0.0
        self.difficulty_weight_alpha = float(config.get("difficulty_weight_alpha", 2.0))

        d_model = int(config["d_model"])
        mz_dim = int(config["mz_codebook_dim"])
        self.peak_encoder = CodebookPeakEncoder(
            d_model=d_model,
            mz_dim=mz_dim,
            int_dim=int(config["int_codebook_dim"]),
            mz_max=float(self.mz_range[1]),
            coarse_num=int(config.get("coarse_num", 2000)),
        )
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.precursor_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        max_seq_len = self.max_peaks + 2
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        mz_precision = 1.0
        vocab_size = 13 + len(torch.arange(*self.mz_range, mz_precision))
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=int(config["num_layers"]),
            num_attention_heads=int(config["num_heads"]),
            intermediate_size=int(config["dim_feedforward"]),
            hidden_dropout_prob=float(config["dropout"]),
            attention_probs_dropout_prob=float(config["dropout"]),
            max_position_embeddings=max(max_seq_len * 2, 512),
            layer_norm_eps=1e-6,
            is_decoder=False,
            add_cross_attention=False,
        )
        self.encoder = BertModel(bert_config, add_pooling_layer=False)
        self.encoder.embeddings.word_embeddings.requires_grad_(False)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.mz_level0_bins = 200
        self.mz_level1_bins = 10
        self.mz_level2_bins = 10
        self.register_buffer("level0_centers", torch.linspace(5.0, 1995.0, 200))
        self.register_buffer("level1_offsets", torch.linspace(0.5, 9.5, 10))
        self.register_buffer("level2_offsets", torch.linspace(0.05, 0.95, 10))
        self.mz_level0_head = nn.Linear(d_model, self.mz_level0_bins)
        self.mz_level1_head = nn.Sequential(
            nn.Linear(d_model + mz_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.mz_level1_bins),
        )
        self.mz_level2_head = nn.Sequential(
            nn.Linear(d_model + mz_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.mz_level2_bins),
        )
        self.int_predictor = nn.Linear(d_model, self.int_num_bins)

    def encode(
        self, spectra: torch.Tensor, attention_mask: torch.Tensor, precursor_mz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spectra.shape[0]
        peak_embeddings = self.peak_encoder(spectra)
        cls_embedding = self.cls_emb.expand(batch_size, 1, -1)
        precursor_input = torch.stack(
            (precursor_mz, torch.full_like(precursor_mz, 1.1)), dim=-1
        ).unsqueeze(1)
        precursor_embedding = self.peak_encoder(precursor_input) + self.precursor_type_emb
        embeddings = torch.cat((cls_embedding, precursor_embedding, peak_embeddings), dim=1)
        positions = torch.arange(embeddings.shape[1], device=spectra.device).unsqueeze(0)
        embeddings = self.dropout(embeddings + self.pos_emb(positions))
        prefix_attention = torch.ones(
            batch_size, 2, dtype=attention_mask.dtype, device=spectra.device
        )
        full_attention = torch.cat((prefix_attention, attention_mask), dim=1)
        hidden = self.encoder(
            inputs_embeds=embeddings, attention_mask=full_attention
        ).last_hidden_state
        return hidden, hidden[:, 0, :]


class _Head(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(cls_embedding).squeeze(-1)


class UltraMSWithHeads(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.base = UltraMSBackbone(config)
        self.rt_head = _Head(int(config["d_model"]))
        self.pol_head = _Head(int(config["d_model"]))

    def encode(
        self, spectra: torch.Tensor, attention_mask: torch.Tensor, precursor_mz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base.encode(spectra, attention_mask, precursor_mz)
