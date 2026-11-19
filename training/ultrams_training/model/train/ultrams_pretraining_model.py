"""UltraMS spectrum encoder and masked peak reconstruction objective.

Four resolution levels encode m/z through learnable soft codebooks. A BERT
encoder combines the peak, precursor and CLS tokens; the reconstruction heads
predict m/z coarse-to-fine with teacher forcing and difficulty weighting.
Training schedules and data paths are defined in the public training configs.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spectrum_decoder import MsTokenizer
from transformers import BertModel, BertConfig

# ============================================================================
# Config
# ============================================================================
CONFIG = {
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
    # Codebook (no Fourier)
    "mz_codebook_dim": 512,
    "int_codebook_dim": 512,
    "coarse_num": 2000,
    # Teacher Forcing
    "teacher_forcing_ratio": 0.5,
    # Difficulty-adaptive weighting
    "difficulty_weight_alpha": 2.0,
    "int_loss_weight": 0.1,
}


# ============================================================================
# Section 1: 4-level Multi-Resolution Codebook (no Fourier)
# ============================================================================
class SoftCodebookLookup(nn.Module):
    """Soft-attention codebook: per-code learnable temperature, optional circular distance."""

    def __init__(
        self,
        num_codes: int,
        d_out: int,
        val_min: float = 0.0,
        val_max: float = 1.0,
        alpha_init: float = 0.1,
        circular: bool = False,
    ):
        super().__init__()
        self.circular = circular
        self.val_range = val_max - val_min

        self.codebook = nn.Embedding(num_codes, d_out)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.register_buffer("centers", torch.linspace(val_min, val_max, num_codes))
        self.alpha = nn.Parameter(torch.full((num_codes,), alpha_init))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Soft lookup instead of hard binning:
        # each scalar value attends to all code centers and returns a weighted average of code embeddings. alpha controls how sharp each code is.
        x = values.unsqueeze(-1)
        diff = x - self.centers
        if self.circular:
            # Circular distance is used for periodic residual scales such as
            # mz mod 1.0 and mz mod 0.1, so the boundary wraps around.
            abs_diff = diff.abs()
            diff = torch.min(abs_diff, self.val_range - abs_diff)
        weights = torch.softmax(-diff.pow(2) / (self.alpha.abs() + 1e-6), dim=-1)
        return weights @ self.codebook.weight


class MultiResolutionMzCodebook(nn.Module):
    """4-level mz Codebook (Instant-NGP style):
    L0: 200 codes, ~10 Da/code  (global)
    L1: 2000 codes, ~1 Da/code  (integer mass)
    L2: 200 codes, sub-Da        (circular, period=1 Da)
    L3: 200 codes, fine isotope   (circular, period=0.1 Da)
    """

    def __init__(
        self,
        d_out: int,
        mz_max: float = 2000.0,
        l0_num: int = 200,
        l1_num: int = 2000,
        l2_num: int = 200,
        l3_num: int = 200,
    ):
        super().__init__()
        d_each = d_out // 4
        d_last = d_out - 3 * d_each

        self.level0 = SoftCodebookLookup(l0_num, d_each, 0.0, mz_max, alpha_init=100.0)
        self.level1 = SoftCodebookLookup(l1_num, d_each, 0.0, mz_max, alpha_init=1.0)
        self.level2 = SoftCodebookLookup(
            l2_num, d_each, 0.0, 1.0, alpha_init=0.01, circular=True
        )
        self.level3 = SoftCodebookLookup(
            l3_num, d_last, 0.0, 0.1, alpha_init=0.001, circular=True
        )

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        # The same mz is encoded at four resolutions:
        # global coarse position, ~1 Da position, sub-Da phase, and 0.1 Da
        # phase. Concatenating them gives a continuous multi-scale mz token.
        e0 = self.level0(mz)
        e1 = self.level1(mz)
        e2 = self.level2(torch.fmod(mz.abs(), 1.0))
        e3 = self.level3(torch.fmod(mz.abs(), 0.1))
        return torch.cat([e0, e1, e2, e3], dim=-1)


class CodebookIntensityEmbedding(nn.Module):
    def __init__(self, d_out: int, num_codes: int = 100, alpha_init: float = 0.1):
        super().__init__()
        self.lookup = SoftCodebookLookup(
            num_codes, d_out, 0.0, 1.0, alpha_init=alpha_init
        )

    def forward(self, intensity: torch.Tensor) -> torch.Tensor:
        return self.lookup(intensity)


class CodebookPeakEncoder(nn.Module):
    """Peak encoder: 4-level mz Codebook + Intensity Codebook + Type Emb → d_model."""

    def __init__(
        self,
        d_model: int,
        mz_dim: int = 512,
        int_dim: int = 512,
        int_num_codes: int = 100,
        mz_max: float = 2000.0,
        coarse_num: int = 2000,
    ):
        super().__init__()
        self.mz_dim = mz_dim
        self.int_dim = int_dim

        self.mz_codebook = MultiResolutionMzCodebook(
            mz_dim, mz_max=mz_max, l1_num=coarse_num
        )
        self.int_codebook = CodebookIntensityEmbedding(int_dim, int_num_codes)

        self.mz_type_emb = nn.Parameter(torch.zeros(mz_dim))
        self.int_type_emb = nn.Parameter(torch.zeros(int_dim))
        nn.init.normal_(self.mz_type_emb, std=0.02)
        nn.init.normal_(self.int_type_emb, std=0.02)

        self.proj = nn.Linear(mz_dim + int_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., 0] is mz and x[..., 1] is normalized intensity.
        # Both are embedded through soft codebooks, tagged by type embeddings,
        # then projected into the shared transformer hidden space.
        mz_emb = self.mz_codebook(x[..., 0]) + self.mz_type_emb
        int_emb = self.int_codebook(x[..., 1]) + self.int_type_emb
        return self.proj(torch.cat([mz_emb, int_emb], dim=-1))


# ============================================================================
# Section 2: Model
# ============================================================================
class UltraMSPretrainingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_peaks = config["max_peaks"]
        self.mz_range = config["mz_range"]
        self.mz_bin_size = config["mz_bin_size"]
        self.mz_num_bins = (
            int((self.mz_range[1] - self.mz_range[0]) / self.mz_bin_size) + 1
        )
        self.int_bin_size = config["int_bin_size"]
        self.int_num_bins = int(1.0 / self.int_bin_size)
        self.ms_start_idx = 2  # [CLS, PRECURSOR, peaks...]
        self.mask_ratio = config["mask_ratio"]
        self.int_loss_weight = config.get("int_loss_weight", 1.0)
        self.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)
        self.difficulty_weight_alpha = config.get("difficulty_weight_alpha", 2.0)

        d_model = config["d_model"]
        mz_codebook_dim = config["mz_codebook_dim"]
        int_codebook_dim = config["int_codebook_dim"]
        coarse_num = config.get("coarse_num", 2000)

        self.tokenizer = MsTokenizer(mz_precision=1.0, mz_range=self.mz_range)

        # Peak encoder (Codebook-only, no Fourier)
        self.peak_encoder = CodebookPeakEncoder(
            d_model=d_model,
            mz_dim=mz_codebook_dim,
            int_dim=int_codebook_dim,
            mz_max=float(self.mz_range[1]),
            coarse_num=coarse_num,
        )

        # Special token embeddings
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.precursor_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_emb, std=0.02)
        nn.init.normal_(self.precursor_type_emb, std=0.02)

        # Positional embedding
        max_seq_len = self.max_peaks + 2
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # BERT encoder
        bert_config = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=d_model,
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["dim_feedforward"],
            hidden_dropout_prob=config["dropout"],
            attention_probs_dropout_prob=config["dropout"],
            max_position_embeddings=max(max_seq_len * 2, 512),
            layer_norm_eps=1e-6,
            is_decoder=False,
            add_cross_attention=False,
        )
        self.encoder = BertModel(bert_config, add_pooling_layer=False)
        self.encoder.embeddings.word_embeddings.requires_grad_(False)
        self.dropout = nn.Dropout(config["dropout"])

        # SCFR heads (3-level mz prediction)
        self.mz_level0_bins = 200  # 10 Da/bin
        self.mz_level1_bins = 10  # 1 Da/bin within 10 Da
        self.mz_level2_bins = 10  # 0.1 Da/bin within 1 Da

        self.register_buffer("level0_centers", torch.linspace(5.0, 1995.0, 200))
        self.register_buffer("level1_offsets", torch.linspace(0.5, 9.5, 10))
        self.register_buffer("level2_offsets", torch.linspace(0.05, 0.95, 10))

        self.mz_level0_head = nn.Linear(d_model, self.mz_level0_bins)

        # SCFR conditioning: Codebook-only (mz_codebook_dim)
        cond_dim = mz_codebook_dim
        self.mz_level1_head = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.mz_level1_bins),
        )
        self.mz_level2_head = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.mz_level2_bins),
        )

        self.int_predictor = nn.Linear(d_model, self.int_num_bins)

    def forward(self, batch, device):
        masked = batch["masked_spectra"].to(device, non_blocking=True)
        orig = batch["orig_spectra"].to(device, non_blocking=True)
        mflags = batch["mask_flags"].to(device, non_blocking=True)
        ms_attn = batch["attn_mask"].to(device, non_blocking=True)
        pmz = batch["precursor_mz"].to(device, non_blocking=True)
        B, P = masked.shape[:2]

        # === Peak embedding (Codebook-only) ===
        ms_emb = self.peak_encoder(masked)  # (B, P, d_model)

        # === Special tokens ===
        cls_emb = self.cls_emb.expand(B, 1, -1)
        prec_input = torch.stack(
            [pmz, torch.full_like(pmz, 1.1)],
            dim=-1,
        ).unsqueeze(1)  # (B, 1, 2)
        prec_emb = self.peak_encoder(prec_input) + self.precursor_type_emb

        # === Sequence: [CLS, PRECURSOR, peaks] ===
        full_emb = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)  # (B, S, d)
        S = full_emb.shape[1]
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        full_emb = self.dropout(full_emb + self.pos_emb(pos_ids))

        prefix_attn = torch.ones(B, 2, dtype=ms_attn.dtype, device=device)
        full_attn = torch.cat([prefix_attn, ms_attn], dim=1)

        # === Encoder forward ===
        enc_out = self.encoder(
            inputs_embeds=full_emb,
            attention_mask=full_attn,
        )
        hs = enc_out.last_hidden_state  # (B, S, d)

        # === Masked-position gathering ===
        batch_idx, peak_idx = torch.where(mflags)
        if batch_idx.numel() == 0:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            z0 = torch.tensor(0.0)
            return {
                "loss": z,
                "mz_loss": z,
                "level0_loss": z,
                "level1_loss": z,
                "level2_loss": z,
                "int_loss": z,
                "mz_acc": z0,
                "level0_acc": z0,
                "level1_acc": z0,
                "level2_acc": z0,
                "mz_acc_1bin": z0,
                "mz_acc_5bin": z0,
                "int_acc": z0,
                "high_mz_acc": z0,
            }

        hidden_idx = peak_idx + self.ms_start_idx
        valid = hidden_idx < hs.shape[1]
        batch_idx, peak_idx, hidden_idx = (
            batch_idx[valid],
            peak_idx[valid],
            hidden_idx[valid],
        )

        hc = hs[batch_idx, hidden_idx]
        mzv = orig[batch_idx, peak_idx, 0]
        intv = orig[batch_idx, peak_idx, 1]

        int_bins = (
            torch.round(intv.clamp(0, 1) / self.int_bin_size)
            .long()
            .clamp(0, self.int_num_bins - 1)
        )

        # 3-level targets
        mz_offset = mzv.clamp(*self.mz_range) - self.mz_range[0]
        level0_target = (mz_offset / 10.0).long().clamp(0, self.mz_level0_bins - 1)
        level1_target = (mz_offset % 10.0).long().clamp(0, self.mz_level1_bins - 1)
        level2_target = (
            ((mz_offset % 1.0) / 0.1).round().long().clamp(0, self.mz_level2_bins - 1)
        )

        # === Level 0: coarse (200 bins × 10 Da) ===
        level0_logits = self.mz_level0_head(hc)

        # === SCFR conditioning with Teacher Forcing + Codebook-only ===
        with torch.no_grad():
            l0_probs = F.softmax(level0_logits.float(), dim=-1)
            soft_mz_0 = (l0_probs * self.level0_centers).sum(-1)

            if self.training and self.teacher_forcing_ratio > 0:
                tf_mask = (
                    torch.rand(soft_mz_0.shape, device=device)
                    < self.teacher_forcing_ratio
                )
                gt_mz_0 = level0_target.float() * 10.0 + 5.0
                cond_mz_0 = torch.where(tf_mask, gt_mz_0, soft_mz_0)
            else:
                cond_mz_0 = soft_mz_0

            # Re-encode the predicted coarse mz as a continuous codebook
            # embedding, then use that embedding as condition for finer levels.
            cb0 = self.peak_encoder.mz_codebook(cond_mz_0)

        # === Level 1 ===
        level1_logits = self.mz_level1_head(torch.cat([hc, cb0], dim=-1))

        with torch.no_grad():
            l1_probs = F.softmax(level1_logits.float(), dim=-1)
            snapped_0 = (cond_mz_0 / 10.0).floor() * 10.0
            soft_mz_1 = snapped_0 + (l1_probs * self.level1_offsets).sum(-1)

            if self.training and self.teacher_forcing_ratio > 0:
                tf_mask_1 = (
                    torch.rand(soft_mz_1.shape, device=device)
                    < self.teacher_forcing_ratio
                )
                gt_mz_1 = level0_target.float() * 10.0 + level1_target.float() + 0.5
                cond_mz_1 = torch.where(tf_mask_1, gt_mz_1, soft_mz_1)
            else:
                cond_mz_1 = soft_mz_1

            # The same codebook is reused here: prediction -> continuous mz ->
            # codebook embedding -> next refinement stage.
            cb1 = self.peak_encoder.mz_codebook(cond_mz_1)

        # === Level 2 ===
        level2_logits = self.mz_level2_head(torch.cat([hc, cb1], dim=-1))

        int_logits = self.int_predictor(hc)

        # === Difficulty-adaptive weighting ===
        with torch.no_grad():
            l0_entropy = -(l0_probs * (l0_probs + 1e-8).log()).sum(-1)
            max_entropy = math.log(self.mz_level0_bins)
            difficulty = (l0_entropy / max_entropy).clamp(0, 1)
            pw = (
                1.0
                + self.difficulty_weight_alpha * difficulty * (1.0 - difficulty) * 4.0
            )

        level0_loss = (
            F.cross_entropy(level0_logits, level0_target, reduction="none") * pw
        ).mean()
        level1_loss = (
            F.cross_entropy(level1_logits, level1_target, reduction="none") * pw
        ).mean()
        level2_loss = (
            F.cross_entropy(level2_logits, level2_target, reduction="none") * pw
        ).mean()

        mz_loss = level0_loss + level1_loss + level2_loss
        int_loss = F.cross_entropy(int_logits, int_bins)

        # === Metrics ===
        l0p = level0_logits.argmax(-1)
        l1p = level1_logits.argmax(-1)
        l2p = level2_logits.argmax(-1)

        level0_acc = (l0p == level0_target).float().mean()
        level1_acc = (l1p == level1_target).float().mean()
        level2_acc = (l2p == level2_target).float().mean()
        mz_acc = (
            ((l0p == level0_target) & (l1p == level1_target) & (l2p == level2_target))
            .float()
            .mean()
        )
        int_acc = (int_logits.argmax(-1) == int_bins).float().mean()

        l01_ok = (l0p == level0_target) & (l1p == level1_target)
        mz_acc_1bin = (l01_ok & ((l2p - level2_target).abs() <= 1)).float().mean()

        pred_mz = l0p.float() * 10.0 + l1p.float() + l2p.float() * 0.1
        gt_mz_d = (
            level0_target.float() * 10.0
            + level1_target.float()
            + level2_target.float() * 0.1
        )
        mz_acc_5bin = ((pred_mz - gt_mz_d).abs() <= 1.0).float().mean()

        sample_pmz = pmz[batch_idx]
        high_mz_mask = sample_pmz > 500
        if high_mz_mask.any():
            high_mz_acc = (
                (
                    (l0p[high_mz_mask] == level0_target[high_mz_mask])
                    & (l1p[high_mz_mask] == level1_target[high_mz_mask])
                    & (l2p[high_mz_mask] == level2_target[high_mz_mask])
                )
                .float()
                .mean()
            )
        else:
            high_mz_acc = mz_acc

        loss = mz_loss + self.int_loss_weight * int_loss
        return {
            "loss": loss,
            "mz_loss": mz_loss.detach(),
            "level0_loss": level0_loss.detach(),
            "level1_loss": level1_loss.detach(),
            "level2_loss": level2_loss.detach(),
            "int_loss": int_loss.detach(),
            "mz_acc": mz_acc.detach(),
            "level0_acc": level0_acc.detach(),
            "level1_acc": level1_acc.detach(),
            "level2_acc": level2_acc.detach(),
            "mz_acc_1bin": mz_acc_1bin.detach(),
            "mz_acc_5bin": mz_acc_5bin.detach(),
            "int_acc": int_acc.detach(),
            "high_mz_acc": high_mz_acc.detach(),
        }


# ============================================================================
# LR Schedule
# ============================================================================
def build_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
