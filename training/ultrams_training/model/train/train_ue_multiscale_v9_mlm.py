"""
UltraExplorer MLM v9 — Codebook-Only SCFR + Teacher Forcing + Difficulty Weighting
==================================================================================
v9 核心改进 (基于 v6):
  1. 去掉 Fourier Features, 使用 4 级多分辨率 Codebook (Instant-NGP 风格)
     - Level 0: 200 codes, ~10 Da/code (全局粗粒度)
     - Level 1: 2000 codes, ~1 Da/code (整数质量)
     - Level 2: 200 codes, 亚 Da (循环距离, period=1 Da)
     - Level 3: 200 codes, 精细同位素 (循环距离, period=0.1 Da)
     - 每个 code 有独立可学习温度 (per-code alpha)
  2. Teacher Forcing: 训练时 50% 概率用 GT 做 SCFR 条件, 加速级联启动
  3. 难度自适应加权: L0 熵 → bell curve 权重, 替代 PMZ 线性加权
  4. 保留三级 SCFR mz 预测, 条件路径改为 Codebook-only
"""

import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.MS_decoder_v3 import MsTokenizer
from transformers import BertModel, BertConfig
from pretrain.dataset_sharded import (
    create_parquet_dataloader_mlm,
    get_parquet_shard_info,
)

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
    # Training
    "batch_size": 640,
    "lr": 3e-5,
    "min_lr": 5e-7,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "num_epochs": 5,
    "gradient_clip": 5.0,
    "accumulation_steps": 1,
    "resume_from": None,
    # Data
    "shard_dir": "derived/ultramsdata_clean/shards",
    "num_workers": 4,
    # DDP
    "world_size": 6,
    # Logging
    "log_interval": 50,
    "save_step_interval": 5000,
    "wandb_project": "UltraMs-pretrain",
    "wandb_run_name": "ue-mlm-v9-codebook-SCFR-tf-dw",
    # Output
    "output_dir": "./output/ue_multiscale_v9",
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
class UltraExplorerMLM(nn.Module):
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


# ============================================================================
# Training
# ============================================================================
def train(rank, world_size, config):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # --- Data ---
    shard_dir = config["shard_dir"]
    if rank == 0:
        info = get_parquet_shard_info(shard_dir)
        print(
            f"[Data] shards={info['num_shards']}, samples={info['total_samples']:,}",
            flush=True,
        )

    dataloader = create_parquet_dataloader_mlm(
        shard_dir=shard_dir,
        batch_size=config["batch_size"],
        max_peaks=config["max_peaks"],
        mask_ratio=config["mask_ratio"],
        rank=rank,
        world_size=world_size,
        num_workers=config["num_workers"],
    )

    # --- Model ---
    model = UltraExplorerMLM(config).to(device)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] UltraExplorer v9 — Codebook-Only SCFR", flush=True)
        print(
            f"  d_model={config['d_model']}, layers={config['num_layers']}, "
            f"heads={config['num_heads']}, FFN={config['dim_feedforward']}",
            flush=True,
        )
        print(f"  params={n:,} ({n / 1e6:.1f}M), trainable={n_train:,}", flush=True)
        print(
            f"  mz_codebook={config['mz_codebook_dim']}, int_codebook={config['int_codebook_dim']}",
            flush=True,
        )
        print(
            f"  teacher_forcing={config['teacher_forcing_ratio']}, "
            f"difficulty_alpha={config['difficulty_weight_alpha']}",
            flush=True,
        )
        print(
            f"  mask_ratio={config['mask_ratio']}, int_loss_weight={config.get('int_loss_weight', 1.0)}",
            flush=True,
        )

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # --- Optimizer & Scheduler ---
    no_decay = [
        "bias",
        "LayerNorm",
        "layer_norm",
        "Embedding",
        "embedding",
        "codebook",
        "alpha",
        "type_emb",
        "cls_emb",
        "precursor_type_emb",
    ]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    estimated_steps_per_epoch = 53000
    total_steps = estimated_steps_per_epoch * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    min_lr_ratio = config["min_lr"] / config["lr"]
    scheduler = build_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)

    # --- Resume from checkpoint ---
    start_epoch = 1
    global_step_offset = 0
    resume_path = config.get("resume_from", None)
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        ckpt_sd = ckpt["model_state_dict"]
        model_sd = raw_model.state_dict()
        filtered_sd = {
            k: v
            for k, v in ckpt_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        skipped = [k for k in ckpt_sd if k not in filtered_sd]
        missing, _ = raw_model.load_state_dict(filtered_sd, strict=False)
        if rank == 0 and skipped:
            print(
                f"[Resume] Skipped {len(skipped)} shape-mismatched keys (re-initialized)",
                flush=True,
            )
        if rank == 0 and missing:
            print(
                f"[Resume] {len(missing)} new keys initialized from scratch", flush=True
            )
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            if rank == 0:
                print(
                    f"[Resume] Optimizer state incompatible, reinitializing: {e}",
                    flush=True,
                )
        global_step_offset = ckpt["global_step"]
        ckpt_epoch = ckpt["epoch"]

        steps_per_epoch = ckpt.get("steps_per_epoch", estimated_steps_per_epoch)
        if ckpt_epoch > 0 and global_step_offset >= steps_per_epoch:
            steps_per_epoch = round(global_step_offset / ckpt_epoch)

        epoch_steps_done = global_step_offset - (ckpt_epoch - 1) * steps_per_epoch
        if epoch_steps_done >= steps_per_epoch * 0.95:
            start_epoch = ckpt_epoch + 1
        else:
            start_epoch = ckpt_epoch

        new_total = steps_per_epoch * config["num_epochs"]
        new_warmup = int(new_total * config["warmup_ratio"])
        scheduler = build_lr_scheduler(optimizer, new_warmup, new_total, min_lr_ratio)
        for _ in range(global_step_offset):
            scheduler.step()
        if rank == 0:
            print(
                f"[Resume] Loaded checkpoint: epoch={ckpt_epoch}, "
                f"global_step={global_step_offset}, start_epoch={start_epoch}",
                flush=True,
            )
            print(
                f"[Resume] Scheduler rebuilt: steps_per_epoch={steps_per_epoch}, "
                f"total={new_total}, warmup={new_warmup}",
                flush=True,
            )
        del ckpt

    # --- wandb ---
    if rank == 0:
        try:
            import wandb

            wandb.login()
            wandb.init(
                project=config["wandb_project"],
                name=config["wandb_run_name"],
                config=config,
            )
            use_wandb = True
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}", flush=True)
            use_wandb = False

        os.makedirs(config["output_dir"], exist_ok=True)
        with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(
            f"[Train] epochs={config['num_epochs']}, batch={config['batch_size']}, "
            f"lr={config['lr']}, int_loss_weight={config.get('int_loss_weight', 1.0)}, "
            f"{world_size}GPU DDP",
            flush=True,
        )

    # --- Train Loop ---
    model.train()
    global_step = global_step_offset
    t0 = time.time()
    accum = config["accumulation_steps"]
    scheduler_recalibrated = resume_path is not None and os.path.exists(
        resume_path or ""
    )

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        if hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch)

        running = {}
        running_count = 0
        epoch_step = 0
        epoch_t0 = time.time()

        if rank == 0:
            print(f"\n{'=' * 60}", flush=True)
            print(
                f"Epoch {epoch}/{config['num_epochs']} | global_step={global_step}",
                flush=True,
            )
            print(f"{'=' * 60}", flush=True)

        data_iter = iter(dataloader)
        while True:
            try:
                batch = next(data_iter)
                has_data = torch.ones(1, device=device)
            except StopIteration:
                has_data = torch.zeros(1, device=device)
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0:
                break

            with autocast("cuda", dtype=torch.bfloat16):
                res = model(batch, device)

            loss = res["loss"] / accum
            loss.backward()

            if (epoch_step + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            epoch_step += 1

            for k in [
                "loss",
                "mz_loss",
                "level0_loss",
                "level1_loss",
                "level2_loss",
                "int_loss",
                "mz_acc",
                "level0_acc",
                "level1_acc",
                "level2_acc",
                "mz_acc_1bin",
                "mz_acc_5bin",
                "int_acc",
                "high_mz_acc",
            ]:
                v = res[k]
                val = v.item() if isinstance(v, torch.Tensor) else v
                running[k] = running.get(k, 0) + val
            running_count += 1

            if rank == 0 and running_count % config["log_interval"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                parts = " | ".join(
                    f"{k}={running[k] / running_count:.4f}" for k in running
                )
                print(
                    f"  step {epoch_step} (global {global_step}) | "
                    f"lr={lr:.2e} | {parts}",
                    flush=True,
                )

                if use_wandb:
                    log_dict = {
                        f"train/{k}": running[k] / running_count for k in running
                    }
                    log_dict["train/lr"] = lr
                    log_dict["train/epoch"] = epoch
                    log_dict["train/global_step"] = global_step
                    wandb.log(log_dict, step=global_step)

                running = {}
                running_count = 0

            save_interval = config.get("save_step_interval", 0)
            if save_interval and global_step > 0 and global_step % save_interval == 0:
                dist.barrier()
                if rank == 0:
                    ckpt_path = os.path.join(
                        config["output_dir"], f"checkpoint_step_{global_step}.pt"
                    )
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config,
                        },
                        ckpt_path,
                    )
                    print(
                        f"  [Checkpoint] step {global_step} saved: {ckpt_path}",
                        flush=True,
                    )
                dist.barrier()

        # --- Epoch end ---
        epoch_time = time.time() - epoch_t0

        if not scheduler_recalibrated and epoch == 1:
            actual_steps_per_epoch = epoch_step // accum
            new_total = actual_steps_per_epoch * config["num_epochs"]
            new_warmup = int(new_total * config["warmup_ratio"])
            total_steps = new_total
            scheduler = build_lr_scheduler(
                optimizer, new_warmup, new_total, min_lr_ratio
            )
            for _ in range(global_step):
                scheduler.step()
            scheduler_recalibrated = True
            if rank == 0:
                print(
                    f"  [Scheduler] Calibrated: steps_per_epoch={actual_steps_per_epoch}, "
                    f"total={new_total}, warmup={new_warmup}",
                    flush=True,
                )

        if rank == 0:
            print(
                f"  Epoch {epoch} done in {epoch_time:.0f}s ({epoch_time / 3600:.1f}h), "
                f"steps={epoch_step}, global_step={global_step}",
                flush=True,
            )

            ckpt_path = os.path.join(
                config["output_dir"], f"checkpoint_epoch_{epoch}.pt"
            )
            raw_model = model.module if hasattr(model, "module") else model
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

        dist.barrier()

    total_time = time.time() - t0
    if rank == 0:
        print(
            f"\nTraining done in {total_time:.0f}s ({total_time / 3600:.1f}h), "
            f"global_step={global_step}",
            flush=True,
        )
        if use_wandb:
            wandb.finish()

    dist.destroy_process_group()


# ============================================================================
# Entry
# ============================================================================
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"
    os.environ["NCCL_ALGO"] = ""
    world_size = CONFIG["world_size"]

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    import torch.multiprocessing as mp

    mp.spawn(train, args=(world_size, CONFIG), nprocs=world_size, join=True)
