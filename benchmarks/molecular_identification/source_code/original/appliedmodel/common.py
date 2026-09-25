"""Shared paths, backbone loaders, and embedding helpers for applied models."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.environ.get("LIGHT_ULTRA_ROOT", os.path.dirname(_HERE))
_TRAIN_DIR = os.path.join(_ROOT_DIR, "train")
_PUBLIC_MODEL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "..", "training", "ultrams_training", "model"))

sys.path.insert(0, os.path.join(_PUBLIC_MODEL_ROOT, "train"))
sys.path.insert(0, _PUBLIC_MODEL_ROOT)

MAX_PEAKS = 150
ULTRA_STAGE_D_CKPT = os.environ.get(
    "ULTRA_STAGE_D_CKPT",
    os.path.join(_TRAIN_DIR, "output", "phase2_rt_only", "stage_d_epoch_11.pt"),
)
CHEMBERTA_PATH = os.environ.get(
    "CHEMBERTA_PATH",
    os.path.join(_ROOT_DIR, "model", "feature", "ChemBERTa-100M-MLM"),
)
DEFAULT_EMBED_CACHE_DIR = os.path.join(_ROOT_DIR, "output", "appliedmodel", "cache", "embeddings")
EMBED_CACHE_VERSION = 1


def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def get_backbone_hidden_dim(model: nn.Module) -> int:
    if hasattr(model, "d_model"):
        return int(model.d_model)
    if hasattr(model, "cls_emb"):
        return int(model.cls_emb.shape[-1])
    raise AttributeError(f"Cannot infer hidden dim for {type(model).__name__}.")


def unfreeze_ultra_last_n(model: nn.Module, n: int) -> None:
    if n <= 0:
        return
    layers = model.encoder.encoder.layer
    start = max(0, len(layers) - n)
    for i in range(start, len(layers)):
        for p in layers[i].parameters():
            p.requires_grad_(True)


def unfreeze_dreams_last_n(model: nn.Module, n: int) -> None:
    if n <= 0:
        return
    L = model.n_layers
    start = max(0, L - n)
    if model.vanilla_transformer:
        for i in range(start, L):
            for p in model.transformer_encoder.layers[i].parameters():
                p.requires_grad_(True)
        return

    te = model.transformer_encoder
    for i in range(start, L):
        for p in te.atts[i].parameters():
            p.requires_grad_(True)
        for p in te.ffs[i].parameters():
            p.requires_grad_(True)
    if hasattr(te, "scales") and te.scales is not None:
        for i in range(start, L):
            for idx in (2 * i, 2 * i + 1):
                if idx < len(te.scales):
                    for p in te.scales[idx].parameters():
                        p.requires_grad_(True)
        for p in te.scales[-1].parameters():
            p.requires_grad_(True)


def load_ultra_backbone(
    device: torch.device | str,
    checkpoint_path: str | None = None,
    state_key: str = "base_state_dict",
) -> nn.Module:
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG

    checkpoint_path = checkpoint_path or ULTRA_STAGE_D_CKPT
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Ultra backbone checkpoint not found: "
            f"{checkpoint_path}. Pass --ultra-ckpt or set ULTRA_STAGE_D_CKPT."
        )
    model = UltraExplorerMLM(CONFIG)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state_key not in ckpt:
        raise KeyError(
            f"Expected checkpoint key '{state_key}' in {checkpoint_path}, "
            f"but only found keys: {sorted(ckpt.keys())[:20]}"
        )
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    if missing or unexpected:
        print(f"[Ultra] missing={missing} unexpected={unexpected}")
    print(
        f"[Ultra] Loaded {checkpoint_path}\n"
        f"  state_key={state_key} ep={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}"
    )
    return model.to(device).eval()


def load_dreams_backbone(
    device: torch.device | str,
    checkpoint_path: str | None = None,
) -> nn.Module:
    from .dreams_loader import load_dreams_encoder

    return load_dreams_encoder(device, ckpt_path=checkpoint_path) if checkpoint_path else load_dreams_encoder(device)


def _normalize_and_pad(spec_np: np.ndarray, max_peaks: int) -> np.ndarray:
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    mx = float(inten.max())
    if mx > 0:
        inten = inten / mx
    inten = np.clip(inten, 0.0, 1.0)
    if len(mz) > max_peaks:
        idx = np.sort(np.argsort(inten)[-max_peaks:])
        mz, inten = mz[idx], inten[idx]
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


def ultra_prepare(
    batch: List[dict],
    device: torch.device,
    max_peaks: int = MAX_PEAKS,
):
    B = len(batch)
    padded = np.zeros((B, max_peaks, 2), dtype=np.float32)
    attn = np.zeros((B, max_peaks), dtype=np.int64)
    pmzs = np.zeros(B, dtype=np.float32)
    for i, sample in enumerate(batch):
        sp = _normalize_and_pad(np.asarray(sample["spectrum"], dtype=np.float32), max_peaks)
        k = min(len(sp), max_peaks)
        if k > 0:
            padded[i, :k] = sp[:k]
            attn[i, :k] = 1
        pmzs[i] = float(sample["precursor_mz"])
    return (
        torch.from_numpy(padded).to(device),
        torch.from_numpy(attn).to(device),
        torch.from_numpy(pmzs).to(device),
    )


def ultra_forward_tokens(
    model: nn.Module,
    peaks_t: torch.Tensor,
    attn_t: torch.Tensor,
    pmz_t: torch.Tensor,
):
    B = peaks_t.shape[0]
    device = peaks_t.device
    ms_emb = model.peak_encoder(peaks_t)
    cls_emb = model.cls_emb.expand(B, 1, -1)
    pi = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], dim=-1).unsqueeze(1)
    prec_emb = model.peak_encoder(pi) + model.precursor_type_emb
    full = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
    S = full.shape[1]
    pos = torch.arange(S, device=device).unsqueeze(0)
    full = model.dropout(full + model.pos_emb(pos))
    full_attn = torch.cat(
        [torch.ones(B, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1
    )
    out = model.encoder(inputs_embeds=full, attention_mask=full_attn)
    hs = out.last_hidden_state
    return hs[:, 0, :], hs[:, 2:, :]


def forward_ultra_batch(
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
) -> torch.Tensor:
    peaks_t, attn_t, pmz_t = ultra_prepare(samples, device, max_peaks=model.max_peaks)
    cls, _ = ultra_forward_tokens(model, peaks_t, attn_t, pmz_t)
    return cls


def forward_dreams_batch(
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
) -> torch.Tensor:
    from .dreams_loader import build_dreams_batch

    peaks_t = build_dreams_batch(model, samples, device)
    hs = model(peaks_t)
    return hs[:, 0, :]


def forward_backbone_batch(
    backbone: str,
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
) -> torch.Tensor:
    if backbone == "ultra":
        return forward_ultra_batch(model, samples, device)
    if backbone == "dreams":
        return forward_dreams_batch(model, samples, device)
    raise ValueError(f"Unsupported backbone: {backbone}")


def encode_ultra(
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
    batch_size: int = 128,
    no_grad: bool = True,
) -> np.ndarray:
    outs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        if no_grad:
            with torch.no_grad():
                cls = forward_ultra_batch(model, batch, device)
        else:
            cls = forward_ultra_batch(model, batch, device)
        outs.append(cls.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1024), dtype=np.float32)


def encode_dreams(
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
    batch_size: int = 128,
    no_grad: bool = True,
) -> np.ndarray:
    outs = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        if no_grad:
            with torch.no_grad():
                cls = forward_dreams_batch(model, batch, device)
        else:
            cls = forward_dreams_batch(model, batch, device)
        outs.append(cls.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 1024), dtype=np.float32)


def encode_backbone(
    backbone: str,
    model: nn.Module,
    samples: List[dict],
    device: torch.device,
    batch_size: int = 128,
    no_grad: bool = True,
) -> np.ndarray:
    if backbone == "ultra":
        return encode_ultra(model, samples, device, batch_size=batch_size, no_grad=no_grad)
    if backbone == "dreams":
        return encode_dreams(model, samples, device, batch_size=batch_size, no_grad=no_grad)
    raise ValueError(f"Unsupported backbone: {backbone}")


def _stable_smiles_vocab_hash(smiles_vocab: List[str]) -> str:
    digest = hashlib.sha1()
    digest.update(f"{len(smiles_vocab)}\n".encode("utf-8"))
    for smi in smiles_vocab:
        digest.update(smi.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def precompute_shared_molecule_embeddings(
    smiles_vocab: List[str],
    device: torch.device,
    batch_size: int = 256,
    pretrained_path: str = CHEMBERTA_PATH,
    cache_dir: str = DEFAULT_EMBED_CACHE_DIR,
    log_prefix: str = "[mol-cache]",
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    encoder_ref = os.path.realpath(pretrained_path)
    smiles_hash = _stable_smiles_vocab_hash(smiles_vocab)
    encoder_hash = hashlib.sha1(encoder_ref.encode("utf-8")).hexdigest()[:12]
    stem = f"chemberta_{encoder_hash}_{len(smiles_vocab)}_{smiles_hash[:16]}"
    cache_path = os.path.join(cache_dir, f"{stem}.npy")
    meta_path = os.path.join(cache_dir, f"{stem}.json")
    if os.path.isfile(cache_path) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
        if (
            meta.get("cache_version") == EMBED_CACHE_VERSION
            and meta.get("encoder_path") == encoder_ref
            and meta.get("n_smiles") == len(smiles_vocab)
            and meta.get("smiles_hash") == smiles_hash
        ):
            print(f"{log_prefix} reuse molecule emb -> {cache_path}")
            return cache_path

    encoder = MolEncoder(pretrained_path).to(device).eval()
    emb_dim = encoder.hidden_dim
    emb = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(len(smiles_vocab), emb_dim)
    )
    for start in range(0, len(smiles_vocab), batch_size):
        end = min(start + batch_size, len(smiles_vocab))
        batch = smiles_vocab[start:end]
        emb[start:end] = encoder.encode_batch(batch, device).cpu().numpy()
        if start % 50000 == 0:
            print(f"{log_prefix} molecule emb {start:,}/{len(smiles_vocab):,}")
    emb.flush()
    with open(meta_path, "w") as fh:
        json.dump(
            {
                "cache_version": EMBED_CACHE_VERSION,
                "encoder_path": encoder_ref,
                "n_smiles": len(smiles_vocab),
                "smiles_hash": smiles_hash,
                "emb_dim": emb_dim,
            },
            fh,
            indent=2,
        )
    return cache_path


class MolEncoder(nn.Module):
    def __init__(self, pretrained_path: str = CHEMBERTA_PATH):
        super().__init__()
        from transformers import AutoTokenizer, RobertaModel

        self.bert = RobertaModel.from_pretrained(pretrained_path, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.hidden_dim = int(self.bert.config.hidden_size)

    @torch.no_grad()
    def encode_batch(self, smiles_list: Iterable[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            list(smiles_list),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        out = self.bert(**encoded, return_dict=True)
        return out.last_hidden_state[:, 0, :]
