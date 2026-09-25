"""
Neutral Loss Detection Probe — UltraMSMS vs DreaMS
multilabel binary classification: nl_{H2O, NH3, CO2, HF, HCl, CO, CH3}

Usage (from train/):
  Run through `benchmarks/spectral_properties/run_experiment.py neutral_loss_train_and_test`.
"""
import os, sys, math, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

_TRAIN = os.environ.get('ULTRAMS_SOURCE_TRAIN_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRAIN)
if os.environ.get('ULTRAMS_SOURCE_SUPPLEMENT_ROOT'):
    sys.path.insert(0, os.environ['ULTRAMS_SOURCE_SUPPLEMENT_ROOT'])

from showcase.data import load_fold, PARQUET_PATH

PARQUET       = PARQUET_PATH
NL_COLS       = ['nl_H2O', 'nl_NH3', 'nl_CO2', 'nl_HF', 'nl_HCl', 'nl_CO', 'nl_CH3']
NL_NAMES      = ['H2O',    'NH3',    'CO2',    'HF',    'HCl',    'CO',    'CH3']
PLOT_NAMES    = ['H2O', 'NH3', 'CO2', 'CO', 'CH3']   # top-5 common, skip HF/HCl

ULTRA_CKPT    = os.environ.get('ULTRAMS_STAGE_D_CKPT', os.path.join(_TRAIN, 'output/phase2_rt_only/stage_d_epoch_11.pt'))
ULTRA_KEY     = 'base_state_dict'
MAX_PEAKS     = 150
D_MODEL       = 1024


# ── Spectrum preprocessing ─────────────────────────────────────────────────────
def prep_spectrum(spec_np, max_peaks=MAX_PEAKS):
    if spec_np.ndim != 2 or spec_np.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    mz, inten = spec_np[:, 0], spec_np[:, 1]
    valid = mz > 0
    mz, inten = mz[valid], inten[valid]
    if len(mz) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inten = inten / (inten.max() + 1e-9)
    order = np.argsort(mz)
    return np.stack([mz[order], inten[order]], axis=-1).astype(np.float32)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_ultra(device):
    from train_ue_multiscale_v9_mlm import UltraExplorerMLM, CONFIG
    model = UltraExplorerMLM(CONFIG)
    ckpt  = torch.load(ULTRA_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt[ULTRA_KEY])
    print(f"  [Ultra] ep={ckpt.get('epoch','?')}, step={ckpt.get('global_step','?')}")
    return model.to(device).eval()


def load_dreams(device):
    sys.path.insert(0, os.path.join(_TRAIN, 'comparison'))
    from dreams_loader import load_dreams_encoder
    return load_dreams_encoder(device)


# ── Probe heads ────────────────────────────────────────────────────────────────
def build_head(d_in, n_out, dropout=0.1):
    return nn.Sequential(
        nn.LayerNorm(d_in), nn.Dropout(dropout),
        nn.Linear(d_in, d_in // 2), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(d_in // 2, n_out),
    )


class CrossAttnAgg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q   = nn.Linear(d, d, bias=False)
        self.kv  = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.scale = d ** -0.5

    def forward(self, hs, mask):          # hs (B,S,d)  mask (B,S) bool True=valid
        q = self.q(hs.mean(1, keepdim=True))   # (B,1,d)
        k, v = self.kv(hs).chunk(2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,1,S)
        attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = attn.softmax(-1)
        out  = (attn @ v).squeeze(1)
        return self.norm(self.out(out))


class UltraProbeNL(nn.Module):
    """Linear probe on top of frozen UltraExplorerMLM (CLS + aggregated peaks)."""
    def __init__(self, encoder, n_out, dropout=0.1):
        super().__init__()
        self.enc = encoder
        self.agg = CrossAttnAgg(D_MODEL)
        self.head = build_head(2 * D_MODEL, n_out, dropout)

    @torch.no_grad()
    def _encode(self, peaks_t, attn_t, pmz_t):
        """Replicates extract_v9 logic; returns (B, 2*d) representation."""
        B = peaks_t.shape[0]
        device = peaks_t.device
        ms_emb   = self.enc.peak_encoder(peaks_t)
        cls_emb  = self.enc.cls_emb.expand(B, 1, -1)
        pi       = torch.stack([pmz_t, torch.full_like(pmz_t, 1.1)], dim=-1).unsqueeze(1)
        prec_emb = self.enc.peak_encoder(pi) + self.enc.precursor_type_emb
        full     = torch.cat([cls_emb, prec_emb, ms_emb], dim=1)
        S        = full.shape[1]
        pos      = torch.arange(S, device=device).unsqueeze(0)
        full     = self.enc.dropout(full + self.enc.pos_emb(pos))
        fa       = torch.cat([torch.ones(B, 2, dtype=attn_t.dtype, device=device), attn_t], dim=1)
        out      = self.enc.encoder(inputs_embeds=full, attention_mask=fa)
        hs       = out.last_hidden_state          # (B, S, d)
        cls      = hs[:, 0, :]                    # (B, d)
        peak_hs  = hs[:, 2:, :]                   # (B, MAX_PEAKS, d)
        return cls, peak_hs

    def forward(self, batch, device):
        B = len(batch)
        padded = np.zeros((B, MAX_PEAKS, 2), dtype=np.float32)
        attn   = np.zeros((B, MAX_PEAKS),    dtype=np.int64)
        pmzs   = []
        for i, s in enumerate(batch):
            sp = prep_spectrum(np.asarray(s['spectrum'], dtype=np.float32))
            k  = min(len(sp), MAX_PEAKS)
            if k > 0:
                padded[i, :k] = sp[:k]
                attn[i, :k]   = 1
            pmzs.append(float(s['precursor_mz']))
        peaks_t = torch.from_numpy(padded).to(device)
        attn_t  = torch.from_numpy(attn).to(device)
        pmz_t   = torch.tensor(pmzs, dtype=torch.float32, device=device)

        cls, peak_hs = self._encode(peaks_t, attn_t, pmz_t)
        agg  = self.agg(peak_hs, attn_t.bool())
        feat = torch.cat([cls, agg], dim=-1)
        return self.head(feat)


class DreamsProbeNL(nn.Module):
    def __init__(self, encoder, n_out, dropout=0.1):
        super().__init__()
        self.enc  = encoder
        d = int(encoder.d_model)
        self.agg  = CrossAttnAgg(d)
        self.head = build_head(2 * d, n_out, dropout)

    def forward(self, batch, device):
        from dreams_loader import build_dreams_batch
        peaks_t  = build_dreams_batch(self.enc, batch, device)
        hs       = self.enc(peaks_t)
        cls      = hs[:, 0, :]
        peak_hs  = hs[:, 1:, :]
        peak_mask = (peaks_t[:, 1:, 0] != 0)
        agg  = self.agg(peak_hs, peak_mask)
        return self.head(torch.cat([cls, agg], dim=-1))


# ── Dataset ────────────────────────────────────────────────────────────────────
class NLDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels  = labels
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i], self.labels[i]

def collate(batch):
    return [b[0] for b in batch], torch.from_numpy(np.stack([b[1] for b in batch]))


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(probs, targets):
    aucs, aps = {}, {}
    for i, name in enumerate(NL_NAMES):
        if targets[:, i].sum() > 0:
            aucs[name] = float(roc_auc_score(targets[:, i], probs[:, i]))
            aps[name]  = float(average_precision_score(targets[:, i], probs[:, i]))
        else:
            aucs[name] = aps[name] = 0.0
    return {
        'mean_auc': float(np.mean(list(aucs.values()))),
        'mean_ap':  float(np.mean(list(aps.values()))),
        **{f'auc_{k}': v for k, v in aucs.items()},
        **{f'ap_{k}':  v for k, v in aps.items()},
    }


def run_eval(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch, labels in loader:
            logits = model(batch, device)
            all_logits.append(logits.cpu().float().numpy())
            all_targets.append(labels.numpy())
    logits  = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs   = 1 / (1 + np.exp(-logits))
    return probs, targets


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone',      choices=('ultra', 'dreams'), required=True)
    ap.add_argument('--device',        default='auto')
    ap.add_argument('--epochs',        type=int,   default=15)
    ap.add_argument('--batch-size',    type=int,   default=256)
    ap.add_argument('--head-lr',       type=float, default=3e-4)
    ap.add_argument('--weight-decay',  type=float, default=0.01)
    ap.add_argument('--grad-clip',     type=float, default=1.0)
    ap.add_argument('--warmup-ratio',  type=float, default=0.1)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--log-interval',  type=int,   default=30)
    ap.add_argument('--out-dir',       default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    device = torch.device(device_name)
    args.device = device_name

    ts = time.strftime('%m%d_%H%M')
    if args.out_dir is None:
        args.out_dir = os.path.join(_TRAIN, 'output', 'showcase',
                                    f'nl_{args.backbone}_{ts}')
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f'[{args.backbone}/neutral_loss]'
    print(f'{tag} out_dir={args.out_dir}', flush=True)
    print(f'{tag} Loading data...', flush=True)
    tr_s, tr_y, _ = load_fold(PARQUET, 'train', 'neutral_loss')
    va_s, va_y, _ = load_fold(PARQUET, 'val',   'neutral_loss')
    te_s, te_y, _ = load_fold(PARQUET, 'test',  'neutral_loss')
    print(f'  train={len(tr_s)}  val={len(va_s)}  test={len(te_s)}', flush=True)

    tr_loader = DataLoader(NLDataset(tr_s, tr_y), batch_size=args.batch_size,
                           shuffle=True,  collate_fn=collate, num_workers=0)
    va_loader = DataLoader(NLDataset(va_s, va_y), batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate, num_workers=0)
    te_loader = DataLoader(NLDataset(te_s, te_y), batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate, num_workers=0)

    # ── Load backbone (frozen) ────────────────────────────────────────────────
    print(f'{tag} Loading backbone...', flush=True)
    n_out = len(NL_NAMES)
    if args.backbone == 'ultra':
        enc   = load_ultra(device)
        for p in enc.parameters(): p.requires_grad_(False)
        model = UltraProbeNL(enc, n_out).to(device)
    else:
        enc   = load_dreams(device)
        for p in enc.parameters(): p.requires_grad_(False)
        model = DreamsProbeNL(enc, n_out).to(device)

    probe_params = [p for p in model.parameters() if p.requires_grad]
    print(f'  trainable params: {sum(p.numel() for p in probe_params):,}', flush=True)

    optimizer = torch.optim.AdamW(probe_params, lr=args.head_lr,
                                  weight_decay=args.weight_decay)
    total_steps  = len(tr_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_score   = -1e9
    best_ckpt    = os.path.join(args.out_dir, f'best_{args.backbone}.pt')
    best_probs_v = None
    history      = []
    gs = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for bi, (batch, labels) in enumerate(tr_loader, 1):
            labels = labels.float().to(device)
            logits = model(batch, device)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            gs += 1
            if bi % args.log_interval == 0:
                print(f'{tag} ep{ep}/{args.epochs} b{bi}/{len(tr_loader)} '
                      f'loss={loss.item():.4f} avg={running_loss/bi:.4f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)

        probs_v, targets_v = run_eval(model, va_loader, device)
        m = compute_metrics(probs_v, targets_v)
        per = '  '.join(f"ap_{n}={m[f'ap_{n}']:.3f}" for n in NL_NAMES)
        print(f'{tag} ep{ep}  loss={running_loss/len(tr_loader):.4f}  '
              f"mean_ap={m['mean_ap']:.4f}  mean_auc={m['mean_auc']:.4f}  [{per}]",
              flush=True)
        history.append({'ep': ep, 'train_loss': running_loss / len(tr_loader), **m})

        if m['mean_ap'] > best_score:
            best_score   = m['mean_ap']
            best_probs_v = probs_v.copy()
            torch.save({'epoch': ep, 'val_mean_ap': best_score,
                        'head': model.head.state_dict(),
                        'agg':  model.agg.state_dict(),
                        'backbone': args.backbone}, best_ckpt)
            print(f'  ★ best  mean_ap={best_score:.4f}', flush=True)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f'\n{tag} Loading best checkpoint for test...', flush=True)
    ck = torch.load(best_ckpt, map_location=device)
    model.head.load_state_dict(ck['head'])
    model.agg.load_state_dict(ck['agg'])

    probs_te, targets_te = run_eval(model, te_loader, device)
    m_te = compute_metrics(probs_te, targets_te)
    print(f'{tag} TEST  mean_ap={m_te["mean_ap"]:.4f}  mean_auc={m_te["mean_auc"]:.4f}',
          flush=True)
    for n in NL_NAMES:
        print(f'  {n:6s}  auc={m_te[f"auc_{n}"]:.3f}  ap={m_te[f"ap_{n}"]:.3f}', flush=True)

    # ── Save intermediate results ──────────────────────────────────────────────
    out = args.out_dir
    np.save(os.path.join(out, 'val_probs.npy'),    best_probs_v)
    np.save(os.path.join(out, 'val_targets.npy'),  targets_v)
    np.save(os.path.join(out, 'test_probs.npy'),   probs_te)
    np.save(os.path.join(out, 'test_targets.npy'), targets_te)

    # ROC curve data for top-5 types
    roc_data = {}
    for name in PLOT_NAMES:
        i = NL_NAMES.index(name)
        if targets_te[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(targets_te[:, i], probs_te[:, i])
            roc_data[name] = {
                'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                'auc': float(roc_auc_score(targets_te[:, i], probs_te[:, i])),
            }

    with open(os.path.join(out, 'roc_data.json'), 'w') as f:
        json.dump(roc_data, f)

    meta = {**vars(args), 'nl_names': NL_NAMES, 'history': history,
            'best_val_mean_ap': best_score, 'test_metrics': m_te}
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{tag} Saved:', flush=True)
    print(f'  val_probs/targets.npy  test_probs/targets.npy', flush=True)
    print(f'  roc_data.json  ({PLOT_NAMES})', flush=True)
    print(f'  config.json', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
