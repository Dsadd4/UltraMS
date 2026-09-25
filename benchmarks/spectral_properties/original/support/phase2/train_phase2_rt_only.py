"""
Phase 2 Training: RT-Only (no VICReg)
Stage Ae: HQ data RT + MLM (no VICReg)
Stage C: 160M incremental learning with self-consistent RT

Usage:
  python phase2/train_phase2_rt_only.py --stage Ae --gpus 6
  python phase2/train_phase2_rt_only.py --stage C --gpus 6
"""
import os, sys, time, math, json, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get('ULTRAMS_SOURCE_SUPPLEMENT_ROOT'):
    sys.path.insert(0, os.environ['ULTRAMS_SOURCE_SUPPLEMENT_ROOT'])

from train_ue_multiscale_v9_mlm import (
    UltraExplorerMLM, CONFIG as V9_CONFIG, build_lr_scheduler,
)
from pretrain.dataset_sharded import create_parquet_dataloader_mlm

PHASE2_CONFIG = {
    **V9_CONFIG,
    'phase1_checkpoint': './output/ue_multiscale_v9/checkpoint_epoch_3.pt',
    'output_dir': './output/phase2_rt_only',
    'freeze_layers': 12,
    'rt_weight': 0.5, 'rt_huber_delta': 0.1,
    'mlm_weight': 0.3,
    'lr': 1e-5, 'min_lr': 1e-7, 'warmup_ratio': 0.05,
    'batch_size': 192, 'num_epochs': 5,
    'grad_accum_steps': 3,
    'log_interval': 50, 'save_step_interval': 2000,
    'gradient_clip': 5.0,
    'rt_norm_scale': 600.0,
    'rt_max_seconds': 1500.0,
    'rt_self_consist_tau': 0.1,
    'stage_c_epochs': 2,
}


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


class Phase2Model(nn.Module):
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

    def forward(self, batch, device, mode='stage_ae'):
        res = {}
        mlm_r = self.base(batch, device)
        res['mlm_loss'] = mlm_r['loss'].detach()
        for k in ['mz_acc', 'level0_acc', 'level1_acc', 'level2_acc', 'int_acc']:
            if k in mlm_r:
                res[k] = mlm_r[k]
        total = self.config['mlm_weight'] * mlm_r['loss']

        ma = batch['attn_mask'].to(device, non_blocking=True)
        pm = batch['precursor_mz'].to(device, non_blocking=True)
        og = batch['orig_spectra'].to(device, non_blocking=True)
        _, cls = self._encode(og, ma, pm, device)
        prt = self.rt_head(cls)
        ppol = self.pol_head(cls)
        dummy_rt = (prt * 0).sum()
        dummy_pol = (ppol * 0).sum()
        total = total + dummy_rt + dummy_pol

        rt_t = batch.get('rt')
        if rt_t is not None:
            if isinstance(rt_t, torch.Tensor):
                rt_t = rt_t.to(device)
            else:
                rt_t = torch.tensor(rt_t, dtype=torch.float32, device=device)
            rt_max = self.config.get('rt_max_seconds', 1500.0)
            if mode in ('stage_c', 'stage_d'):
                rt_t = rt_t / self.config['rt_norm_scale']
                ok = (rt_t > 0) & (rt_t <= rt_max / self.config['rt_norm_scale'])
            else:
                ok = (rt_t > 0) & (rt_t <= rt_max)
            if ok.sum() >= 2:
                rl = F.huber_loss(prt[ok], rt_t[ok], delta=self.config['rt_huber_delta'])
                scale = self.config['rt_norm_scale'] if mode in ('stage_c', 'stage_d') else 1.0
                res['rt_mae'] = (prt[ok] - rt_t[ok]).abs().mean().detach() * scale
                res['rt_n'] = int(ok.sum())
                if mode in ('stage_c', 'stage_d'):
                    tau = self.config.get('rt_self_consist_tau', 0.1)
                    with torch.no_grad():
                        self_err = (prt.detach() - rt_t).abs()
                        sc_mask = self_err < tau
                    cm = ok & sc_mask
                    if cm.sum() >= 2:
                        rl = F.huber_loss(prt[cm], rt_t[cm], delta=self.config['rt_huber_delta'])
                        res['rt_mae_sc'] = (prt[cm] - rt_t[cm]).abs().mean().detach() * scale
                    else:
                        rl = torch.tensor(0.0, device=device, requires_grad=True)
                    res['rt_n_sc'] = int(cm.sum())
                res['rt_loss'] = rl.detach()
                total = total + self.config['rt_weight'] * rl

        if mode == 'stage_d':
            pol_t = batch.get('polarity')
            if pol_t is not None:
                if isinstance(pol_t, torch.Tensor):
                    pol_t = pol_t.to(device)
                else:
                    pol_t = torch.tensor(pol_t, dtype=torch.long, device=device)
                ok_pol = (pol_t >= 0)
                if ok_pol.sum() >= 2:
                    pol_loss = F.binary_cross_entropy_with_logits(
                        ppol[ok_pol], pol_t[ok_pol].float())
                    res['pol_loss'] = pol_loss.detach()
                    with torch.no_grad():
                        pred = (ppol[ok_pol] > 0).long()
                        res['pol_acc'] = (pred == pol_t[ok_pol]).float().mean()
                        res['pol_n'] = int(ok_pol.sum())
                    total = total + self.config.get('pol_weight', 0.1) * pol_loss

        res['loss'] = total
        return res


def _safe_scalar(v):
    if isinstance(v, torch.Tensor):
        return v.item()
    return float(v) if v else 0.0


def train_stage_ae(rank, world_size, cfg):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}')
    from phase2.dataset_hq import create_hq_dataloader
    if rank == 0:
        print(f"\n{'='*60}\nStage Ae: RT+MLM on HQ Data (no VICReg)\n{'='*60}")
    loader, ds = create_hq_dataloader(
        batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
        max_peaks=cfg['max_peaks'], rank=rank, world_size=world_size)
    base = UltraExplorerMLM(V9_CONFIG)
    ck = torch.load(cfg['phase1_checkpoint'], map_location='cpu', weights_only=False)
    base.load_state_dict(ck['model_state_dict'])
    if rank == 0:
        print(f"  Phase1 ckpt: epoch={ck.get('epoch','?')}, step={ck.get('global_step','?')}")
    del ck
    model = Phase2Model(base, cfg).to(dev)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    pg = [
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and 'LayerNorm' not in n and 'bias' not in n],
         'weight_decay': cfg['weight_decay']},
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and ('LayerNorm' in n or 'bias' in n)],
         'weight_decay': 0.0}]
    opt = torch.optim.AdamW(pg, lr=cfg['lr'])
    spe = len(loader)
    tot = spe * cfg['num_epochs']
    wu = int(tot * cfg['warmup_ratio'])
    sch = build_lr_scheduler(opt, wu, tot, cfg['min_lr']/cfg['lr'])
    scaler = GradScaler()
    os.makedirs(cfg['output_dir'], exist_ok=True)
    lp = os.path.join(cfg['output_dir'], 'stage_ae.log')
    ga = cfg.get('grad_accum_steps', 1)
    if rank == 0:
        nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Train params: {nt:,}, Steps/ep: {spe}, Total: {tot}, Warmup: {wu}")
        print(f"  batch_size={cfg['batch_size']}, grad_accum={ga}, eff_batch={cfg['batch_size']*ga}")
    gs = 0
    for ep in range(1, cfg['num_epochs']+1):
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(ep)
        model.train()
        micro = 0
        for batch in loader:
            with autocast('cuda', dtype=torch.bfloat16):
                r = model(batch, dev, mode='stage_ae')
            scaler.scale(r['loss'] / ga).backward()
            micro += 1
            if micro % ga != 0:
                continue
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sch.step()
            gs += 1
            if rank == 0 and gs % cfg['log_interval'] == 0:
                msg = (f"[Ae] ep={ep} step={gs} lr={opt.param_groups[0]['lr']:.2e} "
                       f"loss={r['loss'].item():.4f} mlm={_safe_scalar(r.get('mlm_loss')):.4f} "
                       f"rt_mae={_safe_scalar(r.get('rt_mae')):.1f}s "
                       f"rt_n={r.get('rt_n',0)} mz_acc={_safe_scalar(r.get('mz_acc')):.4f}")
                print(msg, flush=True)
                with open(lp, 'a') as f:
                    f.write(msg+'\n')
            if rank == 0 and gs % cfg['save_step_interval'] == 0:
                raw = model.module
                torch.save({'model_state_dict': raw.state_dict(),
                            'base_state_dict': raw.base.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'epoch': ep, 'global_step': gs, 'config': cfg},
                           os.path.join(cfg['output_dir'], f'stage_ae_step_{gs}.pt'))
        if rank == 0:
            raw = model.module
            torch.save({'model_state_dict': raw.state_dict(),
                        'base_state_dict': raw.base.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'epoch': ep, 'global_step': gs, 'config': cfg},
                       os.path.join(cfg['output_dir'], f'stage_ae_epoch_{ep}.pt'))
            print(f"  Saved epoch {ep}", flush=True)
    dist.destroy_process_group()


def _find_latest_ckpt(d, prefix):
    import re
    cs = [f for f in os.listdir(d) if f.startswith(prefix) and f.endswith('.pt')]
    def _num(fn):
        m = re.search(r'(\d+)\.pt$', fn)
        return int(m.group(1)) if m else 0
    cs.sort(key=_num)
    return os.path.join(d, cs[-1]) if cs else None


def train_stage_c(rank, world_size, cfg):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}')
    if rank == 0:
        print(f"\n{'='*60}\nStage C: 160M Incremental (RT+MLM, no VICReg)\n{'='*60}")
    dl = create_parquet_dataloader_mlm(
        shard_dir=cfg['shard_dir'], batch_size=cfg['batch_size'],
        max_peaks=cfg['max_peaks'], mask_ratio=cfg['mask_ratio'],
        rank=rank, world_size=world_size, num_workers=min(cfg['num_workers'], 2))
    base = UltraExplorerMLM(V9_CONFIG)
    model = Phase2Model(base, cfg)
    step_ck = _find_latest_ckpt(cfg['output_dir'], 'stage_c_step_')
    epoch_ck = _find_latest_ckpt(cfg['output_dir'], 'stage_c_epoch_')
    best_ck, best_step, best_ep = None, 0, 0
    if step_ck:
        ck_tmp = torch.load(step_ck, map_location='cpu', weights_only=False)
        best_ck, best_step, best_ep = step_ck, ck_tmp.get('global_step', 0), ck_tmp.get('epoch', 1)
        del ck_tmp
    if epoch_ck:
        ck_tmp = torch.load(epoch_ck, map_location='cpu', weights_only=False)
        ep_step = ck_tmp.get('global_step', 0)
        ep_num = ck_tmp.get('epoch', 1)
        if ep_step > best_step:
            best_ck, best_step, best_ep = epoch_ck, ep_step, ep_num + 1
        del ck_tmp
    if best_ck:
        ck = torch.load(best_ck, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'], strict=False)
        resume_step = best_step
        resume_ep = best_ep
        if rank == 0:
            print(f"  Resumed Stage C from: {best_ck} (step={resume_step}, ep={resume_ep})")
        del ck
    else:
        ckpt_path = os.path.join(cfg['output_dir'], 'stage_ae_epoch_5.pt')
        if not os.path.exists(ckpt_path):
            ckpt_path = _find_latest_ckpt(cfg['output_dir'], 'stage_ae_')
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'], strict=False)
        resume_step = 0
        resume_ep = 1
        if rank == 0:
            print(f"  Loaded Stage Ae: {ckpt_path}")
        del ck
    cfg['rt_weight'] = 0.1
    if rank == 0:
        print(f"  Stage C rt_weight={cfg['rt_weight']}, tau={cfg['rt_self_consist_tau']}")
    model = DDP(model.to(dev), device_ids=[rank], find_unused_parameters=False)
    pg = [
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and 'LayerNorm' not in n and 'bias' not in n],
         'weight_decay': cfg['weight_decay']},
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and ('LayerNorm' in n or 'bias' in n)],
         'weight_decay': 0.0}]
    opt = torch.optim.AdamW(pg, lr=cfg['lr']*0.5)
    scaler = GradScaler()
    lp = os.path.join(cfg['output_dir'], 'stage_c.log')
    gs = resume_step
    te = cfg['stage_c_epochs']
    for ep in range(resume_ep, te+1):
        dl.dataset.set_epoch(ep)
        model.train()
        batch_iter = iter(dl)
        while True:
            try:
                batch = next(batch_iter)
                has_data = torch.ones(1, device=dev, dtype=torch.int32)
            except StopIteration:
                has_data = torch.zeros(1, device=dev, dtype=torch.int32)
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0:
                if rank == 0:
                    print(f"  [Sync] All ranks finished epoch {ep} at step {gs}", flush=True)
                break
            with autocast('cuda', dtype=torch.bfloat16):
                r = model(batch, dev, mode='stage_c')
            scaler.scale(r['loss']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            gs += 1
            if rank == 0 and gs % cfg['log_interval'] == 0:
                rt_sc_mae = _safe_scalar(r.get('rt_mae_sc')) if 'rt_mae_sc' in r else _safe_scalar(r.get('rt_mae'))
                msg = (f"[C] ep={ep} step={gs} loss={r['loss'].item():.4f} "
                       f"mlm={_safe_scalar(r.get('mlm_loss')):.4f} "
                       f"rt_mae={_safe_scalar(r.get('rt_mae')):.1f}s "
                       f"rt_sc_mae={rt_sc_mae:.1f}s "
                       f"rt_n={r.get('rt_n',0)} rt_sc={r.get('rt_n_sc',0)}")
                print(msg, flush=True)
                with open(lp, 'a') as f:
                    f.write(msg+'\n')
            if gs % 2000 == 0:
                gc.collect()
            if rank == 0 and gs % cfg['save_step_interval'] == 0:
                raw = model.module
                torch.save({'model_state_dict': raw.state_dict(),
                            'base_state_dict': raw.base.state_dict(),
                            'epoch': ep, 'global_step': gs, 'config': cfg},
                           os.path.join(cfg['output_dir'], f'stage_c_step_{gs}.pt'))
        dist.barrier()
        if rank == 0:
            raw = model.module
            torch.save({'model_state_dict': raw.state_dict(),
                        'base_state_dict': raw.base.state_dict(),
                        'epoch': ep, 'global_step': gs, 'config': cfg},
                       os.path.join(cfg['output_dir'], f'stage_c_epoch_{ep}.pt'))
            print(f"  Saved Stage C epoch {ep}", flush=True)
    dist.destroy_process_group()


def train_stage_d(rank, world_size, cfg):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    dev = torch.device(f'cuda:{rank}')
    if rank == 0:
        print(f"\n{'='*60}\nStage D: Polarity + RT + MLM Joint Training\n{'='*60}")
    pol_shard_dir = cfg.get('pol_shard_dir', 'datasets/shards_polarity_balanced')
    dl = create_parquet_dataloader_mlm(
        shard_dir=pol_shard_dir, batch_size=cfg['batch_size'],
        max_peaks=cfg['max_peaks'], mask_ratio=cfg['mask_ratio'],
        rank=rank, world_size=world_size, num_workers=min(cfg['num_workers'], 2))
    base = UltraExplorerMLM(V9_CONFIG)
    model = Phase2Model(base, cfg)

    step_ck = _find_latest_ckpt(cfg['output_dir'], 'stage_d_step_')
    epoch_ck = _find_latest_ckpt(cfg['output_dir'], 'stage_d_epoch_')
    best_ck, best_step, best_ep = None, 0, 0
    if step_ck:
        ck_tmp = torch.load(step_ck, map_location='cpu', weights_only=False)
        best_ck, best_step, best_ep = step_ck, ck_tmp.get('global_step', 0), ck_tmp.get('epoch', 1)
        del ck_tmp
    if epoch_ck:
        ck_tmp = torch.load(epoch_ck, map_location='cpu', weights_only=False)
        ep_step = ck_tmp.get('global_step', 0)
        ep_num = ck_tmp.get('epoch', 1)
        if ep_step > best_step:
            best_ck, best_step, best_ep = epoch_ck, ep_step, ep_num + 1
        del ck_tmp

    if best_ck:
        ck = torch.load(best_ck, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        resume_step = best_step
        resume_ep = best_ep
        if rank == 0:
            print(f"  Resumed Stage D from: {best_ck} (step={resume_step}, ep={resume_ep})")
        del ck
    else:
        src_ckpt = cfg.get('stage_d_init_ckpt',
                           os.path.join(cfg['output_dir'], 'stage_c_epoch_1.pt'))
        ck = torch.load(src_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'], strict=False)
        resume_step = 0
        resume_ep = 1
        if rank == 0:
            print(f"  Loaded Stage C ckpt for Stage D init: {src_ckpt}")
        del ck

    cfg['rt_weight'] = 0.1
    cfg['pol_weight'] = cfg.get('pol_weight', 0.1)
    if rank == 0:
        print(f"  Stage D rt_weight={cfg['rt_weight']}, pol_weight={cfg['pol_weight']}, "
              f"tau={cfg['rt_self_consist_tau']}")
        print(f"  Polarity shard_dir: {pol_shard_dir}")
    model = DDP(model.to(dev), device_ids=[rank], find_unused_parameters=False)
    pg = [
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and 'LayerNorm' not in n and 'bias' not in n],
         'weight_decay': cfg['weight_decay']},
        {'params': [p for n,p in model.named_parameters()
                     if p.requires_grad and ('LayerNorm' in n or 'bias' in n)],
         'weight_decay': 0.0}]
    opt = torch.optim.AdamW(pg, lr=cfg['lr']*0.5)
    scaler = GradScaler()
    lp = os.path.join(cfg['output_dir'], 'stage_d.log')
    gs = resume_step
    te = cfg.get('stage_d_epochs', 1)
    for ep in range(resume_ep, te+1):
        dl.dataset.set_epoch(ep)
        model.train()
        batch_iter = iter(dl)
        while True:
            try:
                batch = next(batch_iter)
                has_data = torch.ones(1, device=dev, dtype=torch.int32)
            except StopIteration:
                has_data = torch.zeros(1, device=dev, dtype=torch.int32)
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0:
                if rank == 0:
                    print(f"  [Sync] All ranks finished epoch {ep} at step {gs}", flush=True)
                break
            with autocast('cuda', dtype=torch.bfloat16):
                r = model(batch, dev, mode='stage_d')
            scaler.scale(r['loss']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            gs += 1
            if rank == 0 and gs % cfg['log_interval'] == 0:
                rt_sc_mae = _safe_scalar(r.get('rt_mae_sc')) if 'rt_mae_sc' in r else _safe_scalar(r.get('rt_mae'))
                msg = (f"[D] ep={ep} step={gs} loss={r['loss'].item():.4f} "
                       f"mlm={_safe_scalar(r.get('mlm_loss')):.4f} "
                       f"rt_mae={_safe_scalar(r.get('rt_mae')):.1f}s "
                       f"rt_sc_mae={rt_sc_mae:.1f}s "
                       f"rt_n={r.get('rt_n',0)} rt_sc={r.get('rt_n_sc',0)} "
                       f"pol_loss={_safe_scalar(r.get('pol_loss')):.4f} "
                       f"pol_acc={_safe_scalar(r.get('pol_acc')):.4f} "
                       f"pol_n={r.get('pol_n',0)}")
                print(msg, flush=True)
                with open(lp, 'a') as f:
                    f.write(msg+'\n')
            if gs % 2000 == 0:
                gc.collect()
            if rank == 0 and gs % cfg['save_step_interval'] == 0:
                raw = model.module
                torch.save({'model_state_dict': raw.state_dict(),
                            'base_state_dict': raw.base.state_dict(),
                            'epoch': ep, 'global_step': gs, 'config': cfg},
                           os.path.join(cfg['output_dir'], f'stage_d_step_{gs}.pt'))
        dist.barrier()
        if rank == 0:
            raw = model.module
            torch.save({'model_state_dict': raw.state_dict(),
                        'base_state_dict': raw.base.state_dict(),
                        'epoch': ep, 'global_step': gs, 'config': cfg},
                       os.path.join(cfg['output_dir'], f'stage_d_epoch_{ep}.pt'))
            print(f"  Saved Stage D epoch {ep}", flush=True)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['Ae','C','D'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--gpus', type=int, default=6)
    parser.add_argument(
        '--stage-c-epochs', type=int, default=None,
        help='Stage C only: inclusive last epoch index te (for ep in range(resume_ep, te+1)). '
             'Example: finished epoch 2 → set 7 to run epochs 3..7 (five more). '
             'Default: PHASE2_CONFIG stage_c_epochs value.')
    parser.add_argument(
        '--stage-d-epochs', type=int, default=None,
        help='Stage D only: inclusive last epoch index ``te`` (loop ``for ep in range(resume_ep, te+1)``). '
             'Example: finished epoch 1 → set 11 to train epochs 2..11 (10 more). '
             'Default: ``cfg[\'stage_d_epochs\']`` or 1.')
    args = parser.parse_args()
    cfg = PHASE2_CONFIG.copy()
    if args.checkpoint:
        cfg['phase1_checkpoint'] = args.checkpoint
    if args.stage_c_epochs is not None:
        cfg['stage_c_epochs'] = args.stage_c_epochs
    if args.stage_d_epochs is not None:
        cfg['stage_d_epochs'] = args.stage_d_epochs
    os.makedirs(cfg['output_dir'], exist_ok=True)
    with open(os.path.join(cfg['output_dir'], 'rt_only_config.json'), 'w') as f:
        json.dump({k: str(v) if not isinstance(v,(int,float,bool,str,list)) else v
                   for k,v in cfg.items()}, f, indent=2)

    import torch.multiprocessing as mp
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29508'
    os.environ['NCCL_ALGO'] = ''

    if args.stage == 'Ae':
        mp.spawn(train_stage_ae, args=(args.gpus, cfg), nprocs=args.gpus, join=True)
    elif args.stage == 'C':
        mp.spawn(train_stage_c, args=(args.gpus, cfg), nprocs=args.gpus, join=True)
    elif args.stage == 'D':
        mp.spawn(train_stage_d, args=(args.gpus, cfg), nprocs=args.gpus, join=True)


if __name__ == '__main__':
    main()
