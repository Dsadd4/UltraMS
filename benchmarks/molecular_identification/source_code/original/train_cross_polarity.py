"""Train Figure 3f reciprocal cross-ionization retrieval from public inputs.

Imports the original feature, projection and loss implementations. Frozen
Figure 3 per-mode role assignments and full reference libraries are retained.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temp, path)


def prepare(root, out):
    manifest = root/'train/output/comparison/molecular_benchmarks/same_adduct_manifest'
    data, roles, protocols = {}, {}, {}
    with h5py.File(manifest/'msnlib_library_spectra.h5', 'r') as handle:
        for mode in ['pos', 'neg']:
            group = handle[mode]
            vocab = json.loads((manifest/f'smiles_{mode}.json').read_text())
            labels = np.asarray(group['smiles_index'])
            names = np.asarray(vocab)[labels]
            protocol = np.load(manifest/f'protocol_{mode}.npz')
            protocols[mode] = {key: protocol[key] for key in protocol.files}
            roles[mode] = {}
            for role, key in [('train', 'train_rows'), ('val', 'val_query_rows'), ('test', 'test_query_rows')]:
                roles[mode][role] = set(names[protocol[key]])
            groups = {}
            for i, name in enumerate(names):
                groups.setdefault(str(name), []).append(i)
            data[mode] = dict(spectra=np.asarray(group['spectra']),
                counts=np.asarray(group['n_peaks']), precursor=np.asarray(group['precursor_mz']),
                source=np.asarray(group['source_row']), names=names, groups=groups)
    shared = {role: sorted(roles['pos'][role] & roles['neg'][role]) for role in ['train', 'val', 'test']}
    assert {key:len(value) for key,value in shared.items()} == dict(train=8265, val=380, test=354)
    for mode in ['pos', 'neg']:
        for role in ['val', 'test']:
            assert not set(shared[role]) & roles[mode]['train']
    assert not set(shared['train']) & (set(shared['val']) | set(shared['test']))
    assert not set(shared['val']) & set(shared['test'])
    fixed = {}
    for role in ['val', 'test']:
        fixed[role] = {}
        for mode in ['pos', 'neg']:
            rows = protocols[mode][f'{role}_query_rows']
            mapping = {str(data[mode]['names'][row]): int(row) for row in rows}
            fixed[role][mode] = np.asarray([mapping[name] for name in shared[role]])
    save_json(out/'molecule_roles.json', shared)
    save_json(out/'population.json', {'split':'intersection of the unchanged original per-polarity roles; no new split',
        'counts':{r:len(v) for r,v in shared.items()},
        'libraries':{m:len(protocols[m]['test_library_rows']) for m in data},
        'evaluation_queries':{m:len(protocols[m]['test_query_rows']) for m in data},
        'missing_reference_policy':'retain all original test queries; save reference support per direction'})
    rows = []
    for role in ['val', 'test']:
        for mode in ['pos', 'neg']:
            for name, row in zip(shared[role], fixed[role][mode]):
                rows.append(dict(role=role,mode=mode,smiles=name,h5_row=int(row),source_row=int(data[mode]['source'][row])))
    pd.DataFrame(rows).to_csv(out/'frozen_queries.csv', index=False)
    return data, shared, fixed, protocols


def samples(data, rows):
    return [dict(spectrum=data['spectra'][i,:data['counts'][i]],
                 precursor_mz=float(data['precursor'][i])) for i in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone', choices=['ultra', 'dreams'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['ULTRAMS_BENCHMARK_ROOT'] = str(root)
    import train_collision_energy_backbones as method
    torch.set_num_threads(8)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    data, shared, fixed, protocols = prepare(root, out)
    if args.prepare_only:
        return
    from element.models import freeze_all, unfreeze_ultra_last_n, unfreeze_dreams_last_n
    device = torch.device(args.device)
    if args.backbone == 'ultra':
        from selection.selection_data_build import load_ultrams_encoder_with_heads
        checkpoint = root/'train/output/pretrained/ultrams_unsupervised.pt'
        model = load_ultrams_encoder_with_heads(device, checkpoint_path=str(checkpoint))
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        del state
        freeze_all(model); unfreeze_ultra_last_n(model, 2)
        encode = method.encode_ultra
        dimension = 2048
    else:
        from dreams_loader import load_dreams_encoder
        model = load_dreams_encoder(device)
        freeze_all(model); unfreeze_dreams_last_n(model, 2)
        encode = method.encode_dreams
        dimension = int(model.d_model)
    projection = method.ProjectionHead(dimension, 256).to(device)
    config = dict(backbone=args.backbone, seed=42,epochs=20,batch=256,projection_dim=256,
        input_dim=dimension,unfreeze_last=2,projection_lr=3e-4,encoder_lr=3e-5,weight_decay=.01,
        temperature=.07,warmup_ratio=.1,gradient_clip=1.,
        objective='original symmetric in-batch InfoNCE; same molecule positive/negative ion pair',
        initialization='same original pretrained encoder initialization, fresh original projection head',
        torch_version=torch.__version__,
        checkpoint_selection='mean bidirectional validation Recall@1; test evaluated only after freezing selection')
    save_json(out/'config.json', config)
    if args.verify_only:
        model.train();projection.train()
        names=shared['train'][:2]
        rows={mode:[data[mode]['groups'][name][0] for name in names] for mode in ['pos','neg']}
        a=projection(encode(model,samples(data['pos'],rows['pos']),device))
        b=projection(encode(model,samples(data['neg'],rows['neg']),device))
        loss,_=method.info_nce(a,b);loss.backward()
        updated=[name for name,p in model.named_parameters() if p.requires_grad]
        nonzero=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().max()>0)]
        assert nonzero and torch.isfinite(loss)
        result=dict(status='pass',purpose='original feature path and real-pair backward execution check; not an experimental result',
            input_dim=dimension,projection_shape=list(a.shape),encoder_trainable_tensors=len(updated),encoder_nonzero_finite_gradient_tensors=len(nonzero),
            first_trainable_names=updated[:3],last_trainable_names=updated[-3:])
        save_json(out/'EXECUTION_CHECK.json',result);print(json.dumps(result),flush=True)
        return
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=projection.parameters(),lr=3e-4),dict(params=parameters,lr=3e-5)],weight_decay=.01)
    batches = math.ceil(len(shared['train'])/256)
    total = batches*20; warmup = int(total*.1)
    def rate(step):
        if step < warmup: return step/max(1,warmup)
        return max(0.,.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    rng = random.Random(42)
    history=[]; best=-1.; start=1
    if (out/'last.pt').exists():
        old=torch.load(out/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(old['enc'],strict=True);projection.load_state_dict(old['proj'],strict=True)
        optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
        history=old['history'];best=old['best'];start=old['epoch']+1
        rng.setstate(old['pair_rng']);torch.set_rng_state(old['torch_rng'].cpu())
        torch.cuda.set_rng_state_all(old['cuda_rng']);del old
    @torch.no_grad()
    def vectors(mode, rows):
        model.eval();projection.eval()
        return np.concatenate([projection(encode(model,samples(data[mode],rows[i:i+256]),device)).cpu().float().numpy()
            for i in range(0,len(rows),256)])
    for epoch in range(start,21):
        model.train();projection.train()
        order=list(shared['train']);rng.shuffle(order)
        loss_sum=0.; n=0; began=time.time(); pair_rows=[]
        for bi in range(0,len(order),256):
            names=order[bi:bi+256]
            rows={mode:[rng.choice(data[mode]['groups'][name]) for name in names] for mode in ['pos','neg']}
            pair_rows.extend(dict(smiles=name,pos_h5_row=int(p),neg_h5_row=int(q),pos_source_row=int(data['pos']['source'][p]),neg_source_row=int(data['neg']['source'][q]))
                for name,p,q in zip(names,rows['pos'],rows['neg']))
            a=projection(encode(model,samples(data['pos'],rows['pos']),device))
            b=projection(encode(model,samples(data['neg'],rows['neg']),device))
            loss,acc=method.info_nce(a,b)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(projection.parameters()),1.)
            optimizer.step();scheduler.step();loss_sum+=loss.item()*len(names);n+=len(names)
            if bi==0 or bi+256>=len(order):
                print(json.dumps(dict(epoch=epoch,batch=bi//256+1,batches=batches,loss=loss.item(),acc=acc)),flush=True)
        pd.DataFrame(pair_rows).to_csv(out/f'epoch_{epoch:02d}_training_pairs.csv.gz',index=False)
        pos=vectors('pos',fixed['val']['pos']);neg=vectors('neg',fixed['val']['neg'])
        scores=pos@neg.T;gt=np.arange(len(pos))
        p=float(np.mean(scores.argmax(1)==gt));q=float(np.mean(scores.argmax(0)==gt))
        row=dict(epoch=epoch,train_loss=loss_sum/n,validation_pos_to_neg=p,validation_neg_to_pos=q,validation_mean=(p+q)/2,seconds=time.time()-began)
        history.append(row);print(json.dumps(row),flush=True)
        if row['validation_mean']>best:
            best=row['validation_mean']
            torch.save(dict(epoch=epoch,enc=model.state_dict(),proj=projection.state_dict(),best_r1=best,unfreeze_last=2),out/'best.pt.part')
            os.replace(out/'best.pt.part',out/'best.pt')
        state=dict(epoch=epoch,enc=model.state_dict(),proj=projection.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,best=best,pair_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(state,out/'last.pt.part');os.replace(out/'last.pt.part',out/'last.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['enc'],strict=True);projection.load_state_dict(checkpoint['proj'],strict=True)
    selected_epoch=checkpoint['epoch'];del checkpoint
    summaries=[]
    for qm,lm in [('pos','neg'),('neg','pos')]:
        qr=protocols[qm]['test_query_rows'];lr=protocols[lm]['test_library_rows']
        queries=vectors(qm,qr)
        path=out/f'full_library_{lm}.npy'
        # Persistent row-aligned export retains every original reference.
        library=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(lr),256))
        for i in range(0,len(lr),256):
            library[i:i+256]=vectors(lm,lr[i:i+256])
            if i%25600==0:print('full-library',lm,i,'/',len(lr),flush=True)
        library.flush();np.save(out/f'queries_{qm}.npy',queries)
        qnames=data[qm]['names'][qr];lnames=data[lm]['names'][lr]
        has=np.isin(qnames,lnames);correct=[];scores=[];bestrows=[]
        for i in range(0,len(qr),128):
            similarity=queries[i:i+128]@library.T;top=similarity.argmax(1)
            scores.extend(similarity[np.arange(len(top)),top].tolist());bestrows.extend(top.tolist())
        bestrows=np.asarray(bestrows);correct=lnames[bestrows]==qnames
        pd.DataFrame(dict(query_smiles=qnames,query_h5_row=qr,query_source_row=data[qm]['source'][qr],
            has_library_positive=has,top1_library_row=bestrows,top1_h5_row=lr[bestrows],
            top1_source_row=data[lm]['source'][lr[bestrows]],top1_smiles=lnames[bestrows],
            top1_score=scores,top1_correct=correct)).to_csv(out/f'{qm}_to_{lm}_per_query.csv',index=False)
        summary=dict(direction=f'{qm}_to_{lm}',n_query=len(qr),n_library=len(lr),n_library_positive=int(has.sum()),
            n_correct=int(correct.sum()),accuracy=float(correct.mean()),reference_supported_recall=float(correct.sum()/has.sum()))
        summaries.append(summary);print(json.dumps(summary),flush=True)
        del library
    save_json(out/'COMPLETE.json',dict(status='complete',selected_epoch=selected_epoch,best_validation_mean=best,
        test_evaluated_after_selection=True,results=summaries))


if __name__=='__main__':main()
