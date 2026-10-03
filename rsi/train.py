"""Fixed-budget, validation-only A/B/D pilot with resumable epoch checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .datasets import IMAMultiReferenceDataset, collate_multi_reference, restore_logits
from .models import ControlledModel
from .objectives import (per_reference_loss, objective_from_losses,
                         weighted_reference_quantile)
from .runtime import (PROJECT, append_csv, atomic_json, code_provenance, environment,
                      load_config, restore_rng, rng_state, save_checkpoint, set_seed, sha256_file)


def loaders(cfg, seed, *, image_ids=None, manifest=None):
    common = dict(manifest=manifest or cfg['manifest'], seed=seed, size=cfg['input_size'][0],
                  cache_dir=cfg['cache_dir'])
    train = IMAMultiReferenceDataset(split='train', subset='M', train=True, image_ids=image_ids, **common)
    canonical = IMAMultiReferenceDataset(split='train', subset='M', train=False, image_ids=image_ids, **common)
    val = IMAMultiReferenceDataset(split='val', subset='M', train=False, **common)
    return train, canonical, val


def loader(dataset, batch_size, workers=4, *, seed=17, epoch=0, shuffle=False):
    generator = torch.Generator().manual_seed(seed + epoch * 100003)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, pin_memory=True, persistent_workers=False,
                      collate_fn=collate_multi_reference, drop_last=False)


def device_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k,v in batch.items()}


def forward_pair(model, batch, stage, alpha=1.):
    if stage == 'A_CNN':
        z = model.forward_cnn(batch['image'])
        return dict(z0=z.detach(), z1=z, message=None)
    if stage == 'A_T':
        z = model.forward_transformer(batch['image'])
        with torch.no_grad(): z0 = model.forward_cnn(batch['image'])
        return dict(z0=z0, z1=z, message=None)
    return model.forward_pair(batch['image'], valid=batch['pixel_valid'], alpha=alpha)


def calculate(model, batch, stage, method='d0', weight=0., q=None, alpha=1.):
    out = forward_pair(model, batch, stage, alpha)
    ell = per_reference_loss(out['z1'], batch['masks'], batch['rater_present'], batch['pixel_valid'])
    with torch.no_grad():
        ell0 = per_reference_loss(out['z0'], batch['masks'], batch['rater_present'], batch['pixel_valid'])
    objective = objective_from_losses(ell, ell0, batch['rater_present'], method=method, weight=weight, q=q)
    return out, objective, ell, ell0


def scalar(x):
    value = float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
    return value if math.isfinite(value) else None


def truncate_uncommitted_epochs(path, epoch_start):
    """Discard CSV rows from epochs absent from the last atomic checkpoint."""
    path = Path(path)
    if not path.is_file(): return
    with path.open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames
        rows = [row for row in reader if int(row['epoch']) < epoch_start]
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _mask_original(ref, cache):
    path = ref['mask_path']
    if path not in cache:
        with Image.open(path) as f: cache[path] = np.asarray(f.convert('L')) > 0
    return cache[path]


def hard_overlap(pred, gt):
    tp = np.count_nonzero(pred & gt)
    total = np.count_nonzero(pred) + np.count_nonzero(gt)
    union = np.count_nonzero(pred | gt)
    return (2*tp/total if total else 1., tp/union if union else 1.)


@torch.no_grad()
def evaluate(model, dataset, cfg, stage, *, method='d0', weight=0., q=None,
             alpha=1., original_metrics=True, mask_cache=None, per_image_path=None,
             epoch=None, split_label=None):
    if dataset.split == 'test': raise PermissionError('Pilot model scoring test is locked')
    model.eval()
    totals = {k: 0. for k in ['loss','seg','risk','weighted_risk','rho','J','kappa_pm','activation_rate','mean_weight']}
    n = 0; dice_values=[]; iou_values=[]; positive=[]; harm=[]; any_harm=[]; flips=[]; j_values=[]; mean_gains=[]; negative=[]
    reference_losses=[]; reference_presence=[]
    mask_cache = mask_cache if mask_cache is not None else {}
    for cpu_batch in loader(dataset, cfg.get('eval_batch',4), cfg['workers']):
        b = device_batch(cpu_batch, next(model.parameters()).device)
        with torch.autocast('cuda', enabled=cfg['amp']):
            out, obj, ell, ell0 = calculate(model,b,stage,method,weight,q,alpha)
        size = len(b['image_id']); n += size
        for k in totals: totals[k] += scalar(obj[k]) * size
        j_values.extend(obj['J_per_image'].cpu().tolist())
        reference_losses.extend([ell[i, b['rater_present'][i]].cpu() for i in range(size)])
        for i in range(size):
            gains=[]; ds=[]; js=[]
            if original_metrics:
                pred = restore_logits(out['z1'][i].float(), b['geometry'][i]).cpu().numpy() >= 0
                anchor = restore_logits(out['z0'][i].float(), b['geometry'][i]).cpu().numpy() >= 0
                for ref in b['references'][i]:
                    gt = _mask_original(ref, mask_cache)
                    d,j = hard_overlap(pred,gt); d0,_ = hard_overlap(anchor,gt)
                    ds.append(d);js.append(j);gains.append(d-d0)
                dice_values.append(float(np.mean(ds)));iou_values.append(float(np.mean(js)))
                positive.append(float(np.maximum(gains,0).mean()))
                mean_gains.append(float(np.mean(gains)));negative.append(float(np.maximum(-np.array(gains),0).mean()))
                harm.append(float(np.maximum(-np.array(gains)-.005,0).mean()))
                any_harm.append(float(min(gains)<-.005)); flips.append(float(min(gains)<-.005 and max(gains)>.005))
            if per_image_path:
                dvals = obj['d'][i,b['rater_present'][i]].cpu().numpy()
                append_csv(per_image_path, dict(epoch=epoch,split=split_label or dataset.split,
                    image_id=b['image_id'][i], group_id=b['group_id'][i],reference_count=len(dvals),
                    J=float(obj['J_per_image'][i]),rho=float((dvals>0).mean()),
                    kappa_pm=int(dvals.min()<-1e-6 and dvals.max()>1e-6),
                    mean_loss=float(ell[i,b['rater_present'][i]].mean()),
                    mean_anchor_loss=float(ell0[i,b['rater_present'][i]].mean()),
                    min_d=float(dvals.min()), max_d=float(dvals.max())))
    result={k:v/n for k,v in totals.items()}
    result.update(images=n,J_p50=float(np.quantile(j_values,.5)),J_p90=float(np.quantile(j_values,.9)),
                  J_p99=float(np.quantile(j_values,.99)))
    if original_metrics:
        result.update(dice=float(np.mean(dice_values)),iou=float(np.mean(iou_values)),
                      G_plus=float(np.mean(positive)),H_epsilon=float(np.mean(harm)),
                      mean_gain=float(np.mean(mean_gains)),H_minus=float(np.mean(negative)),
                      any_harm=float(np.mean(any_harm)),sign_flip=float(np.mean(flips)))
    return result, reference_losses


def gradient_diagnostic(model, batch, stage, method, weight, q):
    was_training = model.training
    model.eval(); params=[p for p in model.parameters() if p.requires_grad]
    out,obj,_,_=calculate(model,batch,stage,method,weight,q)
    gseg=torch.autograd.grad(obj['seg'],params,retain_graph=True,allow_unused=True)
    grisk=torch.autograd.grad(obj['weighted_risk'],params,allow_unused=True)
    ns=sum((g.float().square().sum() for g in gseg if g is not None),start=torch.tensor(0.,device=batch['image'].device)).sqrt()
    nr=sum((g.float().square().sum() for g in grisk if g is not None),start=torch.tensor(0.,device=batch['image'].device)).sqrt()
    dot=sum((a.float().mul(b.float()).sum() for a,b in zip(gseg,grisk) if a is not None and b is not None),start=ns*0)
    result=dict(seg_gradient_norm=scalar(ns),weighted_risk_gradient_norm=scalar(nr),
                risk_seg_gradient_ratio=scalar(nr/ns) if ns>0 else None,
                risk_seg_gradient_cosine=scalar(dot/(ns*nr)) if ns>0 and nr>0 else None)
    model.train(was_training);return result


def make_optimizer(model,stage,cfg):
    if stage in ['B','D']:
        groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=cfg['lr'][stage])]
    else:
        encoder = model.cnn.encoder if stage=='A_CNN' else model.transformer.encoder
        encoder_ids={id(p) for p in encoder.parameters()}
        enc=[p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids]
        dec=[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        groups=[dict(params=enc,lr=cfg['lr']['encoder']),dict(params=dec,lr=cfg['lr']['decoder'])]
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    return optimizer,[g['lr'] for g in groups]


def train_stage(cfg,seed,stage,*,init=None,method='d0',weight=0.,q=None,run_name=None,resume=True):
    audit=json.loads(Path(cfg['audit']).read_text())
    if not audit.get('file_audit_complete',False) or audit.get('scope') not in ['full','all_selected_M_H_T1']:
        raise RuntimeError('Full file audit must pass before full-budget training')
    set_seed(seed);torch.set_num_threads(4)
    run_name=run_name or (stage if stage!='D' else f'{method}_{weight:g}')
    run_dir=Path(cfg['run_root'])/f'seed{seed}'/run_name;run_dir.mkdir(parents=True,exist_ok=True)
    completed=run_dir/'DONE.json'
    manifest_hash=sha256_file(cfg['manifest']); provenance=code_provenance()
    init_hash=sha256_file(init) if init else None
    if audit['final_references_sha256']!=manifest_hash:
        raise ValueError('Full file audit is not bound to the configured final manifest')
    signature=dict(config=cfg,seed=seed,stage=stage,method=method,weight=weight,q=q,
                   init_checkpoint_hash=init_hash,manifest_hash=manifest_hash,
                   source_bundle_hash=provenance['source_bundle_hash'])
    if completed.is_file():
        done=json.loads(completed.read_text())
        if done.get('signature')!=signature: raise ValueError('Completed run signature changed')
        if done['best_sha256']!=sha256_file(run_dir/'best.pth'): raise ValueError('Completed best checkpoint changed')
        return run_dir/'best.pth'
    train,canonical,val=loaders(cfg,seed)
    model=ControlledModel(pretrained=True,input_size=cfg['input_size'][0]).cuda()
    if init:
        checkpoint=torch.load(init,map_location='cpu')
        expected_stage={'A_T':'A_CNN','B':'A_T','D':'B'}.get(stage)
        if checkpoint.get('stage')!=expected_stage:
            raise ValueError(f'{stage} requires the selected {expected_stage} initialization')
        if checkpoint.get('seed')!=seed or checkpoint.get('manifest_hash')!=manifest_hash:
            raise ValueError('Initialization seed or manifest differs')
        init_done_path=Path(init).parent/'DONE.json'
        if not init_done_path.is_file(): raise ValueError('The initialization stage has not completed its full budget')
        init_done=json.loads(init_done_path.read_text())
        if init_done['epochs']!=cfg['epochs'][expected_stage] or init_done['best_sha256']!=init_hash:
            raise ValueError('Initialization is not the full-budget selected checkpoint')
        model.load_state_dict(checkpoint['model'])
    elif stage!='A_CNN':
        raise ValueError(f'{stage} needs its completed prior-stage checkpoint')
    model.set_stage(stage); optimizer,base_lrs=make_optimizer(model,stage,cfg)
    budget=cfg['epochs'][stage]
    warmup=cfg['scheduler']['A_warmup_epochs' if stage.startswith('A_') else 'BD_warmup_epochs']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e: (e+1)/warmup if e<warmup else
         .5*(1+math.cos(math.pi*(e-warmup)/max(1,budget-warmup))))
    scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    epoch_start=1;best=-math.inf;best_epoch=None;steps=0;elapsed_before=0.
    if resume and (run_dir/'latest.pth').is_file():
        last=torch.load(run_dir/'latest.pth',map_location='cpu')
        if last['manifest_hash']!=manifest_hash: raise ValueError('Resume manifest changed')
        if last['config']!=cfg: raise ValueError('Resume configuration changed')
        if any(last[k]!=v for k,v in [('seed',seed),('stage',stage),('method',method),('weight',weight),('q',q)]):
            raise ValueError('Resume seed/stage/objective changed')
        if last['init_checkpoint_hash']!=init_hash: raise ValueError('Resume shared initialization changed')
        if last['code']['source_bundle_hash']!=provenance['source_bundle_hash']:
            raise ValueError('Resume implementation changed; audit before continuing')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler']);scaler.load_state_dict(last['scaler'])
        restore_rng(last['rng']);epoch_start=last['epoch']+1;best=last['best_dice'];best_epoch=last['best_epoch']
        steps=last['optimizer_steps'];elapsed_before=last['wall_seconds']
        if last['epoch'] == best_epoch:
            # latest is saved before best; an interruption between these atomic
            # writes still contains the complete selected checkpoint.
            save_checkpoint(run_dir/'best.pth', last)
        else:
            best_path = run_dir/'best.pth'
            if not best_path.is_file():
                raise ValueError('Selected checkpoint is missing and latest is not the selected epoch')
            selected = torch.load(best_path, map_location='cpu')
            if selected['epoch'] != best_epoch or selected['best_dice'] != best:
                raise ValueError('Selected checkpoint does not match the resumed selection history')
    for name in ['epoch_diagnostics.csv', 'per_image_canonical_diagnostics.csv']:
        truncate_uncommitted_epochs(run_dir/name, epoch_start)
    atomic_json(run_dir/'config_resolved.json',dict(config=cfg,seed=seed,stage=stage,method=method,weight=weight,q=q))
    atomic_json(run_dir/'environment.json',environment())
    fixed_batch=device_batch(next(iter(loader(canonical,cfg['microbatch_start'],0))),torch.device('cuda'))
    frozen_hash=model.frozen_state_hash() if stage in ['B','D'] else None
    if epoch_start>1 and frozen_hash is not None and frozen_hash!=last['frozen_hash']:
        raise ValueError('Resumed frozen state differs from saved state')
    with torch.no_grad(): fixed_anchor=model.forward_cnn(fixed_batch['image']).detach().clone()
    wall_start=time.monotonic();mask_cache={}
    fields=['loss','seg','risk','weighted_risk','rho','J','kappa_pm','activation_rate','mean_weight']
    for epoch in range(epoch_start,budget+1):
        tick=time.monotonic();model.train();train.set_epoch(epoch)
        batches=loader(train,cfg['microbatch_start'],cfg['workers'],seed=seed,epoch=epoch,shuffle=True)
        accumulation=cfg['effective_batch']//cfg['microbatch_start']
        if accumulation*cfg['microbatch_start']!=cfg['effective_batch']: raise ValueError('Invalid effective batch')
        totals={k:0. for k in fields};seen=0;grad_norms=[];clipped=0;message_rms=[];delta_rms=[]
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        for batch_index,cpu_batch in enumerate(batches):
            b=device_batch(cpu_batch,torch.device('cuda'));bs=len(b['image_id'])
            window_index=batch_index//accumulation
            first=window_index*accumulation*cfg['microbatch_start']
            window_samples=min(cfg['effective_batch'],len(train)-first)
            with torch.autocast('cuda',enabled=cfg['amp']):
                out,obj,_,_=calculate(model,b,stage,method,weight,q)
            scaler.scale(obj['loss']*(bs/window_samples)).backward()
            for k in fields: totals[k]+=scalar(obj[k])*bs
            seen+=bs
            if out.get('message') is not None:
                message_rms.append(scalar(out['message'].float().square().mean().sqrt()))
                delta_rms.append(scalar((out['z1'].float()-out['z0'].float()).square().mean().sqrt()))
            if (batch_index+1)%accumulation==0 or batch_index+1==len(batches):
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'])
                grad_norms.append(scalar(norm));clipped+=int(norm>cfg['gradient_clip'])
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);steps+=1
        if frozen_hash is not None:
            if model.frozen_state_hash()!=frozen_hash: raise RuntimeError('Anchor/frozen parameters or buffers changed')
            model.eval()
            with torch.no_grad(): now=model.forward_cnn(fixed_batch['image'])
            anchor_max_absolute_error=float((now-fixed_anchor).abs().max())
            if not torch.allclose(now,fixed_anchor,atol=1e-6,rtol=1e-6):
                raise RuntimeError('Fixed FP32 anchor logits changed')
        else:
            anchor_max_absolute_error=None
        train_metrics=None
        if stage in ['B','D']:
            train_metrics,_=evaluate(model,canonical,cfg,stage,method=method,weight=weight,q=q,
                original_metrics=False,per_image_path=run_dir/'per_image_canonical_diagnostics.csv',epoch=epoch,split_label='train')
        val_metrics,_=evaluate(model,val,cfg,stage,method=method,weight=weight,q=q,mask_cache=mask_cache,
            per_image_path=run_dir/'per_image_canonical_diagnostics.csv',epoch=epoch,split_label='val')
        diagnostics=dict(seg_gradient_norm=None,weighted_risk_gradient_norm=None,risk_seg_gradient_ratio=None,risk_seg_gradient_cosine=None)
        if stage in ['B','D'] and epoch%cfg['gradient_diagnostic_every_epochs']==0:
            diagnostics=gradient_diagnostic(model,fixed_batch,stage,method,weight,q)
        improved=val_metrics['dice']>best
        if improved: best=val_metrics['dice'];best_epoch=epoch
        epoch_seconds=time.monotonic()-tick
        row=dict(seed=seed,stage=stage,run_id=run_name,epoch=epoch,method=method,weight=weight,q=q,
                 **{f'augmented_train_{k}':v/seen for k,v in totals.items()},
                 **{f'canonical_train_{k}':(train_metrics or {}).get(k) for k in fields},
                 **{f'val_{k}':val_metrics[k] for k in val_metrics},**diagnostics,
                 gradient_norm_before_clip=float(np.mean(grad_norms)),gradient_clip_fraction=clipped/len(grad_norms),
                 augmented_message_rms=float(np.mean(message_rms)) if message_rms else None,
                 augmented_logit_delta_rms=float(np.mean(delta_rms)) if delta_rms else None,
                 lr_group0=optimizer.param_groups[0]['lr'],lr_group1=optimizer.param_groups[1]['lr'] if len(optimizer.param_groups)>1 else None,
                 epoch_seconds=epoch_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                 anchor_max_absolute_error=anchor_max_absolute_error,
                 best_epoch=best_epoch,best_val_dice=best,optimizer_steps=steps)
        append_csv(run_dir/'epoch_diagnostics.csv',row)
        scheduler.step()
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(),rng=rng_state(),epoch=epoch,seed=seed,stage=stage,method=method,
                     weight=weight,q=q,best_dice=best,best_epoch=best_epoch,optimizer_steps=steps,
                     config=cfg,manifest_hash=manifest_hash,code=provenance,frozen_hash=frozen_hash,
                     init_checkpoint_hash=init_hash,
                     wall_seconds=elapsed_before+time.monotonic()-wall_start,selection_rule='max_val_macro_mean_rater_hard_dice_earlier_tie')
        save_checkpoint(run_dir/'latest.pth',payload)
        if improved: save_checkpoint(run_dir/'best.pth',payload)
        print(json.dumps(dict(run=run_name,epoch=epoch,budget=budget,val_dice=val_metrics['dice'],
                              best_dice=best,best_epoch=best_epoch,seconds=epoch_seconds)),flush=True)
    report=dict(run=run_name,seed=seed,stage=stage,epochs=budget,best_epoch=best_epoch,best_val_dice=best,
                signature=signature,
                best_sha256=sha256_file(run_dir/'best.pth'),latest_sha256=sha256_file(run_dir/'latest.pth'),
                wall_seconds=elapsed_before+time.monotonic()-wall_start,test_scoring_locked=True)
    atomic_json(completed,report)
    # Preserve selected/last weights and logs on the shared workspace after each stage.
    mirror=PROJECT/'runs'/f'seed{seed}'/run_name;mirror.mkdir(parents=True,exist_ok=True)
    for p in run_dir.iterdir():
        if p.is_file(): shutil.copy2(p,mirror/p.name)
    return run_dir/'best.pth'


def prepare_q(checkpoint,cfg):
    state=torch.load(checkpoint,map_location='cpu')
    if state.get('stage')!='B' or state.get('manifest_hash')!=sha256_file(cfg['manifest']):
        raise ValueError('q requires the shared B checkpoint and its training manifest')
    done=json.loads((Path(checkpoint).parent/'DONE.json').read_text())
    if done['epochs']!=cfg['epochs']['B'] or done['best_sha256']!=sha256_file(checkpoint):
        raise ValueError('q source is not the completed, selected B checkpoint')
    model=ControlledModel(pretrained=False).cuda()
    model.load_state_dict(state['model']);model.set_stage('D');model.eval()
    dataset=IMAMultiReferenceDataset(cfg['manifest'],split='train',subset='M',cache_dir=cfg['cache_dir'])
    _,losses=evaluate(model,dataset,cfg,'B',original_metrics=False)
    r=max(len(x) for x in losses);ell=torch.zeros(len(losses),r);present=torch.zeros_like(ell,dtype=torch.bool)
    for i,x in enumerate(losses): ell[i,:len(x)]=x;present[i,:len(x)]=True
    q=float(weighted_reference_quantile(ell,present,.75))
    atomic_json(Path(checkpoint).parent/'abs_hard_q.json',dict(q=q,quantile=.75,source='B_canonical_M_train',
        checkpoint_hash=sha256_file(checkpoint),manifest_hash=sha256_file(cfg['manifest']),fixed=True,gradient=False))
    return q


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--stage',choices=['A_CNN','A_T','B','D'],required=True);p.add_argument('--init')
    p.add_argument('--method',default='d0');p.add_argument('--weight',type=float,default=0.)
    p.add_argument('--q',type=float);p.add_argument('--run-name')
    a=p.parse_args();cfg=load_config(a.config)
    train_stage(cfg,a.seed,a.stage,init=a.init,method=a.method,weight=a.weight,q=a.q,run_name=a.run_name)


if __name__=='__main__':main()
