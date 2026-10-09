"""Isolated fixed-budget trainer and real-data acceptance for TailProbe."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from rsi.datasets import IMAMultiReferenceDataset, collate_multi_reference, restore_logits
from rsi.models import ControlledModel
from rsi.objectives import per_reference_loss, objective_from_losses
from rsi.runtime import (PROJECT, atomic_json, append_csv, environment, set_seed,
                         rng_state, restore_rng, sha256_file, save_checkpoint)
from rsi.tail_probe_model import TailProbeModel, EXPECTED_TRAINABLE, interpolate_update_logits
from rsi.train import (loader, device_batch, hard_overlap, _mask_original,
                       finish_optimizer_step, gradient_norm_summary, truncate_uncommitted_epochs)


def provenance(cfg):
    paths = sorted({p for directory, pattern in [('rsi','*.py'),('tools','*.py'),
                   ('tests','*.py'),('configs','*.yaml')] for p in (PROJECT/directory).glob(pattern)})
    hashes = {str(p.relative_to(PROJECT)): sha256_file(p) for p in paths}
    return dict(review_commit=cfg['review_commit'],
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
                source_sha256=hashes,
                source_bundle_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                taskbook_sha256=sha256_file(cfg['taskbook']))


def build_model(cfg, group, checkpoint=None):
    base = ControlledModel(pretrained=False,input_size=cfg['input_size'][0])
    initial = torch.load(cfg['teacher_B'], map_location='cpu')
    base.load_state_dict(initial['model'],strict=True)
    model = TailProbeModel(base, group)
    del base, initial
    if checkpoint:
        saved = torch.load(checkpoint,map_location='cpu')
        if any(k.startswith('student_tail.') for k in saved['model']):
            if saved['group'] != group: raise ValueError('Checkpoint group differs')
            model.load_state_dict(saved['model'],strict=True)
        else:
            model.load_legacy_state_dict(saved['model'],strict=True)
        del saved
    return model.cuda()


def calculate(model,batch):
    out=model.forward_details(batch['image'],batch['pixel_valid'])
    ell=per_reference_loss(out['z1'],batch['masks'],batch['rater_present'],batch['pixel_valid'])
    with torch.no_grad():
        ell0=per_reference_loss(out['z0'],batch['masks'],batch['rater_present'],batch['pixel_valid'])
    rsi=model.group.endswith('RSI')
    obj=objective_from_losses(ell,ell0,batch['rater_present'],method='rsi' if rsi else 'd0',weight=3. if rsi else 0.)
    return out,obj,ell,ell0


def learned_state_hash(state,names):
    h=hashlib.sha256()
    for name in sorted(names):
        value=state[name].detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


FIELDS=('loss','seg','risk','weighted_risk','rho','J','kappa_pm','activation_rate','mean_weight')


class MemoryDataset(IMAMultiReferenceDataset):
    """Exact inherited preprocessing and augmentation, retaining only uint8 data."""
    def preload(self):
        parent_canonical=super()._canonical
        self.memory={refs[0]['image_id']:parent_canonical(refs) for refs in self.samples}
        return self

    def _canonical(self,refs):
        if hasattr(self,'memory'): return self.memory[refs[0]['image_id']]
        return super()._canonical(refs)


def datasets(cfg):
    kwargs=dict(manifest=cfg['manifest'],seed=cfg['seed'],size=cfg['input_size'][0],cache_dir=cfg['cache_dir'])
    canonical=MemoryDataset(split='train',subset='M',train=False,**kwargs)
    val=MemoryDataset(split='val',subset='M',train=False,**kwargs)
    if cfg.get('cache_preload_uint8',True):
        print('PRELOAD canonical uint8 train/val',flush=True)
        canonical.preload();val.preload()
    train=MemoryDataset(split='train',subset='M',train=True,**kwargs)
    if hasattr(canonical,'memory'): train.memory=canonical.memory
    return train,canonical,val


@torch.no_grad()
def evaluate(model,dataset,cfg,*,original=True,mask_cache=None,per_image_path=None,epoch=None):
    if dataset.split=='test': raise PermissionError('TailProbe test scoring is locked')
    model.eval();n=0;totals={k:0. for k in FIELDS};js=[];ds=[];ious=[];gplus=[];hminus=[];heps=[];means=[]
    mean_harm=[];any_harm=[];all_harm=[];conflict=[]
    mask_cache={} if mask_cache is None else mask_cache
    for cpu in loader(dataset,cfg['eval_batch'],cfg['workers']):
        b=device_batch(cpu,torch.device('cuda'));bs=len(b['image_id']);n+=bs
        with torch.autocast('cuda',enabled=cfg['amp']): out,obj,ell,ell0=calculate(model,b)
        for k in FIELDS: totals[k]+=float(obj[k])*bs
        js.extend(obj['J_per_image'].cpu().tolist())
        for i in range(bs):
            if original:
                pred=restore_logits(out['z1'][i].float(),b['geometry'][i]).cpu().numpy()>=0
                refpred=restore_logits(out['z0'][i].float(),b['geometry'][i]).cpu().numpy()>=0
                dice=[];iou=[];gain=[]
                for ref in b['references'][i]:
                    gt=_mask_original(ref,mask_cache)
                    d,j=hard_overlap(pred,gt);d0,_=hard_overlap(refpred,gt)
                    dice.append(d);iou.append(j);gain.append(d-d0)
                g=np.asarray(gain);ds.append(np.mean(dice));ious.append(np.mean(iou))
                gplus.append(np.maximum(g,0).mean());hminus.append(np.maximum(-g,0).mean())
                heps.append(np.maximum(-g-.005,0).mean());means.append(g.mean())
                mean_harm.append(g.mean()<-.005);any_harm.append(g.min()<-.005)
                all_harm.append(g.max()<-.005);conflict.append(g.min()<-.005 and g.max()>.005)
            if per_image_path:
                v=obj['d'][i,b['rater_present'][i]].cpu().numpy()
                append_csv(per_image_path,dict(epoch=epoch,split=dataset.split,image_id=b['image_id'][i],
                    group_id=b['group_id'][i],reference_count=len(v),J=float(obj['J_per_image'][i]),
                    rho=float((v>0).mean()),kappa_pm=int(v.min()<-1e-6 and v.max()>1e-6),
                    mean_loss=float(ell[i,b['rater_present'][i]].mean()),
                    mean_anchor_loss=float(ell0[i,b['rater_present'][i]].mean()),min_d=float(v.min()),max_d=float(v.max())))
    result={k:v/n for k,v in totals.items()}
    result.update(images=n,J_p50=float(np.quantile(js,.5)),J_p90=float(np.quantile(js,.9)),J_p99=float(np.quantile(js,.99)))
    if original:
        result.update(dice=float(np.mean(ds)),iou=float(np.mean(ious)),G_plus=float(np.mean(gplus)),
                      H_minus=float(np.mean(hminus)),H_epsilon=float(np.mean(heps)),mean_gain=float(np.mean(means)),
                      mean_harm=float(np.mean(mean_harm)),any_harm=float(np.mean(any_harm)),
                      all_harm=float(np.mean(all_harm)),sign_conflict=float(np.mean(conflict)))
        if abs(result['mean_gain']-(result['G_plus']-result['H_minus']))>1e-12:
            raise AssertionError('gain/harm identity failed')
    return result


def gradient_diagnostic(model,batch):
    was=model.training;model.eval();params=list(model.trainable_parameters())
    _,obj,_,_=calculate(model,batch)
    a=torch.autograd.grad(obj['seg'],params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(obj['weighted_risk'],params,allow_unused=True)
    zero=batch['image'].new_tensor(0.)
    na=sum((v.float().square().sum() for v in a if v is not None),start=zero).sqrt()
    nb=sum((v.float().square().sum() for v in b if v is not None),start=zero).sqrt()
    dot=sum((x.float().mul(y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None),start=zero)
    result=dict(seg_gradient_norm=float(na),weighted_risk_gradient_norm=float(nb),
                risk_seg_gradient_ratio=float(nb/na) if na>0 else None,
                risk_seg_gradient_cosine=float(dot/(na*nb)) if na>0 and nb>0 else None)
    model.train(was);return result


def check_inputs(cfg):
    root=Path(cfg['output_root']).resolve()
    if not Path(cfg['run_root']).resolve().is_relative_to(root): raise ValueError('Run writes outside isolated output')
    if not Path(cfg['cache_dir']).resolve().is_relative_to(root): raise ValueError('Cache writes outside isolated output')
    for protected in ['/home/featurize/rsi_runs/P2-IMA-M-v2','/home/featurize/work/RSI_Implementation_20261003','/home/featurize/rsi_data']:
        for candidate in [root,Path(cfg['run_root']).resolve(),Path(cfg['cache_dir']).resolve()]:
            if candidate==Path(protected) or candidate.is_relative_to(protected) or Path(protected).is_relative_to(candidate):
                raise ValueError('Outputs overlap historical protected assets')
    if cfg['seed']!=17 or cfg['epochs']['D']!=40 or cfg['effective_batch']!=8:
        raise ValueError('Registered seed/budget/batch changed')
    if (cfg['input_size']!=[256,256] or cfg['learning_rate']!=1e-4
        or cfg['weight_decay']!=1e-4 or cfg['gradient_clip']!=1.
        or cfg['rsi_lambda']!=3. or cfg['microbatch_start'] not in [1,2,4,8]
        or cfg['loss']!={'bce':.5,'soft_dice':.5,'epsilon_L':0.}):
        raise ValueError('Registered architecture/loss/optimizer settings changed')
    if not cfg['test_scoring_locked'] or not cfg['strict_test_scoring_lock']:
        raise PermissionError('Test lock required')
    if cfg['taskbook_sha256']!=sha256_file(cfg['taskbook']): raise ValueError('Taskbook changed')
    audit=json.loads((root/'data_audit.json').read_text())
    manifest_hash=sha256_file(cfg['manifest'])
    old_audit=json.loads(Path(cfg['audit']).read_text())
    if (audit.get('status')!='PASS' or audit.get('file_audit_complete') is not True
        or audit.get('errors')!=[] or audit.get('scope')!='all_selected_M_H_T1'
        or audit.get('final_references_sha256')!=manifest_hash
        or audit.get('historical_audit_sha256')!=sha256_file(cfg['audit'])
        or audit.get('historical_final_manifest_binding_verified') is not True
        or audit.get('test_scoring_locked') is not True
        or audit.get('test_model_scoring_performed') is not False):
        raise ValueError('New full data audit not complete, passing and identity-bound')
    for a,b,expected in [('unique_image_files','decoded_image_count',2145),('unique_mask_files','decoded_mask_count',4533)]:
        if audit.get(a)!=expected or audit.get(b)!=expected: raise ValueError('Incomplete real data decoding')
    for subset,split,images,refs in [('M','train',1471,3102),('M','val',223,471),('M','test',451,948),('H','val',47,104),('T1','val',20,49)]:
        if audit['counts'][subset][split]!={'images':images,'references':refs}: raise ValueError('Audited cohort differs')
    if (old_audit.get('file_audit_complete') is not True or old_audit.get('file_errors')!=[]
        or old_audit.get('near_duplicate_unresolved_candidates')!=[]):
        raise ValueError('Historical audit has missing or unresolved files')
    if old_audit['final_references_sha256']!=manifest_hash: raise ValueError('Historical audit/manifest mismatch')
    if manifest_hash!='3d99874f14ae5f316eec36e21141043d75bd4fd17ccfb31e3174198dee269596':
        raise ValueError('Final registered manifest differs')
    expected='1703ca22a1c39afeca05ffd0c12fe01606f41d5af81442d20102f130cc2ca491'
    if sha256_file(cfg['teacher_B'])!=expected: raise ValueError('Common selected B differs')
    assets=json.loads((root/'asset_audit.json').read_text())
    b_asset=next((a for a in assets.get('assets',[]) if a.get('run')=='B'),{})
    if (assets.get('status')!='PASS' or b_asset.get('status')!='PASS'
        or b_asset.get('path')!=cfg['teacher_B'] or b_asset.get('sha256')!=expected
        or b_asset.get('bytes')!=125077601 or b_asset.get('budget_epochs')!=40
        or b_asset.get('best_epoch')!=22 or b_asset.get('budget_completed') is not True
        or b_asset.get('checkpoint_read_success') is not True or b_asset.get('strict_model_load_success') is not True):
        raise ValueError('Common B asset audit incomplete or mismatched')
    return root,manifest_hash,expected


def train_group(cfg,group):
    root,manifest_hash,init_hash=check_inputs(cfg)
    if group not in EXPECTED_TRAINABLE: raise ValueError('Unknown group')
    acceptance=json.loads((root/'engineering_acceptance.json').read_text())
    if acceptance.get('status')!='PASS': raise ValueError('Engineering acceptance not passed')
    prereg=json.loads((root/'preregistration.json').read_text())
    if prereg['taskbook_sha256']!=cfg['taskbook_sha256']: raise ValueError('Preregistration/task mismatch')
    frozen=json.loads((root/'code_provenance.json').read_text())
    code=provenance(cfg)
    if code['source_bundle_hash']!=frozen['source_bundle_hash']: raise ValueError('Code changed after full freeze')
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    if (acceptance.get('configuration_sha256')!=config_hash
        or acceptance.get('manifest_sha256')!=manifest_hash or acceptance.get('teacher_B_sha256')!=init_hash
        or acceptance.get('taskbook_sha256')!=cfg['taskbook_sha256']
        or acceptance.get('code_provenance',{}).get('source_bundle_hash')!=code['source_bundle_hash']):
        raise ValueError('Engineering acceptance does not bind current config/code/teacher/data/task')
    cpu=json.loads((root/'engineering_cpu_tests.json').read_text())
    if (cpu.get('status')!='PASS' or cpu.get('passed_tests',0)<47
        or acceptance.get('cpu_tests_sha256')!=sha256_file(root/'engineering_cpu_tests.json')):
        raise ValueError('Necessary CPU control-flow acceptance missing or changed')
    signature=dict(config=cfg,group=group,seed=17,epochs=40,init_checkpoint_hash=init_hash,
                   manifest_hash=manifest_hash,source_bundle_hash=code['source_bundle_hash'],
                   preregistration_sha256=sha256_file(root/'preregistration.json'),
                   p0_audit_sha256={name:sha256_file(root/name) for name in ['asset_audit.json','data_audit.json','environment.json','resolved_paths.json']})
    run=Path(cfg['run_root'])/'seed17'/group;run.mkdir(parents=True,exist_ok=True)
    done_path=run/'DONE.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        if done.get('reused'):
            raise ValueError('Reused legacy group cannot be retrained in this directory')
        if (done['signature']!=signature or sha256_file(run/'best.pth')!=done['best_sha256']
            or sha256_file(run/'latest.pth')!=done['latest_sha256'] or done['epochs']!=40
            or done['optimizer_step_attempts']!=7360):
            raise ValueError('Completed run identity changed')
        with (run/'epoch_diagnostics.csv').open(newline='') as f: completed_rows=list(csv.DictReader(f))
        if [int(r['epoch']) for r in completed_rows]!=list(range(1,41)): raise ValueError('Completed epoch log missing/duplicated')
        return run/'best.pth'
    set_seed(17);torch.set_num_threads(4)
    model=build_model(cfg,group);model.train()
    parameters=list(model.trainable_parameters())
    assert sum(p.numel() for p in parameters)==EXPECTED_TRAINABLE[group]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:(e+1)/2 if e<2 else .5*(1+math.cos(math.pi*(e-2)/38)))
    scaler=torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    set_seed(17)
    epoch_start=1;best=-math.inf;best_epoch=None;steps=0;attempts=0;elapsed=0.;best_learned_hash=None
    frozen_hash=model.frozen_state_hash();teacher_hash=model.teacher_state_hash()
    if (run/'latest.pth').exists():
        last=torch.load(run/'latest.pth',map_location='cpu')
        if last['signature']!=signature: raise ValueError('Resume config/code/data/init/preregistration mismatch')
        model.load_state_dict(last['model'],strict=True);optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler']);scaler.load_state_dict(last['scaler'])
        epoch_start=last['epoch']+1;best=last['best_dice'];best_epoch=last['best_epoch']
        steps=last['optimizer_steps'];attempts=last['optimizer_step_attempts'];elapsed=last['wall_seconds']
        best_learned_hash=last['best_learned_state_hash']
        if model.frozen_state_hash()!=last['frozen_hash'] or model.teacher_state_hash()!=last['teacher_hash']:
            raise ValueError('Resume frozen teacher differs')
        if last['epoch']==best_epoch: save_checkpoint(run/'best.pth',last)
        else:
            selected=torch.load(run/'best.pth',map_location='cpu')
            if (selected['epoch']!=best_epoch or selected['best_dice']!=best or selected['signature']!=signature
                or selected['teacher_hash']!=teacher_hash or selected['frozen_hash']!=frozen_hash
                or selected['manifest_hash']!=manifest_hash or selected['init_checkpoint_hash']!=init_hash
                or selected['code']['source_bundle_hash']!=code['source_bundle_hash']
                or learned_state_hash(selected['model'],model.trainable_parameter_names())!=best_learned_hash):
                raise ValueError('Resume selected checkpoint missing/different')
            del selected
        saved_rng=last['rng'];del last
    else: saved_rng=None
    train,canonical,val=datasets(cfg)
    assert len(train)==1471 and len(val)==223
    fixed=device_batch(next(iter(loader(canonical,4,0))),torch.device('cuda'))
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',enabled=False): fixed_ref=model.forward_reference(fixed['image']).detach().clone()
    if saved_rng: restore_rng(saved_rng)
    else: set_seed(17)
    for name in ['epoch_diagnostics.csv','per_image_canonical_diagnostics.csv']:
        truncate_uncommitted_epochs(run/name,epoch_start)
    atomic_json(run/'config_resolved.json',dict(config=cfg,group=group,signature=signature))
    atomic_json(run/'environment.json',environment())
    optimizer_names=[n for n,p in model.named_parameters() if any(p is q for q in parameters)]
    atomic_json(run/'parameter_audit.json',dict(**model.architecture_audit(),optimizer_parameter_names=optimizer_names,
               learning_rate=cfg['learning_rate'],optimizer='AdamW',weight_decay=cfg['weight_decay']))
    wall_start=time.monotonic();mask_cache={}
    for epoch in range(epoch_start,41):
        tick=time.monotonic();model.train();train.set_epoch(epoch)
        batches=loader(train,cfg['microbatch_start'],cfg['workers'],seed=17,epoch=epoch,shuffle=True)
        accumulation=8//cfg['microbatch_start']
        assert accumulation*cfg['microbatch_start']==8
        totals={k:0. for k in FIELDS};seen=0;norms=[];clipped=0;skipped=0;order=[];delta_sum=0.;msg_sum=0.
        torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
        for bi,cpu in enumerate(batches):
            b=device_batch(cpu,torch.device('cuda'));bs=len(b['image_id']);order.extend(b['image_id'])
            first=(bi//accumulation)*accumulation*cfg['microbatch_start']
            window_samples=min(8,len(train)-first)
            with torch.autocast('cuda',enabled=cfg['amp']): out,obj,_,_=calculate(model,b)
            scaler.scale(obj['loss']*(bs/window_samples)).backward()
            for k in FIELDS: totals[k]+=float(obj[k].detach())*bs
            valid=b['pixel_valid'].bool();delta=(out['z1'].float()-out['z0'].float()).detach()
            delta_sum+=float(delta[valid].square().mean().sqrt())*bs
            if out['message'] is not None: msg_sum+=float(out['message'].detach().float().square().mean().sqrt())*bs
            seen+=bs
            if (bi+1)%accumulation==0 or bi+1==len(batches):
                info=finish_optimizer_step(optimizer,scaler,parameters,cfg['gradient_clip'])
                norms.append(info['norm']);clipped+=int(info['clipped']);skipped+=int(info['skipped'])
                steps+=int(not info['skipped']);attempts+=1;optimizer.zero_grad(set_to_none=True)
        if seen!=1471 or len(norms)!=184: raise RuntimeError('Dropped images or changed update budget')
        if skipped/len(norms)>.05: raise FloatingPointError('Excessive AMP overflows; INVALID engineering run')
        if model.frozen_state_hash()!=frozen_hash or model.teacher_state_hash()!=teacher_hash:
            raise RuntimeError('Frozen teacher/encoder hash drift')
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',enabled=False): now=model.forward_reference(fixed['image'])
        anchor_error=float((now-fixed_ref).abs().max())
        if not torch.equal(now,fixed_ref): raise RuntimeError('Fixed FP32 teacher drift')
        canonical_metrics=evaluate(model,canonical,cfg,original=False,
            per_image_path=run/'per_image_canonical_diagnostics.csv',epoch=epoch)
        val_metrics=evaluate(model,val,cfg,mask_cache=mask_cache,
            per_image_path=run/'per_image_canonical_diagnostics.csv',epoch=epoch)
        diagnostic=dict(seg_gradient_norm=None,weighted_risk_gradient_norm=None,
                        risk_seg_gradient_ratio=None,risk_seg_gradient_cosine=None)
        if epoch%cfg['gradient_diagnostic_every_epochs']==0: diagnostic=gradient_diagnostic(model,fixed)
        if model.frozen_state_hash()!=frozen_hash: raise RuntimeError('Diagnostic changed frozen state')
        if model.trainable_parameter_names()!=optimizer_names: raise RuntimeError('Trainable set changed')
        improved=val_metrics['dice']>best
        if improved:
            best=val_metrics['dice'];best_epoch=epoch
            best_learned_hash=learned_state_hash(model.state_dict(),optimizer_names)
        row=dict(seed=17,stage='TailProbe',run_id=group,group=group,epoch=epoch,
                 method='rsi' if group.endswith('RSI') else 'mean',weight=3. if group.endswith('RSI') else 0.,
                 **{f'augmented_train_{k}':v/seen for k,v in totals.items()},
                 **{f'canonical_train_{k}':v for k,v in canonical_metrics.items()},
                 **{f'val_{k}':v for k,v in val_metrics.items()},**diagnostic,**gradient_norm_summary(norms),
                 gradient_clip_fraction=clipped/184,amp_skipped_steps=skipped,
                 amp_skipped_step_fraction=skipped/184,augmented_message_rms=msg_sum/seen,
                 augmented_logit_delta_rms=delta_sum/seen,lr_group0=optimizer.param_groups[0]['lr'],
                 epoch_seconds=time.monotonic()-tick,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                 anchor_max_absolute_error=anchor_error,teacher_hash=teacher_hash,frozen_hash=frozen_hash,
                 best_epoch=best_epoch,best_val_dice=best,optimizer_steps=steps,
                 optimizer_step_attempts=attempts,skipped_optimizer_steps=attempts-steps,
                 train_images=seen,train_order_sha256=hashlib.sha256('\n'.join(order).encode()).hexdigest())
        append_csv(run/'epoch_diagnostics.csv',row);scheduler.step()
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(),rng=rng_state(),epoch=epoch,seed=17,stage='TailProbe',group=group,
                     method=row['method'],weight=row['weight'],best_dice=best,best_epoch=best_epoch,
                     optimizer_steps=steps,optimizer_step_attempts=attempts,skipped_optimizer_steps=attempts-steps,
                     config=cfg,manifest_hash=manifest_hash,code=code,frozen_hash=frozen_hash,teacher_hash=teacher_hash,
                     init_checkpoint_hash=init_hash,signature=signature,
                     best_learned_state_hash=best_learned_hash,
                     wall_seconds=elapsed+time.monotonic()-wall_start,
                     selection_rule='max_val_macro_mean_rater_hard_dice_original_coordinates_earlier_tie')
        save_checkpoint(run/'latest.pth',payload)
        if improved: save_checkpoint(run/'best.pth',payload)
        atomic_json(root/'progress.json',dict(status='training',group=group,epoch=epoch,budget=40,
            best_epoch=best_epoch,best_val_dice=best,actual_updates=steps,attempts=attempts,
            teacher_stable=True,test_scoring_locked=True))
        print(json.dumps({k:row[k] for k in ['group','epoch','val_dice','best_epoch','best_val_dice',
                          'epoch_seconds','optimizer_steps','optimizer_step_attempts','amp_skipped_steps','anchor_max_absolute_error']}),flush=True)
    if attempts!=7360: raise RuntimeError('Final update budget mismatch')
    done=dict(run=group,group=group,seed=17,stage='TailProbe',epochs=40,best_epoch=best_epoch,best_val_dice=best,
              signature=signature,best_sha256=sha256_file(run/'best.pth'),latest_sha256=sha256_file(run/'latest.pth'),
              teacher_hash=teacher_hash,frozen_hash=frozen_hash,optimizer_steps=steps,optimizer_step_attempts=attempts,
              wall_seconds=elapsed+time.monotonic()-wall_start,test_scoring_locked=True,reused=False)
    atomic_json(done_path,done)
    del model,optimizer,canonical,val,train;gc.collect();torch.cuda.empty_cache()
    return run/'best.pth'
