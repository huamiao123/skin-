"""T0 real-16 engineering feasibility; never evidence of method effectiveness."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .datasets import IMAMultiReferenceDataset, collate_multi_reference, restore_logits
from .models import ControlledModel
from .objectives import per_reference_loss, objective_from_losses, weighted_reference_quantile, image_reference_mean
from .runtime import PROJECT, atomic_json, environment, load_config, save_checkpoint, set_seed, sha256_file
from .train import device_batch, forward_pair, loader, make_optimizer


def overlays(dataset, destination):
    fig,axes=plt.subplots(4,4,figsize=(16,12))
    for ax,refs in zip(axes.flat,dataset.samples):
        with Image.open(refs[0]['image_path']) as f: image=np.asarray(f.convert('RGB'))
        ax.imshow(image)
        for j,ref in enumerate(refs):
            with Image.open(ref['mask_path']) as f: mask=np.asarray(f.convert('L'))>0
            ax.contour(mask.astype(float),levels=[.5],colors=[plt.cm.tab10(j)],linewidths=.7)
        ax.set_title(f'{refs[0]["image_id"]} ({len(refs)} refs)',fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(destination,dpi=110);plt.close(fig)


def run(manifest=None,steps=30):
    cfg=load_config();cfg['amp']=False
    manifest=Path(manifest or PROJECT/'data_manifests/train16/final_references.csv')
    audit=json.loads((manifest.parent/'file_audit.json').read_text())
    if not audit['file_audit_complete'] or audit['scope']!='T0_training_subset':
        raise ValueError('T0 requires its separately audited training-only manifest')
    set_seed(17,deterministic=True);torch.set_num_threads(4)
    ds=IMAMultiReferenceDataset(manifest,train=False,cache_dir=cfg['cache_dir'])
    if len(ds)!=16: raise ValueError('The preselected T0 cohort has exactly16 training images')
    output=PROJECT/'outputs/T0';output.mkdir(parents=True,exist_ok=True)
    overlays(ds,output/'reference_alignment_overlays.png')
    batches=[device_batch(b,torch.device('cuda')) for b in loader(ds,4,0)]
    model=ControlledModel(pretrained=True).cuda()
    results=dict(scope='T0_engineering_only',environment=environment(),manifest_sha256=sha256_file(manifest),
                 images=16,references=sum(len(x) for x in ds.samples),test_scoring_locked=True,
                 no_mechanism_claim=True,pretrained_audit=str(PROJECT/'outputs/pretrained_audit.json'))
    curves=[]
    for stage in ['A_CNN','A_T']:
        model.set_stage(stage);optimizer,_=make_optimizer(model,stage,cfg)
        model.eval()
        with torch.no_grad():
            initial=np.mean([float(image_reference_mean(per_reference_loss(
                forward_pair(model,b,stage)['z1'],b['masks'],b['rater_present'],b['pixel_valid']),
                b['rater_present'])) for b in batches])
        torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
        for step in range(steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            # Real effective batch8: two microbatches4, identically weighted.
            loss_value=0.
            for offset in [0,1]:
                b=batches[(2*step+offset)%len(batches)]
                out=forward_pair(model,b,stage)
                ell=per_reference_loss(out['z1'],b['masks'],b['rater_present'],b['pixel_valid'])
                obj=objective_from_losses(ell,ell.detach(),b['rater_present'])
                (obj['loss']*.5).backward();loss_value+=float(obj['loss'])*.5
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.);optimizer.step()
            curves.append(dict(stage=stage,step=step+1,loss=loss_value))
        model.eval()
        with torch.no_grad():
            final=np.mean([float(objective_from_losses(
                per_reference_loss(forward_pair(model,b,stage)['z1'],b['masks'],b['rater_present'],b['pixel_valid']),
                torch.zeros_like(b['rater_present'],dtype=torch.float32),b['rater_present'])['seg']) for b in batches])
        results[stage]=dict(steps=steps,initial_loss=float(initial),final_loss=float(final),
                           learns=bool(final<initial),seconds=time.monotonic()-tick,
                           peak_cuda_bytes=torch.cuda.max_memory_allocated())
        print(stage,results[stage],flush=True)
    model.set_stage('B');model.eval();b=batches[0]
    with torch.no_grad():
        pair=model.forward_pair(b['image'],b['pixel_valid']);zero=model(b['image'],b['pixel_valid'],alpha=0)
    torch.testing.assert_close(pair['z0'],zero,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(pair['z0'],pair['z1'],atol=1e-6,rtol=1e-6)
    anchor_hash=model.anchor_state_hash();frozen_hash=model.frozen_state_hash();anchor=pair['z0'].clone()
    opt=torch.optim.AdamW(model.trainable_parameters(),lr=3e-4,weight_decay=1e-4)
    for step in range(8):
        model.train();opt.zero_grad(set_to_none=True);b=batches[step%len(batches)]
        pair=model.forward_pair(b['image'],b['pixel_valid'])
        ell=per_reference_loss(pair['z1'],b['masks'],b['rater_present'],b['pixel_valid'])
        ell0=per_reference_loss(pair['z0'],b['masks'],b['rater_present'],b['pixel_valid'])
        obj=objective_from_losses(ell,ell0,b['rater_present'])
        obj['loss'].backward()
        if step==0:
            results['Wo_first_gradient_norm']=float(model.message.Wo.weight.grad.norm())
            assert results['Wo_first_gradient_norm']>0
        if step==1:
            results['Q_second_gradient_norm']=float(model.message.q_proj.weight.grad.norm())
            assert results['Q_second_gradient_norm']>0
        assert all(p.grad is None for p in model.tail.parameters())
        opt.step();curves.append(dict(stage='B',step=step+1,loss=float(obj['loss'])))
    model.eval()
    with torch.no_grad(): now=model.forward_cnn(batches[0]['image'])
    torch.testing.assert_close(anchor,now,atol=1e-6,rtol=1e-6)
    assert anchor_hash==model.anchor_state_hash() and frozen_hash==model.frozen_state_hash()
    results['anchor_max_absolute_error']=float((anchor-now).abs().max())
    # q is exactly image-then-reference weighted on this diagnostic training cohort.
    q_losses=[];max_r=max(len(refs) for refs in ds.samples)
    with torch.no_grad():
        for b in batches:
            pair=model.forward_pair(b['image'],b['pixel_valid'])
            ell=per_reference_loss(pair['z1'],b['masks'],b['rater_present'],b['pixel_valid'])
            q_losses.extend(ell[i,b['rater_present'][i]].cpu() for i in range(len(b['image_id'])))
    qs=torch.zeros(16,max_r);qp=torch.zeros_like(qs,dtype=torch.bool)
    for i,x in enumerate(q_losses):qs[i,:len(x)]=x;qp[i,:len(x)]=True
    q=float(weighted_reference_quantile(qs,qp,.75));start={k:v.detach().clone() for k,v in model.message.state_dict().items()}
    results['objectives']={}
    for method in ['d0','rsi','mean_hinge','abs_hard']:
        model.message.load_state_dict(start);model.set_stage('D')
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=1e-4,weight_decay=1e-4)
        for step in range(5):
            b=batches[step%4];opt.zero_grad(set_to_none=True)
            pair=model.forward_pair(b['image'],b['pixel_valid'])
            ell=per_reference_loss(pair['z1'],b['masks'],b['rater_present'],b['pixel_valid'])
            ell0=per_reference_loss(pair['z0'],b['masks'],b['rater_present'],b['pixel_valid'])
            obj=objective_from_losses(ell,ell0,b['rater_present'],method=method,weight=0 if method=='d0' else 1,q=q)
            obj['loss'].backward();opt.step()
            assert torch.isfinite(obj['loss'])
            curves.append(dict(stage=method,step=step+1,loss=float(obj['loss'])))
        results['objectives'][method]={k:float(obj[k]) for k in ['loss','seg','risk','rho','J','kappa_pm']}
        assert anchor_hash==model.anchor_state_hash()
    model.message.load_state_dict(start);model.eval();b=batches[0]
    with torch.no_grad():
        shrink=[]
        for alpha in [0.,1/3,2/3,1.]:
            pair=model.forward_pair(b['image'],b['pixel_valid'],alpha=alpha)
            shrink.append(dict(alpha=alpha,logit_delta_rms=float((pair['z1']-pair['z0']).square().mean().sqrt())))
    results['alpha_actual_tail_engineering_diagnostic_from_B']=shrink
    results['learning_checks_passed']=all(results[s]['learns'] for s in ['A_CNN','A_T'])
    results['invariants_passed']=True
    results['T0_train16_engineering_passed']=results['learning_checks_passed']
    results['full_data_T1_requires_separate_complete_file_audit']=True
    with (output/'train16_curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=curves[0].keys());w.writeheader();w.writerows(curves)
    atomic_json(output/'feasibility_report.json',results)
    save_checkpoint('/home/featurize/rsi_runs/T0/train16_debug.pth',dict(model=model.state_dict(),scope='debug_only_not_full_stage_init'))
    print(json.dumps(results,indent=2),flush=True)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--steps',type=int,default=30)
    a=p.parse_args();run(a.manifest,a.steps)


if __name__=='__main__':main()
