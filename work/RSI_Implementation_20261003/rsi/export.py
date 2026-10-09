"""Original-coordinate, per-reference validation exports for actual interventions."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from .datasets import IMAMultiReferenceDataset, restore_logits
from .metrics import evaluate_reference, update_diagnostics
from .models import ControlledModel
from .objectives import per_reference_loss
from .runtime import PROJECT, atomic_json, load_config, sha256_file
from .train import device_batch, loader, _mask_original


@torch.no_grad()
def export_checkpoint(checkpoint,cfg,*,run_id,alpha=1.,weight=0.,method='d0',d0_checkpoint=None,
                      include_anchor=False,destination=None,seed=17):
    model=ControlledModel(pretrained=False,input_size=cfg['input_size'][0]).cuda()
    state=torch.load(checkpoint,map_location='cpu');model.load_state_dict(state['model']);model.set_stage('D');model.eval()
    checkpoint_hash=sha256_file(checkpoint);manifest_hash=sha256_file(cfg['manifest']);d0=None
    if d0_checkpoint:
        d0=ControlledModel(pretrained=False,input_size=cfg['input_size'][0]).cuda()
        d0.load_state_dict(torch.load(d0_checkpoint,map_location='cpu')['model']);d0.set_stage('D');d0.eval()
        if d0.anchor_state_hash()!=model.anchor_state_hash(): raise ValueError('Methods do not share a fixed anchor')
    rows=[];mask_cache={};output_cache={};d0_cache={};tick=time.monotonic()
    for subset in ['M','H','T1']:
        dataset=IMAMultiReferenceDataset(cfg['manifest'],split='val',subset=subset,cache_dir=cfg['cache_dir'])
        for cpu in loader(dataset,cfg['eval_batch'],cfg['workers']):
            b=device_batch(cpu,torch.device('cuda'))
            if subset == 'M':
                with torch.autocast('cuda',enabled=cfg['amp']):
                    pair=model.forward_pair(b['image'],b['pixel_valid'],alpha=alpha)
                    d0_pair=d0.forward_pair(b['image'],b['pixel_valid']) if d0 is not None else None
                for i,image_id in enumerate(b['image_id']):
                    output_cache[image_id]={key:pair[key][i].detach().cpu() for key in ['z0','z1','message']}
                    if d0_pair is not None: d0_cache[image_id]=d0_pair['z1'][i].detach().cpu()
            else:
                # Source sensitivity changes only the selected references. Use
                # the exact M outputs, including for different last-batch sizes.
                pair={key:torch.stack([output_cache[i][key] for i in b['image_id']]).to(b['image'].device)
                      for key in ['z0','z1','message']}
                d0_pair=({'z1':torch.stack([d0_cache[i] for i in b['image_id']]).to(b['image'].device)}
                         if d0 is not None else None)
            ell=per_reference_loss(pair['z1'],b['masks'],b['rater_present'],b['pixel_valid'],return_components=True)
            ell0=per_reference_loss(pair['z0'],b['masks'],b['rater_present'],b['pixel_valid'],return_components=True)
            ell_d0=per_reference_loss(d0_pair['z1'],b['masks'],b['rater_present'],b['pixel_valid']) if d0_pair else None
            for i in range(len(b['image_id'])):
                z=restore_logits(pair['z1'][i],b['geometry'][i]).cpu().numpy()
                z0=restore_logits(pair['z0'][i],b['geometry'][i]).cpu().numpy()
                zd0=restore_logits(d0_pair['z1'][i],b['geometry'][i]).cpu().numpy() if d0_pair else None
                update=update_diagnostics(pair['message'][i:i+1],pair['z1'][i:i+1],pair['z0'][i:i+1],b['pixel_valid'][i:i+1])
                pred=z>=0;anchor=z0>=0
                update['changed_pixel_fraction_canonical']=update['changed_pixel_fraction']
                update['changed_pixel_fraction']=float(np.mean(pred!=anchor))
                for r,ref in enumerate(b['references'][i]):
                    gt=_mask_original(ref,mask_cache)
                    metrics=evaluate_reference(z,gt,z0,zd0)
                    # Preserve two loss views: training-coordinate canonical loss
                    # determines d/risk, while the metrics evaluator also exposes
                    # original-coordinate loss for descriptive reporting.
                    for name in ['loss','bce','soft_dice','loss_anchor','loss_D0','gain_loss']:
                        if name in metrics: metrics[name+'_orig']=metrics.pop(name)
                    canonical=dict(loss=float(ell['loss'][i,r]),bce=float(ell['bce'][i,r]),
                        soft_dice=float(ell['soft_dice'][i,r]),loss_anchor=float(ell0['loss'][i,r]),
                        gain_loss=float(ell0['loss'][i,r]-ell['loss'][i,r]),loss_view='canonical_letterbox_valid')
                    if ell_d0 is not None: canonical['loss_D0']=float(ell_d0[i,r])
                    transitions=dict(fp_added=int(np.count_nonzero(pred & ~anchor & ~gt)),
                        fn_added=int(np.count_nonzero(~pred & anchor & gt)),
                        fp_removed=int(np.count_nonzero(~pred & anchor & ~gt)),
                        fn_removed=int(np.count_nonzero(pred & ~anchor & gt)))
                    common=dict(protocol_id=cfg['protocol_id'],seed=seed,run_id=run_id,
                        checkpoint_hash=checkpoint_hash,manifest_hash=manifest_hash,
                        image_id=b['image_id'][i],group_id=b['group_id'][i],split='val',subset=subset,
                        annotator_id=ref.get('annotator_id',ref.get('annotator')),
                        reference_id=ref['reference_id'],seg_filename=ref['seg_filename'],
                        tool=ref['tool'],skill_level=ref['skill_level'],reference_count=len(b['references'][i]),
                        action=alpha,alpha=alpha,method=method,**{'lambda':weight})
                    rows.append({**common,**metrics,**canonical,**transitions,**update})
                    if include_anchor:
                        anchor_metrics=evaluate_reference(z0,gt,z0,zd0)
                        for name in ['loss','bce','soft_dice','loss_anchor','loss_D0','gain_loss']:
                            if name in anchor_metrics: anchor_metrics[name+'_orig']=anchor_metrics.pop(name)
                        ar=dict(common,run_id='Anchor',action=0.,alpha=0.,method='anchor',**{'lambda':0.})
                        rows.append(dict(ar,**dict(anchor_metrics,loss=float(ell0['loss'][i,r]),bce=float(ell0['bce'][i,r]),
                            soft_dice=float(ell0['soft_dice'][i,r]),loss_anchor=float(ell0['loss'][i,r]),gain_loss=0.,
                            loss_view='canonical_letterbox_valid',fp_added=0,fn_added=0,fp_removed=0,fn_removed=0,
                            message_rms=0.,logit_delta_rms=0.,changed_pixel_fraction=0.)))
        print(f'Exported {run_id} {subset}: {len(dataset)} validation images',flush=True)
    destination=Path(destination or PROJECT/'outputs/per_reference'/f'seed{seed}'/f'{run_id}.csv')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(set(r) for r in rows)))
    temp=destination.with_suffix('.tmp.csv')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(destination)
    atomic_json(destination.with_suffix('.json'),dict(run_id=run_id,checkpoint_hash=checkpoint_hash,
        reference_rows=len(rows),seconds=time.monotonic()-tick,split='val',test_scoring_locked=True,
        actual_tail_recomputation=True,alpha=alpha))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config')
    p.add_argument('--run-id',required=True);p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--weight',type=float,default=0.);p.add_argument('--method',default='d0')
    p.add_argument('--d0-checkpoint');p.add_argument('--include-anchor',action='store_true')
    a=p.parse_args();export_checkpoint(a.checkpoint,load_config(a.config),run_id=a.run_id,alpha=a.alpha,
        weight=a.weight,method=a.method,d0_checkpoint=a.d0_checkpoint,include_anchor=a.include_anchor)


if __name__=='__main__':main()
