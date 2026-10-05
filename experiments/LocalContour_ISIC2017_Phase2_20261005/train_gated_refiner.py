"""Explicit keep/repair ablation on the same frozen CNN and dense candidates."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from refiner_model import GatedNodeRefiner,parameter_count
from train_refiner import NodeCases,RANGES
from train_local import CACHE,ROOT,to_device

SEEDS=(17,23,42)

def losses(scores,gate_logits,b):
    valid=b['valid'].bool();has=b['has_contour'].bool()
    distance=b['gt_distance'].float()
    other=valid.clone();other[...,32]=False
    closest_nonzero=distance.masked_fill(~other,1e4).min(dim=-1).values
    zero_distance=distance[...,32]
    repair=(zero_distance>2)&(closest_nonzero+0.5<zero_distance)&(closest_nonzero<=4)
    repair=repair&has[:,None]
    soft=F.softmax((-distance/1.5).masked_fill(~other,-1e4),dim=-1)
    zero=F.one_hot(torch.zeros_like(repair,dtype=torch.long),65).float()
    target=torch.where(repair[...,None],soft,zero)
    masked=scores.float().masked_fill(~valid,-1e4)
    candidate_loss=-(target*F.log_softmax(masked,dim=-1)).sum(dim=-1)
    gate_loss=F.binary_cross_entropy_with_logits(gate_logits.float(),repair.float(),
                                                  pos_weight=torch.tensor(2.,device=gate_logits.device),reduction='none')
    use=has[:,None].expand_as(repair)
    if not use.any():return masked.sum()*0,0.,0.,0.
    c=candidate_loss[use].mean();g=gate_loss[use].mean()
    gate_accuracy=((gate_logits[use]>=0)==repair[use]).float().mean()
    return c+g,float(c.detach()),float(g.detach()),float(gate_accuracy.detach())

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();total=0.;candidate=0.;gate=0.;accuracy=0.;count=0
    for raw in loader:
        b=to_device(raw,device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            score,gate_logit=model(b['context'],b['local_scores'],b['valid'])
        loss,c,g,a=losses(score,gate_logit,b)
        weight=int(b['has_contour'].sum());total+=float(loss)*weight
        candidate+=c*weight;gate+=g*weight;accuracy+=a*weight;count+=weight
    return {'loss':total/max(1,count),'candidate_loss':candidate/max(1,count),
            'gate_loss':gate/max(1,count),'gate_accuracy':accuracy/max(1,count)}

@torch.no_grad()
def save_outputs(model,seed,device,path_scores,path_gate):
    records=json.loads((CACHE/'COMPLETE.json').read_text())['records'];n=len(records)
    score_out=np.lib.format.open_memmap(path_scores,mode='w+',dtype=np.float16,shape=(n,256,65))
    gate_out=np.lib.format.open_memmap(path_gate,mode='w+',dtype=np.float16,shape=(n,256))
    model.eval()
    for split in ('train','val'):
        loader=DataLoader(NodeCases(split,seed),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
        for raw in loader:
            b=to_device(raw,device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                score,gate_logit=model(b['context'],b['local_scores'],b['valid'])
            s=score.float().cpu().numpy().astype(np.float16)
            g=gate_logit.float().cpu().numpy().astype(np.float16)
            for j,index in enumerate(b['index'].cpu().numpy()):score_out[index]=s[j];gate_out[index]=g[j]
    score_out.flush();gate_out.flush()

def train(seed,attention,epochs=8):
    if attention not in ('local8','global'):raise ValueError('The gate ablation compares only local8 and global')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit source changes before training')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=GatedNodeRefiner(RANGES[attention]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    training=DataLoader(NodeCases('train',seed),batch_size=16,shuffle=True,num_workers=0,pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))
    validation=DataLoader(NodeCases('val',seed),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    output=ROOT/'models'/f'{attention}_gate_seed{seed}';output.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_epoch=None;rows=[]
    for epoch in range(1,epochs+1):
        model.train();train_losses=[]
        for step,raw in enumerate(training,1):
            b=to_device(raw,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                scores,gate=model(b['context'],b['local_scores'],b['valid'])
                loss,_,_,_=losses(scores,gate,b)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            train_losses.append(float(loss.detach()))
            if step%50==0:print(f'{attention}_gate seed={seed} epoch={epoch} step={step}/{len(training)}',flush=True)
        validation_result=evaluate(model,validation,device)
        row={'attention':attention,'seed':seed,'epoch':epoch,'train_loss':float(np.mean(train_losses)),
             'val_loss':validation_result['loss'],'val_candidate_loss':validation_result['candidate_loss'],
             'val_gate_loss':validation_result['gate_loss'],'val_gate_accuracy':validation_result['gate_accuracy'],
             'updates':len(training)*epoch}
        rows.append(row);print(json.dumps(row),flush=True)
        if validation_result['loss']<best-1e-6:
            best=validation_result['loss'];best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'seed':seed,'attention':attention,
                        'epoch':epoch,'val_loss':best,'parameter_count':parameter_count(model)},output/'best.pth')
    with (output/'training_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    best_model=torch.load(output/'best.pth',map_location=device,weights_only=True)
    model.load_state_dict(best_model['state_dict'],strict=True)
    save_outputs(model,seed,device,output/'candidate_scores.npy',output/'gate_logits.npy')
    manifest={'status':'PASS','attention':attention,'attention_nodes':RANGES[attention],
              'seed':seed,'source_git_commit':commit,'parameters':parameter_count(model),
              'best_epoch':best_epoch,'epochs_run':epochs,'best_val_loss':best,
              'training_images':2000,'validation_images':150,'official_test_images_opened':0,
              'GT_repair_label':'d0>2 and best_nonzero+0.5<d0 and best_nonzero<=4, where distances are EDT proxy at candidate points',
              'candidate_loss':'soft nearest-boundary target if repair, zero one-hot if keep',
              'gate_loss':'BCEWithLogits pos_weight2; total=candidate+gate',
              'optimizer':'AdamW lr3e-4 weight_decay1e-4','batch_size':16,
              'selection':'minimum val150 combined candidate+gate loss',
              'checkpoint_path':str(output/'best.pth')}
    (output/'DONE.json').write_text(json.dumps(manifest,indent=2))
    print('DONE '+str(output),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--attention',choices=('local8','global'),required=True);p.add_argument('--epochs',type=int,default=8)
    a=p.parse_args();train(a.seed,a.attention,a.epochs)

if __name__=='__main__':main()
