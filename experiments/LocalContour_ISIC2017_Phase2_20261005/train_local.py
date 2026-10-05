"""Three-seed strong local baseline, using only official train GT and val selection."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from local_model import StrongLocal,parameter_count

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'
RADIUS=32
SEEDS=(17,23,42)

class CachedCases(Dataset):
    def __init__(self,split):
        self.split=split
        done=json.loads((CACHE/'COMPLETE.json').read_text())
        self.records=[r for r in done['records'] if r['split']==split]
        self.index=np.array([r['index'] for r in self.records],dtype=int)
        target=CACHE/f'candidates_R{RADIUS}'
        if json.loads((target/'COMPLETE.json').read_text())['status']!='PASS':raise RuntimeError('Candidate cache incomplete')
        self.m={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('features','rgb','probability','boundary')}
        self.c={name:np.load(target/(name+'.npy'),mmap_mode='r') for name in ('points','source_points','normals','valid','gt_distance','has_contour')}

    def __len__(self):return len(self.index)

    def __getitem__(self,i):
        j=int(self.index[i])
        return {'index':j,'image_id':self.records[i]['image_id'],
          **{name:torch.from_numpy(np.array(arr[j],copy=True)) for name,arr in self.m.items()},
          **{name:torch.from_numpy(np.array(arr[j],copy=True)) for name,arr in self.c.items()}}

def to_device(batch,device):
    return {key:(value.to(device,non_blocking=True) if torch.is_tensor(value) else value) for key,value in batch.items()}

def forward(model,batch):
    return model(batch['features'].float(),batch['rgb'].float(),
                 batch['probability'].float(),batch['boundary'].float(),
                 batch['points'].float(),batch['source_points'].float(),batch['normals'].float())

def loss_and_accuracy(logits,batch):
    valid=batch['valid'].bool()
    has=batch['has_contour'].bool()
    masked=logits.float().masked_fill(~valid,-1e4)
    distance=batch['gt_distance'].float()
    target=F.softmax((-distance/1.5).masked_fill(~valid,-1e4),dim=-1)
    # No nearby candidate is evidence to preserve the CNN boundary here.
    fixable=distance.masked_fill(~valid,1e4).min(dim=-1).values<=4.0
    keep=F.one_hot(torch.zeros_like(fixable,dtype=torch.long),masked.shape[-1]).float()
    target=torch.where(fixable[...,None],target,keep)
    per_node=-(target*F.log_softmax(masked,dim=-1)).sum(dim=-1)
    mask=has[:,None].expand_as(per_node)
    loss=per_node[mask].mean() if mask.any() else masked.sum()*0
    label=target.argmax(dim=-1)
    acc=(masked.argmax(dim=-1)[mask]==label[mask]).float().mean() if mask.any() else torch.tensor(0.,device=logits.device)
    return loss,acc

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); total_loss=0.;total_acc=0.;count=0
    for raw in loader:
        b=to_device(raw,device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            score=forward(model,b)
        loss,acc=loss_and_accuracy(score,b)
        size=int(b['has_contour'].sum().item())
        total_loss+=float(loss)*size;total_acc+=float(acc)*size;count+=size
    return total_loss/max(1,count),total_acc/max(1,count)

@torch.no_grad()
def save_scores(model,device,path):
    records=json.loads((CACHE/'COMPLETE.json').read_text())['records']
    out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(records),256,65))
    model.eval()
    for split in ('train','val'):
        loader=DataLoader(CachedCases(split),batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
        for raw in loader:
            b=to_device(raw,device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                scores=forward(model,b)
            scores=scores.float().cpu().numpy().astype(np.float16)
            for j,index in enumerate(b['index'].cpu().numpy()):out[index]=scores[j]
    out.flush()

def train_seed(seed,epochs=8,width=64):
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit phase-2 source changes before training')
    feature_meta=json.loads((CACHE/'COMPLETE.json').read_text())
    candidate_meta=json.loads((CACHE/f'candidates_R{RADIUS}/COMPLETE.json').read_text())
    if feature_meta.get('gt_shape_verified_for_all')!=2150 or candidate_meta.get('gt_distance_label_source') not in ('complete 2D official GT mask','repaired complete 2D official GT mask'):
        raise RuntimeError('Complete 2D GT and candidate labels must pass the repaired-cache audit')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=StrongLocal(width=width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    training=DataLoader(CachedCases('train'),batch_size=4,shuffle=True,num_workers=0,pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))
    validation=DataLoader(CachedCases('val'),batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    output=ROOT/'models'/(f'local_seed{seed}' if width==64 else f'local_large_seed{seed}')
    output.mkdir(parents=True,exist_ok=True)
    best=np.inf;best_epoch=None;log=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[];accs=[]
        for number,raw in enumerate(training,1):
            b=to_device(raw,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                scores=forward(model,b)
                loss,acc=loss_and_accuracy(scores,b)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();losses.append(float(loss.detach()));accs.append(float(acc.detach()))
            if number%100==0:print(f'seed={seed} epoch={epoch} step={number}/{len(training)} train_loss={np.mean(losses):.4f}',flush=True)
        val_loss,val_acc=evaluate(model,validation,device)
        row={'seed':seed,'epoch':epoch,'train_loss':float(np.mean(losses)),'train_candidate_acc':float(np.mean(accs)),
             'val_loss':val_loss,'val_candidate_acc':val_acc,'updates':len(training)*epoch}
        log.append(row);print(json.dumps(row),flush=True)
        if val_loss<best-1e-6:
            best=val_loss;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'seed':seed,'epoch':epoch,
                        'val_loss':best,'parameter_count':parameter_count(model)},output/'best.pth')
    with (output/'training_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=log[0].keys());w.writeheader();w.writerows(log)
    checkpoint=torch.load(output/'best.pth',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    save_scores(model,device,output/'candidate_scores.npy')
    manifest={'status':'PASS','seed':seed,'best_epoch':best_epoch,'epochs_run':epochs,'width':width,
              'source_git_commit':source_commit,
              'best_val_candidate_loss':best,'parameters':parameter_count(model),'receptive_field':'9x9 on frozen CNN feature maps',
              'candidate_config':'dense 65 positions, R32, 256 nodes','training_images':2000,'validation_images':150,
              'test_images_opened':0,'optimizer':'AdamW lr1e-3 weight_decay1e-4','batch_size':4,
              'input':'frozen CNN decoder 32 + normalized RGB 3 + CNN probability 1 + frozen boundary probability 1; candidate offset/normal/position',
              'selection':'lowest validation soft candidate cross entropy',
              'checkpoint_path':str(output/'best.pth')}
    (output/'DONE.json').write_text(json.dumps(manifest,indent=2))
    print('DONE '+str(output),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--width',type=int,choices=(64,96),default=64)
    args=parser.parse_args();train_seed(args.seed,args.epochs,args.width)

if __name__=='__main__':main()
