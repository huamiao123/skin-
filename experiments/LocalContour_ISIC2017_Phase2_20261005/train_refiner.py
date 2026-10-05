"""Matched candidate labels and budget for local/global node attention."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

from refiner_model import NodeRefiner,parameter_count
from train_local import CACHE,ROOT,loss_and_accuracy,to_device

RANGES={'local8':8,'medium32':32,'global':None}
SEEDS=(17,23,42)

class NodeCases(Dataset):
    def __init__(self,split,seed):
        done=json.loads((CACHE/'COMPLETE.json').read_text())
        node_done=json.loads((CACHE/'node_context_COMPLETE.json').read_text())
        if done['status']!='PASS' or node_done['status']!='PASS':raise RuntimeError('Node cache incomplete')
        local_done=json.loads((ROOT/'models'/f'local_seed{seed}'/'DONE.json').read_text())
        if local_done['status']!='PASS':raise RuntimeError('Strong local model incomplete')
        self.records=[r for r in done['records'] if r['split']==split]
        self.context=np.load(CACHE/'node_context.npy',mmap_mode='r')
        self.local=np.load(ROOT/'models'/f'local_seed{seed}'/'candidate_scores.npy',mmap_mode='r')
        target=CACHE/'candidates_R32'
        self.labels={k:np.load(target/(k+'.npy'),mmap_mode='r') for k in ('valid','gt_distance','has_contour')}

    def __len__(self):return len(self.records)

    def __getitem__(self,i):
        rec=self.records[i];j=rec['index']
        return {'index':j,'image_id':rec['image_id'],
            'context':torch.from_numpy(np.array(self.context[j],copy=True)),
            'local_scores':torch.from_numpy(np.array(self.local[j],copy=True)),
            **{name:torch.from_numpy(np.array(value[j],copy=True)) for name,value in self.labels.items()}}

def forward(model,b):return model(b['context'],b['local_scores'],b['valid'])

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();losses=[];accs=[];weights=[]
    for raw in loader:
        b=to_device(raw,device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits=forward(model,b)
        loss,acc=loss_and_accuracy(logits,b)
        weight=int(b['has_contour'].sum())
        losses.append(float(loss)*weight);accs.append(float(acc)*weight);weights.append(weight)
    return sum(losses)/max(1,sum(weights)),sum(accs)/max(1,sum(weights))

@torch.no_grad()
def save_scores(model,seed,device,path):
    records=json.loads((CACHE/'COMPLETE.json').read_text())['records']
    out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(records),256,65))
    model.eval()
    for split in ('train','val'):
        loader=DataLoader(NodeCases(split,seed),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
        for raw in loader:
            b=to_device(raw,device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                scores=forward(model,b)
            values=scores.float().cpu().numpy().astype(np.float16)
            for j,index in enumerate(b['index'].cpu().numpy()):out[index]=values[j]
    out.flush()

def train(seed,attention,epochs=8):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit source changes before training')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=NodeRefiner(RANGES[attention]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    training=DataLoader(NodeCases('train',seed),batch_size=16,shuffle=True,num_workers=0,pin_memory=True,
                        generator=torch.Generator().manual_seed(seed))
    validation=DataLoader(NodeCases('val',seed),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    root=ROOT/'models'/f'{attention}_seed{seed}';root.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_epoch=None;rows=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[];accs=[]
        for step,raw in enumerate(training,1):
            b=to_device(raw,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits=forward(model,b)
                loss,acc=loss_and_accuracy(logits,b)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()));accs.append(float(acc.detach()))
            if step%50==0:print(f'{attention} seed={seed} epoch={epoch} step={step}/{len(training)}',flush=True)
        val_loss,val_acc=evaluate(model,validation,device)
        row={'attention':attention,'seed':seed,'epoch':epoch,
             'train_loss':float(np.mean(losses)),'train_candidate_acc':float(np.mean(accs)),
             'val_loss':val_loss,'val_candidate_acc':val_acc,'updates':len(training)*epoch}
        rows.append(row);print(json.dumps(row),flush=True)
        if val_loss<best-1e-6:
            best=val_loss;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'seed':seed,'attention':attention,
                        'epoch':epoch,'val_loss':best,'parameter_count':parameter_count(model)},root/'best.pth')
    with (root/'training_log.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    checkpoint=torch.load(root/'best.pth',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    save_scores(model,seed,device,root/'candidate_scores.npy')
    manifest={'status':'PASS','attention':attention,'attention_nodes':RANGES[attention],
              'seed':seed,'source_git_commit':commit,'parameters':parameter_count(model),
              'best_epoch':best_epoch,'epochs_run':epochs,'best_val_candidate_loss':best,
              'training_images':2000,'validation_images':150,'official_test_images_opened':0,
              'input':'frozen CNN node context44 + selected-seed strong local candidate scores65',
              'optimizer':'AdamW lr3e-4 weight_decay1e-4','batch_size':16,
              'selection':'minimum validation soft candidate cross entropy','checkpoint_path':str(root/'best.pth')}
    (root/'DONE.json').write_text(json.dumps(manifest,indent=2))
    print('DONE '+str(root),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--attention',choices=RANGES,required=True);p.add_argument('--epochs',type=int,default=8)
    a=p.parse_args();train(a.seed,a.attention,a.epochs)

if __name__=='__main__':main()
