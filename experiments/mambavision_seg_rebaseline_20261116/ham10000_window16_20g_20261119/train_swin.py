"""Matched 16/8 MambaVision-S experiments on the frozen HAM10000 split."""
import argparse
import csv
import fcntl
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from prepare import CACHE, digest, write_json
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent

BASE=ROOT.parent
SDI=BASE/'decoder_literature_20261119/sdi_formal_20261119'
sys.path.insert(0,str(SDI))
from fit_sdi import build as build_sdi, setup, train as original
from model import HF, WEIGHT_SHA256, segmentation_loss
from stage3_confirmation.data import ImageData
sys.path.insert(0,str(BASE/'swin_local_window_20261119'))
from fit_local import LocalSystem
from train_stageB_color import rng_state, restore_rng

ARMS=('uper64','sdi_weak16','swin_local16')
SEED=42
BATCH=64
MAX_EPOCHS=200
PATIENCE=25

class HamImages(ImageData):
    def __init__(self, partitions):
        with (ROOT/'split.csv').open() as f: rows=list(csv.DictReader(f))
        self.seed=SEED
        self.images,self.masks,self.ids={},{},{}
        xs=np.load(CACHE/'images.npy',mmap_mode='r')
        ys=np.load(CACHE/'masks.npy',mmap_mode='r')
        for split in partitions:
            subset=[r for r in rows if r['split']==split]
            ix=[int(r['index']) for r in subset]
            self.ids[split]=[r['image_id'] for r in subset]
            self.images[split]=torch.from_numpy(xs[ix]).permute(0,3,1,2).contiguous().cuda()
            self.masks[split]=torch.from_numpy(ys[ix]).unsqueeze(1).to(device='cuda',dtype=torch.float32).div_(255.)
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    def batch(self,split,indices,matrices=None):
        images=self.images[split][indices].float().div_(255.)
        masks=self.masks[split][indices]
        if matrices is not None:
            grid=F.affine_grid(matrices[indices],images.shape,align_corners=False)
            images=F.grid_sample(images,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            masks=F.grid_sample(masks,grid,mode='nearest',padding_mode='zeros',align_corners=False)
        return (images-self.mean)/self.std,masks


class PrefixCache:
    """Keep only frozen features and paired augmented labels. Trainable stages run on every update."""
    def __init__(self,n):
        self.features=[torch.empty((n,c,h,h),dtype=torch.bfloat16,device='cuda') for c,h in [(96,64),(192,32),(384,16)]]
        self.targets=torch.empty((n,1,256,256),dtype=torch.float32,device='cuda')

    @torch.no_grad()
    def fill(self,model,data,split,order,geometry=None,check=False):
        start=0
        for ix in batches(order):
            stop=start+len(ix)
            x,y=data.batch(split,ix,geometry)
            with torch.autocast('cuda',dtype=torch.bfloat16):fs=model.prefix(x)
            assert all(v.dtype==torch.bfloat16 for v in fs)
            for dest,src in zip(self.features,fs):dest[start:stop].copy_(src)
            self.targets[start:stop].copy_(y)
            if check and start==0:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cached=self.logits(model,slice(0,2))
                    direct=model(x[:2])['main']
                    # Match batch geometry for the direct reference as well.
                    direct_same=model.from_cache(*(v[:2] for v in fs),(256,256))['main']
                assert torch.equal(direct_same,cached) and torch.isfinite(direct).all()
            start=stop
        assert start==len(order)<=len(self.targets)

    def training_batches(self,model,data,order,geometry):
        for block in order.split(len(self.targets)):
            self.fill(model,data,'train',block,geometry)
            start=0
            for ix in batches(block):
                yield ix,slice(start,start+len(ix))
                start+=len(ix)

    def logits(self,model,sl):
        return model.from_cache(*(v[sl] for v in self.features),(256,256))['main']

    @torch.inference_mode()
    def dice(self,model):
        model.eval()
        scores=[]
        for start in range(0,len(self.targets),BATCH):
            sl=slice(start,start+BATCH)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.logits(model,sl)
            if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite validation logits')
            pred=logits[:,0].float().sigmoid()>=.5
            gt=self.targets[sl,0]>=.5
            den=pred.sum((1,2))+gt.sum((1,2))
            scores.append(torch.where(den>0,2*(pred&gt).sum((1,2)).double()/den,1.).cpu())
        return float(torch.cat(scores).numpy().mean())


def weak_gain(module,args,delta):
    energy=delta.detach().float().square().mean(1,keepdim=True)
    q=energy.flatten(1).median(1).values[:,None,None,None]
    return (delta.float()*(1.+(energy<q).float())).to(delta.dtype)

def make_model(arm):
    setup(SEED)
    if arm=='swin_local16':
        model=LocalSystem('local16').cuda()
    else:
        model=build_sdi('product' if arm=='sdi_weak16' else 'add',SEED)
        for level,window in zip(model.encoder.model.levels[2:4],(16,8)):level.window_size=window
        if arm=='sdi_weak16':
            model.probe.decode_head.out.register_forward_hook(weak_gain)
    assert [v.window_size for v in model.encoder.model.levels[2:4]]==[16,8]
    for i,stage in enumerate(model.encoder.model.levels):
        assert all(p.requires_grad==(i in (2,3)) for p in stage.parameters())
    return model

def config(arm):
    return dict(arm=arm,seed=SEED,windows=[16,8],backbone='MambaVision-S ImageNet-1K',
                backbone_sha256=WEIGHT_SHA256,split_sha256=digest(ROOT/'split.csv'),training_code=digest(Path(__file__)),
                prefix_cache='BF16 blocked train F1,F2,mid (5632 examples) and full validation; refresh after augmentation',max_epochs=MAX_EPOCHS,patience=PATIENCE,selection='highest mean per-image foreground validation Dice, strict improvement, earliest tie',
                batch=BATCH,image_size=256,lr_encoder=1e-4,lr_decoder=1e-3,weight_decay=.01,
                scheduler='cosine to 1% over 200 epochs',threshold=.5,stage3_4_trainable=True,encoder_eval=True,
                sdi='weak2 on F2 SDI residual during training, validation and test' if arm=='sdi_weak16' else None)

def save(path,obj):
    tmp=path.with_suffix('.partial')
    torch.save(obj,tmp)
    os.replace(tmp,path)

def batches(order):
    sizes=list(order.split(BATCH))
    if len(sizes)>1 and len(sizes[-1])==1:
        pair=torch.cat(sizes[-2:])
        sizes[-2:]=[pair[:-2],pair[-2:]]
    return sizes

@torch.inference_mode()
def evaluate(model,data,split):
    model.eval()
    scores=[]
    ids=data.ids[split]
    for ix in torch.arange(len(ids),device='cuda').split(BATCH):
        x,y=data.batch(split,ix)
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x)['main']
        if not torch.isfinite(logits).all():raise FloatingPointError('evaluation logits')
        pred=logits[:,0].float().sigmoid()>=.5
        gt=y[:,0]>=.5
        den=pred.sum((1,2))+gt.sum((1,2))
        dice=torch.where(den>0,2*(pred&gt).sum((1,2)).double()/den,1.)
        scores.extend(dice.cpu().tolist())
    return float(np.mean(scores)),list(zip(ids,scores))

def test_best(model,dst,cfg):
    summary=json.loads((dst/'fit_complete.json').read_text())
    ckpt=dst/summary['best_checkpoint']
    assert digest(ckpt)==summary['best_sha256']
    if (dst/'test_result.json').exists():return
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    assert state['config']==cfg and state['epoch']==summary['best_epoch']
    model.load_state_dict(state['model'])
    del state
    data=HamImages(['test'])
    score,rows=evaluate(model,data,'test')
    tmp=dst/'test_per_image.csv.new'
    with tmp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','foreground_dice']);w.writerows(rows)
    tmp.replace(dst/'test_per_image.csv')
    write_json(dst/'test_result.json',dict(test_macro_dice=score,n=len(rows),best_epoch=summary['best_epoch'],
                                          epochs_trained=summary['epochs_trained'],best_sha256=summary['best_sha256']))
    print('TEST',cfg['arm'],len(rows),score,flush=True)

def fit(arm):
    dst=ROOT/arm;dst.mkdir(exist_ok=True)
    cfg=config(arm)
    if (dst/'config.json').exists():
        assert json.loads((dst/'config.json').read_text())==cfg
    write_json(dst/'config.json',cfg)
    assert digest(HF/'model.safetensors')==WEIGHT_SHA256
    model=make_model(arm)
    if (dst/'fit_complete.json').exists():
        test_best(model,dst,cfg);return
    data=HamImages(['train','val'])
    opt=original.optimizer(model)
    model.eval()
    valcache=PrefixCache(len(data.ids['val']))
    valcache.fill(model,data,'val',torch.arange(len(data.ids['val']),device='cuda'),check=True)
    del data.images['val'],data.masks['val']
    traincache=PrefixCache(min(5632,len(data.ids['train'])))
    history=[]
    latest=dst/'resume.pth'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['config']==cfg
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng']);history=ck['history'];del ck
    best=max(history,key=lambda r:r['val_macro_dice']) if history else None
    print('start',arm,'train',len(data.ids['train']),'val',len(data.ids['val']),'resume_epoch',len(history),flush=True)
    for epoch in range(len(history)+1,MAX_EPOCHS+1):
        if best is not None and epoch-1-best['epoch']>=PATIENCE:break
        tick=time.monotonic()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        order,geometry=data.geometry(epoch)
        factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/MAX_EPOCHS))/2
        for g in opt.param_groups:g['lr']=g['initial_lr']*factor
        loss_sum=0.
        for j,(ix,sl) in enumerate(traincache.training_batches(model,data,order,geometry)):
            target=traincache.targets[sl]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=traincache.logits(model,sl)
            loss=segmentation_loss(logits,target)
            if not torch.isfinite(loss):raise FloatingPointError('training loss')
            loss.backward()
            norms={g['name']:nn.utils.clip_grad_norm_(g['params'],1.,error_if_nonfinite=True) for g in opt.param_groups}
            if epoch==1 and j==0:
                assert all(float(v)>0 for v in norms.values())
                assert all(not s.training for s in model.encoder.model.levels)
                write_json(dst/'first_update.json',dict(batch=len(ix),loss=loss.item(),norms={k:float(v) for k,v in norms.items()},
                          actual_windows=[s.window_size for s in model.encoder.model.levels[2:]],decoder_parameters=sum(v.numel() for v in model.probe.parameters())))
                print('verified first step',arm,flush=True)
            opt.step()
            loss_sum+=loss.item()*len(ix)
            if epoch==1 and j in (19,59):print('step',arm,j+1,'elapsed',round(time.monotonic()-tick,1),flush=True)
        score=valcache.dice(model)
        row=dict(epoch=epoch,train_loss=loss_sum/len(order),val_macro_dice=score,compute_s=round(time.monotonic()-tick,3),
                 peak_GiB=round(torch.cuda.max_memory_allocated()/1024**3,2),reserved_GiB=round(torch.cuda.max_memory_reserved()/1024**3,2))
        history.append(row)
        improved=best is None or score>best['val_macro_dice']
        if improved:
            best=row
            save(dst/f'best_epoch_{epoch}.pth',dict(config=cfg,epoch=epoch,model=model.state_dict()))
        finished=(epoch-best['epoch']>=PATIENCE or epoch==MAX_EPOCHS)
        if improved or epoch%5==0 or finished:
            save(latest,dict(config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),history=history,rng=rng_state()))
            for p in dst.glob('best_epoch_*.pth'):
                if p.name!=f"best_epoch_{best['epoch']}.pth":p.unlink()
        row['wall_s']=round(time.monotonic()-tick,3)
        write_json(dst/'history.json',history)
        print('epoch',arm,epoch,'best_epoch',best['epoch'],'wall_s',row['wall_s'],flush=True)
        if finished:break
    best=max(history,key=lambda r:r['val_macro_dice'])
    checkpoint=dst/f"best_epoch_{best['epoch']}.pth"
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model']);del ck
    replay=valcache.dice(model)
    assert abs(replay-best['val_macro_dice'])<1e-10,(replay,best)
    write_json(dst/'fit_complete.json',dict(best_epoch=best['epoch'],epochs_trained=len(history),early_stopped=len(history)<MAX_EPOCHS,
               best_checkpoint=checkpoint.name,best_sha256=digest(checkpoint),split_sha256=cfg['split_sha256']))
    del opt,data,traincache,valcache
    gc.collect();torch.cuda.empty_cache()
    test_best(model,dst,cfg)
    if (dst/'test_result.json').exists():latest.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('arm',choices=['swin_local16'])
    args=parser.parse_args()
    with (ROOT/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fit(args.arm)
