"""Frozen same-stage readouts, trained only on ISIC2018 training images."""
import csv
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
import official

torch.cuda.set_per_process_memory_fraction(.10)
torch.set_num_threads(2)
cv2.setNumThreads(1)
stages=((2,16,384,{'E':2,'M':3,'H':4,'A':6}),(3,8,768,{'E':1,'M':2,'H':3,'A':4}))
arms={'A':('A','A'),'M':('M','M'),'D':('D','D'),'AM':('A','M'),'MA':('M','A'),'AH':('A','H'),'AE':('A','E')}
fit=json.loads((OUT.parent/'ISIC2018'/'fit_complete.json').read_text())
ck=OUT.parent/'ISIC2018'/fit['best_checkpoint']
assert official.sha(ck)==fit['sha256']
official.recipe.setup(42)
net=official.Segmenter()
net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
net.cuda().eval().requires_grad_(False)
data=official.dataset('ISIC2018',['train','val'])
assert set(data.ids['train']).isdisjoint(data.ids['val'])
taps={}
def grab(key):
    def hook(module,args,value):
        taps[key]=value
    return hook
gen=torch.Generator().manual_seed(835)
projections={}
for s,w,c,indices in stages:
    for name,b in indices.items():
        block=net.backbone.levels[s].blocks[b]
        expected='MambaVisionMixer' if name in ('M','E') else 'Attention'
        assert block.mixer.__class__.__name__==expected
        block.register_forward_hook(grab((s,name)))
    projections[s]=torch.linalg.qr(torch.randn(c,128,generator=gen)).Q.cuda()
def batch(split,ix):
    x,y=data.batch(split,torch.as_tensor(ix.copy(),device='cuda'))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        feats=net.backbone(x)
        base=F.interpolate(net.probe.decode_head(feats),size=(256,256),mode='bilinear',align_corners=False)
    base=base.float().sigmoid()
    states={k:[[],[]] for k in ('A','M','H','E','D')}
    for s,w,c,inds in stages:
        norm=getattr(net.backbone,f'outnorm{s}')
        raw={}
        for k in inds:
            v=taps[s,k]
            assert v.shape==(len(ix),w*w,c)
            raw[k]=v.transpose(1,2).reshape(len(ix),c,w,w).contiguous()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            assert torch.equal(norm(raw['A']).contiguous(),feats[s])
        for k in states:
            ft=(raw['M'].float()-raw['A'].float()) if k=='D' else raw[k].float()
            if k!='D': ft=norm(ft)
            v=ft.permute(0,2,3,1) @ projections[s]
            if w==8:v=F.interpolate(v.permute(0,3,1,2),size=(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1)
            for side,z in enumerate(v.split(64,dim=-1)):
                states[k][side].append(z)
    states={k:[torch.cat(z,-1).reshape(len(ix),256,128) for z in val] for k,val in states.items()}
    return states,y,base

def chunks(ids):
    for i in range(0,len(ids),4):yield ids[i:i+4]
rng=np.random.default_rng(4359)
order=rng.permutation(len(data.ids['train']))
cal=order[:189]
train=order[189:]
def fit_heads():
    gram={k:torch.zeros(257,257,device='cuda',dtype=torch.float64) for k in arms}
    xy={k:torch.zeros(257,2,device='cuda',dtype=torch.float64) for k in arms}
    n=0
    for ix in chunks(train):
        vals,y,base=batch('train',ix)
        sample=torch.as_tensor(np.stack([rng.choice(256,48,replace=False) for _ in ix]),device='cuda')
        b=torch.arange(len(ix),device='cuda')[:,None]
        get=lambda z:F.avg_pool2d(z,16).flatten(1)[b,sample].reshape(-1)
        yy=get((y>=.5).float())
        target=torch.stack([yy,yy-get(base)],1)
        for k,(a,c) in arms.items():
            z=torch.cat([vals[a][0][b,sample],vals[c][1][b,sample]],-1).reshape(-1,256)
            z=F.pad(z,(0,1),value=1)
            gram[k]+=(z.T@z).double()
            xy[k]+=(z.T@target).double()
        n+=target.shape[0]
    weights={}
    for k,g in gram.items():
        variance=(g.diag()[:-1]-g[:-1,-1].square()/n).clamp_min(1e-8)
        reg=torch.diag(F.pad(variance*.01,(0,1)))
        weights[k]=torch.linalg.solve(g+reg,xy[k]).float()
    return weights
def logits(k,vals,weights):
    a,c=arms[k];w=weights[k]
    return F.interpolate((vals[a][0]@w[:128]+vals[c][1]@w[128:256]+w[256]).transpose(1,2).reshape(-1,2,16,16),size=(256,256),mode='bilinear',align_corners=False)
def bins(v):
    return np.minimum(3071,np.maximum(0,np.floor((v+1)*1024).astype(np.int32)))
def get_thresholds(weights):
    bg={k:np.zeros(3072,dtype='int64') for k in arms}
    for ix in chunks(cal):
        vals,y,base=batch('train',ix)
        neg=y[:,0].cpu().numpy()<.5
        for k in arms:
            v=logits(k,vals,weights)[:,0].cpu().numpy()
            bg[k]+=np.bincount(bins(v[neg]),minlength=3072)
    return {k:min(3071,int(np.searchsorted(np.cumsum(b),.95*b.sum())+1)) for k,b in bg.items()}
def dice(m,g):
    den=int(m.sum()+g.sum())
    return 2*np.count_nonzero(m&g)/den if den else 1.
el=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
def contour(x):
    v=x.astype('uint8')
    return v-cv2.erode(v,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
def boundary(m,gtc,gtd):
    c=contour(m); cm=c.sum();gm=gtc.sum()
    if cm==0 or gm==0:return float(cm==gm)
    p=(c*gtd).sum()/cm;r=(gtc*cv2.dilate(c,el)).sum()/gm
    return float(2*p*r/(p+r)) if p+r else 0.
def evaluate(weights,thresholds):
    hist={k:np.zeros((2,3072),dtype='int64') for k in arms}
    ct={k:np.zeros(8,dtype='int64') for k in arms}
    rows=[]; rr=np.random.default_rng(657)
    for ix in chunks(np.arange(len(data.ids['val']))):
        vals,y,base=batch('val',ix)
        gt=y[:,0].cpu().numpy()>=.5
        probs=base[:,0].cpu().numpy()
        outs={k:logits(k,vals,weights).cpu().numpy() for k in arms}
        for j,idx in enumerate(ix):
            g=gt[j];p=probs[j]; cur=p>=.5
            interior=cv2.distanceTransform(g.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>8
            hard=interior&(p<.1)
            bgidx=np.flatnonzero(~g);goodidx=np.flatnonzero(interior&cur)
            # Match negative/easy-positive counts within the same image whenever all three pools exist.
            nm=min(int(hard.sum()),len(bgidx),len(goodidx))
            neg=rr.choice(bgidx,nm,replace=False);easy=rr.choice(goodidx,nm,replace=False)
            hd=rr.choice(np.flatnonzero(hard),nm,replace=False)
            gc=contour(g);gd=cv2.dilate(gc,el)
            row={'id':data.ids['val'][idx],'baseline':dice(cur,g),'boundary_baseline':boundary(cur,gc,gd)}
            for k in arms:
                pure,res=outs[k][j];h=bins(pure)
                for flag,subset in enumerate((~g,g)):hist[k][flag]+=np.bincount(h[subset],minlength=3072)
                on=h>=thresholds[k]
                ct[k]+=[np.count_nonzero(on&hard),hard.sum(),on.ravel()[hd].sum(),on.ravel()[neg].sum(),on.ravel()[easy].sum(),nm,np.count_nonzero(on&~g),np.count_nonzero(~g)]
                corrected=p+res>=.5
                row[k]=dice(pure>=.5,g)
                row[k+'_corrected']=dice(corrected,g)
                row[k+'_bf']=boundary(corrected,gc,gd)
            rows.append(row)
    def ci(delta):
        rnd=np.random.default_rng(824)
        return [round(float(v),4) for v in np.percentile(rnd.choice(delta,size=(3000,len(rows))).mean(1),[2.5,97.5])]
    vec=lambda k:np.array([r[k] for r in rows])
    stats={}
    for k in arms:
        neg,pos=hist[k][:,::-1].cumsum(axis=1)
        ap=np.sum(np.diff(np.r_[0,pos])/pos[-1]*pos/np.maximum(1,pos+neg))
        h,tot,h2,neg2,easy,nm,fp,nb=ct[k]
        delta=100*(vec(k)-vec('A'))
        corr=100*(vec(k+'_corrected')-vec('baseline'))
        stats[k]=dict(ap=round(100*ap,4),dice=round(100*vec(k).mean(),4),dice_vs_A_ci=ci(delta),hard_recall=round(100*h/tot,3),bg_fpr=round(100*fp/nb,3),matched_hard_recall=round(100*h2/nm,3),matched_bg_fpr=round(100*neg2/nm,3),matched_good_recall=round(100*easy/nm,3),corrected_dice_gain_pp=round(corr.mean(),4),corrected_gain_ci=ci(corr),corrected_bf_gain_pp=round(100*(vec(k+'_bf')-vec('boundary_baseline')).mean(),4))
    result=dict(checkpoint=fit['sha256'],fit_images=len(train),calibration_images=len(cal),validation_images=len(rows),baseline=100*vec('baseline').mean(),baseline_bf=100*vec('boundary_baseline').mean(),hard_pixels=int(ct['A'][1]),matched_each=int(ct['A'][5]),train_tokens=len(train)*48,columns=256,thresholds=thresholds,stats=stats)
    for other in ('AH','AE','MA','A'):
        for suff in ('','_corrected'):
            diff=100*(vec('AM'+suff)-vec(other+suff))
            result['AM_vs_'+other+suff]=[round(diff.mean(),4),ci(diff)]
    assert abs(result['baseline']-88.8293)<.02
    with (OUT/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    official.write(OUT/'result.json',result)
    print('DONE',len(rows),flush=True)
if __name__ == '__main__':
    with torch.inference_mode():
        print('fit started',len(train),len(cal),flush=True)
        weights=fit_heads()
        print('calibrating thresholds',flush=True)
        th=get_thresholds(weights)
        print('validating',th,flush=True)
        evaluate(weights,th)
    
