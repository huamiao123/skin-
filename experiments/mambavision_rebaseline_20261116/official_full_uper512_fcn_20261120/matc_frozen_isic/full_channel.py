"""Full-channel frozen ridge probes: conditioning on every final feature channel."""
import sys,json,csv
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.nn import functional as F
import probe as q

dest=q.OUT
parts={'A':(0,), 'M':(1,), 'AM':(0,1), 'AH':(0,2)}
def extract(split,ix):
    x,y=q.data.batch(split,torch.as_tensor(ix.copy(),device='cuda'))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ff=q.net.backbone(x)
        base=F.interpolate(q.net.probe.decode_head(ff),size=(256,256),mode='bilinear',align_corners=False).float().sigmoid()
    columns=[]
    for name in ['A','M','H']:
        group=[]
        for s,w,c,indices in q.stages:
            v=q.taps[s,name].transpose(1,2).reshape(len(ix),c,w,w).contiguous()
            norm=getattr(q.net.backbone,f'outnorm{s}')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                v=norm(v)
                if name=='A': assert torch.equal(ff[s],v)
            v=v.float()
            if w==8:v=F.interpolate(v,size=(16,16),mode='bilinear',align_corners=False)
            group.append(v.flatten(2).transpose(1,2))
        columns.append(torch.cat(group,dim=-1))
    return torch.cat(columns,dim=-1), y, base

def fit_full():
    d=3457
    gram=torch.zeros(d,d,device='cuda',dtype=torch.float64)
    cross=torch.zeros(d,2,device='cuda',dtype=torch.float64)
    buf=[];labels=[];n=0
    plans=list(q.chunks(q.train))
    for ci,ix in enumerate(plans):
        v,y,p=extract('train',ix)
        idx=torch.as_tensor(np.stack([q.rng.choice(256,48,replace=False) for _ in ix]),device='cuda')
        bi=torch.arange(len(ix),device='cuda')[:,None]
        get=lambda z:F.avg_pool2d(z,16).flatten(1)[bi,idx].flatten()
        g=get((y>=.5).float())
        buf.append(v[bi,idx].flatten(0,1));labels.append(torch.stack([g,g-get(p)],1))
        n+=len(g)
        if len(buf)>=6 or ci==len(plans)-1:
            z=F.pad(torch.cat(buf),(0,1),value=1)
            gram+=(z.T@z).double()
            cross+=(z.T@torch.cat(labels)).double()
            buf=[];labels=[]
    ret={}
    for name,items in parts.items():
        cols=np.r_[np.concatenate([np.arange(1152*i,1152*(i+1)) for i in items]),3456]
        ids=torch.as_tensor(cols,device='cuda')
        g=gram[ids[:,None],ids[None,:]]
        variances=(g.diag()[:-1]-g[:-1,-1].square()/n).clamp_min(1e-8)
        w=torch.linalg.solve(g+torch.diag(F.pad(.01*variances,(0,1))),cross[ids]).float()
        ret[name]=(ids[:-1],w[:-1],w[-1])
    return ret

def metric_at_fpr(hist,fpr=.05):
    neg,fg,hard=hist[:,::-1].cumsum(1)
    j=np.searchsorted(neg, fpr*neg[-1])
    assert j>0
    mix=(fpr*neg[-1]-neg[j-1])/(neg[j]-neg[j-1])
    interp=lambda a:(a[j-1]+mix*(a[j]-a[j-1]))/a[-1]
    ap=np.sum(np.diff(np.r_[0,fg])/fg[-1]*fg/np.maximum(1,fg+neg))
    return [100*float(a) for a in (ap,interp(fg),interp(hard))]
def assess(weights):
    hist={k:np.zeros((3,3072),dtype='int64') for k in parts};rows=[]
    for ix in q.chunks(np.arange(len(q.data.ids['val']))):
        v,y,base=extract('val',ix)
        gold=y[:,0].cpu().numpy()>=.5;probs=base[:,0].cpu().numpy()
        outputs={}
        for k,(cols,w,b) in weights.items():
            low=v.index_select(2,cols)@w+b
            outputs[k]=F.interpolate(low.transpose(1,2).reshape(-1,2,16,16),size=(256,256),mode='bilinear',align_corners=False).cpu().numpy()
        for j,idx in enumerate(ix):
            g=gold[j];p=probs[j];cur=p>=.5
            hard=(cv2.distanceTransform(g.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>8)&(p<.1)
            row={'id':q.data.ids['val'][idx],'base':q.dice(cur,g)}
            for k in parts:
                pure,res=outputs[k][j];h=q.bins(pure)
                for t,mask in enumerate((~g,g,hard)):hist[k][t]+=np.bincount(h[mask],minlength=3072)
                row[k]=q.dice(pure>=.5,g);row[k+'_res']=q.dice(p+res>=.5,g)
            rows.append(row)
    vec=lambda k:np.array([r[k] for r in rows])
    def compare(a,b):
        diff=100*(vec(a)-vec(b));ci=np.percentile(np.random.default_rng(42).choice(diff,(3000,len(rows))).mean(1),[2.5,97.5])
        return [round(float(x),4) for x in [diff.mean(),*ci]]
    metrics={}
    for k in parts:
        ap,rec,hr=metric_at_fpr(hist[k])
        metrics[k]=dict(dice=100*vec(k).mean(),ap=ap,recall_at_val_fpr5=rec,hard_recall_at_val_fpr5=hr,dice_vs_A=compare(k,'A'),residual_vs_base=compare(k+'_res','base'))
    assert abs(100*vec('base').mean()-88.8293)<.02
    q.official.write(dest/'full_channel.json',dict(checkpoint=q.fit['sha256'],train_fit=len(q.train),validation=len(rows),arm_channels={'A':1152,'M':1152,'AM':2304,'AH':2304},hard_pixels=int(hist['A'][2].sum()),scores=metrics,AM_vs_AH=compare('AM','AH'),AMres_vs_AHres=compare('AM_res','AH_res')))
    with (dest/'full_channel_cases.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('DONE',len(rows),flush=True)
with torch.inference_mode():
    print('fit full channels',flush=True)
    weights=fit_full()
    print('validating full channels',flush=True)
    assess(weights)

