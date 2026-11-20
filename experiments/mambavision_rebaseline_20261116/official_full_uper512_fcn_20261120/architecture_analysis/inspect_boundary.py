import json
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from inspect_validation import dice, official, dest, root

def guided(rgb, z):
    colors = rgb.reshape(64, 4, 64, 4, 3).mean(axis=(1, 3))
    pos = (np.arange(256, dtype=np.float32) + .5) / 4 - .5
    left = np.floor(pos).astype(int)
    frac = pos - left
    numer = np.zeros((256, 256), dtype=np.float64)
    denom = np.zeros_like(numer)
    plain = np.zeros_like(numer)
    for a in (0, 1):
        for b in (0, 1):
            i = np.clip(left + a, 0, 63)[:,None]
            j = np.clip(left + b, 0, 63)[None,:]
            spatial = (frac if a else 1 - frac)[:,None] * (frac if b else 1 - frac)[None,:]
            dist2 = np.square(rgb-colors[i,j]).sum(axis=2)
            w = spatial * np.exp(-dist2.astype(np.float64) / (2 * .1**2))
            numer += w*z[i,j]
            denom += w
            plain += spatial*z[i,j]
    assert np.all(denom>0)
    return plain, numer/denom

def band(mask):
    if not mask.any() or mask.all():return np.zeros_like(mask)
    d = lambda v: cv2.distanceTransform(v.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return np.where(mask,d(mask),d(~mask)) <= 8

def run(ds):
    folder=root/ds
    fit=json.loads((folder/'fit_complete.json').read_text())
    s=official.Segmenter()
    s.load_state_dict(torch.load(folder/fit['best_checkpoint'],weights_only=True,map_location='cpu'))
    s.cuda().eval()
    data=official.dataset(ds,['val'])
    rows=[]
    err=eligible=afn=fpix=correct=broken=0
    with torch.inference_mode():
        for i in range(0,len(data.ids['val']),4):
            ix=torch.arange(i,min(i+4,len(data.ids['val'])),device='cuda')
            x,y=data.batch('val',ix)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                low=s.probe.decode_head(s.backbone(x))
                main=F.interpolate(low,scale_factor=4,mode='bilinear',align_corners=False)
            rgb=(x*data.std+data.mean).permute(0,2,3,1).cpu().numpy()
            for im,lo,base,gt in zip(rgb,low[:,0].float().cpu().numpy(),main[:,0].float().cpu().numpy(),y[:,0].cpu().numpy()):
                gold=gt>=.5
                plain,pred=guided(im,lo)
                m=base>=0
                b=band(m)
                bad=gold!=m
                new=pred>=0
                rows.append((dice(m,gold),dice(plain>=0,gold),dice(new,gold),dice(np.where(b,gold,m),gold)))
                err+=bad.sum();eligible+=(bad&b).sum();afn+=(gold&~m).sum();fpix+=(gold&~m&~b).sum();correct+=(new==gold)[bad].sum();broken+=(new!=gold)[~bad].sum()
    arr=np.array(rows)
    assert abs(arr[:,0].mean()-json.loads((dest/(ds+'.json')).read_text())['macro']['main'])<1.e-5
    gain=100*(arr[:,2]-arr[:,1])
    rng=np.random.default_rng(42);ci=np.percentile(rng.choice(gain,(4000,len(gain)),replace=True).mean(axis=1),[2.5,97.5]).tolist()
    out=dict(n=len(rows),err_in_pred_band_pct=100*float(eligible)/err,fn_outside_pred_band_pct=100*float(fpix)/afn,perfect_pred_band_gain_pp=float((arr[:,3]-arr[:,0]).mean()*100),rgb_guided_vs_fp32_bilinear_pp=float(gain.mean()),paired_ci_pp=ci,fp32_ref_delta_pp=float(100*(arr[:,1]-arr[:,0]).mean()),corrected=int(correct),damaged=int(broken),rgb_vs_deployed_pp=float(100*(arr[:,2]-arr[:,0]).mean()))
    (dest/(ds+'_boundary.json')).write_text(json.dumps(out,indent=2))
    print('DONE',ds,out,flush=True)
if __name__=='__main__':
    for ds in ('PH2','Waterloo','ISIC2018'):
        run(ds)

