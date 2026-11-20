"""Fixed coefficient controls on the read-only decoder skip; original encoder path stays intact."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from torch.nn import functional as F
dest=Path(__file__).resolve().parent
sys.path.insert(0,str(dest.parent))
import official
torch.cuda.set_per_process_memory_fraction(.09)
torch.set_num_threads(2);cv2.setNumThreads(1)
official.recipe.setup(42)
fit=json.loads((dest.parent/'ISIC2018'/'fit_complete.json').read_text())
net=official.Segmenter()
net.load_state_dict(torch.load(dest.parent/'ISIC2018'/fit['best_checkpoint'],map_location='cpu',weights_only=True))
net.eval().requires_grad_(False).cuda()
data=official.dataset('ISIC2018',['val'])
taps={}
def grab(key):
    def hook(mod,args,out): taps[key]=out
    return hook
for s,inds in [(2,(3,4,6)),(3,(2,3,4))]:
    for name,i in zip(('M','H','A'),inds):
        net.backbone.levels[s].blocks[i].register_forward_hook(grab((s,name)))
keys={'base':{},'M3':{2:'M'},'M4':{3:'M'},'M34':{2:'M',3:'M'},'H34':{2:'H',3:'H'}}
rows=[];hardn=0
with torch.inference_mode():
    for start in range(0,len(data.ids['val']),4):
        ix=torch.arange(start,min(start+4,len(data.ids['val'])),device='cuda')
        x,y=data.batch('val',ix)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feats=net.backbone(x)
            lut={}
            for s in (2,3):
                shape=feats[s].shape
                raw={k:taps[s,k].transpose(1,2).reshape(shape).contiguous() for k in ('A','M','H')}
                norm=getattr(net.backbone,f'outnorm{s}')
                assert torch.equal(norm(raw['A']),feats[s])
                for k in ('M','H'):
                    lut[s,k]=norm(torch.lerp(raw['A'].float(),raw[k].float(),.25).to(raw['A'].dtype)).contiguous()
            outputs={}
            for arm,locations in keys.items():
                f=list(feats)
                for s,k in locations.items(): f[s]=lut[s,k]
                z=F.interpolate(net.probe.decode_head(f),size=(256,256),mode='bilinear',align_corners=False)
                outputs[arm]=z[:,0].float().cpu().numpy()
        for j,idx in enumerate(ix.cpu().tolist()):
            gt=y[j,0].cpu().numpy()>=.5
            p=outputs['base'][j]
            hard=(cv2.distanceTransform(gt.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>8)&(p<np.log(.1/.9))
            hardn+=hard.sum()
            rec={'id':data.ids['val'][idx]}
            for k,a in outputs.items():
                pred=a[j]>=0; den=pred.sum()+gt.sum()
                rec[k]=(2*np.count_nonzero(pred&gt)/den) if den else 1.
                rec[k+'_rescued']=int((pred&hard).sum())
            rows.append(rec)
def vec(k):return np.array([r[k] for r in rows])
res={}
for arm in keys:
    d=(vec(arm)-vec('base'))*100
    ci=np.percentile(np.random.default_rng(42).choice(d,(3000,len(rows))).mean(1),[2.5,97.5])
    res[arm]=dict(dice_gain_pp=round(float(d.mean()),4),ci_pp=ci.tolist(),recovered_hard=int(vec(arm+'_rescued').sum()))
assert abs(100*vec('base').mean()-88.8293)<.02
official.write(dest/'intervention.json',dict(arms=res,hard_pixels=int(hardn),n=len(rows),alpha=.25,unchanged_backbone=True))
with (dest/'intervention_per_image.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('DONE',len(rows),flush=True)

