"""Original-coordinate RefCOCO evaluation; one score per natural expression."""
from common import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from PIL import Image
from models import masked_loss
from evaluation import aggregate_counts,summarize_rows
from stage1_visual.analyze import metrics as full_metrics

cv2.setNumThreads(1)

@lru_cache(maxsize=6000)
def native_gt(ann_id):return np.asarray(Image.open(ASSETS/'masks'/f'{ann_id}.png'))>=128

def restore(low,image):
    z=cv2.resize(low,(256,256),interpolation=cv2.INTER_LINEAR)
    t,l,h,w=image['top'],image['left'],image['new_height'],image['new_width']
    return cv2.resize(z[t:t+h,l:l+w],(image['width'],image['height']),interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def predict(probe,cache,split,limit=None,batch=16,fixed_intervention=False):
    data=cache.data;rs=data.rows[split];n=len(rs) if limit is None else min(limit,len(rs))
    probe.eval();logits=np.empty((n,64,64),np.float32);total_loss=0.
    for start in range(0,n,batch):
        pos=torch.arange(start,min(start+batch,n),device='cuda');features,target,valid,text=cache.batch(split,pos)
        if probe.name=='dinov3_fixed_text' or fixed_intervention:text=data.text[data.fixed_index].expand(len(pos),-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=probe(features,text)
        total_loss+=float(masked_loss(z,target,valid))*len(pos);logits[start:start+len(pos)]=z[:,0].cpu().numpy()
    return logits,total_loss/n

def score(logits,data,split,full=False):
    rs=data.rows[split][:len(logits)]
    def sample(i):
        r=rs[i];pred=restore(logits[i],data.images[r['image_index']])>=0;gt=native_gt(r['ann_id'])
        if full:s=full_metrics(pred,gt)
        else:
            s=dict(tp=int((pred&gt).sum()),fp=int((pred&~gt).sum()),fn=int((~pred&gt).sum()),tn=int((~pred&~gt).sum()))
        return dict(sample_id=r['sample_id'],ref_id=r['ref_id'],image_id=r['image_id'],ann_id=r['ann_id'],**s)
    with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(rs))))
    summary=summarize_rows(rows) if full else aggregate_counts(*[[r[k] for r in rows] for k in ('tp','fp','fn','tn')])
    groups={}
    for row in rows:
        a,b,c=row['tp'],row['fp'],row['fn'];groups.setdefault(row['image_id'],[]).append(a/(a+b+c) if a+b+c else 1.)
    summary['image_balanced_mean_iou']=float(np.mean([np.mean(v) for v in groups.values()]))
    summary['evaluation_resolution']='native image dimensions; logits upsample->unpad->resize using OpenCV INTER_LINEAR; threshold 0'
    summary['aggregation']='macro and Prec@X over expressions; pooled sums expression-pixel counts; image-balanced score separately reported'
    return summary,rows
