"""Native-resolution evaluation with the project's shared foreground metrics."""
from common import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, torch
from PIL import Image
from models import masked_loss
from evaluation import aggregate_counts, summarize_rows
from stage1_visual.analyze import metrics as full_metrics
cv2.setNumThreads(1)

@lru_cache(maxsize=1024)
def native_gt(path): return np.asarray(Image.open(path).convert('L'))>0

def restore(low,row):
    z=cv2.resize(low,(256,256),interpolation=cv2.INTER_LINEAR)
    t,l,h,w=[row[k] for k in ('top','left','new_height','new_width')]
    return cv2.resize(z[t:t+h,l:l+w],(row['width'],row['height']),interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def predict(probe,cache,split,limit=None,batch=BATCH):
    data=cache.data; n=len(data.rows[split]) if limit is None else min(limit,len(data.rows[split]))
    probe.eval(); logits=np.empty((n,64,64),np.float32); total_loss=0.
    for start in range(0,n,batch):
        pos=torch.arange(start,min(start+batch,n),device='cuda'); fs,y,valid,text=cache.batch(split,pos)
        if probe.name==NAMES[1]: text=data.text[data.fixed_index].expand(len(pos),-1)
        with torch.autocast('cuda',dtype=torch.bfloat16): z=probe(fs,text)
        total_loss+=float(masked_loss(z,y,valid))*len(pos); logits[start:start+len(pos)]=z[:,0].float().cpu().numpy()
    return logits,total_loss/n

def score(logits,data,split,full=False):
    rows=data.rows[split][:len(logits)]
    def sample(i):
        r=rows[i]; p=restore(logits[i],r)>=0; g=native_gt(r['mask_path'])
        if full: s=full_metrics(p,g)
        else: s=dict(tp=int((p&g).sum()),fp=int((p&~g).sum()),fn=int((~p&g).sum()),tn=int((~p&~g).sum()))
        return dict(id=r['id'],group_id=r['group_id'],**s)
    with ThreadPoolExecutor(4) as pool: records=list(pool.map(sample,range(len(rows))))
    result=summarize_rows(records) if full else aggregate_counts(*[[r[k] for r in records] for k in ('tp','fp','fn','tn')])
    result.pop('threshold_jaccard_065',None)
    result['evaluation_resolution']='native image; 64 logits -> 256 -> unpad -> native; INTER_LINEAR; threshold 0'
    return result,records
