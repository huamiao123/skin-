"""Historical native-size segmentation metrics for the new predecoder models."""
from common import *
import numpy as np,torch
from models import masked_loss
from stage9_isic_text.engine import native_gt,restore,score

@torch.no_grad()
def predict(probe,cache,split,limit=None,batch=BATCH):
    data=cache.data;n=len(data.rows[split]) if limit is None else min(limit,len(data.rows[split]))
    probe.eval();low=np.empty((n,64,64),np.float32);loss=0.
    for start in range(0,n,batch):
        pos=torch.arange(start,min(start+batch,n),device='cuda')
        fs,y,valid,text=cache.batch(split,pos,fixed=probe.name==NAMES[0])
        with torch.autocast('cuda',dtype=torch.bfloat16):z=probe(fs,text)
        loss+=float(masked_loss(z,y,valid))*len(pos);low[start:start+len(pos)]=z[:,0].float().cpu().numpy()
    return low,loss/n
