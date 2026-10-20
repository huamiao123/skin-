"""Historical native-size ISIC metric definitions; no text mismatch experiments."""
from common import *
import numpy as np,torch
from models import masked_loss
from stage9_isic_text.engine import native_gt,restore,score

@torch.no_grad()
def predict(probe,cache,split,limit=None,batch=BATCH):
    data=cache.data;n=len(data.rows[split]) if limit is None else min(limit,len(data.rows[split]))
    probe.eval();logits=np.empty((n,64,64),np.float32);total_loss=0.
    for start in range(0,n,batch):
        pos=torch.arange(start,min(start+batch,n),device='cuda');fs,y,valid,text=cache.batch(split,pos)
        if probe.name==NAMES[1]:text=data.text[data.fixed_index].expand(len(pos),-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):z=probe(fs,text)
        total_loss+=float(masked_loss(z,y,valid))*len(pos);logits[start:start+len(pos)]=z[:,0].float().cpu().numpy()
    return logits,total_loss/n
