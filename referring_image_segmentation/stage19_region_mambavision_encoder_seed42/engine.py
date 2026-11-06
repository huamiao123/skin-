"""Recompute the trainable encoder on each validation pass."""
from common import *
from evaluation import aggregate_counts
from models import segmentation_loss


@torch.no_grad()
def evaluate(model, data, batch, export=False):
    model.eval()
    n=len(data.ids['val'])
    probs=np.empty((n,256,256),dtype=np.float32) if export else None
    counts=[]
    loss_sum=0.
    for start in range(0,n,batch):
        end=min(start+batch,n)
        ids=torch.arange(start,end,device='cuda')
        images,target=data.batch('val',ids)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_=model(images)
        loss=segmentation_loss(logits,target)
        prob=logits.float().sigmoid()
        pred,truth=prob>=.5,target>=.5
        vals=[(pred&truth).sum((1,2,3)),(pred&~truth).sum((1,2,3)),
              (~pred&truth).sum((1,2,3)),(~pred&~truth).sum((1,2,3))]
        counts.append(torch.stack(vals,1).cpu().numpy())
        loss_sum+=float(loss.item())*(end-start)
        if export:probs[start:end]=prob[:,0].cpu().numpy()
    counts=np.concatenate(counts,axis=0).astype(np.int64)
    result=aggregate_counts(*counts.T)
    result['loss']=loss_sum/n
    return result,probs,counts
