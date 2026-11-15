"""Evaluation with the project's shared metrics and bounded GPU batches."""
from common import *
from evaluation import aggregate_counts
from models import segmentation_loss


@torch.no_grad()
def evaluate(model, features, targets, batch=16, export=False):
    model.eval()
    n = len(targets)
    counts, loss_sum = [], 0.
    route_hard = torch.zeros((2,3),device='cuda',dtype=torch.float64)
    route_soft = torch.zeros_like(route_hard)
    probabilities = np.empty((n,256,256),dtype=np.float32) if export else None
    for start in range(0,n,batch):
        end = min(start+batch,n)
        fs = [f[start:end].to('cuda') for f in features]
        gt = targets[start:end].to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits, regions = model(fs,return_regions=True)
        assert len(regions)==2
        for i,logp in enumerate(regions):
            p=logp.float().exp()
            hard=logp.argmax(1)
            route_soft[i] += p.sum((0,2,3)).double()
            route_hard[i] += torch.stack([(hard==k).sum() for k in range(3)]).double()
        loss = segmentation_loss(logits,gt)
        prob = logits.float().sigmoid()
        pred, truth = prob>=.5, gt>=.5
        vals = [(pred & truth).sum((1,2,3)), (pred & ~truth).sum((1,2,3)),
                (~pred & truth).sum((1,2,3)), (~pred & ~truth).sum((1,2,3))]
        counts.append(torch.stack(vals,1).cpu().numpy())
        loss_sum += float(loss.item())*(end-start)
        if export:
            probabilities[start:end] = prob[:,0].cpu().numpy()
    counts = np.concatenate(counts,axis=0).astype(np.int64)
    result = aggregate_counts(*counts.T)
    result['loss'] = loss_sum/n
    denom=n*32*32
    result['routing']={name:dict(hard_fraction=(route_hard[i]/denom).cpu().tolist(),
                                  mean_soft_probability=(route_soft[i]/denom).cpu().tolist())
                       for i,name in enumerate(('coarse','skip'))}
    return result, probabilities, counts
