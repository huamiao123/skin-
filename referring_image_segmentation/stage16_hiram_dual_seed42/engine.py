"""Evaluation with the project's shared metrics and bounded GPU batches."""
from common import *
from evaluation import aggregate_counts
from models import segmentation_loss


@torch.no_grad()
def evaluate(model, features, targets, batch=16, export=False):
    model.eval()
    n = len(targets)
    counts, loss_sum = [], 0.
    probabilities = np.empty((n,256,256),dtype=np.float32) if export else None
    for start in range(0,n,batch):
        end = min(start+batch,n)
        fs = [f[start:end].to('cuda') for f in features]
        gt = targets[start:end].to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(fs)
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
    return result, probabilities, counts
