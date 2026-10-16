"""Unchanged stage-2 cached encoding and region evaluation functions."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from stage1_visual.models import segmentation_loss
from evaluation import aggregate_counts

@torch.no_grad()
def encode(encoder, data, split, batch, epoch=0):
    n = len(data.ids[split])
    order, matrices = data.geometry(epoch) if split == 'train' else (torch.arange(n, device='cuda'), None)
    cache, targets = None, torch.empty(n, 1, 256, 256, device='cuda', dtype=torch.float32)
    for start in range(0, n, batch):
        indices = order[start:start+batch]
        images, masks = data.batch(split, indices, matrices)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features = encoder(images)
        if cache is None:
            cache = [torch.empty((n, *f.shape[1:]), device='cuda', dtype=torch.bfloat16) for f in features]
        for dst, src in zip(cache, features):
            dst[start:start+len(indices)].copy_(src)
        targets[start:start+len(indices)].copy_(masks)
    return cache, targets


@torch.no_grad()
def evaluate(probe, cache, targets, export=False):
    probe.eval()
    n = len(targets)
    sums = dict(loss=0., dice=0., iou=0., precision=0., recall=0.)
    tp = fp = fn = tn = 0
    count_batches = []
    probabilities = np.empty((n, 256, 256), dtype=np.float32) if export else None
    per_image = []
    for start in range(0, n, 64):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = probe([f[start:start+64] for f in cache])
        gt = targets[start:start+64]
        size = len(gt)
        loss = segmentation_loss(logits, gt)
        prob = logits.float().sigmoid()
        pred, target = prob >= .5, gt >= .5
        a = (pred & target).sum((1, 2, 3)).double()
        b = (pred & ~target).sum((1, 2, 3)).double()
        c = (~pred & target).sum((1, 2, 3)).double()
        d = (~pred & ~target).sum((1, 2, 3)).double()
        count_batches.append(torch.stack((a,b,c,d), dim=1).cpu().numpy())
        dice = torch.where(2*a+b+c > 0, 2*a/(2*a+b+c), 1.)
        iou = torch.where(a+b+c > 0, a/(a+b+c), 1.)
        precision = torch.where(a+b > 0, a/(a+b), (a+c == 0).double())
        recall = torch.where(a+c > 0, a/(a+c), 1.)
        sums['loss'] += loss.item()*size
        for key, values in [('dice',dice),('iou',iou),('precision',precision),('recall',recall)]:
            sums[key] += values.sum().item()
        tp += a.sum().item(); fp += b.sum().item(); fn += c.sum().item(); tn += d.sum().item()
        if export:
            probabilities[start:start+size] = prob[:, 0].cpu().numpy()
            for j in range(size):
                per_image.append(dict(index=start+j, dice=dice[j].item(), iou=iou[j].item(),
                    precision=precision[j].item(), recall=recall[j].item(),
                    tp=int(a[j]), fp=int(b[j]), fn=int(c[j]), tn=int(d[j])))
    counts = np.concatenate(count_batches, axis=0).astype(np.int64)
    metrics = aggregate_counts(*counts.T)
    metrics['loss'] = sums['loss']/n
    return metrics, probabilities, per_image

