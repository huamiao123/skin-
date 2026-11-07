"""Common historical frozen-encoder extraction and evaluation, unchanged math."""
from common import *
from stage3_confirmation.models import segmentation_loss
from evaluation import aggregate_counts


@torch.no_grad()
def extract(encoder, images, batch):
    parts = None
    for i in range(0, len(images), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = encoder(images[i:i + batch])
        if parts is None:
            parts = [[] for _ in out]
        for dest, x in zip(parts, out):
            dest.append(x.detach())
    return [torch.cat(x, 0) for x in parts]


@torch.no_grad()
def encode_val(encoder, data, batch):
    n = len(data.ids['val'])
    cache = None
    targets = torch.empty((n, 1, 256, 256), dtype=torch.float32, device='cuda')
    for i in range(0, n, batch):
        end = min(i + batch, n)
        images, gt = data.batch('val', torch.arange(i, end, device='cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats = encoder(images)
        if cache is None:
            cache = [torch.empty((n, *x.shape[1:]), dtype=torch.bfloat16, device='cuda') for x in feats]
        for dst, x in zip(cache, feats):
            dst[i:end].copy_(x)
        targets[i:end].copy_(gt)
    return cache, targets


@torch.no_grad()
def evaluate(model, features, targets, batch=64, export=False):
    model.eval()
    n = len(targets)
    counts = []
    loss_sum = 0.
    probabilities = np.empty((n, 256, 256), dtype=np.float32) if export else None
    for i in range(0, n, batch):
        end = min(i + batch, n)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model([x[i:end] for x in features])
        gt = targets[i:end]
        loss_sum += float(segmentation_loss(logits, gt)) * (end - i)
        prob = logits.float().sigmoid()
        pred, truth = prob >= .5, gt >= .5
        vals = [(pred & truth).sum((1, 2, 3)), (pred & ~truth).sum((1, 2, 3)),
                (~pred & truth).sum((1, 2, 3)), (~pred & ~truth).sum((1, 2, 3))]
        counts.append(torch.stack(vals, 1).cpu().numpy())
        if export:
            probabilities[i:end] = prob[:, 0].cpu().numpy()
    counts = np.concatenate(counts).astype(np.int64)
    result = aggregate_counts(*counts.T)
    result['loss'] = loss_sum / n
    return result, probabilities, counts
