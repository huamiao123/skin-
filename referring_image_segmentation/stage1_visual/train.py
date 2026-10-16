"""Stage 1 frozen visual probes: fixed optimization, exact per-epoch feature caching."""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data import ImageData
from models import FrozenEncoder, Probe, segmentation_loss, tensor_hash

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def save_checkpoint(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


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
    metrics = dict(loss=sums['loss']/n, macro={k:v/n for k,v in sums.items() if k!='loss'},
            pooled=dict(dice=2*tp/(2*tp+fp+fn), iou=tp/(tp+fp+fn),
                        precision=tp/(tp+fp), recall=tp/(tp+fn), specificity=tn/(tn+fp)),
            counts=dict(tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn)))
    return metrics, probabilities, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['dinov3','vmamba'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    budget = 2 if args.smoke else 300
    out = ROOT / ('smoke' if args.smoke else 'results') / args.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE.json').exists():
        print('Already complete', flush=True); return
    benchmark = json.loads((ROOT / f'benchmark_{args.name}.json').read_text())
    batch = benchmark['encoder_batch']
    encoder = FrozenEncoder(args.name).cuda()
    encoder_hash = tensor_hash(encoder)
    torch.manual_seed(42)
    probe = Probe(encoder.channels).cuda()
    assert tensor_hash(probe.decoder) == benchmark['decoder_initial_hash']
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=.01,
                                  betas=(.9,.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    config = dict(model=args.name, seed=42, epochs=budget, input_size=256, optimizer_batch=64,
        encoder_batch=batch, evaluation_batch=64, optimizer='AdamW', lr=1e-3, weight_decay=.01,
        scheduler='CosineAnnealingLR', T_max=300, eta_min=1e-5, grad_clip=1., precision='BF16 autocast; FP32 loss/optimizer',
        frozen_encoder=True, backbone_state='eval', decoder='shared 64-channel FPN pixel decoder',
        decoder_initial_hash=tensor_hash(probe.decoder), encoder_initial_hash=encoder_hash,
        loss='BCEWithLogits + per-image soft Dice, equal weights, smooth=1; no deep supervision',
        augmentation='H/V flip independently .5; uniform rotation [0,360) with p=.5; GPU bilinear image/nearest mask',
        normalization='ImageNet mean .485/.456/.406 and std .229/.224/.225',
        selection='minimum sample-weighted validation loss; earliest epoch on exact ties',
        early_stopping=False, split='legacy ISIC2018 1886 train / 808 development validation; no independent test',
        pretrained=encoder.audit, trainable_params=sum(p.numel() for p in probe.parameters()),
        total_params=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in probe.parameters()))
    save_json(out/'config.json',config)
    data = ImageData()
    if not (ROOT/'data_manifest.json').exists(): data.save_audit(ROOT/'data_manifest.json')
    best, best_epoch, start, history = float('inf'), 0, 1, []
    if (out/'latest.pth').exists():
        checkpoint = torch.load(out/'latest.pth',map_location='cpu')
        assert checkpoint['config'] == config
        probe.load_state_dict(checkpoint['probe']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        best, best_epoch = checkpoint['best'],checkpoint['best_epoch']
        start, history = checkpoint['epoch']+1,checkpoint['history']
    print('START',json.dumps(config),flush=True)
    t = time.perf_counter()
    val_features, val_targets = encode(encoder,data,'val',batch)
    torch.cuda.synchronize()
    print('Validation feature cache seconds',time.perf_counter()-t,flush=True)
    for epoch in range(start,budget+1):
        t = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        features, targets = encode(encoder,data,'train',batch,epoch)
        probe.train()
        loss_sum = 0.
        lr_used = optimizer.param_groups[0]['lr']
        for offset in range(0,len(targets),64):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = probe([f[offset:offset+64] for f in features])
            loss = segmentation_loss(logits,targets[offset:offset+64])
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient')
            optimizer.step()
            loss_sum += loss.item()*len(logits)
        scheduler.step()
        val,_,_ = evaluate(probe,val_features,val_targets)
        del features,targets,logits,loss
        if val['loss'] < best:
            best,best_epoch = val['loss'],epoch
            save_checkpoint(out/'best.pth',probe.state_dict())
        if epoch in [1,budget] or epoch%25 == 0:
            assert not encoder.training and not any(p.grad is not None for p in encoder.parameters())
            assert tensor_hash(encoder) == encoder_hash, 'Frozen encoder changed'
        torch.cuda.synchronize()
        row=dict(epoch=epoch,train_loss=loss_sum/1886,val=val,best_epoch=best_epoch,
            best_loss=best,lr=lr_used,seconds=time.perf_counter()-t,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        history.append(row)
        save_checkpoint(out/'latest.pth',dict(probe=probe.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,history=history,
            config=config,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        save_json(out/'history.json',history);save_json(out/'status.json',row)
        print('EPOCH',json.dumps(row),flush=True)
    probe.load_state_dict(torch.load(out/'best.pth',map_location='cpu'))
    final,probabilities,per_image = evaluate(probe,val_features,val_targets,export=True)
    for row,sample_id in zip(per_image,data.ids['val']):row['id']=sample_id
    if not args.smoke:
        np.save(out/'probabilities.npy',probabilities)
        np.save(out/'masks.npy',(probabilities>=.5).astype(np.uint8))
        save_json(out/'per_image.json',per_image)
        save_json(out/'ids.json',data.ids['val'])
    save_json(out/'DONE.json',dict(model=args.name,epochs=budget,seed=42,best_epoch=best_epoch,
        best_val_loss=best,final=final,smoke_only=args.smoke,evaluation_split='V_dev',
        encoder_unchanged=tensor_hash(encoder)==encoder_hash))
    print('DONE',final,flush=True)


if __name__ == '__main__':
    main()
