"""Independent ordinary CNN + global Transformer, corrected seed-42 recipe."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('ISIC2018_DATA_ROOT', str(ROOT / 'data')))
sys.path.insert(0, str(ROOT / 'source'))
from utils import (BceDiceLoss, myNormalize, myToTensor, myRandomHorizontalFlip,
                   myRandomVerticalFlip, myRandomRotation, myResize,
                   set_seed, seed_worker, make_data_generator)
from datasets.dataset import NPY_datasets


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def build_model(name, out):
    from cnn_transformer import CNNTransformerUNet
    return CNNTransformerUNet().cuda()


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, tp, fp, fn, tn = [], 0, 0, 0, 0
    for images, masks in loader:
        images, masks = images.cuda().float(), masks.cuda().float()
        pred = model(images).sigmoid()
        losses.append(criterion(pred, masks).item())
        p, g = pred >= .5, masks >= .5
        tp += (p & g).sum().item()
        fp += (p & ~g).sum().item()
        fn += (~p & g).sum().item()
        tn += (~p & ~g).sum().item()
    return dict(loss=float(np.mean(losses)), dice=2*tp/(2*tp+fp+fn),
                iou=tp/(tp+fp+fn), sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['cnn_transformer'])
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke-epochs', type=int, default=0)
    args = parser.parse_args()
    budget = args.smoke_epochs or 300
    assert 0 <= args.smoke_epochs <= 2
    out = ROOT / ('smoke' if args.smoke_epochs else 'results') / args.model
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE.json').exists() and not args.preflight:
        print('Already complete', flush=True)
        return
    torch.set_num_threads(4)
    set_seed(42)
    config = type('Config', (), {
        'train_transformer': transforms.Compose([
            myNormalize('isic18', train=True), myToTensor(),
            myRandomHorizontalFlip(.5), myRandomVerticalFlip(.5),
            myRandomRotation(.5, [0, 360]), myResize(256, 256)]),
        'test_transformer': transforms.Compose([
            myNormalize('isic18', train=False), myToTensor(), myResize(256, 256)])})
    train = NPY_datasets(DATA_ROOT, config, train=True)
    val = NPY_datasets(DATA_ROOT, config, train=False)
    assert (len(train), len(val)) == (1886, 808)
    generator = make_data_generator(42)
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=4,
                              pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val, batch_size=8, num_workers=4, pin_memory=True,
                            worker_init_fn=seed_worker)
    model = build_model(args.model, out)
    criterion = BceDiceLoss(1, 1)
    t_params = list(model.t_encoder.parameters())
    t_ids = {id(p) for p in t_params}
    cnn_params = [p for p in model.parameters() if id(p) not in t_ids]
    optimizer = torch.optim.AdamW([{'params': cnn_params, 'lr': 1e-3}, {'params': t_params, 'lr': 1e-4}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    resolved = dict(model=args.model, seed=42, epochs=budget, train_batch=64, val_batch=8,
                    final_batch=1, accumulation=1, workers=4, amp=False, cnn_lr=1e-3, transformer_lr=1e-4,
                    optimizer='AdamW', weight_decay=.01, eta_min=1e-5,
                    loss='final BCE+Dice, equal weights, no deep supervision',
                    selection='minimum validation loss (mean batch losses, legacy EGE convention)',
                    split='legacy 1886/808 development split, no independent test',
                    depths=[2,2,2,2], params=sum(p.numel() for p in model.parameters()),
                    torch=torch.__version__, device=torch.cuda.get_device_name(),
                    augmentation='HFlip .5, VFlip .5, resampled rotation .5 [0,360]; nearest mask interpolation')
    save_json(out / 'config.json', resolved)
    if args.model == 'cnn_transformer':
        resolved['activation_checkpointing'] = True
        resolved['architecture'] = 'independent plain convolution U-Net + global nn.TransformerEncoderLayer encoder'
        resolved['pretrained'] = False
        resolved['patch_size'] = 8
        resolved['attention'] = 'global standard multihead self-attention; no windows or spatial reduction'
        resolved['transformer_dims'] = [64,128,256,512]
        resolved['fusion'] = 'scalar protected residual at 32/16/8/4, gamma init0.1'
        resolved['transformer_encoder_params'] = sum(p.numel() for p in t_params)
        resolved['cnn_fusion_params'] = sum(p.numel() for p in cnn_params)
        save_json(out / 'config.json', resolved)
    if args.preflight:
        model.train()
        encoder_before = model.t_encoder.patch_embed.weight.detach().clone()
        cnn_before = model.stem[0].weight.detach().clone()
        images, masks = next(iter(train_loader))
        # Legacy pre-resized source masks contain gray edge pixels; preserve
        # these targets exactly as in the corrected seed-42 EGE training.
        assert torch.isfinite(masks).all() and masks.min() >= 0 and masks.max() <= 1
        images, masks = images.cuda().float(), masks.cuda().float()
        pred = model(images).sigmoid()
        loss = criterion(pred, masks)
        assert pred.shape == masks.shape and torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        # A second update covers steady-state Adam moment storage as well.
        optimizer.zero_grad(set_to_none=True)
        pred = model(images).sigmoid()
        loss = criterion(pred, masks)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        assert not torch.equal(encoder_before, model.t_encoder.patch_embed.weight)
        assert not torch.equal(cnn_before, model.stem[0].weight)
        metrics = evaluate(model, DataLoader(torch.utils.data.Subset(val, range(8)), batch_size=8), criterion)
        save_json(out / 'preflight.json', dict(loss=loss.item(), gradient_norm=norm.item(),
                  validation=metrics, peak_gpu_bytes=torch.cuda.max_memory_allocated(), passed=True))
        print('PREFLIGHT PASSED', resolved, flush=True)
        return
    start, best_loss, best_epoch, history = 1, float('inf'), 0, []
    latest = out / 'latest.pth'
    if latest.exists():
        ckpt = torch.load(latest, map_location='cpu')
        assert ckpt['config'] == resolved
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        random.setstate(ckpt['python_rng']); np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng']); torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        generator.set_state(ckpt['loader_rng'])
        start, best_loss, best_epoch, history = ckpt['epoch']+1, ckpt['best_loss'], ckpt['best_epoch'], ckpt['history']
    print('START', resolved, 'epoch', start, flush=True)
    for epoch in range(start, budget + 1):
        t0 = time.monotonic()
        model.train()
        train_losses = []
        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(non_blocking=True).float(), masks.cuda(non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(images).sigmoid()
            loss = criterion(pred, masks)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient')
            optimizer.step()
            train_losses.append(loss.item())
            if step % 20 == 0:
                print(f'train epoch={epoch} iter={step} loss={loss.item():.6f}', flush=True)
        scheduler.step()
        metrics = evaluate(model, val_loader, criterion)
        if metrics['loss'] < best_loss:
            best_loss, best_epoch = metrics['loss'], epoch
            save_checkpoint(out / 'best.pth', model.state_dict())
        row = dict(epoch=epoch, train_loss=float(np.mean(train_losses)), val=metrics,
                   best_epoch=best_epoch, best_loss=best_loss, gamma={k: v.detach().mean().item() for k,v in model._gamma_stats.items()}, seconds=time.monotonic()-t0)
        history.append(row)
        save_checkpoint(latest, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, best_loss=best_loss, best_epoch=best_epoch,
            history=history, config=resolved, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), loader_rng=generator.get_state()))
        save_json(out / 'history.json', history)
        save_json(out / 'status.json', row)
        print('EPOCH', json.dumps(row), flush=True)
    model.load_state_dict(torch.load(out / 'best.pth', map_location='cpu'))
    final = evaluate(model, DataLoader(val, batch_size=1, num_workers=4, pin_memory=True,
                                      worker_init_fn=seed_worker), criterion)
    save_json(out / 'DONE.json', dict(model=args.model, seed=42, epochs=budget,
              best_epoch=best_epoch, best_val_loss=best_loss, evaluation_split='V_dev', smoke_only=bool(args.smoke_epochs), final=final))
    print('DONE', final, flush=True)


if __name__ == '__main__':
    main()
