"""Compare unchanged validation semantics at batch size 1 and a larger batch."""
import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from configs.config_setting import setting_config
from configs.config_ege_dual_difflr import ege_dual_difflr_config
from configs.config_ege_dual_difflr_scf import ege_dual_difflr_scf_config
from datasets.dataset import NPY_datasets
from engine import binary_confusion_counts
from models.ege_dual import EGEDualUNet
from models.egeunet import EGEUNet


def make_model(kind, config):
    cfg = config.model_config
    common = dict(num_classes=cfg['num_classes'], input_channels=cfg['input_channels'],
                  c_list=cfg['c_list'], bridge=cfg['bridge'], gt_ds=cfg['gt_ds'])
    if kind == 'baseline':
        return EGEUNet(**common)
    return EGEDualUNet(
        **common,
        t_embed=getattr(config, 't_embed', 48),
        t_depths=getattr(config, 't_depths', (2, 2, 2, 2)),
        t_head_dim=getattr(config, 't_head_dim', 16),
        t_sr_ratios=getattr(config, 't_sr_ratios', (4, 2, 1, 1)),
        t_mlp_ratio=getattr(config, 't_mlp_ratio', 4.0),
        t_drop_path_rate=getattr(config, 't_drop_path_rate', 0.1),
        fusion_type=getattr(config, 'fusion_type', 'scalar'),
    )


def evaluate(model, criterion, dataset, batch_size, workers, threshold):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, drop_last=False)
    outputs, gate_rows, losses = [], [], []
    first_batch_reduction_diff = None
    confusion = np.zeros((2, 2), dtype=np.int64)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for images, masks in loader:
            images = images.cuda(non_blocking=True).float()
            masks = masks.cuda(non_blocking=True).float()
            gt_pre, out = model(images)
            batch_loss = criterion(gt_pre, out, masks).item()
            losses.append(batch_loss)
            if first_batch_reduction_diff is None and len(images) > 1:
                sample_losses = [criterion(
                    tuple(gt[i:i + 1] for gt in gt_pre),
                    out[i:i + 1], masks[i:i + 1],
                ).item() for i in range(len(images))]
                first_batch_reduction_diff = abs(batch_loss - float(np.mean(sample_losses)))
            out_np = out.squeeze(1).cpu().numpy()
            mask_np = masks.squeeze(1).cpu().numpy()
            outputs.append(out_np)
            confusion += binary_confusion_counts(out_np, mask_np, threshold)
            if getattr(model, '_gamma_stats', None):
                gate_rows.append(torch.cat([
                    model._gamma_stats[key].detach().reshape(len(images), -1)
                    for key in ('g2', 'g3', 'g4', 'g5')
                ], dim=1).cpu().numpy())
    torch.cuda.synchronize()
    return dict(
        output=np.concatenate(outputs, axis=0),
        gamma=np.concatenate(gate_rows, axis=0) if gate_rows else None,
        confusion=confusion,
        loss=float(np.mean(losses)),
        first_batch_reduction_diff=first_batch_reduction_diff,
        seconds=time.perf_counter() - start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('dual', 'scf', 'baseline'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=808)
    parser.add_argument('--repeat-one', action='store_true')
    args = parser.parse_args()
    if args.batch < 2 or args.limit % args.batch:
        raise ValueError('limit must be divisible by batch so mean batch loss matches mean image loss')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.30)
    config = {'dual': ege_dual_difflr_config,
              'scf': ege_dual_difflr_scf_config,
              'baseline': setting_config}[args.kind]
    model = make_model(args.kind, config).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    dataset = Subset(NPY_datasets(args.data_root, config, train=False), range(args.limit))
    one = evaluate(model, config.criterion, dataset, 1, args.workers, config.threshold)
    repeat_one = evaluate(model, config.criterion, dataset, 1, args.workers, config.threshold) if args.repeat_one else None
    many = evaluate(model, config.criterion, dataset, args.batch, args.workers, config.threshold)
    diff = np.abs(one['output'] - many['output'])
    a = one['output'] >= config.threshold
    b = many['output'] >= config.threshold
    gamma_diff = None if one['gamma'] is None else float(np.max(np.abs(one['gamma'] - many['gamma'])))
    report = {
        'kind': args.kind, 'images': args.limit, 'batch': args.batch,
        'loss_batch1': one['loss'], 'loss_batchN': many['loss'],
        'loss_abs_diff': abs(one['loss'] - many['loss']),
        'confusion_batch1': one['confusion'].tolist(),
        'confusion_batchN': many['confusion'].tolist(),
        'threshold_flip_pixels': int(np.count_nonzero(a != b)),
        'prediction_max_abs_diff': float(np.max(diff)),
        'prediction_mean_abs_diff': float(np.mean(diff)),
        'gamma_max_abs_diff': gamma_diff,
        'batch_loss_vs_mean_sample_loss_abs_diff': many['first_batch_reduction_diff'],
        'repeat_batch1_loss_abs_diff': None if repeat_one is None else abs(one['loss'] - repeat_one['loss']),
        'repeat_batch1_prediction_max_abs_diff': None if repeat_one is None else float(np.max(np.abs(one['output'] - repeat_one['output']))),
        'repeat_batch1_threshold_flip_pixels': None if repeat_one is None else int(np.count_nonzero((one['output'] >= config.threshold) != (repeat_one['output'] >= config.threshold))),
        'seconds_batch1': one['seconds'], 'seconds_batchN': many['seconds'],
        'equivalent': bool(np.array_equal(one['confusion'], many['confusion'])
                           and np.count_nonzero(a != b) == 0
                           and abs(one['loss'] - many['loss']) < 1e-5
                           and (gamma_diff is None or gamma_diff < 1e-5)),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
