"""Train exactly one small score head; the public CNN is never optimized."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from contour.data import ImageBoundaryDataset, read_train_manifest, read_val_split
from contour.models import BoundaryHead, FrozenMSGUNet, file_sha256

DEFAULT_CONFIG = {
    'seed': 17, 'epochs': 10, 'batch_size': 8, 'learning_rate': 0.001,
    'device': 'cuda', 'num_workers': 4, 'input_size': 256,
    'feature_channels': 32, 'feature_dtype': 'float16',
    'feature_resolution': 256, 'boundary_kernel_size': 3,
    'boundary_definition': 'foreground inner 1px 3x3-erosion boundary, then one 3x3 dilation; 3px band',
    'optimizer': 'Adam', 'augmentation': 'none', 'amp': False,
    'checkpoint_selection': 'minimum calibration50 balanced BCE; exact tie keeps earlier epoch',
    'positive_weight': 'negative_pixels / positive_pixels, measured on train2000 boundary targets once',
    'balanced_normalizer': '1 / (2 * train_negative_pixel_fraction)',
    'head': 'Conv1x1(32,16)+GroupNorm(4,16)+ReLU+Conv3x3(16,1); 705 parameters',
    'cnn_frozen': True, 'test_scoring': False,
}


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve_config(cfg: dict | None = None) -> dict:
    result = DEFAULT_CONFIG | (cfg or {})
    if result['seed'] != 17 or not isinstance(result['epochs'], int) or not 1 <= result['epochs'] <= 10:
        raise ValueError('Only seed17 and at most ten boundary-head epochs are allowed')
    fixed = ['batch_size', 'learning_rate', 'input_size', 'feature_channels', 'feature_dtype', 'feature_resolution', 'boundary_kernel_size', 'boundary_definition', 'optimizer', 'augmentation', 'amp', 'checkpoint_selection', 'positive_weight', 'balanced_normalizer', 'head', 'cnn_frozen', 'test_scoring']
    if any(result[k] != DEFAULT_CONFIG[k] for k in fixed):
        raise ValueError('The preregistered boundary head/cache/optimizer settings cannot be changed silently')
    return result


def training_signature(root: Path, cfg: dict) -> dict:
    return {'config': cfg,
            'code_sha256': {str(p.relative_to(root)): file_sha256(p) for p in [root / 'contour/data.py', root / 'contour/models.py', root / 'tools/train_boundary.py']},
            'asset_sha256': {str(p.relative_to(root)): file_sha256(p) for p in [root / 'assets/isic2017_train_asset_manifest.csv', root / 'assets/val_split_seed17.csv', root / 'third_party/msgu_net/source_manifest.json']}}


def build_feature_cache(root: Path, cfg: dict, cnn: FrozenMSGUNet) -> dict:
    cache_dir = root / 'boundary_head/feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = read_train_manifest(root / 'assets/isic2017_train_asset_manifest.csv') + read_val_split(root / 'assets/val_split_seed17.csv', 'calibration50')
    if len(records) != 2050:
        raise AssertionError('Only train2000 + calibration50 may produce head-training cache')
    signature = training_signature(root, cfg)
    signature_sha = canonical_sha(signature)
    index_path = cache_dir / 'index.json'
    before = cnn.state_hash()
    if index_path.is_file():
        existing = json.loads(index_path.read_text())
        if existing['signature_sha256'] != signature_sha or existing['cnn_state_hash'] != before:
            raise RuntimeError('Existing feature cache identity differs; preserve it instead of silently overwriting')
        for name, expected in existing['files_sha256'].items():
            if file_sha256(cache_dir / name) != expected:
                raise RuntimeError('Frozen feature cache file changed: ' + name)
        return existing
    shape = (len(records), 32, 256, 256)
    features = np.lib.format.open_memmap(cache_dir / 'features.npy', mode='w+', dtype=np.float16, shape=shape)
    targets = np.lib.format.open_memmap(cache_dir / 'boundary_targets.npy', mode='w+', dtype=np.uint8, shape=(len(records), 1, 256, 256))
    masks = np.lib.format.open_memmap(cache_dir / 'segmentation_targets.npy', mode='w+', dtype=np.uint8, shape=(len(records), 1, 256, 256))
    logits = np.lib.format.open_memmap(cache_dir / 'cnn_logits.npy', mode='w+', dtype=np.float32, shape=(len(records), 1, 256, 256))
    loader = DataLoader(ImageBoundaryDataset(records), batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=cfg['device'].startswith('cuda'))
    cursor = positive_train = 0
    began = time.perf_counter()
    for batch in loader:
        details = cnn.forward_with_features(batch['rgb'].to(cfg['device'], non_blocking=True))
        size = len(batch['image_id'])
        dest = slice(cursor, cursor + size)
        features[dest] = details['features'].cpu().numpy().astype(np.float16)
        logits[dest] = details['logits'].cpu().numpy()
        boundary = batch['boundary'].numpy().astype(np.uint8)
        targets[dest] = boundary
        masks[dest] = batch['mask'].numpy().astype(np.uint8)
        if cursor < 2000:
            positive_train += int(boundary[:min(size, 2000 - cursor)].sum())
        cursor += size
        if cursor % 200 == 0 or cursor == len(records):
            print(f'Frozen CNN feature cache {cursor}/{len(records)}; seconds={time.perf_counter()-began:.1f}', flush=True)
    for array in [features, targets, masks, logits]:
        array.flush()
    del features, targets, masks, logits
    after = cnn.state_hash()
    if before != after or any(module.training for module in cnn.cnn.modules()) or any(p.requires_grad for p in cnn.cnn.parameters()):
        raise RuntimeError('Frozen CNN parameters/buffers/modes changed while producing features')
    train_pixels = 2000 * 256 * 256
    if not 0 < positive_train < train_pixels:
        raise RuntimeError('The train boundary labels require positive and negative examples')
    fraction = positive_train / train_pixels
    result = {'status': 'PASS', 'signature_sha256': signature_sha, 'cnn_state_hash': before, 'cnn_state_hash_after': after, 'cnn_architecture': cnn.architecture_audit(), 'rows': [{k: r[k] for k in ['image_id', 'image_path', 'mask_path']} | {'role': 'train2000' if i < 2000 else 'calibration50'} for i, r in enumerate(records)], 'counts': {'train': 2000, 'calibration': 50, 'locked_verification': 0, 'official_test': 0}, 'feature_shape': list(shape), 'feature_dtype': 'float16', 'feature_resolution': 'native dec1 full256; no spatial downsampling', 'feature_quantization_also_applied_at_inference': True, 'train_boundary_positive_pixels': positive_train, 'train_boundary_total_pixels': train_pixels, 'train_boundary_positive_fraction': fraction, 'positive_weight': (1 - fraction) / fraction, 'balanced_normalizer': 1 / (2 * (1 - fraction)), 'cache_seconds': time.perf_counter() - began, 'files_sha256': {name: file_sha256(cache_dir / name) for name in ['features.npy', 'boundary_targets.npy', 'segmentation_targets.npy', 'cnn_logits.npy']}}
    atomic_json(index_path, result)
    return result


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir: Path, begin: int, end: int):
        if (begin, end) not in [(0, 2000), (2000, 2050)]:
            raise ValueError('Head training/calibration must not access any other GT scope')
        self.cache_dir, self.begin, self.end = cache_dir, begin, end
        self._features = self._targets = None

    def __len__(self):
        return self.end - self.begin

    def __getitem__(self, index):
        if self._features is None:
            self._features = np.load(self.cache_dir / 'features.npy', mmap_mode='r')
            self._targets = np.load(self.cache_dir / 'boundary_targets.npy', mmap_mode='r')
        row = self.begin + int(index)
        return torch.from_numpy(np.array(self._features[row], copy=True)), torch.from_numpy(self._targets[row].astype(np.float32))


def balanced_bce(logits, targets, positive_weight: float, normalizer: float):
    weight = torch.as_tensor(positive_weight, dtype=torch.float32, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=weight) * normalizer


@torch.no_grad()
def evaluate_head(head, loader, device, positive_weight, normalizer):
    head.eval()
    count = 0
    loss_sum = 0.0
    tp = fp = fn = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = head(features)
        loss_sum += float(balanced_bce(logits, targets, positive_weight, normalizer)) * len(features)
        count += len(features)
        prediction, truth = torch.sigmoid(logits) > 0.5, targets > 0.5
        tp += int((prediction & truth).sum())
        fp += int((prediction & ~truth).sum())
        fn += int((~prediction & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'balanced_bce': loss_sum / count, 'boundary_band_pixel_precision_at_0_5': precision, 'boundary_band_pixel_recall_at_0_5': recall, 'boundary_band_pixel_f1_at_0_5': 2 * precision * recall / (precision + recall) if precision + recall else 0.0, 'images': count}


def train_boundary_head(cfg: dict | None = None, *, root: str | Path | None = None) -> dict:
    cfg = resolve_config(cfg)
    root = Path(root or ROOT).resolve()
    out = root / 'boundary_head'
    out.mkdir(parents=True, exist_ok=True)
    signature = training_signature(root, cfg)
    signature_sha = canonical_sha(signature)
    done_path = out / 'DONE.json'
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        if done['signature_sha256'] != signature_sha or done['epochs_completed'] != cfg['epochs']:
            raise RuntimeError('Completed head signature or budget differs')
        for name in ['best.pth', 'latest.pth', 'training_log.csv']:
            if file_sha256(out / name) != done['files_sha256'][name]:
                raise RuntimeError('Completed boundary-head artifact hash differs: ' + name)
        return done
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Real GPU is required for the authorized head training')
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    config_path = out / 'config.json'
    if config_path.is_file():
        previous_config = json.loads(config_path.read_text())
        if previous_config['signature_sha256'] != signature_sha:
            raise RuntimeError('Existing preregistered head source/config differs; preserve the original run')
    else:
        atomic_json(config_path, signature | {'signature_sha256': signature_sha, 'preregistered_before_training_utc': datetime.now(timezone.utc).isoformat()})
    cnn = FrozenMSGUNet(root / 'third_party/msgu_net').to(cfg['device'])
    cnn_before = cnn.state_hash()
    cache = build_feature_cache(root, cfg, cnn)
    head = BoundaryHead().to(cfg['device'])
    if sum(p.numel() for p in head.parameters()) != 705:
        raise AssertionError('The single preregistered score head must have 705 parameters')
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg['learning_rate'])
    cached_train = CachedFeatureDataset(out / 'feature_cache', 0, 2000)
    cached_calibration = CachedFeatureDataset(out / 'feature_cache', 2000, 2050)
    calibration_loader = DataLoader(cached_calibration, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=cfg['device'].startswith('cuda'))
    positive_weight, normalizer = cache['positive_weight'], cache['balanced_normalizer']
    start, best_value, best_epoch, updates = 0, math.inf, None, 0
    rows = []
    elapsed_seconds = 0.0
    latest_path = out / 'latest.pth'
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location=cfg['device'], weights_only=True)
        if latest['signature_sha256'] != signature_sha or latest['cnn_state_hash'] != cnn_before:
            raise RuntimeError('Unfinished head cannot resume with a different source/CNN/config')
        head.load_state_dict(latest['state_dict'], strict=True)
        optimizer.load_state_dict(latest['optimizer'])
        start, best_value, best_epoch, updates = latest['epoch'], latest['best_balanced_bce'], latest['best_epoch'], latest['optimizer_updates']
        rows, elapsed_seconds = latest['training_rows'], latest['head_training_seconds']
        torch.set_rng_state(latest['torch_rng'].cpu())
        if cfg['device'].startswith('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in latest['cuda_rng']])
        # Rebuild an interrupted selected checkpoint transaction from latest.
        if latest.get('epoch') == latest['best_epoch'] and (not (out / 'best.pth').is_file() or torch.load(out / 'best.pth', map_location='cpu', weights_only=True)['epoch'] != best_epoch):
            atomic_torch_save({k: latest[k] for k in ['state_dict', 'epoch', 'signature_sha256', 'cnn_state_hash', 'calibration_metrics', 'boundary_head_architecture']}, out / 'best.pth')
        print(f'Resume boundary head after complete epoch {start}', flush=True)
    for epoch in range(start + 1, cfg['epochs'] + 1):
        began = time.perf_counter()
        generator = torch.Generator().manual_seed(17 + epoch * 100003)
        train_loader = DataLoader(cached_train, batch_size=cfg['batch_size'], shuffle=True, generator=generator, num_workers=0, pin_memory=cfg['device'].startswith('cuda'))
        head.train()
        loss_sum, count, epoch_updates = 0.0, 0, 0
        for features, targets in train_loader:
            features, targets = features.to(cfg['device'], non_blocking=True), targets.to(cfg['device'], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_bce(head(features), targets, positive_weight, normalizer)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite boundary head objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('Nonfinite boundary head gradient')
            optimizer.step()
            epoch_updates += 1
            loss_sum += float(loss.detach()) * len(features)
            count += len(features)
        if count != 2000 or epoch_updates != 250:
            raise AssertionError('Every epoch requires all train2000 images and 250 actual head optimizer updates')
        updates += epoch_updates
        calibration = evaluate_head(head, calibration_loader, cfg['device'], positive_weight, normalizer)
        frozen_after = cnn.state_hash()
        if frozen_after != cnn_before or any(module.training for module in cnn.cnn.modules()):
            raise RuntimeError('CNN changed during score-head training')
        elapsed_seconds += time.perf_counter() - began
        is_best = calibration['balanced_bce'] < best_value
        if is_best:
            best_value, best_epoch = calibration['balanced_bce'], epoch
        row = {'epoch': epoch, 'train_images': count, 'train_balanced_bce': loss_sum / count, 'calibration_images': 50, 'calibration_balanced_bce': calibration['balanced_bce'], 'calibration_boundary_band_pixel_precision': calibration['boundary_band_pixel_precision_at_0_5'], 'calibration_boundary_band_pixel_recall': calibration['boundary_band_pixel_recall_at_0_5'], 'calibration_boundary_band_pixel_f1': calibration['boundary_band_pixel_f1_at_0_5'], 'optimizer_updates': epoch_updates, 'cumulative_optimizer_updates': updates, 'best_epoch': best_epoch, 'epoch_seconds': time.perf_counter() - began, 'cnn_parameter_buffer_hash_unchanged': True}
        rows.append(row)
        snapshot = {'state_dict': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'signature_sha256': signature_sha, 'cnn_state_hash': cnn_before, 'best_balanced_bce': best_value, 'best_epoch': best_epoch, 'optimizer_updates': updates, 'training_rows': rows, 'head_training_seconds': elapsed_seconds, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if cfg['device'].startswith('cuda') else [], 'calibration_metrics': calibration, 'boundary_head_architecture': head.architecture_audit()}
        atomic_torch_save(snapshot, latest_path)
        if is_best:
            atomic_torch_save({k: snapshot[k] for k in ['state_dict', 'epoch', 'signature_sha256', 'cnn_state_hash', 'calibration_metrics', 'boundary_head_architecture']}, out / 'best.pth')
        with (out / 'training_log.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Boundary head epoch {epoch}/{cfg["epochs"]}; train BCE={row["train_balanced_bce"]:.6f}; calibration BCE={calibration["balanced_bce"]:.6f}; best={best_epoch}; updates={updates}; seconds={row["epoch_seconds"]:.1f}', flush=True)
    if len(rows) != cfg['epochs'] or updates != cfg['epochs'] * 250:
        raise AssertionError('Head budget/readout incomplete')
    best = torch.load(out / 'best.pth', map_location=cfg['device'], weights_only=True)
    head.load_state_dict(best['state_dict'], strict=True)
    final_calibration = evaluate_head(head, calibration_loader, cfg['device'], positive_weight, normalizer)
    complete = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'signature_sha256': signature_sha, 'epochs_completed': cfg['epochs'], 'best_epoch': best_epoch, 'optimizer_updates': updates, 'cnn_state_hash_before': cnn_before, 'cnn_state_hash_after': cnn.state_hash(), 'cnn_parameters_and_buffers_unchanged': cnn.state_hash() == cnn_before, 'cnn_training_performed': False, 'transformer_training_performed': False, 'training_GT_scope': 'official ISIC2017 train2000 only', 'checkpoint_selection_GT_scope': 'calibration50 only; no locked_verification100', 'locked_verification_GT_opened_during_head_training': 0, 'test_GT_opened': 0, 'feature_cache_seconds': cache['cache_seconds'], 'head_training_seconds': elapsed_seconds, 'selected_head_calibration': final_calibration, 'head_ability_metric_note': 'Pixel precision/recall/F1 of the 3px GT-derived boundary band at probability>0.5; these are calibration diagnostics and do not replace candidate coverage or final segmentation metrics.', 'boundary_head_architecture': head.architecture_audit(), 'head_best_path': str(out / 'best.pth'), 'feature_cache_dir': str(out / 'feature_cache'), 'files_sha256': {name: file_sha256(out / name) for name in ['best.pth', 'latest.pth', 'training_log.csv']}, 'historical_val_exposure': 'Public CNN trained/selected after pooled official train+val 70/30 split; all probe results are development diagnosis.'}
    atomic_json(out / 'DONE.json', complete)
    del cnn, head, optimizer
    if cfg['device'].startswith('cuda'):
        torch.cuda.empty_cache()
    return complete


def load_selected_head(*, root: str | Path | None = None, device='cuda') -> BoundaryHead:
    root = Path(root or ROOT)
    done = json.loads((root / 'boundary_head/DONE.json').read_text())
    path = root / 'boundary_head/best.pth'
    if done['status'] != 'PASS' or file_sha256(path) != done['files_sha256']['best.pth']:
        raise RuntimeError('The selected head has not been verified complete')
    selected = torch.load(path, map_location=device, weights_only=True)
    head = BoundaryHead().to(device)
    head.load_state_dict(selected['state_dict'], strict=True)
    head.eval()
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    result = train_boundary_head({'epochs': args.epochs, 'device': args.device, 'num_workers': args.workers}, root=args.root)
    print(json.dumps({'status': result['status'], 'epochs_completed': result['epochs_completed'], 'best_epoch': result['best_epoch'], 'optimizer_updates': result['optimizer_updates']}), flush=True)


if __name__ == '__main__':
    main()
