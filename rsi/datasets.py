from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def letterbox_geometry(height: int, width: int, size: int = 256) -> dict:
    scale = size / max(height, width)
    new_h = max(1, min(size, round(height * scale)))
    new_w = max(1, min(size, round(width * scale)))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    return dict(original_h=height, original_w=width, resized_h=new_h,
                resized_w=new_w, top=top, left=left, size=size)


def restore_logits(logits: torch.Tensor, geometry: dict) -> torch.Tensor:
    """Remove letterbox padding, interpolate logits, then let the caller sigmoid."""
    if logits.ndim == 2:
        logits = logits[None, None]
    elif logits.ndim == 3:
        logits = logits[None]
    t, l = geometry['top'], geometry['left']
    cropped = logits[..., t:t + geometry['resized_h'], l:l + geometry['resized_w']]
    return F.interpolate(cropped.float(),
                         size=(geometry['original_h'], geometry['original_w']),
                         mode='bilinear', align_corners=False)[0, 0]


def _binary_mask(path: str) -> np.ndarray:
    with Image.open(path) as handle:
        a = np.asarray(handle.convert('L'))
    unique = np.unique(a)
    if not np.isin(unique, [0, 1, 255]).all():
        raise ValueError(f'Original reference is not binary: {path}: {unique[:12]}')
    return (a > 0).astype(np.uint8)


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    required = {'image_id', 'reference_id', 'subset', 'split', 'group_id'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest requires {sorted(required)}')
    for row in rows:
        # Accept acquisition's explicit canonical path names.
        row['image_path'] = row.get('image_path') or row.get('rgb_path')
        row['mask_path'] = row.get('mask_path') or row.get('seg_path')
        if not row['image_path'] or not row['mask_path']:
            raise ValueError('Manifest needs image_path/rgb_path and mask_path/seg_path')
    keys = [(r['image_id'], r['subset'], r['reference_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate image/subset/reference manifest key')
    splits = defaultdict(set)
    group_splits = defaultdict(set)
    for row in rows:
        splits[row['image_id']].add(row['split'])
        group_splits[row['group_id']].add(row['split'])
    if any(len(x) > 1 for x in splits.values()):
        raise ValueError('An image has references crossing splits')
    if any(len(x) > 1 for x in group_splits.values()):
        raise ValueError('A known image/patient/lesion group crosses splits')
    return rows


class IMAMultiReferenceDataset(Dataset):
    def __init__(self, manifest, split='train', subset='M', train=False, seed=17,
                 size=256, cache_dir=None, image_ids=None, allow_test_files=False):
        if split == 'test' and not allow_test_files:
            raise PermissionError('Pilot loaders cannot expose test for model scoring')
        if train and (split != 'train' or subset != 'M'):
            raise ValueError('All pilot supervised training uses M train only')
        self.rows = read_manifest(manifest)
        grouped = defaultdict(list)
        requested = set(image_ids) if image_ids is not None else None
        for row in self.rows:
            if row['split'] == split and row['subset'] == subset:
                if requested is None or row['image_id'] in requested:
                    grouped[row['image_id']].append(row)
        self.samples = []
        for image_id in sorted(grouped):
            refs = sorted(grouped[image_id], key=lambda r: r['reference_id'])
            if len(refs) < 2 or len({r.get('annotator_id', r.get('annotator')) for r in refs}) < 2:
                raise ValueError(f'{subset}/{image_id} needs two distinct annotators')
            if len({r.get('annotator_id', r.get('annotator')) for r in refs}) != len(refs):
                raise ValueError(f'{subset}/{image_id} has repeated annotators')
            for row in refs:
                for field in ['image_path', 'mask_path']:
                    if not Path(row[field]).is_file():
                        raise FileNotFoundError(row[field])
            self.samples.append(refs)
        if not self.samples:
            raise ValueError(f'No samples for {split}/{subset}')
        self.split, self.subset, self.train = split, subset, bool(train)
        self.seed, self.size, self.epoch = int(seed), int(size), 0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def _canonical(self, refs):
        key_data = [(r['image_id'], r['reference_id'], r.get('image_sha256', ''),
                     r.get('mask_sha256', ''), r['image_path'], r['mask_path']) for r in refs]
        key = hashlib.sha256(json.dumps([self.size, key_data]).encode()).hexdigest()
        cached = self.cache_dir / (key + '.npz') if self.cache_dir else None
        if cached and cached.is_file():
            with np.load(cached, allow_pickle=False) as data:
                return data['image'].copy(), data['masks'].copy(), data['valid'].copy(), json.loads(str(data['geometry']))
        with Image.open(refs[0]['image_path']) as handle:
            rgb = handle.convert('RGB')
            w, h = rgb.size
            geometry = letterbox_geometry(h, w, self.size)
            shape = (geometry['resized_w'], geometry['resized_h'])
            rgb = rgb.resize(shape, resample=Image.Resampling.BILINEAR)
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            t, l = geometry['top'], geometry['left']
            image[t:t + shape[1], l:l + shape[0]] = np.asarray(rgb)
        masks = np.zeros((len(refs), 1, self.size, self.size), dtype=np.uint8)
        for i, ref in enumerate(refs):
            mask = _binary_mask(ref['mask_path'])
            if mask.shape != (h, w):
                raise ValueError(f'RGB/reference spatial mismatch: {ref["reference_id"]}')
            resized = Image.fromarray(mask).resize(shape, resample=Image.Resampling.NEAREST)
            masks[i, 0, t:t + shape[1], l:l + shape[0]] = np.asarray(resized)
        valid = np.zeros((1, self.size, self.size), dtype=np.uint8)
        valid[:, t:t + shape[1], l:l + shape[0]] = 1
        if cached:
            temp = cached.with_suffix('.tmp.npz')
            np.savez(temp, image=image, masks=masks, valid=valid, geometry=json.dumps(geometry))
            temp.replace(cached)
        return image, masks, valid, geometry

    def __getitem__(self, index):
        refs = self.samples[index]
        image, masks, valid, geometry = self._canonical(refs)
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        masks, valid = torch.from_numpy(masks.copy()).float(), torch.from_numpy(valid.copy()).float()
        if self.train:
            digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{refs[0]["image_id"]}'.encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], 'big'))
            if rng.random() < .5:
                image, masks, valid = [x.flip(-1) for x in (image, masks, valid)]
            if rng.random() < .5:
                image, masks, valid = [x.flip(-2) for x in (image, masks, valid)]
            k = rng.randrange(4)
            image, masks, valid = [x.rot90(k, (-2, -1)) for x in (image, masks, valid)]
            operations = [(TF.adjust_brightness, rng.uniform(.9, 1.1)),
                          (TF.adjust_contrast, rng.uniform(.9, 1.1)),
                          (TF.adjust_saturation, rng.uniform(.9, 1.1)),
                          (TF.adjust_hue, rng.uniform(-.02, .02))]
            rng.shuffle(operations)
            for func, value in operations:
                image = func(image, value)
            image *= valid
        return dict(image=image, masks=masks, rater_present=torch.ones(len(refs), dtype=torch.bool),
                    pixel_valid=valid, image_id=refs[0]['image_id'], group_id=refs[0]['group_id'],
                    references=refs, geometry=geometry, subset=self.subset)


def collate_multi_reference(samples):
    count = max(len(s['references']) for s in samples)
    batch, _, h, w = len(samples), *samples[0]['image'].shape
    masks = torch.zeros(batch, count, 1, h, w)
    present = torch.zeros(batch, count, dtype=torch.bool)
    for i, sample in enumerate(samples):
        r = len(sample['references'])
        masks[i, :r], present[i, :r] = sample['masks'], sample['rater_present']
    return dict(image=torch.stack([s['image'] for s in samples]), masks=masks,
                rater_present=present, pixel_valid=torch.stack([s['pixel_valid'] for s in samples]),
                image_id=[s['image_id'] for s in samples], group_id=[s['group_id'] for s in samples],
                references=[s['references'] for s in samples], geometry=[s['geometry'] for s in samples],
                subset=[s['subset'] for s in samples])
