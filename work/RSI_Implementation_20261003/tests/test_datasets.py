import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from rsi.datasets import IMAMultiReferenceDataset, collate_multi_reference, letterbox_geometry, restore_logits


def fixture_manifest(tmp_path):
    rows = []
    for i, shape in enumerate([(19, 31), (25, 13)]):
        h, w = shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[4:10, 3:9] = 255
        ip = tmp_path / f'image{i}.png'
        Image.fromarray(image).save(ip)
        for r in range(i + 2):
            mask = (image[..., 0] > 0).astype(np.uint8) * 255
            mp = tmp_path / f'mask{i}_{r}.png'
            Image.fromarray(mask).save(mp)
            rows.append(dict(image_id=f'I{i}', reference_id=f'R{i}_{r}', subset='M',
                             split='train', group_id=f'G{i}', annotator=f'A{r}',
                             image_path=str(ip), mask_path=str(mp)))
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return manifest


def test_geometry_inverse_shape_and_constant():
    g = letterbox_geometry(37, 61, 256)
    z = torch.zeros(256, 256)
    z[g['top']:g['top'] + g['resized_h'], g['left']:g['left'] + g['resized_w']] = 3
    restored = restore_logits(z, g)
    assert restored.shape == (37, 61)
    assert torch.allclose(restored, torch.full((37, 61), 3.))


def test_references_and_valid_augmented_together(tmp_path):
    manifest = fixture_manifest(tmp_path)
    ds = IMAMultiReferenceDataset(manifest, train=True, size=64, cache_dir=tmp_path/'cache')
    ds.set_epoch(2)
    sample = ds[0]
    assert torch.equal(sample['masks'][0], sample['masks'][1])
    assert not (sample['masks'] * (1-sample['pixel_valid'])).any()
    assert torch.equal(sample['masks'], ds[0]['masks'])
    assert torch.equal(sample['image'], ds[0]['image'])
    changed = False
    for e in range(3, 12):
        ds.set_epoch(e)
        changed |= not torch.equal(sample['image'], ds[0]['image'])
    assert changed


def test_variable_raters_collation(tmp_path):
    ds = IMAMultiReferenceDataset(fixture_manifest(tmp_path), size=64)
    b = collate_multi_reference([ds[0], ds[1]])
    assert b['masks'].shape == (2, 3, 1, 64, 64)
    assert b['rater_present'].tolist() == [[True,True,False],[True,True,True]]
    assert not b['masks'][0,2].any()


def test_test_loader_is_locked(tmp_path):
    with pytest.raises(PermissionError):
        IMAMultiReferenceDataset(fixture_manifest(tmp_path), split='test')
