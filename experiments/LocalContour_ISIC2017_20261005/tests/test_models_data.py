from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contour.data import ImageBoundaryDataset, boundary_target, load_mask, preprocess_mask, preprocess_rgb, read_train_manifest, read_val_split
from contour.models import BoundaryHead, FrozenMSGUNet, PUBLIC_WEIGHT_SHA256, file_sha256, segmentation_mask
from tools.train_boundary import CachedFeatureDataset, balanced_bce, resolve_config


@pytest.fixture(autouse=True)
def fewer_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_author_strict_threshold_handles_relu_zero():
    logits = torch.tensor([0.0, 6.0, -6.0])
    assert segmentation_mask(logits).tolist() == [False, True, False]
    assert bool((torch.sigmoid(torch.zeros(20)) >= .5).all())  # why strict > matters


def test_black_rgb_has_reference_imagenet_normalization():
    normalized = preprocess_rgb(np.zeros((9, 13, 3), np.uint8))
    assert normalized.shape == (3, 256, 256)
    assert normalized.dtype == torch.float32
    reference = torch.tensor([-2.11790393, -2.03571415, -1.80444443])
    assert torch.allclose(normalized[:, 30, 40], reference, atol=3e-7, rtol=0)


def test_mask_nearest_keeps_binary_labels():
    raw = np.zeros((7, 5), np.uint8)
    raw[2:5, 1:4] = 255
    out = preprocess_mask(raw)
    assert out.shape == (1, 256, 256)
    assert set(out.unique().tolist()) == {0.0, 1.0}
    assert out[0, 128, 128] == 1
    assert out[0, 0, 0] == 0


def test_three_pixel_boundary_band_definition():
    mask = np.zeros((11, 11), np.uint8)
    mask[3:8, 3:8] = 1
    band = boundary_target(mask)[0].numpy().astype(bool)
    assert int(band.sum()) == 48  # 7x7 outer extent minus one remaining centre pixel
    assert not band[5, 5]
    assert band[2, 5] and band[3, 5] and band[4, 5]


def test_locked_and_test_GT_refused_before_file_open():
    with pytest.raises(ValueError, match='Locked'):
        load_mask({'role': 'locked_verification100', 'mask_path': '/file-that-does-not-exist'})
    with pytest.raises(ValueError, match='test'):
        load_mask({'split': 'test', 'mask_path': '/file-that-does-not-exist'}, allow_verification=True)
    with pytest.raises(ValueError, match='train and calibration'):
        ImageBoundaryDataset([{'role': 'locked_verification100'}])


def test_cache_dataset_scope_is_only_train_and_calibration(tmp_path):
    CachedFeatureDataset(tmp_path, 0, 2000)
    CachedFeatureDataset(tmp_path, 2000, 2050)
    with pytest.raises(ValueError, match='GT scope'):
        CachedFeatureDataset(tmp_path, 2050, 2150)


def test_train_manifest_rejects_mixed_years(tmp_path):
    path = tmp_path / 'train.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'dataset', 'split'])
        w.writeheader()
        w.writerow({'image_id': 'a', 'dataset': 'ISIC2018', 'split': 'train'})
    with pytest.raises(ValueError, match='ISIC2017'):
        read_train_manifest(path, expected_count=1)


def test_frozen_50_100_split_acceptance(tmp_path):
    path = tmp_path / 'split.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'role'])
        w.writeheader()
        w.writerows({'image_id': str(i), 'role': 'calibration50' if i < 50 else 'locked_verification100'} for i in range(150))
    assert len(read_val_split(path, 'calibration50')) == 50
    assert len(read_val_split(path, 'locked_verification100')) == 100


@pytest.mark.parametrize('change', [{'epochs': 11}, {'epochs': 0}, {'seed': 29}, {'feature_resolution': 128}, {'amp': True}, {'learning_rate': .002}])
def test_fixed_budget_and_settings_refuse_unplanned_changes(change):
    with pytest.raises(ValueError):
        resolve_config(change)


def test_head_705_parameters_gradient_and_cache_precision_match():
    head = BoundaryHead()
    assert sum(p.numel() for p in head.parameters()) == 705
    raw = torch.rand(2, 32, 16, 16, requires_grad=True)
    from_raw = head(raw)
    from_cache = head(raw.detach().half())
    assert torch.equal(from_raw, from_cache)
    balanced_bce(from_raw, torch.zeros_like(from_raw), 10.0, .55).backward()
    assert raw.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_public_trained_checkpoint_strict_load_and_freeze():
    path = ROOT / 'third_party/msgu_net/weights/best_model_isic2017.pth'
    assert file_sha256(path) == PUBLIC_WEIGHT_SHA256
    cnn = FrozenMSGUNet()
    before = cnn.state_hash()
    cnn.train(True)
    assert not cnn.training
    assert all(not module.training for module in cnn.cnn.modules())
    assert all(not p.requires_grad for p in cnn.cnn.parameters())
    logits = cnn(torch.zeros(1, 3, 32, 32))
    assert logits.shape == (1, 1, 32, 32)
    assert not logits.requires_grad
    assert torch.isfinite(logits).all()
    assert logits.min() >= 0 and logits.max() <= 6
    assert cnn.state_hash() == before
