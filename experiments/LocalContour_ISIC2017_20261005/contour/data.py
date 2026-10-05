"""Official ISIC2017 records and the author's fixed 256px preprocessing."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

INPUT_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * np.float32(255.0)
IMAGENET_DENOMINATOR = np.reciprocal(np.array([0.229, 0.224, 0.225], dtype=np.float32) * np.float32(255.0))


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    """PIL RGB -> cv2 INTER_LINEAR -> A.Normalize-compatible float32 CHW."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB must be an HxWx3 uint8 array')
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - IMAGENET_MEAN) * IMAGENET_DENOMINATOR
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def preprocess_mask(mask: np.ndarray) -> torch.Tensor:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError('GT must be a two-dimensional binary mask')
    resized = cv2.resize(mask.astype(np.uint8), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((resized > 0).astype(np.float32)[None])


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def load_image(path: str | Path) -> torch.Tensor:
    return preprocess_rgb(load_rgb(path))


def load_mask(record: dict[str, Any], *, allow_verification: bool = False) -> torch.Tensor:
    """Locked100 GT access is explicit and disabled in all head-training paths."""
    if record.get('split') == 'test' or record.get('role') == 'test':
        raise ValueError('Official test GT is outside this probe')
    if record.get('role') == 'locked_verification100' and not allow_verification:
        raise ValueError('Locked verification GT cannot be read before head and DP are frozen')
    with Image.open(record['mask_path']) as mask:
        return preprocess_mask(np.asarray(mask.convert('L')))


def boundary_target(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """A fixed 3px band: inner 1px boundary, dilated once with a 3x3 square."""
    array = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError('Boundary target requires a single two-dimensional mask')
    foreground = array > 0
    structure = np.ones((3, 3), dtype=bool)
    inner_boundary = foreground & ~ndimage.binary_erosion(foreground, structure=structure, border_value=0)
    target = ndimage.binary_dilation(inner_boundary, structure=structure, border_value=0)
    return torch.from_numpy(target.astype(np.float32)[None])


def read_train_manifest(path: str | Path, *, expected_count: int = 2000) -> list[dict[str, str]]:
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_count or len({r['image_id'] for r in rows}) != expected_count:
        raise ValueError('Official train manifest must contain every unique train image')
    if any(r.get('dataset') != 'ISIC2017' or r.get('split') != 'train' for r in rows):
        raise ValueError('Only official ISIC2017 train records may train the boundary head')
    return sorted(rows, key=lambda r: r['image_id'])


def read_val_split(path: str | Path, role: str | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 150 or len({r['image_id'] for r in rows}) != 150:
        raise ValueError('Official validation split requires 150 unique image IDs')
    if sum(r['role'] == 'calibration50' for r in rows) != 50 or sum(r['role'] == 'locked_verification100' for r in rows) != 100:
        raise ValueError('The frozen split must have exactly 50 calibration and 100 locked images')
    if role not in (None, 'calibration50', 'locked_verification100'):
        raise ValueError('Unknown validation role')
    return rows if role is None else [r for r in rows if r['role'] == role]


class ImageBoundaryDataset(Dataset):
    """The cache producer only permits train2000 and calibration50 GT."""

    def __init__(self, records: list[dict[str, str]]):
        if any(r.get('split') != 'train' and r.get('role') != 'calibration50' for r in records):
            raise ValueError('Boundary-head cache may contain train and calibration only')
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        mask = load_mask(record)
        return {'image_id': record['image_id'], 'rgb': load_image(record['image_path']), 'mask': mask, 'boundary': boundary_target(mask)}
