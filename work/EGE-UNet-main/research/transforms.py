from __future__ import annotations

import random

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ResearchTransform:
    """Paired research_v1 preprocessing and augmentation."""

    def __init__(self, size=(256, 256), train=False, rotation_p=0.5):
        self.size = tuple(size)
        self.train = bool(train)
        self.rotation_p = float(rotation_p)

    def __call__(self, data):
        image, mask = data
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.as_tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        mask = (mask >= 0.5).to(torch.float32)
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() < self.rotation_p:
                angle = random.uniform(0.0, 360.0)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)
        return image, (mask >= 0.5).to(torch.float32)

