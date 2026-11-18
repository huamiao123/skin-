from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from model import SegmentationSystem, load_pretrained_encoder


def resize(x, shape):
    if x.shape[-2:] == tuple(shape):
        return x
    return F.interpolate(x, size=shape, mode="bilinear", align_corners=False)


def block(source, width, kernel):
    return nn.Sequential(nn.Conv2d(source, width, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(8, width), nn.ReLU(inplace=False))


class AnchorDecoder(nn.Module):
    def __init__(self, guide, width=64):
        super().__init__()
        self.guide = guide
        self.f3 = block(384, width, 1)
        self.f4 = block(768, width, 1)
        self.coarse = nn.Sequential(block(width, width, 3), nn.Conv2d(width, 1, 1))
        self.shallow = nn.ModuleList([block(192, width, 1), block(96, width, 1)])
        self.delta = nn.ModuleList([
            nn.Sequential(block(width * 2, width, 3), nn.Conv2d(width, 1, 1))
            for _ in self.shallow
        ])

    def forward(self, features, output_size):
        f1, f2, f3, f4 = features
        mid = self.f3(f3)
        deep = resize(self.f4(f4), mid.shape[-2:])
        anchor = mid if self.guide == "f3" else deep
        coarse = self.coarse(mid + deep)
        out = coarse
        for feature, project, predict in zip((f2, f1), self.shallow, self.delta):
            shape = feature.shape[-2:]
            evidence = torch.cat((project(feature), resize(anchor, shape)), dim=1)
            out = resize(out, shape) + predict(evidence)
        coarse = resize(coarse, output_size)
        return {"main": resize(out, output_size), "aux": coarse}


def build(arm):
    model = SegmentationSystem(load_pretrained_encoder(), "A-64-noaux", seed=42)
    if arm in ("f3", "f4"):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            model.probe = AnchorDecoder(arm)
    elif arm != "uper64":
        raise ValueError(arm)
    return model.cuda()
