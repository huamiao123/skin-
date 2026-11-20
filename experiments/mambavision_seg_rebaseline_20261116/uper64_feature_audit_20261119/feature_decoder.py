"""Single-site feature corrections for the project's pinned UPer64 decoder."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from model import SegmentationSystem, load_pretrained_encoder

ARMS = ("add", "concat", "gated_concat", "f3_difference", "f4_difference")


class FeatureCorrection(nn.Module):
    def __init__(self, width, arm):
        super().__init__()
        self.include_difference = arm in ("f3_difference", "f4_difference")
        self.mix = nn.Conv2d(2 * width, width, 1)
        self.residual = nn.Conv2d(width, width, 1)
        self.gate = None if arm == "concat" else nn.Conv2d(width, width, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, lateral, anchor):
        evidence = self.mix(torch.cat((lateral, anchor), dim=1))
        if self.include_difference:
            evidence = evidence + (lateral - anchor).abs()
        update = self.residual(F.relu(evidence))
        if self.gate is not None:
            update = self.gate(evidence).sigmoid() * update
        return update


class CorrectedUPerHead(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        self.native = native
        self.use_f4 = arm == "f4_difference"
        self.correct = FeatureCorrection(native.channels, arm)

    def resize(self, feature, shape):
        return F.interpolate(feature, size=shape, mode="bilinear",
                             align_corners=self.native.align_corners)

    def forward(self, features):
        head = self.native
        inputs = head._transform_inputs(features)
        laterals = [conv(x) for conv, x in zip(head.lateral_convs, inputs)]
        laterals.append(head.psp_forward(inputs))
        # Capture the reference BEFORE top-down additions, retaining the native PPM for F4.
        anchor = laterals[3 if self.use_f4 else 2]
        for i in range(3, 0, -1):
            lateral = laterals[i - 1]
            shape = lateral.shape[-2:]
            merged = lateral + self.resize(laterals[i], shape)
            if i == 2:
                merged = merged + self.correct(lateral, self.resize(anchor, shape))
            laterals[i - 1] = merged
        pyramid = [conv(x) for conv, x in zip(head.fpn_convs, laterals)]
        pyramid.append(laterals[-1])
        shape = pyramid[0].shape[-2:]
        aligned = [pyramid[0]] + [self.resize(x, shape) for x in pyramid[1:]]
        return head.cls_seg(head.fpn_bottleneck(torch.cat(aligned, dim=1)))


def build(arm, seed=42):
    if arm not in ARMS:
        raise ValueError(f"unknown feature correction: {arm}")
    model = SegmentationSystem(load_pretrained_encoder(), "A-64-noaux", seed=seed)
    if arm != "add":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model.probe.decode_head = CorrectedUPerHead(model.probe.decode_head, arm)
    return model.cuda()
