"""Decoder-side use of both native MambaVision branches before out_proj."""
from functools import partial
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
BASE = OUT.parent / 'official_full_uper512_fcn_20261120'
sys.path.insert(0, str(BASE))
import official

TAPS = ((2, 3, 384), (3, 2, 768))
WIDTH = 64

class BranchAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ssm = nn.Conv2d(channels // 2, WIDTH, 1, bias=False)
        self.non_ssm = nn.Conv2d(channels // 2, WIDTH, 1, bias=False)
        self.context = nn.Conv2d(channels, WIDTH, 1, bias=False)
        self.gates = nn.Conv2d(4 * WIDTH, 2, 1)
        self.expand = nn.Conv2d(WIDTH, channels, 1, bias=False)
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, feature, token_map):
        a, b = token_map.chunk(2, dim=1)
        s = self.ssm(a)
        n = self.non_ssm(b)
        context = self.context(feature)
        weights = self.gates(torch.cat((context, s, n, (s - n).abs()), dim=1)).softmax(1)
        delta = self.expand(weights[:, :1] * s + weights[:, 1:] * n)
        return feature + self.strength.tanh() * delta

class NativeBranchReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.paths = nn.ModuleList(BranchAdapter(c) for _, _, c in TAPS)
        self.tokens = {}

    def capture(self, slot, module, inputs):
        self.tokens[slot] = inputs[0]

    def adapt(self, module, inputs):
        (original,) = inputs
        features = list(original)
        for slot, (stage, _, c) in enumerate(TAPS):
            feature = features[stage]
            batch, _, h, w = feature.shape
            tokens = self.tokens.pop(slot)
            # At the fixed 256px input and 16/8 windows, each stage is exactly one window.
            if tokens.shape != (batch, h * w, c):
                raise ValueError(f'Unexpected branch token layout: {tokens.shape} for {feature.shape}')
            token_map = tokens.transpose(1, 2).reshape(batch, c, h, w)
            features[stage] = self.paths[slot](feature, token_map)
        return (features,)

def attach(net):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261120)
        readout = NativeBranchReadout()
    net.probe.native_branches = readout.to(next(net.parameters()).device)
    for slot, (stage, block, c) in enumerate(TAPS):
        layer = net.backbone.levels[stage]
        mixer = layer.blocks[block].mixer
        assert type(mixer).__name__ == 'MambaVisionMixer' and mixer.out_proj.in_features == c
        assert layer.window_size == (16, 8)[slot]
        mixer.out_proj.register_forward_pre_hook(partial(readout.capture, slot))
    # Only UPer receives the adapted skip list. NativeHeads' auxiliary FCN receives the original F3.
    net.probe.decode_head.register_forward_pre_hook(readout.adapt)
    return net

def load_branch_model():
    return attach(official.load_model())

def method_config():
    return dict(name='native_preprojection_branch_readout', taps=[[3,3],[4,2]],
                projected_width=WIDTH, residual='zero-initialized scalar tanh',
                scope='UPer feature inputs only', source_sha=official.sha(Path(__file__)))

