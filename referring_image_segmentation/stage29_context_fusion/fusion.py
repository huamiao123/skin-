"""Residual fusion adapters: equal capacity; only context routing differs.
Original implementation informed by EMCAD/DCA; no third-party code copied.
"""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False),
                                    nn.GroupNorm(4, 32), nn.GELU())
        self.dw3 = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.dw5 = nn.Conv2d(32, 32, 5, padding=2, groups=32, bias=False)
        self.norm = nn.GroupNorm(4, 32)
        self.gate = nn.Conv2d(32, 1, 1)
        self.out = nn.Conv2d(32, 64, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, context, skip):
        h = self.reduce(torch.cat([context, skip], dim=1))
        h = F.gelu(self.norm((self.dw3(h) + self.dw5(h)) * 0.5))
        # Gate weights a new residual; it never suppresses the original skip.
        return self.out(h * (2 * self.gate(h).sigmoid()))

class FusionDecoder(nn.Module):
    def __init__(self, original, route):
        super().__init__()
        assert route in ('local_fusion', 'anchor_fusion')
        self.blocks = original.blocks
        self.readout = original.readout
        self.route = route
        self.disable_fusion = False
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(290042)
            self.fusion = nn.ModuleList([ResidualFusion() for _ in range(3)])

    def forward(self, pyramid):
        anchor = self.blocks[3](pyramid[3])
        x = anchor
        for i in (2, 1, 0):
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            merged = x + pyramid[i]
            if not self.disable_fusion:
                context = x if self.route == 'local_fusion' else F.interpolate(
                    anchor, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
                merged = merged + self.fusion[i](context, pyramid[i])
            x = self.blocks[i](merged)
        return F.interpolate(self.readout(x), size=(256, 256), mode='bilinear', align_corners=False)
