"""Frozen dual encoders with independent adapters and simple concatenation."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from stage3_confirmation.models import Probe as SingleProbe
from stage1_visual.models import FrozenEncoder, tensor_hash, segmentation_loss

CHANNELS = {'dinov3': (384,384,384,384), 'vmamba': (96,192,384,768)}
NAMES = tuple(f'{encoder}_{decoder}' for encoder in ('dinov3','vmamba','fusion')
              for decoder in ('light','swin'))


class FusionProbe(nn.Module):
    def __init__(self, decoder, seed):
        super().__init__()
        # Equal branch-specific adapter and decoder initialization across controls.
        dino = SingleProbe(decoder, seed, CHANNELS['dinov3'])
        mamba = SingleProbe(decoder, seed, CHANNELS['vmamba'])
        self.dino_adapter = dino.adapter
        self.mamba_adapter = mamba.adapter
        self.decoder = dino.decoder
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.fusion = nn.ModuleList([nn.Conv2d(128,64,1,bias=True) for _ in range(4)])

    def forward(self, features):
        assert len(features) == 8
        pyramid = []
        for i, side in enumerate((64,32,16,8)):
            a = self.dino_adapter[i](features[i])
            b = self.mamba_adapter[i](features[i+4])
            a = F.interpolate(a, (side,side), mode='bilinear', align_corners=False)
            b = F.interpolate(b, (side,side), mode='bilinear', align_corners=False)
            pyramid.append(self.fusion[i](torch.cat((a,b),dim=1)))
        return self.decoder(pyramid)


def build_probes(seed):
    return {name: FusionProbe(name.split('_')[1], seed) if name.startswith('fusion_')
            else SingleProbe(name.split('_')[1],seed,CHANNELS[name.split('_')[0]])
            for name in NAMES}


def select_features(name, features):
    return features if name.startswith('fusion_') else features[:4] if name.startswith('dinov3_') else features[4:]


class EncoderPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov3 = FrozenEncoder('dinov3')
        self.vmamba = FrozenEncoder('vmamba')
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        return self.dinov3(x) + self.vmamba(x)

