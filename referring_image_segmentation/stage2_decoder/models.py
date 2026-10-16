"""Swin pixel decoder ablation with the stage-1 adapter and frozen encoders."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from stage1_visual.models import FrozenEncoder, Probe as LightProbe, segmentation_loss, tensor_hash
from timm.models.swin_transformer import SwinTransformerBlock

DECODER_SPEC = dict(channels=64, resolutions=[64,32,16,8], blocks_per_level=2,
                    window_size=8, shift_size=4, num_heads=4, mlp_ratio=4.,
                    dropout=0., drop_path=0., norm='LayerNorm', output_size=256,
                    fusion='bilinear upsample and add; same topology as light FPN',
                    implementation='timm 1.0.22 SwinTransformerBlock')


class SwinLevel(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.blocks = nn.Sequential(*[SwinTransformerBlock(
            dim=64, input_resolution=(side,side), num_heads=4, window_size=8,
            shift_size=shift, mlp_ratio=4., proj_drop=0., attn_drop=0., drop_path=0.)
            for shift in (0,4)])
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        return self.norm(self.blocks(x)).permute(0,3,1,2).contiguous()


class SwinPixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([SwinLevel(side) for side in (64,32,16,8)])
        self.readout = nn.Conv2d(64,1,1)

    def forward(self, pyramid):
        x = self.blocks[3](pyramid[3])
        for i in (2,1,0):
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.blocks[i](x + pyramid[i])
        return F.interpolate(self.readout(x), size=(256,256), mode='bilinear', align_corners=False)


class Probe(LightProbe):
    def __init__(self, feature_channels):
        # The adapter and readout start from the exact same tensors as stage 1.
        super().__init__(feature_channels)
        readout = self.decoder.readout.state_dict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.decoder = SwinPixelDecoder()
        self.decoder.readout.load_state_dict(readout)
