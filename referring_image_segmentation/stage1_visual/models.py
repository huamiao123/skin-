"""Frozen pretrained visual encoders, equal-output adapters, one shared decoder."""
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))


def tensor_hash(module):
    h = hashlib.sha256()
    for key, value in module.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == 'dinov3':
            import timm
            from safetensors.torch import load_file
            self.net = timm.create_model('vit_small_patch16_dinov3.lvd1689m',
                                         pretrained=False, img_size=256)
            path = ROOT / 'weights/dinov3_vits16/model.safetensors'
            result = self.net.load_state_dict(load_file(str(path)), strict=True)
            self.channels = [384] * 4
        elif name == 'vmamba':
            sys.path.insert(0, str(ROOT / 'vendor/VMamba'))
            import vmamba
            if not vmamba.WITH_SELECTIVESCAN_MAMBA:
                raise RuntimeError('Official selective_scan CUDA extension required')
            self.net = vmamba.VSSM(depths=[2, 2, 8, 2], dims=96,
                ssm_d_state=1, ssm_ratio=1., ssm_dt_rank='auto', ssm_conv=3,
                ssm_conv_bias=False, forward_type='v05_noz', mlp_ratio=4.,
                downsample_version='v3', patchembed_version='v2',
                norm_layer='ln2d', drop_path_rate=.2)
            path = ROOT / 'weights/vmamba_tiny_imagenet.pth'
            state = torch.load(path, map_location='cpu')['model']
            result = self.net.load_state_dict(state, strict=True)
            # Exclude the pretrained ImageNet classification head entirely.
            del self.net.classifier
            self.channels = [96, 192, 384, 768]
        else:
            raise ValueError(name)
        self.requires_grad_(False)
        self.eval()
        self.audit = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          strict_load=True, missing=list(result.missing_keys),
                          unexpected=list(result.unexpected_keys), channels=self.channels,
                          params=sum(p.numel() for p in self.parameters()))

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        if self.name == 'dinov3':
            features = self.net.forward_intermediates(x, indices=[2, 5, 8, 11],
                            norm=True, output_fmt='NCHW', intermediates_only=True)
        else:
            x = self.net.patch_embed(x)
            features = []
            for layer in self.net.layers:
                x = layer.blocks(x)
                features.append(x)
                x = layer.downsample(x)
        return [f.contiguous() for f in features]


class ConvBlock(nn.Sequential):
    def __init__(self, channels=64):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, channels), nn.GELU())


class LightDecoder(nn.Module):
    """Same FPN-style pixel decoder for both branches; one binary linear readout."""
    def __init__(self, channels=64):
        super().__init__()
        self.blocks = nn.ModuleList([ConvBlock(channels) for _ in range(4)])
        self.readout = nn.Conv2d(channels, 1, 1)

    def forward(self, pyramid):
        x = self.blocks[3](pyramid[3])
        for i in [2, 1, 0]:
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.blocks[i](x + pyramid[i])
        return F.interpolate(self.readout(x), size=(256, 256), mode='bilinear', align_corners=False)


class Probe(nn.Module):
    def __init__(self, feature_channels):
        super().__init__()
        # No raw-image shortcut: this probe measures the selected encoder features.
        self.adapter = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 64, 1, bias=False),
                    nn.GroupNorm(8, 64), nn.GELU()) for c in feature_channels])
        # Identical initial decoder tensors, independent of adapter channel count.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.decoder = LightDecoder()

    def forward(self, features):
        pyramid = []
        for feature, adapter, side in zip(features, self.adapter, [64, 32, 16, 8]):
            x = adapter(feature)
            if x.shape[-2:] != (side, side):
                x = F.interpolate(x, size=(side, side), mode='bilinear', align_corners=False)
            pyramid.append(x)
        return self.decoder(pyramid)


def segmentation_loss(logits, targets):
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    pred = logits.sigmoid().flatten(1)
    target = targets.flatten(1)
    dice = (2 * (pred * target).sum(1) + 1) / (pred.sum(1) + target.sum(1) + 1)
    return bce + 1 - dice.mean()
