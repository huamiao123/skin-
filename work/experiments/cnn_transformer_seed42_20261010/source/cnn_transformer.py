"""Independent ordinary CNN U-Net with global-attention Transformer fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


class GlobalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, 8, stride=8)
        dims = [64, 128, 256, 512]
        self.positions = nn.ParameterList([nn.Parameter(torch.zeros(1, (32//2**i)**2, d))
                                          for i, d in enumerate(dims)])
        self.stages = nn.ModuleList([nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, dim_feedforward=4*d, dropout=0.,
                activation='gelu', batch_first=True, norm_first=True) for _ in range(2)])
            for d, h in zip(dims, [2, 4, 8, 16])])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.down = nn.ModuleList([nn.Conv2d(a, b, 2, stride=2)
                                  for a, b in zip(dims[:-1], dims[1:])])
        for p in self.positions:
            nn.init.trunc_normal_(p, std=.02)

    def forward(self, image):
        x = self.patch_embed(image)
        features = []
        for i, blocks in enumerate(self.stages):
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2) + self.positions[i]
            for block in blocks:
                tokens = checkpoint(block, tokens, use_reentrant=False) if self.training else block(tokens)
            x = self.norms[i](tokens).transpose(1, 2).reshape(b, c, h, w)
            features.append(x)
            if i < 3:
                x = self.down[i](x)
        return features


class ResidualFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))

    def forward(self, cnn, transformer):
        return cnn + self.gamma_logit.sigmoid() * self.proj(transformer)


class CNNTransformerUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_encoder = GlobalTransformer()
        self.stem = DoubleConv(3, 16)
        self.encoder = nn.ModuleList([DoubleConv(a, b) for a, b in
                                     zip([16, 32, 64, 128, 256], [32, 64, 128, 256, 512])])
        self.fuse = nn.ModuleList([ResidualFusion(d) for d in [64, 128, 256, 512]])
        self.decoder = nn.ModuleList([DoubleConv(a+b, b) for a, b in
                                     zip([512, 256, 128, 64, 32], [256, 128, 64, 32, 16])])
        self.head = nn.Conv2d(16, 1, 1)
        self._gamma_stats = {}

    def block(self, module, x):
        return checkpoint(module, x, use_reentrant=False) if self.training else module(x)

    def forward(self, image, cnn_only=False):
        # CNN grids 128/64/32/16/8/4; attention grids 32/16/8/4.
        # Pool raw image once to keep the ordinary convolution path affordable.
        x = self.block(self.stem, F.avg_pool2d(image, 2))
        features = [x]
        auxiliary = None if cnn_only else self.t_encoder(image)
        for i, enc in enumerate(self.encoder):
            x = self.block(enc, F.max_pool2d(x, 2))
            if i >= 1 and not cnn_only:
                x = self.fuse[i-1](x, auxiliary[i-1])
            features.append(x)
        self._gamma_stats = {f'g{i+1}': f.gamma_logit.sigmoid() for i, f in enumerate(self.fuse)}
        for dec, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = self.block(dec, torch.cat([x, skip], dim=1))
        return F.interpolate(self.head(x), size=image.shape[-2:], mode='bilinear', align_corners=False)
