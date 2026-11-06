"""Region supervision control and batched adaptation of the official HiRAM.

Official source: deepnoid-ai/HiRAM, commit d8aac2de3c33f8f2e79e281fcea8e5239db29645.
The original sources and CC BY-NC 4.0 license are preserved in vendor_hiram/.
Only the 16x16 pair is changed; legacy adapters, concat and Swin are reused.
"""
from common import *
from torch import nn
from torch.nn import functional as F

from stage4_fusion.models import FusionProbe
from stage1_visual.models import segmentation_loss, tensor_hash
from vendor_hiram.HiRAM import HiRAM


def route_logits(router, x):
    b, _, h, w = x.shape
    return F.log_softmax(router.route_proj(x.permute(0, 2, 3, 1)), dim=-1).permute(0, 3, 1, 2).reshape(b, 3, h, w)


def _mug_batched(mug, hidden, stochastic):
    """Independent per-image region scans, padded only after valid tokens.

    Within each sample/region the sequence and prior equal the official B=1
    implementation. Padding is placed after valid tokens; causal SSM outputs
    for valid tokens cannot depend on later padding. Evaluation uses argmax
    rather than the official stochastic Gumbel route; this is predeclared.
    """
    b, _, h, w = hidden.shape
    length = h * w
    xz = mug.in_proj(hidden.permute(0, 2, 3, 1))
    x, z = xz.chunk(2, dim=-1)
    z = mug.act(z).permute(0, 3, 1, 2).contiguous()
    x = x.permute(0, 3, 1, 2).contiguous()
    c = x.shape[1]
    logmap = route_logits(mug.prior_generate, hidden)
    logflat = logmap.flatten(2).transpose(1, 2).contiguous()
    if stochastic:
        chosen = F.gumbel_softmax(logflat, hard=True, dim=-1).detach().argmax(-1)
    else:
        chosen = logflat.detach().argmax(-1)
    flat = x.flatten(2)
    merged = flat.new_zeros((b, c, length), dtype=torch.float32)
    for region, ssm in enumerate((mug.ssm_bg, mug.ssm_bd, mug.ssm_in)):
        valid = chosen == region
        confidence = logflat[:, :, region]
        indices = torch.argsort(confidence.detach().masked_fill(~valid, -torch.inf), dim=1, descending=True)
        sorted_valid = valid.gather(1, indices)
        ordered = flat.gather(2, indices[:, None, :].expand(-1, c, -1))
        ordered_conf = confidence.gather(1, indices)
        bank = mug.prior_generate.shared_emb.weight @ mug.prior_generate.region_emb_list[region].weight
        prior = (ordered_conf.unsqueeze(-1) * bank).permute(0, 2, 1).contiguous()
        values = ssm(ordered, prior).float() * sorted_valid[:, None, :]
        merged = merged.scatter_add(2, indices[:, None, :].expand(-1, c, -1), values)
    y = mug.ssm_full(merged, None, skip_conv=True)
    y_rev = mug.ssm_full_bwd(torch.flip(merged, dims=[-1]), None, skip_conv=True)
    y = y + torch.flip(y_rev, dims=[-1])
    y = mug.out_act(mug.out_norm(y.reshape(b, c, h, w)))
    return y.to(x.dtype), z, logmap


def batched_hiram(module, a, b, stochastic):
    ya, za, ra = _mug_batched(module.mug_sam, module.norm_sam(a), stochastic)
    yb, zb, rb = _mug_batched(module.mug_cnn, module.norm_cnn(b), stochastic)
    ya = module.out_proj_sam((ya * zb).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    yb = module.out_proj_cnn((yb * za).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return a + ya, b + yb, [ra, rb]


class BaselineProbe(FusionProbe):
    """Rerun the stage4 baseline to match both controls on this hardware."""
    def __init__(self):
        super().__init__('swin', 42)

    def forward(self, features, return_regions=False):
        logits = super().forward(features)
        return (logits, []) if return_regions else logits


class RegionProbe(FusionProbe):
    def __init__(self, mode):
        if mode not in ('region_only', 'hiram'):
            raise ValueError(mode)
        super().__init__('swin', 42)
        self.mode = mode
        # Construct the same official dual module in both arms, then retain only
        # the routing heads in the auxiliary-supervision control.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1042)
            official = HiRAM(64)
        if mode == 'hiram':
            self.hiram = official
            # Official global scans use skip_conv=True, so these convolutions
            # are not in the computational graph and must not be optimized.
            for mug in (self.hiram.mug_sam, self.hiram.mug_cnn):
                mug.ssm_full.conv.requires_grad_(False)
                mug.ssm_full_bwd.conv.requires_grad_(False)
        else:
            self.route_dino = official.mug_sam.prior_generate.route_proj
            self.route_vmamba = official.mug_cnn.prior_generate.route_proj

    def forward(self, features, return_regions=False):
        assert len(features) == 8
        pyramid, regions = [], None
        for i, side in enumerate((64, 32, 16, 8)):
            a = F.interpolate(self.dino_adapter[i](features[i]), (side, side), mode='bilinear', align_corners=False)
            b = F.interpolate(self.mamba_adapter[i](features[i + 4]), (side, side), mode='bilinear', align_corners=False)
            if i == 2:
                if self.mode == 'hiram':
                    a, b, regions = batched_hiram(self.hiram, a, b, self.training)
                else:
                    regions = [F.log_softmax(self.route_dino(a.permute(0, 2, 3, 1)), -1).permute(0, 3, 1, 2),
                               F.log_softmax(self.route_vmamba(b.permute(0, 2, 3, 1)), -1).permute(0, 3, 1, 2)]
            pyramid.append(self.fusion[i](torch.cat((a, b), dim=1)))
        logits = self.decoder(pyramid)
        return (logits, regions) if return_regions else logits


@torch.no_grad()
def region_targets(mask):
    """Official width=2 morphology and bilinear region-label resize."""
    # Historical ISIC masks can contain gray levels; region morphology needs
    # binary ground truth, using the project's evaluation threshold of 0.5.
    # The original grayscale targets remain unchanged for the segmentation loss.
    fg = (mask >= .5).float()
    kernel = torch.ones((1, 1, 5, 5), device=fg.device)
    counts = F.conv2d(fg, kernel, padding=2)
    boundary = ((counts > 0) ^ (counts == 25)).float()
    interior = fg * (1 - boundary)
    return [F.interpolate(boundary, (16, 16), mode='bilinear', align_corners=False),
            F.interpolate(interior, (16, 16), mode='bilinear', align_corners=False)]


def region_loss_numerators(regions, targets):
    """Return the two additive numerators; normalize by whole effective batch."""
    boundary, interior = targets
    zero = regions[0].new_zeros((), dtype=torch.float32)
    bd, inside = zero, zero
    for pred in regions:
        bd = bd + F.kl_div(pred[:, 1:2].float(), boundary.float(), reduction='sum')
        inside = inside + F.kl_div(pred[:, 2:3].float(), interior.float(), reduction='sum')
    return bd, inside
