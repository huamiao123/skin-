import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetailBranch(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ProtectedScaleUncertaintyRefinement(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        scale_hidden=16,
        beta_init=0.05,
    ):
        super().__init__()

        self.branch_d1 = DetailBranch(encoder_channels, decoder_channels, dilation=1)
        self.branch_d2 = DetailBranch(encoder_channels, decoder_channels, dilation=2)
        self.branch_d3 = DetailBranch(encoder_channels, decoder_channels, dilation=3)

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, scale_hidden),
            nn.GELU(),
            nn.Linear(scale_hidden, 3)
        )

        beta_logit = math.log(beta_init / (1.0 - beta_init))
        self.beta_logit = nn.Parameter(torch.tensor(beta_logit, dtype=torch.float32))

    def forward(self, gab_feature, encoder_feature, mask_logits):
        if encoder_feature.shape[-2:] != gab_feature.shape[-2:]:
            encoder_feature = F.interpolate(
                encoder_feature,
                size=gab_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if mask_logits.shape[-2:] != gab_feature.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits,
                size=gab_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(mask_logits.detach())
        uncertainty = 4.0 * prob * (1.0 - prob)

        r1 = self.branch_d1(encoder_feature)
        r2 = self.branch_d2(encoder_feature)
        r3 = self.branch_d3(encoder_feature)

        area_ratio = prob.mean(dim=(2, 3))
        scale_logits = self.scale_mlp(area_ratio)
        scale_weights = torch.softmax(scale_logits, dim=-1)

        B = gab_feature.shape[0]
        w1 = scale_weights[:, 0].view(B, 1, 1, 1)
        w2 = scale_weights[:, 1].view(B, 1, 1, 1)
        w3 = scale_weights[:, 2].view(B, 1, 1, 1)

        detail = w1 * r1 + w2 * r2 + w3 * r3

        correction = uncertainty * detail

        beta = torch.sigmoid(self.beta_logit)
        out = gab_feature + beta * correction

        return {
            "feature": out,
            "uncertainty": uncertainty,
            "scale_weights": scale_weights,
            "beta": beta,
            "correction": correction,
        }
