"""Capacity control and paired static/dynamic residual fusion; Swin fixed."""
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from stage4_fusion.models import EncoderPair, FusionProbe, CHANNELS
from stage3_confirmation.models import Probe as SingleProbe
from stage1_visual.models import tensor_hash, segmentation_loss

NAMES=('capacity_concat','static_residual','dynamic_gate')
SPEC=dict(decoder='stage2 SwinPixelDecoder',channels=64,resolutions=[64,32,16,8],
          initial_alpha=.1,phi='Conv2d(64,64,1,bias=True)',dynamic_gate='Conv2d(128,1,1,bias=True), FP32 logits/sigmoid',
          static_gate='one learned scalar logit per scale',adapter_initial_seed=42,
          capacity_control='same DINO features; independent adapter parameter storage; same seeded initial tensors')


class CapacityProbe(FusionProbe):
    def __init__(self,seed):
        super().__init__('swin',seed)
        self.mamba_adapter=SingleProbe('swin',seed,CHANNELS['dinov3']).adapter


class ResidualProbe(nn.Module):
    def __init__(self,seed,dynamic):
        super().__init__()
        dino=SingleProbe('swin',seed,CHANNELS['dinov3'])
        mamba=SingleProbe('swin',seed,CHANNELS['vmamba'])
        self.dino_adapter=dino.adapter;self.mamba_adapter=mamba.adapter;self.decoder=dino.decoder
        self.dynamic=dynamic;self.gate_mode='learned'
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.phi=nn.ModuleList([nn.Conv2d(64,64,1,bias=True) for _ in range(4)])
            if dynamic:
                self.gate=nn.ModuleList([nn.Conv2d(128,1,1,bias=True) for _ in range(4)])
                for layer in self.gate:
                    nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,math.log(.1/.9))
            else:
                self.beta=nn.ParameterList([nn.Parameter(torch.tensor(math.log(.1/.9))) for _ in range(4)])

    def forward(self,features,return_gates=False):
        assert len(features)==8
        pyramid=[];gates=[]
        for i,side in enumerate((64,32,16,8)):
            a=self.dino_adapter[i](features[i]);b=self.mamba_adapter[i](features[i+4])
            a=F.interpolate(a,(side,side),mode='bilinear',align_corners=False)
            b=F.interpolate(b,(side,side),mode='bilinear',align_corners=False)
            correction=self.phi[i](b)
            # FP32 gate logits make static/dynamic initial functions identical under BF16.
            with torch.autocast(device_type=a.device.type,enabled=False):
                alpha=(self.gate[i](torch.cat((a.float(),b.float()),dim=1)).sigmoid() if self.dynamic
                       else self.beta[i].sigmoid().expand(len(a),1,side,side))
            if self.gate_mode=='zero':alpha=torch.zeros_like(alpha)
            elif self.gate_mode=='spatial_mean':alpha=alpha.mean((-2,-1),keepdim=True).expand_as(alpha)
            elif self.gate_mode=='shift':alpha=torch.roll(alpha,(side//2,side//2),(-2,-1))
            else:assert self.gate_mode=='learned'
            gates.append(alpha)
            pyramid.append(a+alpha.to(correction.dtype)*correction)
        logits=self.decoder(pyramid)
        return (logits,gates) if return_gates else logits


def build_probes(seed):
    return dict(capacity_concat=CapacityProbe(seed),static_residual=ResidualProbe(seed,False),dynamic_gate=ResidualProbe(seed,True))


def select_features(name,features):
    if name=='capacity_concat':
        return features[:4]+features[:4]
    assert name in ('static_residual','dynamic_gate')
    return features
