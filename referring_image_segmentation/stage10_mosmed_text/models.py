"""Three paired baselines: common frozen DINO features, adapter and Swin decoder."""
from common import *
import math, torch
from torch import nn
from torch.nn import functional as F
from stage3_confirmation.models import Probe as VisualProbe
from stage1_visual.models import FrozenEncoder, tensor_hash

class Probe(nn.Module):
    def __init__(self, name, seed):
        super().__init__(); assert name in NAMES; self.name = name
        base = VisualProbe('swin', seed, (384,)*4)
        self.adapter, self.decoder = base.adapter, base.decoder
        if name != NAMES[0]:
            del self.decoder.readout
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.text_projection = nn.Linear(512,64)
                self.pixel_projection = nn.Conv2d(64,64,1)
            self.log_tau = nn.Parameter(torch.tensor(math.log(10.)))
            self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, features, text=None):
        pyramid = [F.interpolate(a(f), (side,side), mode='bilinear', align_corners=False)
                   for a,f,side in zip(self.adapter, features, (64,32,16,8))]
        x = self.decoder.blocks[3](pyramid[3])
        for i in (2,1,0):
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.decoder.blocks[i](x+pyramid[i])
        if self.name == NAMES[0]: return self.decoder.readout(x)
        assert text is not None
        with torch.autocast('cuda',enabled=False):
            v = F.normalize(self.pixel_projection(x.float()), dim=1, eps=1e-6)
            u = F.normalize(self.text_projection(text.float()), dim=1, eps=1e-6)
            return self.log_tau.clamp(max=math.log(100.)).exp()*(v*u[:,:,None,None]).sum(1,keepdim=True)+self.bias

def build_probes(seed): return {n:Probe(n,seed) for n in NAMES}

def masked_loss(logits, target, valid):
    z = F.interpolate(logits.float(),(256,256),mode='bilinear',align_corners=False)
    dims=(1,2,3); count=valid.sum(dims).clamp_min(1)
    bce=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*valid).sum(dims)/count
    p=z.sigmoid()*valid; t=target*valid
    dice=1-(2*(p*t).sum(dims)+1)/(p.sum(dims)+t.sum(dims)+1)
    return (bce+dice).mean()
