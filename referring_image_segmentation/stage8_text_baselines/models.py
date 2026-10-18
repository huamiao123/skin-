"""Frozen visual features with a common Swin feature decoder and text readout."""
from common import *
import math,torch
from torch import nn
from torch.nn import functional as F
from stage3_confirmation.models import Probe as OldProbe
from stage4_fusion.models import EncoderPair,FusionProbe,CHANNELS
from stage1_visual.models import tensor_hash

class TextPixelProbe(nn.Module):
    def __init__(self,name,seed=42):
        super().__init__();assert name in NAMES;self.name=name
        self.dual=name=='fusion_text'
        branch='vmamba' if name=='vmamba_text' else 'dinov3'
        old=FusionProbe('swin',seed) if self.dual else OldProbe('swin',seed,CHANNELS[branch])
        if self.dual:
            self.dino_adapter=old.dino_adapter;self.mamba_adapter=old.mamba_adapter;self.fusion=old.fusion
        else:self.adapter=old.adapter
        self.decoder=old.decoder;del self.decoder.readout
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.text_projection=nn.Linear(512,64)
            self.pixel_projection=nn.Conv2d(64,64,1)
        self.log_tau=nn.Parameter(torch.tensor(math.log(10.)))
        self.bias=nn.Parameter(torch.tensor(0.))

    def forward(self,features,text):
        pyramid=[]
        for i,side in enumerate((64,32,16,8)):
            if self.dual:
                d=F.interpolate(self.dino_adapter[i](features[i]),(side,side),mode='bilinear',align_corners=False)
                m=F.interpolate(self.mamba_adapter[i](features[i+4]),(side,side),mode='bilinear',align_corners=False)
                p=self.fusion[i](torch.cat((d,m),1))
            else:
                raw=features[i+4] if self.name=='vmamba_text' else features[i]
                p=F.interpolate(self.adapter[i](raw),(side,side),mode='bilinear',align_corners=False)
            pyramid.append(p)
        x=self.decoder.blocks[3](pyramid[3])
        for i in (2,1,0):
            x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            x=self.decoder.blocks[i](x+pyramid[i])
        with torch.autocast('cuda',enabled=False):
            v=F.normalize(self.pixel_projection(x.float()),dim=1,eps=1e-6)
            u=F.normalize(self.text_projection(text.float()),dim=1,eps=1e-6)
            return self.log_tau.clamp(max=math.log(100.)).exp()*(v*u[:,:,None,None]).sum(1,keepdim=True)+self.bias

def build_probes():return {n:TextPixelProbe(n) for n in NAMES}

def masked_loss(logits,target,valid):
    z=F.interpolate(logits.float(),(256,256),mode='bilinear',align_corners=False)
    dims=(1,2,3);count=valid.sum(dims).clamp_min(1)
    bce=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*valid).sum(dims)/count
    p=z.sigmoid()*valid;t=target*valid
    dice=1-(2*(p*t).sum(dims)+1)/(p.sum(dims)+t.sum(dims)+1)
    return (bce+dice).mean()
