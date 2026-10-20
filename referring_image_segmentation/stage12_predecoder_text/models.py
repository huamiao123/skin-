"""Text is used only before Swin. The final binary head receives visual features only."""
from common import *
import torch
from torch import nn
from torch.nn import functional as F
from stage3_confirmation.models import Probe as VisualProbe
from stage1_visual.models import FrozenEncoder,tensor_hash

class Probe(nn.Module):
    def __init__(self,name,seed=SEED):
        super().__init__();assert name in NAMES;self.name=name
        base=VisualProbe('swin',seed,(384,)*4)
        self.adapter,self.decoder=base.adapter,base.decoder
        # The original visual modules keep their exact seed42 initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+12000)
            self.word_norm=nn.LayerNorm(512)
            self.word_projection=nn.Linear(512,64)
            self.visual_norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(4)])
            self.cross_attention=nn.ModuleList([nn.MultiheadAttention(64,4,dropout=0.,batch_first=True) for _ in range(4)])

    def predecoder_features(self,features,text):
        assert text['valid'].dtype==torch.bool and text['valid'].any(1).all()
        words=self.word_projection(self.word_norm(text['tokens']))
        pyramid=[]
        for i,(adapter,f,side) in enumerate(zip(self.adapter,features,(64,32,16,8))):
            x=adapter(f);b,c,h,w=x.shape
            assert (h,w)==(16,16)
            visual=x.flatten(2).transpose(1,2)
            update,_=self.cross_attention[i](self.visual_norms[i](visual),words,words,
                key_padding_mask=~text['valid'],need_weights=False)
            visual=visual+RESIDUAL_SCALE*update
            x=visual.transpose(1,2).reshape(b,c,h,w)
            pyramid.append(F.interpolate(x,(side,side),mode='bilinear',align_corners=False))
        return pyramid

    def decode(self,pyramid):
        x=self.decoder.blocks[3](pyramid[3])
        for i in (2,1,0):
            x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            x=self.decoder.blocks[i](x+pyramid[i])
        # This ordinary head has no text input, dot product, temperature or text bias.
        return self.decoder.readout(x)

    def forward(self,features,text):return self.decode(self.predecoder_features(features,text))

def build_probes():return {n:Probe(n,SEED) for n in NAMES}

def masked_loss(logits,target,valid):
    z=F.interpolate(logits.float(),(256,256),mode='bilinear',align_corners=False)
    dims=(1,2,3);count=valid.sum(dims).clamp_min(1)
    bce=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*valid).sum(dims)/count
    p=z.sigmoid()*valid;t=target*valid
    dice=1-(2*(p*t).sum(dims)+1)/(p.sum(dims)+t.sum(dims)+1)
    return (bce+dice).mean()
