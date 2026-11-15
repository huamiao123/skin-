"""Channel and scale selection on Stage29's deep-context residual path.
ECA-like reference follows global pooling + 1D channel convolution.
The 2*sigmoid identity initialization is an explicit adaptation, not a verbatim ECA reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from stage29_context_fusion.fusion import FusionDecoder as PriorDecoder

class ChannelResidualFusion(nn.Module):
    def __init__(self, original, mode, seed):
        super().__init__()
        assert mode in ('eca_channel','cross_channel','cross_channel_scale')
        self.mode=mode
        self.disable_channel=False
        self.disable_scale=False
        for name in ('reduce','dw3','dw5','norm','gate','out'):
            setattr(self,name,getattr(original,name))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            if mode=='eca_channel':
                self.eca=nn.Conv1d(1,1,3,padding=1,bias=False)
                nn.init.zeros_(self.eca.weight)
            else:
                self.descriptor=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,16),nn.GELU())
                self.channel_head=nn.Linear(16,64)
                nn.init.zeros_(self.channel_head.weight)
                nn.init.zeros_(self.channel_head.bias)
                if mode=='cross_channel_scale':
                    self.scale_head=nn.Linear(16,64)
                    nn.init.zeros_(self.scale_head.weight)
                    nn.init.zeros_(self.scale_head.bias)

    def forward(self, context, skip):
        descriptor=None
        if self.mode=='eca_channel':
            weights=2*self.eca(context.mean((2,3)).unsqueeze(1)).sigmoid()
            weights=weights.transpose(1,2).unsqueeze(-1)
        else:
            descriptor=self.descriptor(torch.cat([context.mean((2,3)),skip.mean((2,3))],dim=1))
            weights=(2*self.channel_head(descriptor).sigmoid())[:,:,None,None]
        selected=context if self.disable_channel else context*weights
        h=self.reduce(torch.cat([selected,skip],dim=1))
        a,b=self.dw3(h),self.dw5(h)
        if self.mode=='cross_channel_scale' and not self.disable_scale:
            alpha=self.scale_head(descriptor).reshape(-1,2,32,1,1).softmax(dim=1)
            mixed=alpha[:,0]*a+alpha[:,1]*b
        else:
            mixed=(a+b)*.5
        h=F.gelu(self.norm(mixed))
        return self.out(h*(2*self.gate(h).sigmoid()))

class ChannelDecoder(PriorDecoder):
    def __init__(self, original, mode):
        super().__init__(original,'anchor_fusion')
        self.fusion=nn.ModuleList([ChannelResidualFusion(block,mode,300042+i)
                                  for i,block in enumerate(self.fusion)])
