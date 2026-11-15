"""Move existing channel modulation from residual-context input to main decoder context.
All parameter names and counts match Stage30 cross_channel_scale.
No new attention parameters, auxiliary loss, or backbone change.
"""
from torch import nn
import torch
from torch.nn import functional as F

class MainRouteDecoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.blocks=original.blocks
        self.readout=original.readout
        self.fusion=original.fusion
        self.disable_fusion=False
        self.route='main_context_channel'

    def forward(self,pyramid):
        anchor=self.blocks[3](pyramid[3])
        x=anchor
        for i in (2,1,0):
            x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            skip=pyramid[i]
            context=F.interpolate(anchor,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            block=self.fusion[i]
            descriptor=block.descriptor(torch.cat([context.mean((2,3)),skip.mean((2,3))],dim=1))
            weights=(2*block.channel_head(descriptor).sigmoid())[:,:,None,None]
            # Sole experimental change: channel gains act on the main upsampled context.
            selected=x if block.disable_channel else x*weights
            merged=selected+skip
            if not self.disable_fusion:
                h=block.reduce(torch.cat([context,skip],dim=1))
                a,b=block.dw3(h),block.dw5(h)
                if block.disable_scale:
                    mixed=(a+b)*.5
                else:
                    alpha=block.scale_head(descriptor).reshape(-1,2,32,1,1).softmax(dim=1)
                    mixed=alpha[:,0]*a+alpha[:,1]*b
                h=F.gelu(block.norm(mixed))
                merged=merged+block.out(h*(2*block.gate(h).sigmoid()))
            x=self.blocks[i](merged)
        return F.interpolate(self.readout(x),size=(256,256),mode='bilinear',align_corners=False)
