"""9x9 local patch classifier with candidate-offset and normal information."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class PixelGroupNorm(nn.Module):
    """Normalize channel groups independently at each spatial position."""
    def __init__(self,groups,channels,eps=1e-5):
        super().__init__()
        if channels%groups:raise ValueError('Channels must divide groups')
        self.groups=groups;self.eps=eps
        self.weight=nn.Parameter(torch.ones(channels))
        self.bias=nn.Parameter(torch.zeros(channels))

    def forward(self,x):
        b,c,h,w=x.shape
        grouped=x.reshape(b,self.groups,c//self.groups,h,w)
        mean=grouped.mean(dim=2,keepdim=True)
        variance=grouped.var(dim=2,unbiased=False,keepdim=True)
        normalized=((grouped-mean)*torch.rsqrt(variance+self.eps)).reshape(b,c,h,w)
        return normalized*self.weight[None,:,None,None]+self.bias[None,:,None,None]

class StrongLocal(nn.Module):
    def __init__(self, width=64, normalization='pixel_group'):
        super().__init__()
        if normalization not in ('pixel_group','spatial_group'):
            raise ValueError('Unknown normalization')
        norm=lambda:PixelGroupNorm(8,width) if normalization=='pixel_group' else nn.GroupNorm(8,width)
        self.encoder=nn.Sequential(
            nn.Conv2d(37,width,3,padding=1),norm(),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),norm(),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),norm(),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),norm(),nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(width+5,width),nn.GELU(),nn.Linear(width,1))
        self.width=width;self.normalization=normalization

    def forward(self,features,rgb,probability,boundary,points,source_points,normals):
        # PixelGroupNorm adds no spatial communication; frozen CNN maps can contain global context.
        x=torch.cat((features,rgb,probability,boundary),dim=1)
        encoded=self.encoder(x)
        grid=torch.stack((points[...,0]*2/255-1,points[...,1]*2/255-1),dim=-1)
        patch=F.grid_sample(encoded.float(),grid,mode='bilinear',padding_mode='border',align_corners=True)
        patch=patch.permute(0,2,3,1)
        offset=((points-source_points[:,:,None,:])*normals[:,:,None,:]).sum(dim=-1,keepdim=True)/32.0
        extra=torch.cat((offset,normals[:,:,None,:].expand(-1,-1,points.shape[2],-1),
                         points/255.0),dim=-1)
        return self.classifier(torch.cat((patch,extra),dim=-1)).squeeze(-1)

def parameter_count(model):return sum(p.numel() for p in model.parameters())
