"""9x9 local patch classifier with candidate-offset and normal information."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class StrongLocal(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(37,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(width+5,width),nn.GELU(),nn.Linear(width,1))
        self.width=width

    def forward(self,features,rgb,probability,boundary,points,source_points,normals):
        # The learned convolution sees only a 9x9 window in the *frozen CNN maps*.
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
