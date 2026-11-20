"""Cache the unchanged batch inputs on this 32 GiB device, without changing training batches."""
import sys
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from branch_model import official
Base=official.MappedImages

class ResidentImages(Base):
    def __init__(self,parts):
        super().__init__(parts)
        # FP32 values are built by exactly the same on-device /255 as the streaming loader.
        # Fall back to byte storage on large sets if the measured training budget needs it.
        import os
        self.bytes_only=os.environ.get('MV_RESIDENT_BYTES','0')=='1'
        dtype=torch.uint8 if self.bytes_only else torch.float32
        for split in parts:
            indexes=self.indices[split]
            images=torch.empty((len(indexes),3,256,256),dtype=dtype,device='cuda')
            masks=torch.empty((len(indexes),1,256,256),dtype=dtype,device='cuda')
            for i in range(0,len(indexes),64):
                ix=indexes[i:i+64]
                x=torch.from_numpy(self.x[ix]).permute(0,3,1,2).contiguous().cuda()
                y=torch.from_numpy(self.y[ix]).unsqueeze(1).cuda()
                if not self.bytes_only:
                    x=x.float().div_(255.);y=y.float().div_(255.)
                images[i:i+len(ix)].copy_(x);masks[i:i+len(ix)].copy_(y)
            self.images[split]=images;self.masks[split]=masks

    def batch(self,split,indices,matrices=None):
        x=self.images[split][indices];y=self.masks[split][indices]
        if self.bytes_only:
            x=x.float().div_(255.);y=y.float().div_(255.)
        if matrices is not None:
            grid=F.affine_grid(matrices[indices],x.shape,align_corners=False)
            x=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            y=F.grid_sample(y,grid,mode='nearest',padding_mode='zeros',align_corners=False)
        return (x-self.mean)/self.std,y

@contextmanager
def loader(cls):
    prev=official.MappedImages
    official.MappedImages=cls
    try: yield
    finally:official.MappedImages=prev

