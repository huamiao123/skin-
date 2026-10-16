"""Stage-1 data and augmentations with an explicit run seed."""
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from stage1_visual.data import ImageData as BaseData


class ImageData(BaseData):
    def __init__(self, seed):
        self.seed=seed
        super().__init__()

    def geometry(self, epoch):
        g=torch.Generator().manual_seed(self.seed+epoch*100003)
        n=len(self.ids['train'])
        order=torch.randperm(n,generator=g)
        h=torch.where(torch.rand(n,generator=g)<.5,-1.,1.)
        v=torch.where(torch.rand(n,generator=g)<.5,-1.,1.)
        rotate=torch.rand(n,generator=g)<.5
        angle=torch.rand(n,generator=g)*(2*torch.pi)*rotate
        matrix=torch.zeros(n,2,3)
        matrix[:,0,0]=angle.cos()*h
        matrix[:,0,1]=-angle.sin()*h
        matrix[:,1,0]=angle.sin()*v
        matrix[:,1,1]=angle.cos()*v
        return order.cuda(),matrix.cuda()
