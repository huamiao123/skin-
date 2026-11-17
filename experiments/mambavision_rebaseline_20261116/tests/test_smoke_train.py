import copy,sys,unittest
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smoke_train import accumulate

class TinySeg(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Conv2d(1,1,1)
    def from_cache(self,f1,f2,mid,output_size):return {'main':self.proj(f1)}

class AccumulationTest(unittest.TestCase):
    def test_short_microbatch_is_weighted_by_sample_count(self):
        torch.manual_seed(17); a=TinySeg();b=copy.deepcopy(a)
        data=torch.randn(5,1,2,2);target=torch.rand(5,1,2,2);cache={'features':[data,data],'mid':data,'target':target}
        oa=torch.optim.SGD(a.parameters(),lr=.1);ob=torch.optim.SGD(b.parameters(),lr=.1);ids=torch.arange(5)
        loss_a,_=accumulate(a,oa,cache,ids,5,device_type='cpu',use_amp=False,clip=None)
        loss_b,_=accumulate(b,ob,cache,ids,2,device_type='cpu',use_amp=False,clip=None)
        self.assertAlmostEqual(loss_a,loss_b,places=6)
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.allclose(x,y,atol=1e-7,rtol=1e-6))

if __name__=='__main__':unittest.main(verbosity=2)
