"""Four training cases only. Strictly frozen parent model, stage-4 auxiliary fraction head."""
import hashlib,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from frozen_probe import ROOT,SEED,setup,model_for,train
from fit_bypass import resize,lower_half
from model import segmentation_loss

class OccupancySDI(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.adapter=nn.Sequential(nn.Conv2d(768,96,1,bias=False),nn.GroupNorm(8,96),nn.GELU(),nn.Conv2d(96,64,3,padding=1,bias=False),nn.GroupNorm(8,64),nn.GELU())
        self.project=nn.Conv2d(64,64,1)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
        self.readout=nn.Conv2d(64,1,1)
        self.capture=False;self.occupancy=None
    def forward(self,x):
        h=self.base.head
        lat=[c(v) for c,v in zip(h.lateral_convs,x)]+[h.psp_forward(x)]
        code=self.adapter(x[-1])
        self.occupancy=self.readout(code) if self.capture else None
        # Create a separate input for the product. Preserve the PPM/FPN tensor.
        factors=self.base.factors([*lat[:-1],lat[-1]+self.project(code)])
        delta=self.base.out(factors[0]*factors[1]*factors[2]*factors[3])
        delta=(delta.float()*(1+lower_half(delta).float())).to(delta.dtype)
        for i in range(3,0,-1):
            lat[i-1]=lat[i-1]+resize(lat[i],lat[i-1].shape[-2:])
            if i==2:lat[i-1]=lat[i-1]+delta
        pyramid=[c(v) for c,v in zip(h.fpn_convs,lat)]+[lat[-1]]
        return h.cls_seg(h.fpn_bottleneck(torch.cat([pyramid[0]]+[resize(v,pyramid[0].shape[-2:]) for v in pyramid[1:]],dim=1)))

def main():
    setup(SEED);data,_=train.reference.load_data();x,truth=data.batch('train',torch.arange(4,device='cuda'))
    obj={}
    for weight in [0.,.2]:
        setup(SEED);net=model_for('control').eval().requires_grad_(False)
        old_state=list(net.state_dict().values())
        def fingerprint():
            h=hashlib.sha256()
            for v in old_state:h.update(v.detach().contiguous().cpu().numpy().tobytes())
            return h.hexdigest()
        before=fingerprint()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):old=net(x)['main'].detach()
        mod=OccupancySDI(net.probe.decode_head).cuda();net.probe.decode_head=mod
        params=[p for p in mod.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.001)
        stats=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);mod.capture=(weight>0)
            with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x)['main']
            if step==0:assert torch.equal(old,z)
            main_loss=segmentation_loss(z,truth)
            if weight:
                fraction=F.avg_pool2d((truth>=.5).float(),32,32)
                assert mod.occupancy.shape==fraction.shape==(4,1,8,8)
                loss=main_loss+weight*F.binary_cross_entropy_with_logits(mod.occupancy.float(),fraction)
            else:loss=main_loss
            loss.backward()
            def peak(p):return 0. if p.grad is None else float(p.grad.abs().max())
            stats.append(dict(adapter=peak(mod.adapter[0].weight),project=peak(mod.project.weight),readout=peak(mod.readout.weight)))
            assert stats[-1]['project']>0 and all(p.grad is None for p in net.encoder.parameters())
            optimizer.step()
        assert stats[1]['adapter']>0
        assert (stats[0]['adapter']>0)==bool(weight) and (stats[0]['readout']>0)==bool(weight)
        assert fingerprint()==before
        mod.capture=False
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):result=net(x)['main']
        assert mod.occupancy is None and torch.isfinite(result).all()
        obj[str(weight)]=dict(train_examples=4,steps=2,initial_logit_max_difference=0,old_parameters_and_buffers_unchanged=True,original_state_hash=before,extra_parameters=sum(p.numel() for p in params),inference_branch_parameters=sum(p.numel() for s in [mod.adapter,mod.project] for p in s.parameters()),gradients=stats)
    train.atomic_json(ROOT/'occupancy_sdi_smoke.json',obj);print(obj)
if __name__=='__main__':main()
