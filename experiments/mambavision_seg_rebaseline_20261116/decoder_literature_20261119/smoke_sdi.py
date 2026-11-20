"""Check the original U-Net v2 SDI against a matched arithmetic-mean control in UPer64."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/featurize/work/mambavision_seg_rebaseline_20261116/uper64_feature_audit_20261119')
from train_feature import build, optimizer, reference
from model import PROJECT, segmentation_loss

def author_class():
    path=ROOT/'verified_sources/sdi.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SDI')
    scope=dict(torch=torch,nn=nn,F=F)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['SDI']

class FusionUPer(nn.Module):
    def __init__(self,head,mode):
        super().__init__()
        self.head=head
        self.mode=mode
        self.sdi=author_class()(head.channels)
        self.out=nn.Conv2d(head.channels,head.channels,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,features):
        h=self.head
        lateral=[c(x) for c,x in zip(h.lateral_convs,features)]
        lateral.append(h.psp_forward(features))
        ref=lateral[1]
        if self.mode=='product':
            joint=self.sdi(lateral,ref)
        else:
            pieces=[]
            for conv,x in zip(self.sdi.convs,lateral):
                if x.shape[-1]>ref.shape[-1]:
                    x=F.adaptive_avg_pool2d(x,ref.shape[-2:])
                elif x.shape[-1]<ref.shape[-1]:
                    x=F.interpolate(x,size=ref.shape[-2:],mode='bilinear',align_corners=True)
                pieces.append(conv(x))
            joint=torch.stack(pieces).mean(0)
        def up(x,other):
            return F.interpolate(x,size=other.shape[-2:],mode='bilinear',align_corners=h.align_corners)
        for i in range(3,0,-1):
            lateral[i-1]=lateral[i-1]+up(lateral[i],lateral[i-1])
            if i==2:lateral[i-1]=lateral[i-1]+self.out(joint)
        pyramid=[c(x) for c,x in zip(h.fpn_convs,lateral)]+[lateral[-1]]
        concat=torch.cat([pyramid[0]]+[up(x,pyramid[0]) for x in pyramid[1:]],dim=1)
        return h.cls_seg(h.fpn_bottleneck(concat))

def inputs():
    root=PROJECT/'data/isic2018/train'
    xs=[];ys=[]
    for sid in ('0','1','10','100'):
        with Image.open(root/'images'/(sid+'.png')) as im,Image.open(root/'masks'/(sid+'.png')) as gt:
            xs.append(np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)))
            ys.append(np.array(gt.convert('L').resize((256,256),Image.Resampling.NEAREST)))
    x=torch.from_numpy(np.stack(xs)).permute(0,3,1,2).cuda().float()/255
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    return x,torch.from_numpy(np.stack(ys)).cuda()[:,None].float()/255.

def main():
    reference.setup();x,y=inputs();model=build('add');model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        f1,f2,mid=model.prefix(x)
        features=[f1,f2,*model.suffix(mid)]
        native=model(x)['main'].float().clone()
    hist={}
    for mode in ['mean','product']:
        reference.setup()
        net=build('add')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261119)
            net.probe.decode_head=FusionUPer(net.probe.decode_head,mode).cuda()
        head=net.probe.decode_head;net.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            same=(native-net(x)['main'].float()).abs().max().item()
        assert same==0,same
        init=hashlib.sha256(b''.join(w.detach().cpu().numpy().tobytes() for part in [head.sdi,head.out] for w in part.state_dict().values())).hexdigest()
        opt=optimizer(net);net.train();grads=[]
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=net(x)['main']
            loss=segmentation_loss(z,y);loss.backward()
            grad=[float(c.weight.grad.abs().max()) for c in head.sdi.convs]
            out_g=float(head.out.weight.grad.abs().max());assert np.isfinite([*grad,out_g]).all() and out_g>0
            for group in opt.param_groups:nn.utils.clip_grad_norm_(group['params'],1.,error_if_nonfinite=True)
            opt.step()
            grads.append(dict(loss=float(loss),four_weight_grad_max=grad,out_weight_grad_max=out_g))
        assert min(grads[1]['four_weight_grad_max'])>0
        hist[mode]=dict(extra_params=sum(p.numel() for m in [head.sdi,head.out] for p in m.parameters()),initial_logit_max_diff=same,initial_branch_sha256=init,steps=grads)
        del net,opt
    assert hist['mean']['initial_branch_sha256']==hist['product']['initial_branch_sha256']
    record=dict(smoke_only=True,validation_or_test_read=False,batch=4,inputs=[list(f.shape) for f in features],source_sha256=hashlib.sha256((ROOT/'verified_sources/sdi.py').read_bytes()).hexdigest(),arms=hist)
    (ROOT/'sdi_smoke.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
if __name__=='__main__':main()
