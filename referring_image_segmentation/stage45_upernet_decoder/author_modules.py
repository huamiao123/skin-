"""Pinned native OpenMMLab UPerHead forward, with real MMCV/MMEngine components."""
from pathlib import Path
import ast,hashlib,json,functools,warnings,inspect
from abc import ABCMeta,abstractmethod
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.registry import Registry
from mmengine.fileio import load
ROOT=Path(__file__).parent
RESEARCH=ROOT.parent.parent.parent/'research/upernet_author_20261112'
AUTHOR=RESEARCH/'official_code/open-mmlab__mmsegmentation/mmseg/models'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(path,names,scope):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
    if names is not None:assert {n.name for n in nodes}==set(names)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    exec(compile(module,str(path),'exec'),scope)
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}
def native_scope():
    registry=Registry('isolated_uper_head')
    def no_sampler(*args,**kwargs):raise RuntimeError('This main-head-only experiment configures sampler=None')
    scope=dict(__name__='isolated_uper_source',torch=torch,nn=nn,F=F,Tensor=Tensor,np=np,
        warnings=warnings,functools=functools,load=load,ABCMeta=ABCMeta,abstractmethod=abstractmethod,
        BaseModule=BaseModule,ConvModule=ConvModule,MODELS=registry,build_loss=registry.build,build_pixel_sampler=no_sampler)
    provenance={}
    for name,names in [
        ('losses/utils.py',None),('losses/cross_entropy_loss.py',None),
        ('utils/wrappers.py',{'resize'}),('decode_heads/decode_head.py',{'BaseDecodeHead'}),
        ('decode_heads/psp_head.py',{'PPM'}),('decode_heads/uper_head.py',{'UPerHead'})]:
        path=AUTHOR/name;provenance[name]=dict(sha256=digest(path),nodes=extract(path,names,scope))
    return scope,provenance
class NativeUPerProbe(nn.Module):
    def __init__(self,width,seed=42):
        super().__init__();assert width in (64,128)
        scope,self.provenance=native_scope()
        self.width=width
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.decoder=scope['UPerHead'](
                in_channels=[96,192,384,768],in_index=[0,1,2,3],channels=width,
                pool_scales=(1,2,3,6),dropout_ratio=.1,num_classes=2,out_channels=1,threshold=.5,
                norm_cfg=dict(type='SyncBN',requires_grad=True),align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0))
            self.decoder.init_weights()
    def forward(self,features):
        assert len(features)==4
        for x,c,h in zip(features,[96,192,384,768],[64,32,16,8]):
            assert x.shape[1:]==(c,h,h)
        out=self.decoder(features)
        return F.interpolate(out,size=(256,256),mode='bilinear',align_corners=False)
