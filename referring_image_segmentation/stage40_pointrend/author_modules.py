"""Pinned author PointRend core; framework/medical coarse-head adapters are explicit."""
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Dict
import ast,copy,hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.common.registry import Registry
from fvcore.nn import weight_init
from stage1_visual.models import segmentation_loss
ROOT=Path(__file__).resolve().parent
AUTHOR=ROOT.parent.parent.parent/'research/pointrend_author_20261112/official_code/projects/PointRend/point_rend'
COMMIT='247691a1c292654867deb35cabb27f18d75a1363'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(path,names,scope):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}

class SwinCoarseAdapter(nn.Module):
    def __init__(self,cfg,input_shape):
        super().__init__()
        self.baseline=cfg.COARSE_INSTANCE
    def layers(self,features):
        pyramid=[features[f'p{i}'] for i in range(4)]
        x=self.baseline.blocks[3](pyramid[3])
        for i in (2,1,0):
            x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            x=self.baseline.blocks[i](x+pyramid[i])
        logits=self.baseline.readout(x)
        return torch.cat([torch.zeros_like(logits),logits],dim=1)
    def losses(self,coarse,targets):
        binary=coarse[:,1:2]-coarse[:,0:1]
        logits=F.interpolate(binary,size=targets.shape[-2:],mode='bilinear',align_corners=False)
        return {'loss_sem_seg_coarse':segmentation_loss(logits,targets[:,None].float())}

def make_scope(random_policy=False):
    scope=dict(np=np,torch=torch,nn=nn,F=F,Dict=Dict,ShapeSpec=NS,cat=torch.cat,
        weight_init=weight_init,POINT_HEAD_REGISTRY=Registry('POINT_HEAD'),SEM_SEG_HEADS_REGISTRY=Registry('SEM_HEAD'))
    scope['SEM_SEG_HEADS_REGISTRY'].register(SwinCoarseAdapter)
    provenance={}
    for file,names in [
        ('point_features.py',{'point_sample','get_uncertain_point_coords_on_grid','get_uncertain_point_coords_with_randomness'}),
        ('point_head.py',{'StandardPointHead','build_point_head'}),
        ('semantic_seg.py',{'calculate_uncertainty','PointRendSemSegHead'})]:
        provenance[file]=dict(sha256=sha(AUTHOR/file),nodes=extract(AUTHOR/file,names,scope))
    if random_policy:
        def uniform_grid(uncertainty_map,num_points):
            n,_,h,w=uncertainty_map.shape
            # Fixed label-independent unique random positions; invariant to batch partition.
            gen=torch.Generator(device='cpu').manual_seed(39000+h*1000+w)
            indices=torch.randperm(h*w,generator=gen)[:min(h*w,num_points)].to(uncertainty_map.device)
            indices=indices[None].expand(n,-1)
            coords=torch.empty(n,indices.shape[1],2,device=uncertainty_map.device,dtype=torch.float32)
            coords[:,:,0]=(indices%w+.5)/w;coords[:,:,1]=(indices//w+.5)/h
            return indices,coords
        scope['get_uncertain_point_coords_on_grid']=uniform_grid
    return scope,provenance

class PointRendDecoder(nn.Module):
    def __init__(self,variant,baseline,seed=42):
        super().__init__();assert variant in ('point_uncertain','point_random')
        self.variant=variant
        scope,provenance=make_scope(variant=='point_random')
        ph=NS(NAME='StandardPointHead',NUM_CLASSES=2,FC_DIM=256,NUM_FC=3,
            CLS_AGNOSTIC_MASK=False,COARSE_PRED_EACH_LAYER=False,
            COARSE_SEM_SEG_HEAD_NAME='SwinCoarseAdapter',IN_FEATURES=['p0'],
            TRAIN_NUM_POINTS=1024,OVERSAMPLE_RATIO=3,
            IMPORTANCE_SAMPLE_RATIO=0. if variant=='point_random' else .75,
            SUBDIVISION_STEPS=2,SUBDIVISION_NUM_POINTS=1024)
        cfg=NS(MODEL=NS(SEM_SEG_HEAD=NS(NUM_CLASSES=2,IGNORE_VALUE=255),POINT_HEAD=ph),
               COARSE_INSTANCE=baseline)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.author_head=scope['PointRendSemSegHead'](cfg,{f'p{i}':NS(channels=64) for i in range(4)})
        self.provenance=provenance
    @property
    def baseline(self):return self.author_head.coarse_sem_seg_head.baseline
    def forward(self,pyramid,targets=None):
        features={f'p{i}':v for i,v in enumerate(pyramid)}
        logits,losses=self.author_head(features,targets[:,0].long() if targets is not None else None)
        if self.training:return losses
        return logits[:,1:2]-logits[:,0:1]
