"""Official segmenter, with a shared binary initialization and fixed 256px windows."""
import ast
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PREVIOUS=PROJECT/'uper512_ade_decoder_only_20261119'
sys.path.insert(0,str(PROJECT/'ham10000_window16_20g_20261119'))
import train_swin as recipe
from model import NativeHeads, objective
from train_stageB_color import rng_state,restore_rng
sys.path.append(str(PREVIOUS))
previous=runpy.run_path(str(PREVIOUS/'train.py'),run_name='decoder_only_archive')
MappedImages=previous['MappedImages']
from fetch_decoder import ID,sha,download

DATASETS=('ISIC2018','PH2','Waterloo','HAM10000')
SPLITS=json.loads((PREVIOUS/'queue_protocol.json').read_text())

def write(path,data):
    tmp=path.with_suffix(path.suffix+'.new')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.part')
    torch.save(data,tmp)
    tmp.replace(path)

def backbone_class():
    path=ROOT/'mamba_vision.py'
    tree=ast.parse(path.read_bytes())
    # Registry integrations are optional here. Preserve the NVIDIA model classes intact.
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('mmseg.registry','mmdet.registry'))]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MM_mamba_vision')
    assert {ast.unparse(x) for x in cls.decorator_list}=={'MODELS_MMDET.register_module()','MODELS_MMSEG.register_module()'}
    cls.decorator_list=[]
    space=dict(__name__='official_seg_arch',__file__=str(path))
    exec(compile(tree,str(path),'exec'),space)
    return space[cls.name]

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=backbone_class()(dim=96,in_dim=64,depths=(3,3,7,5),window_size=(8,8,16,8),
            mlp_ratio=4,num_heads=(2,4,8,16),out_indices=(0,1,2,3),norm_layer='ln2d',pretrained=None)
        self.probe=NativeHeads('C-512-aux',seed=42)
    def forward(self,x):
        return self.probe(self.backbone(x),x.shape[-2:])

def initialize():
    init=ROOT/'common_initial.pth'
    record=ROOT/'initialization.json'
    if record.exists():
        assert sha(init)==json.loads(record.read_text())['initial_sha256']
        return
    recipe.setup(42)
    net=Segmenter()
    full=download()
    assert sha(full)==ID
    state=torch.load(full,map_location='cpu',weights_only=False)['state_dict']
    target=net.state_dict()
    classifiers={f'{head}.conv_seg.{suffix}' for head in ('decode_head','auxiliary_head') for suffix in ('weight','bias')}
    used=set()
    for key in target:
        official=key.removeprefix('probe.')
        if official in classifiers:
            assert state[official].shape[0]==150 and target[key].shape[0]==1
            continue
        candidate=next((k for k in (official,official.replace('.norm.','.bn.'),official.replace('.bn.','.norm.')) if k in state and state[k].shape==target[key].shape),None)
        if candidate is None:raise ValueError(('missing',key))
        assert candidate not in used
        used.add(candidate)
        target[key]=state[candidate]
    assert set(state)-used==classifiers,sorted(set(state)-used-classifiers)
    assert all(f'backbone.outnorm{i}.{v}' in used for i in range(4) for v in ('weight','bias'))
    net.load_state_dict(target,strict=True)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in target.items())
    save(init,net.state_dict())
    rates=[getattr(getattr(b,'drop_path',None),'drop_prob',0.) for level in net.backbone.levels for b in level.blocks]
    write(record,dict(initial_sha256=sha(init),upstream_sha256=ID,upstream_revision='7860a506b2eb844eaaae676f08461ce8c3c26f43',
        source_sha256=sha(ROOT/'mamba_vision.py'),replaced=sorted(classifiers),verified_tensor_count=len(used),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),max_drop_path=max(rates),windows=[16,8],seed=42))
    print('INITIALIZATION VERIFIED',len(used),flush=True)

def load_model():
    recipe.setup(42)
    net=Segmenter()
    net.load_state_dict(torch.load(ROOT/'common_initial.pth',weights_only=True,map_location='cpu'),strict=True)
    assert all(p.requires_grad for p in net.parameters())
    return net.cuda()

def dataset(ds,parts):
    recipe.ROOT=PROJECT/'ham10000_window16_20g_20261119' if ds=='HAM10000' else PROJECT/'four_dataset_matched_20261119'/ds
    recipe.CACHE=Path('/tmp/ham10000_window16_20261119') if ds=='HAM10000' else Path('/tmp/mv_four_matched_20261119')/ds
    stamp=json.loads((recipe.CACHE/'ready.json').read_text())['split_sha256'] if ds=='HAM10000' else (recipe.CACHE/'ready').read_text().strip()
    assert stamp==sha(recipe.ROOT/'split.csv')==SPLITS[ds]['split_sha256']
    d=MappedImages(parts)
    assert all(len(d.ids[p])==SPLITS[ds]['counts'][p] for p in parts)
    return d
