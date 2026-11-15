"""Native MambaIR CAB; preserve MambaVision SSM/MLP and original pretrained weights."""
from pathlib import Path
import ast,hashlib,importlib
import torch
from torch import nn
from torch.nn import functional as F
from stage45_upernet_decoder.author_modules import NativeUPerProbe
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent.parent.parent/'research/topconf_mamba_followup_20261112/author_sources/csguoh__MambaIR'
AUTHOR=SOURCE/'basicsr/archs/mambair_arch.py'
def native_classes():
    nodes=[n for n in ast.parse(AUTHOR.read_text()).body if isinstance(n,ast.ClassDef)
           and n.name in ['ChannelAttention','CAB']]
    assert [n.name for n in nodes]==['ChannelAttention','CAB']
    scope=dict(__name__='stage50_native_mambair',torch=torch,nn=nn)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(AUTHOR),'exec'),scope)
    return scope['CAB'],{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes}

class FixedHalf(nn.Module):
    def forward(self,x):return x*.5

class SpatialAdapter(nn.Module):
    def __init__(self,arm,native_window_functions):
        super().__init__();assert arm in ['cab','conv_control']
        self.arm=arm;self.height=self.width=16;self.window=14;self.padded=28
        self.partition,self.reverse=native_window_functions
        CAB,self.ast_hashes=native_classes()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(142)
            self.cab=CAB(384,is_light_sr=False,compress_ratio=3,squeeze_factor=30)
            if arm=='conv_control':self.cab.cab[3]=FixedHalf()
            self.scale=nn.Parameter(torch.zeros(384))
    def forward(self,tokens):
        assert tokens.ndim==3 and tokens.shape[1:]==(196,384) and tokens.shape[0]%4==0
        dense=self.reverse(tokens,self.window,self.padded,self.padded)
        real=dense[:,:,:self.height,:self.width].contiguous()
        local=self.cab(real)*self.scale[None,:,None,None]
        padded=F.pad(local,(0,self.padded-self.width,0,self.padded-self.height))
        return self.partition(padded,self.window).to(tokens.dtype)

def install_adapter(encoder,arm):
    stage=encoder.model.levels[2];block=stage.blocks[3]
    assert stage.window_size==14 and type(block.mixer).__name__=='MambaVisionMixer'
    assert type(stage.blocks[4].mixer).__name__=='Attention'
    module=importlib.import_module(type(block).__module__)
    adapter=SpatialAdapter(arm,(module.window_partition,module.window_reverse))
    block.mlp.add_module('spatial_adapter',adapter)
    def add_local(module,inputs,output):
        delta=module.spatial_adapter(inputs[0])
        assert delta.shape==output.shape
        return output+delta.to(output.dtype)
    block.mlp.register_forward_hook(add_local)
    return dict(path='model.levels.2.blocks.3.mlp.spatial_adapter',
        location='Parallel to original MLP, using norm2 after the last stage3 Mamba mixer, before stage3 Attention blocks',
        channels=384,real_spatial_size=[16,16],window=14,padded_size=[28,28],
        native_CAB=arm=='cab',ast_hashes=adapter.ast_hashes)
