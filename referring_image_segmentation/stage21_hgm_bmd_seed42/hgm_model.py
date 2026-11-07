"""MambaVision + partial Swin + HGM BMD adaptation, not full HGM.

Uses pinned, locally inspected upstream class definitions. Original HGM
expects PVT skips and an already fused coarse map; here we supply the existing
MambaVision/Swin features with two explicit channel projections. HGM Mamba
layers run FP32 because the upstream autocast-disabled code rejects BF16.
"""
from common import *
import ast
import copy
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from stage3_confirmation.models import Probe, segmentation_loss, tensor_hash
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder

sys.path.insert(0,str(STAGE14/'runtime'))
from mamba_ssm import Mamba

REF=ROOT/'references'
SOURCES={
    'hgm_decoder_source.py':'8c783b14dcedd13d98c433213d2a8f33c8ea0aec5bb4fa8281914b2d73c7a8da',
    'hgm_fsem_source.py':'107b4fd749bf6443b1e55e603216bca43c2e97d9878d747a392d260e10b6367b',
    'hgm_psa_source.py':'367f99050964ef7ab15a2291dd9304d1330da56fc47d846ca2a7c6883cea8c0e',
}


def _extract(path,names,namespace):
    assert sha(path)==SOURCES[path.name]
    source=ast.parse(path.read_text())
    nodes=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names),path
    module=ast.Module(body=nodes,type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module,str(path),'exec'),namespace)


def load_bmd():
    namespace=globals().copy()
    _extract(REF/'hgm_psa_source.py',('kaiming_init','PSA_p'),namespace)
    _extract(REF/'hgm_fsem_source.py',('conv2d','BNPReLU','SELayer','FSEMV2'),namespace)
    _extract(REF/'hgm_decoder_source.py',('BiPixelMambaLayer','BiWindowMambaLayer','BMD'),namespace)
    return namespace['BMD']


class FP32Mamba(nn.Module):
    def __init__(self,block):
        super().__init__()
        self.block=block

    def forward(self,x):
        with torch.autocast('cuda',enabled=False):
            return self.block(x.float())


class MambaVisionHGMProbe(Probe):
    def __init__(self):
        super().__init__('swin',42,channels=CHANNELS)
        # The coarsest three Swin levels remain. HGM BMD replaces the finest
        # Swin level and head, so these baseline components are absent here.
        self.decoder.blocks[0]=nn.Identity()
        self.decoder.readout=nn.Identity()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1042)
            self.prediction_projection=nn.Sequential(nn.Conv2d(64,32,1,bias=False),nn.GroupNorm(8,32),nn.GELU())
            self.second_skip_projection=nn.Sequential(nn.Conv2d(64,128,1,bias=False),nn.GroupNorm(8,128),nn.GELU())
            self.hgm_bmd=load_bmd()()
        # These two batchnorm/PReLU modules are defined but never used by the
        # official FSEMV2.forward. Keep the tensors but do not train them.
        for stage in self.hgm_bmd.stages:
            stage.bn_relu_1.requires_grad_(False)
            stage.bn_relu_2.requires_grad_(False)
        self.hgm_bmd.mamba_layers=nn.ModuleList([FP32Mamba(m) for m in self.hgm_bmd.mamba_layers])
        self.hgm_bmd.w_mamba_layers=nn.ModuleList([FP32Mamba(m) for m in self.hgm_bmd.w_mamba_layers])

    def forward(self,features):
        assert len(features)==4
        pyramid=[a(f) for a,f in zip(self.adapter,features)]
        assert [f.shape[-1] for f in pyramid]==[64,32,16,8]
        x=self.decoder.blocks[3](pyramid[3])
        for i in (2,1):
            x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
            x=self.decoder.blocks[i](x+pyramid[i])
        coarse=self.prediction_projection(x)
        skip1=pyramid[0]
        skip2=self.second_skip_projection(pyramid[1])
        logits=self.hgm_bmd([skip1,skip2],coarse)
        assert list(logits.shape[1:])==[1,256,256]
        return logits
