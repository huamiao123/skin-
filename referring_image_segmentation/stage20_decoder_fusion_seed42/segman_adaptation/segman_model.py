"""Research-only MambaVision + SegMAN decoder adaptation.

The official SegMAN decoder's AST is loaded unchanged from a pinned local
snapshot, except that its decorator and external framework dependencies are
replaced by small compatibility interfaces. Its oflex scan is substituted
with the independently validated standard Mamba selective scan, with FP32
output matching the oflex=True path. This is NOT the official implementation
or an official numerical reproduction. See README.md before reuse.
"""
from pathlib import Path
import ast
import importlib.util
import math
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import *
from torch import nn
from torch.nn import functional as F
from einops import repeat
from timm.layers import DropPath, to_2tuple
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

SRC = ROOT/'references/segman_decoder_source.py'
SRC_SHA256 = '67bea12cb22409344e59908c2ee4b916adcad76be6fb7b2ea43fce988e5b1516'
CSM = ROOT/'references/segman_csm_triton_source.py'

spec=importlib.util.spec_from_file_location('segman_upstream_cross_scan', CSM)
csm=importlib.util.module_from_spec(spec)
spec.loader.exec_module(csm)
CrossScanTriton, CrossMergeTriton=csm.CrossScanTriton,csm.CrossMergeTriton


class SelectiveScanOflex:
    @staticmethod
    def apply(u,delta,A,B,C,D,delta_bias,delta_softplus,nrows,backnrows,oflex):
        assert oflex and delta_softplus and nrows is None and backnrows is None
        # The upstream oflex=True path returns FP32. Its BF16 input is already
        # quantized, so promote that input without silently changing the SSM.
        return selective_scan_fn(u.float().contiguous(),delta.float().contiguous(),A.float(),
            B.float().contiguous(),C.float().contiguous(),D.float(),
            delta_bias=delta_bias.float(),delta_softplus=delta_softplus)


class ConvModule(nn.Module):
    """Only the convolution / SyncBN / ReLU configuration used by this head."""
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,
                 conv_cfg=None,norm_cfg=None,act_cfg=None,**kwargs):
        super().__init__()
        if conv_cfg is not None or kwargs:
            raise ValueError((conv_cfg,kwargs))
        assert norm_cfg and norm_cfg.get('type')=='SyncBN'
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
        self.norm=nn.SyncBatchNorm(out_channels)
        self.activate=nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        return self.activate(self.norm(self.conv(x)))


class BaseDecodeHead(nn.Module):
    """Minimum interface needed by SegMANDecoder; one binary output channel."""
    def __init__(self,in_channels,channels,num_classes,in_index,dropout_ratio=.1,
                 norm_cfg=None,act_cfg=None,conv_cfg=None,align_corners=False,
                 input_transform=None,**kwargs):
        super().__init__()
        assert input_transform=='multiple_select' and num_classes==1
        self.in_channels=in_channels
        self.in_index=in_index
        self.channels=channels
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg or dict(type='ReLU')
        self.conv_cfg=conv_cfg
        self.align_corners=align_corners
        self.conv_seg=nn.Conv2d(channels,1,1)
        self.dropout=nn.Dropout2d(dropout_ratio) if dropout_ratio else nn.Identity()
        nn.init.normal_(self.conv_seg.weight,std=.01)
        nn.init.zeros_(self.conv_seg.bias)

    def _transform_inputs(self,inputs):
        return [inputs[i] for i in self.in_index]

    def cls_seg(self,feature):
        return self.conv_seg(self.dropout(feature))


def resize(x,size,mode,align_corners):
    return F.interpolate(x,size=size,mode=mode,align_corners=align_corners)


def load_head_class():
    assert sha(SRC)==SRC_SHA256
    wanted={'LayerScale','LayerNorm2d','VSSM','FFN','VSSMBlock','MLP','SegMANDecoder'}
    tree=ast.parse(SRC.read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in wanted]
    assert {n.name for n in classes}==wanted
    for node in classes:
        node.decorator_list=[]
    fragment=ast.Module(body=classes,type_ignores=[])
    ast.fix_missing_locations(fragment)
    namespace=globals().copy()
    exec(compile(fragment,str(SRC),'exec'),namespace)
    return namespace['SegMANDecoder']


class MambaVisionSegMAN(nn.Module):
    """Official SegMAN-small head configuration adapted to MambaVision features."""
    def __init__(self):
        super().__init__()
        head=load_head_class()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.decoder=head(image_size=256,in_channels=list(CHANNELS),in_index=[0,1,2,3],
                channels=144,feat_proj_dim=288,num_classes=1,dropout_ratio=.1,
                norm_cfg=dict(type='SyncBN',requires_grad=True),align_corners=False)

    def forward(self,features):
        assert len(features)==4
        return F.interpolate(self.decoder(features),size=(256,256),mode='bilinear',align_corners=False)
