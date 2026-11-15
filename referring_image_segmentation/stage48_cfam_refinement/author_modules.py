"""Stage48: pinned CENet CFAM on one UPer FPN output, with a parameter-matched convolution control."""
from pathlib import Path
from functools import partial
import ast,hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath,trunc_normal_tf_
from timm.models import named_apply
from stage45_upernet_decoder.author_modules import NativeUPerProbe
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent.parent.parent/'research/topconf_mamba_followup_20261112/author_sources/xmindflow__cenet'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_cfam():
    scope=dict(__name__='pinned_cenet_cfam',torch=torch,nn=nn,F=F,math=math,partial=partial,
               named_apply=named_apply,trunc_normal_tf_=trunc_normal_tf_,DropPath=DropPath)
    provenance={}
    for rel,names in [
        ('src/networks/cenet/modules/blocks.py',{'SepConvBN','_init_weights'}),
        ('src/networks/cenet/modules/nlb.py',{'Nonlocal'}),
        ('src/networks/cenet/modules/cfam.py',None)]:
        path=SOURCE/rel;tree=ast.parse(path.read_text())
        nodes=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and (names is None or x.name in names)]
        if names is not None:assert set(n.name for n in nodes)==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        provenance[rel]=dict(sha256=sha(path),nodes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in nodes})
    return scope['CFAModule'],provenance

class ConvControl(nn.Module):
    """Conventional two 3x3 conv residual block; 59 hidden channels match CFAM parameter count."""
    def __init__(self,channels=64,hidden=59):
        super().__init__()
        self.norm=nn.BatchNorm2d(channels)
        self.body=nn.Sequential(nn.Conv2d(channels,hidden,3,padding=1),nn.GELU(),
                                nn.Conv2d(hidden,channels,3,padding=1))
        self.layer_scale=nn.Parameter(torch.full((1,channels,1,1),1e-6))
    def forward(self,x):
        return x+self.layer_scale*self.body(self.norm(x))

class RefinedUPerProbe(NativeUPerProbe):
    def __init__(self,arm,seed=42):
        super().__init__(64,seed)
        self.arm=arm;self.enabled=True
        if arm=='original':return
        assert arm in ('cfam','conv_control')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(142)
            if arm=='cfam':
                cls,provenance=load_cfam()
                self.refiner=cls(embed_dims=64,ffn_ratio=4,drop_rate=0,drop_path_rate=0,
                                norm_type='BN',init_value=1e-6,mca_rates=[1,2,4])
                self.provenance['cenet']=provenance
            else:self.refiner=ConvControl()
        # fpn_convs[1] is the 1/8 feature AFTER top-down addition and native FPN conv.
        self._refiner_hook=self.decoder.fpn_convs[1].register_forward_hook(self._refine)
    def _refine(self,module,args,output):
        assert output.ndim==4 and output.shape[1:]==(64,32,32)
        if not self.enabled:return output
        # Both residual modules can promote BF16 to FP32. Restore the common head
        # interface dtype before interpolation; identity must preserve original math.
        refined=self.refiner(output)
        assert refined.shape==output.shape
        return refined.to(output.dtype)
