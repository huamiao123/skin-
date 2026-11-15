"""Research wiring only. Author module algorithms and production files are unchanged."""
from pathlib import Path
import ast,copy,importlib.util,math,sys
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parent
PROJECT=R.parent.parent/'source/referring_image_segmentation'
sys.path.insert(0,str(PROJECT))
from stage2_decoder.models import SwinPixelDecoder
OLD=R.parent/'module_design_20261111/official_code'
NEW=R/'official_code'
def _import(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
def _mfms_class():
 # Isolate exact author AST nodes to avoid unrelated ResNet/backbone imports.
 # No node is edited; source hash is recorded in compatibility_results.json.
 path=NEW/'Inha-CVAI__MADGNet/IS2D_models/mfmsnet.py'
 wanted={'get_freq_indices','MultiFrequencyChannelAttention','MFMSAttentionBlock'}
 tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
 assert {n.name for n in nodes}==wanted
 scope=dict(torch=torch,nn=nn,F=F,math=math)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
 return scope['MFMSAttentionBlock']
class AuthorModuleDecoder(nn.Module):
 """Four 64-channel feature maps in shallow-to-deep order -> binary logits.
    This class only routes tensors through existing implementations.
    It is a research prototype, not a claim to reproduce full paper training.
    No Stage29/30 custom attention is included.
    """
 def __init__(self,variant,baseline_decoder,seed=42):
  super().__init__()
  if variant not in {'emcad64','swin_cga','swin_mfms'}:raise ValueError(variant)
  self.variant=variant
  self.baseline=copy.deepcopy(baseline_decoder) if variant!='emcad64' else None
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(seed)
   if variant=='emcad64':
    official=_import(OLD/'SLDGroup__EMCAD/lib/decoders.py','original_emcad_for_wiring')
    self.author_decoder=official.EMCAD(channels=[64,64,64,64])
    self.readout=copy.deepcopy(baseline_decoder.readout)
   elif variant=='swin_cga':
    official=_import(NEW/'maklachur__MambaLiteUNet/models/MambaLiteUNet.py','original_cga_for_wiring')
    self.author_gates=nn.ModuleDict({str(i):official.CrossGatedAttentionGate(64,64,32) for i in (2,1,0)})
   else:
    cls=_mfms_class()
    self.author_blocks=nn.ModuleDict({str(i):cls(64) for i in (2,1,0)})
 def forward(self,pyramid):
  if len(pyramid)!=4:raise ValueError('Expected four feature maps')
  for p,h in zip(pyramid,[64,32,16,8]):
   if p.shape[1:]!=(64,h,h):raise ValueError('Expected 64-channel pyramid at 64,32,16,8')
  if self.variant=='emcad64':
   outputs=self.author_decoder(pyramid[3],[pyramid[2],pyramid[1],pyramid[0]])
   return F.interpolate(self.readout(outputs[-1]),size=(256,256),mode='bilinear',align_corners=False)
  if self.variant=='swin_mfms':
   enhanced=[self.author_blocks[str(i)](pyramid[i]) if i<3 else pyramid[i] for i in range(4)]
   return self.baseline(enhanced)
  x=self.baseline.blocks[3](pyramid[3])
  for i in (2,1,0):
   x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)
   x=self.baseline.blocks[i](x+self.author_gates[str(i)](x,pyramid[i]))
  return F.interpolate(self.baseline.readout(x),size=(256,256),mode='bilinear',align_corners=False)

