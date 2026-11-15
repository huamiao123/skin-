"""Stage36: exact author upsampling blocks in a shared MambaVision-S + Swin decoder."""
from pathlib import Path
import ast,copy,importlib.util,math
import torch
from torch import nn
from torch.nn import functional as F
NEW=Path('/home/waas/mambavision_stage22/research/topconf_open_source_20261111/official_code')
UC=NEW/'McGregorWwww__UCTransNet/nets/UCTransNet.py'
MADG=NEW/'Inha-CVAI__MADGNet/IS2D_models/mfmsnet.py'
ARMS=('swin_control','uct_concat','uct_cca','madg_concat','madg_mfms')
def _import(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module
def author_nodes(path,wanted):
 tree=ast.parse(path.read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
 assert {n.name for n in nodes}==set(wanted)
 scope=dict(torch=torch,nn=nn,F=F,math=math)
 # Compile untouched original AST nodes, retaining author filenames/line numbers.
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
 return scope
class RectifiedSkip(nn.Module):
 """CCA ablation: keep its final ReLU, set channel scales to one."""
 def forward(self,g,x):
  return F.relu(x)
class AuthorModuleDecoder(nn.Module):
 def __init__(self,variant,baseline_decoder,seed=42):
  super().__init__()
  assert variant in ARMS
  self.variant=variant
  self.baseline=copy.deepcopy(baseline_decoder)
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(seed)
   if variant.startswith('uct_'):
    scope=author_nodes(UC,{'get_activation','_make_nConv','ConvBatchNorm','Flatten','CCA','UpBlock_attention'})
    cls=scope['UpBlock_attention']
    self.author_blocks=nn.ModuleDict({str(i):cls(128,64,2) for i in (2,1,0)})
    # Construct the full modules first, so paired surrounding conv initialization stays identical.
    if variant=='uct_concat':
     for block in self.author_blocks.values():block.coatt=RectifiedSkip()
   elif variant.startswith('madg_'):
    scope=author_nodes(MADG,{'get_freq_indices','MultiFrequencyChannelAttention','MFMSAttentionBlock','UpsampleBlock'})
    self.author_blocks=nn.ModuleDict({'0':scope['UpsampleBlock'](
      in_channels=64,out_channels=64,skip_connection_channels=64,
      scale_branches=2,frequency_branches=16,frequency_selection='top',
      block_repetition=1,min_channel=64,min_resolution=8)})
    if variant=='madg_concat':self.author_blocks['0'].attention_layer=nn.Identity()
 def forward(self,pyramid):
  assert len(pyramid)==4
  for p,h in zip(pyramid,[64,32,16,8]):
   if p.shape[1:]!=(64,h,h):raise ValueError('Expected 64-channel pyramid at 64,32,16,8')
  if self.variant=='swin_control':
   return self.baseline(pyramid)
  x=self.baseline.blocks[3](pyramid[3])
  for i in (2,1,0):
   if str(i) in self.author_blocks:
    x=self.author_blocks[str(i)](x,pyramid[i])
   else:
    x=F.interpolate(x,size=pyramid[i].shape[-2:],mode='bilinear',align_corners=False)+pyramid[i]
   x=self.baseline.blocks[i](x)
  return F.interpolate(self.baseline.readout(x),size=(256,256),mode='bilinear',align_corners=False)

