import sys,json,time,torch
from pathlib import Path
from torch import nn
from model import PROJECT,SegmentationSystem,load_pretrained_encoder,objective
from smoke_train import accumulate,rng_state,restore_rng,atomic_json
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as ref
from stage28_drop_path import train as base
class Pool(nn.Module):
 def __init__(self,shape):super().__init__();self.shape=(shape,shape) if isinstance(shape,int) else shape
 def forward(self,x):
  h,w=x.shape[-2:];oh,ow=self.shape
  return torch.stack([torch.stack([x[...,i*h//oh:((i+1)*h+oh-1)//oh,j*w//ow:((j+1)*w+ow-1)//ow].mean((-2,-1)) for j in range(ow)],-1) for i in range(oh)],-2)
def replace(module):
 count=0
 for name,m in list(module.named_children()):
  if isinstance(m,nn.AdaptiveAvgPool2d):setattr(module,name,Pool(m.output_size));count+=1
  else:count+=replace(m)
 return count
def analyze(mode):
 ref.setup();torch.use_deterministic_algorithms(True,warn_only=True)
 data,_=ref.load_data();m=SegmentationSystem(load_pretrained_encoder(),'A-64-noaux',seed=42).cuda();num=replace(m.probe) if mode=='explicit_mean' else 0
 opt=torch.optim.AdamW([{'params':list(m.probe.parameters()),'lr':.001,'name':'decoder'},{'params':[p for p in m.encoder.parameters() if p.requires_grad],'lr':.0001,'name':'encoder'}],weight_decay=.01)
 cache=ref.cache_new(len(data.ids['train']));order,g=data.geometry(1);ref.fill_cache(m,data,'train',cache,g);ids=order[:2];m.train()
 initial=base.cpu_tree(m.state_dict());state=rng_state();results=[];snap=[]
 for run in range(3):
  m.load_state_dict(initial);opt.state.clear();restore_rng(state);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   o=m.from_cache(cache['features'][0][ids],cache['features'][1][ids],cache['mid'][ids],(256,256))
  loss,_=objective(o,cache['target'][ids]);loss.backward()
  grad={k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
  for group in opt.param_groups:torch.nn.utils.clip_grad_norm_(group['params'],1.)
  opt.step();params=base.cpu_tree(m.state_dict())
  results.append(float(loss));snap.append((grad,params))
 rows=[]
 for run in [1,2]:
  gd=[];pd=[]
  for k in snap[0][0]:
   a=snap[0][0][k].float();b=snap[run][0][k].float();gd.append((float((a-b).abs().max()),k))
  for k in initial:
   if k not in snap[0][0]:continue
   a=snap[0][1][k].float();b=snap[run][1][k].float();pd.append((float((a-b).abs().max()),k))
  rows.append(dict(gradient_top=sorted(gd,reverse=True)[:10],parameter_top=sorted(pd,reverse=True)[:10]))
 result=dict(mode=mode,pools_replaced=num,losses=results,comparisons=rows)
 atomic_json(Path('reports')/('replay_diagnosis_'+mode+'.json'),result);print(json.dumps(result),flush=True)
for mode in ['native','explicit_mean']:analyze(mode)
