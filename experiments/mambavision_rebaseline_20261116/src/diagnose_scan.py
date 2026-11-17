import sys,json,torch
from pathlib import Path
from model import PROJECT,SegmentationSystem,load_pretrained_encoder
from smoke_train import rng_state,restore_rng,atomic_json
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as ref
from stage28_drop_path import train as base
import torch.nn.functional as F
ref.setup();torch.use_deterministic_algorithms(True,warn_only=True)
data,_=ref.load_data();m=SegmentationSystem(load_pretrained_encoder(),'A-64-noaux',seed=42).cuda().eval()
order,g=data.geometry(1);ids=order[:2]
x,_=data.batch('train',ids,g)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f1,f2,mid=m.prefix(x)
block=m.encoder.model.levels[2].blocks[0].mixer
def audit(kind):
 pairs=[];outs=[]
 for j in range(3):
  for p in block.parameters():p.grad=None
  torch.manual_seed(888);torch.cuda.manual_seed_all(888)
  if kind=='mixer':
   inp=torch.randn(2,196,384,device='cuda',requires_grad=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):y=block(inp);loss=y.float().square().mean()
  else:
   inp=torch.randn(2,192,196,device='cuda',requires_grad=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):y=F.conv1d(inp,block.conv1d_x.weight,padding='same',groups=192);loss=y.float().square().mean()
  loss.backward();outs.append(y.detach().cpu())
  pairs.append({k:p.grad.detach().cpu().clone() for k,p in block.named_parameters() if p.grad is not None})
 rows=[]
 for j in [1,2]:
  grads=sorted([(float((pairs[0][k]-v).abs().max()),k) for k,v in pairs[j].items()],reverse=True)
  rows.append(dict(output_max=float((outs[0]-outs[j]).abs().max()),top_gradient_diffs=grads[:5]))
 return dict(mode=kind,comparisons=rows)
r=[audit('mixer'),audit('conv1d')]
atomic_json(Path('reports/scan_operator_diagnosis.json'),r);print(json.dumps(r),flush=True)
