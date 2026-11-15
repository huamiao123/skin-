"""Isolate mixed precision + original author adaptive-pool backward; no image labels or test data."""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from train import setup,dump,ROOT
from author_modules import AuthorModuleDecoder
from stage2_decoder.models import SwinPixelDecoder
import torch
setup()
d=AuthorModuleDecoder('madg_mfms',SwinPixelDecoder()).cuda().train()
block=d.author_blocks['0']
x=torch.randn(64,64,32,32,device='cuda').bfloat16()
skip=torch.randn(64,64,64,64,device='cuda').bfloat16()
start=copy.deepcopy(block.state_dict())
results={}
for fp32 in [False,True]:
 trials=[]
 losses=[]
 for rep in range(3):
  block.load_state_dict(start);block.zero_grad(set_to_none=True)
  a=x.detach().clone().requires_grad_();b=skip.detach().clone().requires_grad_()
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32):
   output=block(a.float() if fp32 else a,b.float() if fp32 else b)
  loss=output.float().square().mean();loss.backward()
  values={'input':a.grad.float().cpu(),'skip':b.grad.float().cpu()}
  values.update({n:q.grad.float().cpu() for n,q in block.named_parameters()})
  trials.append(values);losses.append(float(loss))
 comparisons=[]
 for trial in trials[1:]:
  d2=g2=0.;worst=[]
  for n,g in trials[0].items():
   delta=trial[n]-g;d2+=float(delta.double().square().sum());g2+=float(g.double().square().sum())
   worst.append((float(delta.abs().max()),n))
  comparisons.append(dict(relative_gradient_l2=(d2/max(g2,1e-30))**.5,worst=sorted(worst,reverse=True)[:6]))
 results['fp32' if fp32 else 'bf16']=dict(losses=losses,comparisons=comparisons)
dump(ROOT/'POOL_PRECISION_DIAGNOSIS.json',results)
print(json.dumps(results),flush=True)
