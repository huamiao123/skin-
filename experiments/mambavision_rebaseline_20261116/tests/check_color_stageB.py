import sys,json,torch,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from color_augmentation import apply_color,wrap_data
class Dummy:
 def __init__(self):
  self.ids={'train':list(range(7)),'val':[0,1]}
  self.images={'train':torch.arange(7*3*4*4).reshape(7,3,4,4).float()/(7*3*4*4)}
  self.mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);self.std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
  self.target=torch.arange(7*4*4).reshape(7,1,4,4).float()/255
 def geometry(self,epoch):return torch.arange(7),torch.eye(2,3)[None].expand(7,-1,-1).clone()
 def batch(self,split,indices,matrices=None):
  return (self.images['train'][indices]-self.mean)/self.std,self.target[indices].clone()
torch.manual_seed(123);rng=torch.get_rng_state().clone()
d=wrap_data(Dummy(),True);_,g=d.geometry(1)
assert torch.equal(rng,torch.get_rng_state())
x,y=d.batch('train',torch.arange(7),g)
parts=[d.batch('train',torch.arange(st,min(st+3,7)),g) for st in range(0,7,3)]
assert torch.equal(x,torch.cat([p[0] for p in parts])) and torch.equal(y,torch.cat([p[1] for p in parts]))
assert torch.equal(y,d.target)
f=d.color_factors.clone();d.geometry(1);assert torch.equal(f,d.color_factors)
d.geometry(2);assert not torch.equal(f,d.color_factors)
v,_=d.batch('val',torch.arange(2),None)
orig=Dummy();vref,_=orig.batch('val',torch.arange(2),None);assert torch.equal(v,vref)
c=wrap_data(Dummy(),False);_,g=c.geometry(1);a,b=c.batch('train',torch.arange(7),g)
aref,bref=orig.batch('train',torch.arange(7),g);assert torch.equal(a,aref) and torch.equal(b,bref)
ones=torch.ones(7,1,1,1);rgb=orig.images['train']
assert torch.equal(apply_color(rgb,ones,ones,ones),rgb)
changed=apply_color(rgb,ones*1.1,ones,ones);assert torch.allclose(changed,(rgb*1.1).clamp(0,1)) and not torch.equal(changed,rgb)
report=dict(passed=True,identity_factors_exact=True,targets_unchanged=True,batching_independent=True,
 deterministic_per_epoch=True,validation_unchanged=True,control_exact=True,no_global_rng_consumption=True,
 brightness_expected_formula=True,source_sha256=hashlib.sha256((ROOT/'src/color_augmentation.py').read_bytes()).hexdigest())
(ROOT/'reports/STAGEB_COLOR_CPU_TESTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
