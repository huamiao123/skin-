from pathlib import Path
import sys,json,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scale_augmentation import compose_scale,wrap_data
class Toy:
 def __init__(self):
  self.ids={'train':list(range(5))}
  self.images=torch.zeros(5,3,64,64)
  self.masks=torch.zeros(5,1,64,64)
  self.images[:,:,16:48,16:48]=1;self.masks[:,:,16:48,16:48]=.8
 def geometry(self,e):return torch.arange(5),torch.eye(2,3)[None].repeat(5,1,1)
 def batch(self,split,ix,matrices=None):
  x=self.images[ix];y=self.masks[ix]
  if matrices is not None:
   grid=F.affine_grid(matrices[ix],x.shape,align_corners=False)
   x=F.grid_sample(x,grid,mode='bilinear',align_corners=False)
   y=F.grid_sample(y,grid,mode='nearest',align_corners=False)
  return x,y
torch.manual_seed(1);rng=torch.get_rng_state().clone()
d=wrap_data(Toy(),True);order,g=d.geometry(1)
assert torch.equal(rng,torch.get_rng_state())
assert float(d.scale_factors.min())>=.85 and float(d.scale_factors.max())<=1.
f=d.scale_factors.clone();_,g2=d.geometry(1);assert torch.equal(g,g2)
_,g3=d.geometry(2);assert not torch.equal(g,g3)
_,g=d.geometry(1);x,y=d.batch('train',order,g)
assert set(y.unique().tolist()).issubset(set(d.masks.unique().tolist()))
x2=[];y2=[]
for st in range(0,5,2):
 a,b=d.batch('train',order[st:st+2],g);x2.append(a);y2.append(b)
assert torch.equal(x,torch.cat(x2)) and torch.equal(y,torch.cat(y2))
assert (y>0).sum()< (d.masks>0).sum()
# Both sampled from the same grid: nearest labels and bilinear image share center and interior.
grid=F.affine_grid(g,d.images.shape,align_corners=False)
assert torch.equal(x,F.grid_sample(d.images,grid,mode='bilinear',align_corners=False))
assert torch.equal(y,F.grid_sample(d.masks,grid,mode='nearest',align_corners=False))
# Bilinear image corners may differ from nearest labels at the threshold;
# both centers must remain at the same known image center.
coords=torch.arange(64).float()
for image,mask in zip(x[:,0],y[:,0]):
 for axis in (0,1):
  a=(image.sum(axis)*coords).sum()/image.sum()
  b=(mask.sum(axis)*coords).sum()/mask.sum()
  assert abs(float(a-b))<.5 and abs(float(a)-31.5)<.5
before=Toy();v1,v2=d.batch('val',order,None);a,b=before.batch('val',order,None)
assert torch.equal(v1,a) and torch.equal(v2,b)
_,identity=before.geometry(1);assert torch.equal(compose_scale(identity,torch.ones(5)),identity)
# Image edge structures survive shrink in output rather than being cropped.
edge=Toy();edge.masks.zero_()
for ys,xs in [(slice(0,8),slice(0,8)),(slice(0,8),slice(56,64)),(slice(56,64),slice(0,8)),(slice(56,64),slice(56,64))]:
 edge.masks[:,:,ys,xs]=1
scaled=compose_scale(identity,torch.full((5,),.85))
_,mask=edge.batch('train',order,scaled)
for ys,xs in [(slice(0,32),slice(0,32)),(slice(0,32),slice(32,64)),(slice(32,64),slice(0,32)),(slice(32,64),slice(32,64))]:
 assert mask[:,:,ys,xs].sum()>0
disabled=wrap_data(Toy(),False);_,g=disabled.geometry(1);assert torch.equal(g,identity)
result=dict(passed=True,deterministic_epoch=True,independent_global_rng=True,range_verified=True,
 common_image_mask_grid=True,nearest_mask_values_preserved=True,batching_equivalent=True,
 validation_unchanged=True,identity_exact=True,edge_structures_not_cropped=True,disabled_geometry_exact=True)
(ROOT/'reports/STAGEB_SCALE_CPU_TESTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
