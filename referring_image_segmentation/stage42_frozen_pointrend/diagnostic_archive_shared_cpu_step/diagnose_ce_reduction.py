from pathlib import Path
import sys,json,torch,gc
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent))
from train import ROOT,setup,System,load_data,optimizer,update,base,dump,cpu_tree
class FlatCE:
 def __getattr__(self,name):return getattr(F,name)
 def cross_entropy(self,input,target,**kw):
  if input.ndim==3:
   n,c,p=input.shape
   return F.cross_entropy(input.permute(0,2,1).reshape(n*p,c),target.reshape(n*p),**kw)
  return F.cross_entropy(input,target,**kw)
def main():
 setup();m=System().cuda();data,_=load_data()
 ids=torch.arange(128,device='cuda');x,y=data.batch('train',ids)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):fine,coarse=m.frozen_features(x)
 cache=dict(fine=fine,coarse=coarse,target=y)
 initial=m.point_state();frozen=m.frozen_digest();scope=m.probe.decoder.author_head.forward.__globals__
 result={}
 for policy in ['native_3d','flattened_2d']:
  scope['F']=F if policy=='native_3d' else FlatCE()
  m.load_point_state(initial);setup();opt=optimizer(m)
  for _ in range(3):update(m,opt,cache,torch.arange(64,device='cuda'),64,True)
  state=dict(model=m.point_state(),opt=cpu_tree(opt.state_dict()),rng=base.rng())
  trials=[]
  for _ in range(4):
   m.load_point_state(state['model']);opt.load_state_dict(state['opt']);base.set_rng(state['rng'])
   loss,_=update(m,opt,cache,torch.arange(64,128,device='cuda'),64,True)
   now=m.point_state()
   if not trials:expected=now;expected_loss=loss
   trials.append(dict(loss=loss,delta_loss=loss-expected_loss,
    max_state_delta=max(float((now[k]-expected[k]).abs().max()) for k in now),
    state_bitwise_equal=all(torch.equal(now[k],expected[k]) for k in now)))
  result[policy]=trials
  del opt
 # Independently verify equal CE mathematical objective and gradient on actual point-logit shape.
 setup();z=torch.randn(64,2,1024,device='cuda',requires_grad=True);target=torch.randint(0,2,(64,1024),device='cuda')
 a=F.cross_entropy(z,target);ga=torch.autograd.grad(a,z)[0]
 b=FlatCE().cross_entropy(z,target);gb=torch.autograd.grad(b,z)[0]
 result['equivalence']=dict(loss_native=float(a),loss_flat=float(b),abs_difference=abs(float(a-b)),
  gradient_max_abs=float((ga-gb).abs().max()),gradient_allclose=bool(torch.allclose(ga,gb,atol=1e-8,rtol=1e-6)))
 assert result['equivalence']['abs_difference']<1e-6 and result['equivalence']['gradient_allclose']
 assert all(x['delta_loss']==0 and x['state_bitwise_equal'] for x in result['flattened_2d'])
 assert m.frozen_digest()==frozen
 dump(ROOT/'CE_REDUCTION_DIAGNOSIS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
