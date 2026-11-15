"""Repeated native GPU updates from the exact same state, independently of checkpoint serialization."""
import sys,copy,gc,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from train import *
def differences(a,b,initial):
 result={}
 for group,prefix in [('encoder','encoder.'),('decoder','probe.')]:
  d2=u2=0.;worst=[]
  for k,v in a.items():
   if not k.startswith(prefix):continue
   delta=(v-b[k]).double()
   d2+=float(delta.square().sum())
   u2+=float((b[k]-initial[k]).double().square().sum())
   mx=float(delta.abs().max())
   if mx:worst.append((mx,k))
  result[group]=dict(relative_update_l2=(d2/max(u2,1e-30))**.5,worst=sorted(worst,reverse=True)[:8])
 return result
def main():
 setup();data,_=load_data();cache=cache_new(128)
 original_ids=data.ids['train']
 data.ids['train']=original_ids[:128]
 ref=System('swin_control').cuda()
 fill_cache(ref,data,'train',cache)
 del ref;gc.collect();torch.cuda.empty_cache()
 result={}
 for arm in ['swin_control','madg_mfms']:
  setup();m=System(arm).cuda().train();opt=base.optimizer(m)
  for _ in range(3):update(m,opt,cache,torch.arange(64,device='cuda'),64)
  initial=cpu_tree(m.state_dict());oinitial=cpu_tree(opt.state_dict());rng=base.rng()
  states=[];losses=[];gradients=[]
  for rep in range(3):
   m.load_state_dict(initial);opt.load_state_dict(copy.deepcopy(oinitial));base.set_rng(rng)
   base.assert_tree_exact(m.state_dict(),initial);base.assert_tree_exact(opt.state_dict(),oinitial)
   assert {int(v["step"]) for v in opt.state.values()}=={3}
   loss,_=update(m,opt,cache,torch.arange(64,128,device='cuda'),64)
   losses.append(loss);states.append(cpu_tree(m.state_dict()))
  comparisons=[differences(v,states[0],initial) for v in states[1:]]
  result[arm]=dict(restored_tensor_state_exact=True,repeated_losses=losses,comparisons=comparisons)
  dump(ROOT/'GPU_REPEAT_UPDATE_DIAGNOSIS.json',result)
  print(arm,json.dumps(result[arm]),flush=True)
  del m,opt,states,initial,oinitial;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
