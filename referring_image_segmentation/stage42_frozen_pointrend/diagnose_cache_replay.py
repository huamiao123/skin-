from pathlib import Path
import sys,json,torch
sys.path.insert(0,str(Path(__file__).parent))
from train import *
def info(a,b):
 d=(a.float()-b.float()).abs()
 return dict(equal=torch.equal(a,b),max_abs=float(d.max()),mean_abs=float(d.mean()),different=int((a!=b).sum()),
  shape=list(a.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),stride_a=list(a.stride()),stride_b=list(b.stride()))
def main():
 setup();m=System().cuda();data,_=load_data()
 val=cache_new(len(data.ids['val']));fill_cache(m,data,'val',val)
 ix=torch.arange(64,device='cuda');x,y=data.batch('train',ix)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f,c=m.frozen_features(x)
 train=dict(fine=f,coarse=c,target=y)
 opt=optimizer(m)
 for _ in range(4):update(m,opt,train,ix,64,True)
 m.eval();x,_=data.batch('val',ix)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  f,c=m.frozen_features(x)
  a=m.from_cache(val,ix);b=m(x);d=m._head(f,c)
 result=dict(fine=info(val['fine'][ix],f),coarse=info(val['coarse'][ix],c),output=info(a,b),
  direct_vs_head=info(b,d),binary_different=int(((a.float().sigmoid()>=.5)!=(b.float().sigmoid()>=.5)).sum()),
  trainable=[k for k,p in m.named_parameters() if p.requires_grad],
  training_modules=[k for k,v in m.named_modules() if v.training])
 dump(ROOT/'CACHE_REPLAY_DIAGNOSIS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
