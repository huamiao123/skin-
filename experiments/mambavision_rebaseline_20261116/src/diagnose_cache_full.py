import sys,torch
from model import SegmentationSystem,load_pretrained_encoder,PROJECT
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as r
r.setup();torch.use_deterministic_algorithms(True,warn_only=True);data,_=r.load_data();m=SegmentationSystem(load_pretrained_encoder(),'A-64-noaux',seed=42).cuda();vc=r.cache_new(len(data.ids['val']));r.fill_cache(m,data,'val',vc);cache=r.cache_new(len(data.ids['train']));order,g=data.geometry(1);r.fill_cache(m,data,'train',cache,g);m.train();ids=order[:2];print('ids',ids.tolist(),flush=True);print('encoder_training',[(n,x.training) for n,x in m.encoder.named_modules() if x.training][:5],flush=True);m.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 for sid in ids.tolist():
  st=sid//64*64;ii=torch.arange(st,min(st+64,len(data.ids['train'])),device='cuda');x,y=data.batch('train',ii,g);a=m.prefix(x)
  for j,v in enumerate([cache['features'][0],cache['features'][1],cache['mid']]):
   delta=(a[j][sid-st].float()-v[sid].float()).abs();print('id',sid,'feat',j,'max',float(delta.max()),'mean',float(delta.mean()),'not_equal',int((delta!=0).sum()),flush=True)
  d=(cache['target']-y[sid-st]).abs().mean((1,2,3));best=torch.topk(d,5,largest=False);print('maskdiff',float((y[sid-st]-cache['target'][sid]).abs().max()),'nearest',best.indices.tolist(),best.values.tolist(),flush=True)
  xx,yy=data.batch('train',ii,g);print('repeat image',float((x-xx).abs().max()),'repeat target',float((y-yy).abs().max()),'train_idslen',len(data.ids['train']),flush=True)
