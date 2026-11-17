import sys,torch
from model import SegmentationSystem,load_pretrained_encoder,PROJECT
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as r
r.setup();data,_=r.load_data();m=SegmentationSystem(load_pretrained_encoder(),'A-64-noaux',seed=42).cuda().eval();order,g=data.geometry(1);ids=order[:2];block=ids//64*64
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 x,y=data.batch('train',ids,g); a=m.prefix(x)
 for j in range(2):
  k=int(ids[j]); st=k//64*64; ii=torch.arange(st,min(st+64,len(data.ids['train'])),device='cuda'); xx,yy=data.batch('train',ii,g); b=m.prefix(xx)
  print('id',k,'image_maxdiff',float((x[j]-xx[k-st]).abs().max()),'target_diff',float((y[j]-yy[k-st]).abs().max()),flush=True)
  for i in range(3):
   print('feature',i,'max',float((a[i][j].float()-b[i][k-st].float()).abs().max()),'mean',float((a[i][j].float()-b[i][k-st].float()).abs().mean()),flush=True)
