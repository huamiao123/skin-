import sys,json,torch,numpy as np
from pathlib import Path
from model import PROJECT,SegmentationSystem,load_pretrained_encoder
from smoke_train import atomic_json
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as ref
ref.setup();torch.use_deterministic_algorithms(True,warn_only=True)
data,_=ref.load_data();m=SegmentationSystem(load_pretrained_encoder(),'A-64-noaux',seed=42).cuda()
ck=torch.load('results/smoke/A-64-noaux/best_dice.pth',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval()
cache=ref.cache_new(len(data.ids['val']));ref.fill_cache(m,data,'val',cache)
n=len(data.ids['val']);rows=[];mx=0;sumabs=0;pixels=0;changed=0
with torch.no_grad():
 for st in range(0,n,2):
  ix=torch.arange(st,min(n,st+2),device='cuda');images,truth=data.batch('val',ix)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   a=m(images)['main'].float();b=m.from_cache(cache['features'][0][ix],cache['features'][1][ix],cache['mid'][ix],(256,256))['main'].float()
  delta=(a-b).abs();mx=max(mx,float(delta.max()));sumabs+=float(delta.sum());pixels+=delta.numel()
  pa=a>=0;pb=b>=0;gt=truth>=.5;changed+=int((pa!=pb).sum())
  for p,q,y in zip(pa,pb,gt):
   def dice(z):return float((2*(z&y).sum()+1e-6)/(z.sum()+y.sum()+1e-6))
   rows.append([dice(p),dice(q)])
r=dict(n=n,checkpoint_epoch=ck['epoch'],max_logit_abs=mx,mean_logit_abs=sumabs/pixels,
 pixel_disagreement=changed,disagreement_fraction=changed/pixels,
 direct_macro_dice=float(np.mean([x[0] for x in rows])),cached_macro_dice=float(np.mean([x[1] for x in rows])),
 mean_image_dice_delta=float(np.mean([x[0]-x[1] for x in rows])),test_read=False)
atomic_json(Path('reports/validation_cache_comparison.json'),r);print(json.dumps(r),flush=True)
