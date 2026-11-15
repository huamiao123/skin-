"""Validation-only fusion audit. No optimizer, no test loading, original checkpoints unchanged."""
from pathlib import Path
import sys,json,hashlib,copy,types,gc,math,importlib.util
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from torch import nn
R=Path(__file__).resolve().parent
P=R.parent.parent/'source/referring_image_segmentation'
S=P/'stage33_author_modules'
sys.path.insert(0,str(S));sys.path.insert(0,str(P))
from stage33_author_modules import train as training
from stage33_author_modules import evaluate as evaluation
base=training.base
base.setup()
def dump(name,data):
 tmp=R/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(R/name)
partition=json.loads((S/'identity_partition.json').read_text())
vids=partition['splits']['val'];tids=partition['splits']['train']
assert len(vids)==762 and len(tids)==1886
val=evaluation.helpers.load_data(vids,P/'data/isic2018/val/images',P/'data/isic2018/val/masks')
result=dict(protocol=json.loads((R/'PROTOCOL.json').read_text()),arms={},test_read=False,trained=False)
def gates(m,arm):
 dec=m.probe.decoder
 return {str(i):dec.author_gates[str(i)] for i in (2,1,0)} if arm=='swin_cga' else {str(i):getattr(dec.author_decoder,'lgag'+str(i)) for i in (3,2,1)}
class Moments:
 def __init__(self):self.n=0;self.s=0.;self.s2=0.;self.neg=0;self.low=0;self.high=0
 def add(self,t):
  t=t.detach().float()
  self.n+=t.numel();self.s+=float(t.double().sum());self.s2+=float(t.double().square().sum())
  self.neg+=int((t<0).sum());self.low+=int((t<.01).sum());self.high+=int((t>.99).sum())
 def out(self):
  return dict(mean=self.s/self.n,std=math.sqrt(max(0,self.s2/self.n-(self.s/self.n)**2)),rms=math.sqrt(self.s2/self.n),negative_fraction=self.neg/self.n,below01_fraction=self.low/self.n,above99_fraction=self.high/self.n)
def hook_stats(m,arm):
 stats={};handles=[]
 for k,gate in gates(m,arm).items():
  stats[k]={name:Moments() for name in ['g','skip','gated_skip','attention','attention_spatial_std']}
  def pre(module,args,kwargs,key=k):
   g=args[0] if args else kwargs['g'];x=args[1] if args else kwargs['x']
   stats[key]['g'].add(g);stats[key]['skip'].add(x)
  def post(module,args,out,key=k):stats[key]['gated_skip'].add(out)
  def psi(module,args,out,key=k):
   stats[key]['attention'].add(out)
   stats[key]['attention_spatial_std'].add(out.float().flatten(1).std(1,unbiased=False))
  handles.append(gate.register_forward_pre_hook(pre,with_kwargs=True))
  handles.append(gate.register_forward_hook(post))
  handles.append((gate.combine[-1] if arm=='swin_cga' else gate.psi[-1]).register_forward_hook(psi))
 return stats,handles
def metrics(m,label,arm):
 metric,rows=evaluation.helpers.evaluate(m,val,vids,base,R/f'{arm}_{label}_predictions.npz')
 evaluation.write_rows(R/f'{arm}_{label}_per_image.csv',rows)
 dump(f'{arm}_{label}_metrics.json',metric)
 print('METRIC',arm,label,'macro',metric['macro']['dice'],'pooled',metric['pooled']['dice'],flush=True)
 return metric
def train_rgb(sid):
 return np.asarray(Image.open(P/'data/isic2018/train/images'/f'{sid}.png').convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
with ThreadPoolExecutor(8) as pool:rgb=np.stack(list(pool.map(train_rgb,tids)))
mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
for arm in ['emcad64','swin_cga']:
 ckpt=S/arm/'results'/f'{arm}_best_dice.pth'
 sha=base.digest(ckpt);state=torch.load(ckpt,map_location='cpu')
 m=training.System(arm).cuda();m.load_state_dict(state['model'],strict=True);m.requires_grad_(False).eval()
 record=dict(checkpoint_sha256=sha,checkpoint_epoch=state['epoch'],cases={})
 stats,handles=hook_stats(m,arm)
 record['cases']['original']=metrics(m,'original',arm)
 assert record['cases']['original']['counts']==state['metric']['counts']
 for h in handles:h.remove()
 record['gate_statistics']={k:{n:x.out() for n,x in v.items()} for k,v in stats.items()}
 if arm=='swin_cga':
  handles=[gate.register_forward_hook(lambda mod,inp,out:inp[1]) for gate in gates(m,arm).values()]
  record['cases']['gates_identity']=metrics(m,'gates_identity',arm)
  for h in handles:h.remove()
  handles=[gate.combine[-1].register_forward_hook(lambda mod,inp,out:out.mean((-2,-1),keepdim=True).expand_as(out)) for gate in gates(m,arm).values()]
  record['cases']['gates_spatial_mean']=metrics(m,'gates_spatial_mean',arm)
  for h in handles:h.remove()
 # Fixed train-only cumulative BN recalibration. No gradient, optimizer, labels or validation-stat updates.
 bn={n:b for n,b in m.probe.decoder.named_modules() if isinstance(b,nn.modules.batchnorm._BatchNorm)}
 before={n:dict(mean=b.running_mean.cpu().clone(),var=b.running_var.cpu().clone(),counter=int(b.num_batches_tracked)) for n,b in bn.items()}
 momentum={n:b.momentum for n,b in bn.items()}
 for b in bn.values():b.reset_running_stats();b.momentum=None;b.train()
 assert all(not q.training for q in m.encoder.modules())
 with torch.no_grad():
  for start in range(0,len(rgb),32):
   x=torch.from_numpy(rgb[start:start+32]).permute(0,3,1,2).contiguous().cuda().float()/255
   with torch.autocast('cuda',dtype=torch.bfloat16):m((x-mean)/std)
 m.eval()
 buffer_summary=[]
 for n,b in bn.items():
  b.momentum=momentum[n]
  normalized=(b.running_mean.cpu()-before[n]['mean'])/torch.sqrt(before[n]['var']+b.eps)
  ratio=b.running_var.cpu()/(before[n]['var']+b.eps)
  buffer_summary.append(dict(name=n,old_batches=before[n]['counter'],new_batches=int(b.num_batches_tracked),mean_shift_in_old_std_rms=float(normalized.square().mean().sqrt()),variance_ratio_median=float(ratio.median())))
 # Ensure every learnable tensor is identical to the archived checkpoint after recalibration.
 for name,p in m.named_parameters():assert torch.equal(p.detach().cpu(),state['model'][name]),name
 record['bn_recalibration']=dict(train_n=len(rgb),batch_size=32,bn_layers=len(bn),learned_parameters_unchanged=True,buffer_changes=buffer_summary)
 record['cases']['bn_train_recalibrated']=metrics(m,'bn_train_recalibrated',arm)
 assert base.digest(ckpt)==sha
 record['validation_deltas']={label:dict(macro_dice_pp=(q['macro']['dice']-record['cases']['original']['macro']['dice'])*100,
        fp_delta=q['counts']['fp']-record['cases']['original']['counts']['fp'],fn_delta=q['counts']['fn']-record['cases']['original']['counts']['fn'])
        for label,q in record['cases'].items() if label!='original'}
 result['arms'][arm]=record;dump('diagnostic_results.json',result)
 del m,state,bn;gc.collect();torch.cuda.empty_cache()
dump('DIAGNOSTICS_DONE.json',dict(complete=True,test_read=False,training=False,cases=6))
print('DONE',flush=True)

