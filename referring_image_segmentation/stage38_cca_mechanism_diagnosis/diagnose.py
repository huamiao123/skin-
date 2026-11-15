"""Bounded validation-only CCA mechanism audit. Original source and checkpoints remain immutable."""
from pathlib import Path
import sys,types,json,gc,csv,datetime,time,traceback,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).parent;PROJECT=ROOT.parent
SOURCE=PROJECT/'stage36_topconf_skip_fusion'
sys.path.insert(0,str(SOURCE))
from train import System,base,digest
from evaluate import helpers,compare
def dump(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def status(state,**kw):
 obj=dict(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=__import__('os').getpid(),training=False,test_read=False,**kw)
 dump(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def writer(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dist(a):
 return dict(mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),q05=float(np.quantile(a,.05)),q95=float(np.quantile(a,.95)))
@torch.inference_mode()
def main():
 base.setup()
 partition=json.loads((SOURCE/'identity_partition.json').read_text());ids=partition['splits']['val'];n=len(ids)
 data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
 xs,ys=data
 mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
 permutations={}
 for seed in [3801,3802,3803]:
  order=np.random.default_rng(seed).permutation(n);perm=np.empty(n,dtype=np.int64);perm[order]=np.roll(order,1)
  assert np.all(perm!=np.arange(n)) and len(set(perm))==n
  permutations[str(seed)]=perm
 dump(ROOT/'permutations.json',{k:v.tolist() for k,v in permutations.items()})
 all_results={}
 for tag,stage in [('old_lr','stage36_topconf_skip_fusion'),('new_lr','stage37_lr_horizon100')]:
  dest=ROOT/tag;dest.mkdir()
  directory=PROJECT/stage/'uct_cca/results';ck=directory/'uct_cca_best_dice.pth';sha=digest(ck)
  saved=torch.load(ck,map_location='cpu')
  for rel,expected in saved['config']['sources'].items():assert digest(PROJECT/rel)==expected,rel
  expected=saved['metric'];epoch=saved['epoch']
  model=System('uct_cca').cuda();model.load_state_dict(saved['model'],strict=True);del saved
  model.requires_grad_(False).eval()
  gates={str(i):model.probe.decoder.author_blocks[str(i)].coatt for i in [2,1,0]}
  raw={i:[] for i in gates};raw_g={i:[] for i in gates};raw_x={i:[] for i in gates};dtypes={};latest={}
  capture=[]
  for i,g in gates.items():
   def hx(mod,args,output,i=i):latest[(i,'x')]=output
   def hg(mod,args,output,i=i):latest[(i,'g')]=output
   def hc(mod,args,output,i=i):
    z=(latest[(i,'x')]+latest[(i,'g')])/2.0
    dtypes[i]=z.dtype;raw[i].append(z.float().cpu().numpy())
    raw_x[i].append(latest[(i,'x')].float().cpu().numpy());raw_g[i].append(latest[(i,'g')].float().cpu().numpy())
   capture.extend([g.mlp_x.register_forward_hook(hx),g.mlp_g.register_forward_hook(hg),g.register_forward_hook(hc)])
  current={'mode':'native','start':0}
  def evaluate_mode(mode):
   current['mode']=mode;rows=[];predictions=[]
   status('evaluating',model=tag,intervention=mode)
   with ThreadPoolExecutor(8) as pool:
    for begin in range(0,n,64):
     current['start']=begin
     x=torch.from_numpy(xs[begin:begin+64]).permute(0,3,1,2).contiguous().cuda().float()/255.
     target=ys[begin:begin+64]>=127.5
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=model((x-mean)/std)
     assert torch.isfinite(logits).all()
     pred=logits.float().sigmoid()[:,0].cpu().numpy()>=.5
     rr=list(pool.map(lambda z:base.boundary_row(*z),zip(pred,target)))
     for sid,row in zip(ids[begin:begin+64],rr):row['id']=sid
     rows.extend(rr);predictions.append(np.packbits(pred.reshape(len(pred),-1),axis=1))
   metric=base.summarize_rows(rows);metric['severe_undercoverage']=sum(r['recall']<.5 for r in rows)
   writer(dest/(mode+'_per_image.csv'),rows);dump(dest/(mode+'_metrics.json'),metric)
   return metric,rows,np.concatenate(predictions)
  native,original_rows,original_pred=evaluate_mode('native')
  assert native['counts']==expected['counts'] and abs(native['macro']['dice']-expected['macro']['dice'])<1e-12
  previous=np.load(directory/'uct_cca_val_predictions.npz')
  assert np.array_equal(previous['ids'],ids) and np.array_equal(previous['predictions_packed'],original_pred)
  for h in capture:h.remove()
  arrays={i:np.concatenate(raw[i]) for i in gates};ag={i:np.concatenate(raw_g[i]) for i in gates};ax={i:np.concatenate(raw_x[i]) for i in gates}
  np.savez_compressed(dest/'native_gate_logits.npz',**{kind+'_'+i:a for kind,d in [('sum',arrays),('g',ag),('x',ax)] for i,a in d.items()})
  statistics={}
  for i in gates:
   weights=torch.from_numpy(arrays[i]).to(dtypes[i]).sigmoid().float().numpy()
   statistics[i]=dict(gate_weights=dist(weights),between_image_std_by_channel=dist(weights.std(0)),fraction_below005=float((weights<.05).mean()),fraction_above095=float((weights>.95).mean()))
  dump(dest/'gate_statistics.json',statistics)
  averages={i:arrays[i].mean(0,keepdims=True) for i in gates}
  avg_g={i:ag[i].mean(0,keepdims=True) for i in gates};avg_x={i:ax[i].mean(0,keepdims=True) for i in gates}
  for i,g in gates.items():
   def replacement(this,g,x,i=i):
    mode=current['mode'];start=current['start'];size=len(x)
    if mode=='identity_all':return this.relu(x*torch.ones((size,x.shape[1],1,1),device=x.device,dtype=dtypes[i]))
    if mode=='recorded_all':z=arrays[i][start:start+size]
    elif mode.startswith('permute_'):z=arrays[i][permutations[mode.split('_')[1]][start:start+size]]
    elif mode=='mean_all' or mode=='mean_level'+i:z=np.repeat(averages[i],size,axis=0)
    else:
     px=this.mlp_x(F.avg_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3))))
     pg=this.mlp_g(F.avg_pool2d(g,(g.size(2),g.size(3)),stride=(g.size(2),g.size(3))))
     if mode=='decoder_mean_all':pg=torch.as_tensor(avg_g[i],device=pg.device,dtype=pg.dtype).expand_as(pg)
     if mode=='skip_mean_all':px=torch.as_tensor(avg_x[i],device=px.device,dtype=px.dtype).expand_as(px)
     z=(px+pg)/2.0
    if isinstance(z,np.ndarray):z=torch.as_tensor(z,device=x.device,dtype=dtypes[i])
    scale=z.sigmoid().unsqueeze(2).unsqueeze(3).expand_as(x)
    return this.relu(x*scale)
   g.forward=types.MethodType(replacement,g)
  replay,_,replay_pred=evaluate_mode('recorded_all')
  assert np.array_equal(replay_pred,original_pred) and replay['counts']==native['counts']
  dump(dest/'REPLAY_VERIFIED.json',dict(passed=True,original_matches_archived_binary_predictions=True,captured_gate_replay_matches_original=True,checkpoint_sha256=sha))
  results={}
  variants=['mean_all','permute_3801','permute_3802','permute_3803','mean_level2','mean_level1','mean_level0','decoder_mean_all','skip_mean_all','identity_all']
  for mode in variants:
   metric,rows,pred=evaluate_mode(mode)
   results[mode]=dict(metrics=metric,paired_vs_native=compare(original_rows,rows),prediction_pixels_changed=int(np.unpackbits(np.bitwise_xor(original_pred,pred)).sum()))
   dump(dest/'interventions.json',results)
  assert digest(ck)==sha
  all_results[tag]=dict(checkpoint=str(ck),sha256=sha,epoch=epoch,native=native,gate_statistics=statistics,interventions=results)
  dump(ROOT/'summary.json',all_results)
  del model,gates,raw,raw_g,raw_x,latest,arrays,ag,ax,original_pred,replay_pred,pred;gc.collect();torch.cuda.empty_cache()
 dump(ROOT/'DONE.json',dict(complete=True,training=False,test_read=False,validation_n=n,models=list(all_results),native_and_recorded_gate_replays_exact=True))
 status('complete',models=list(all_results))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc()))
  status('failed');raise
