from pathlib import Path
import sys,json,csv,datetime,traceback,hashlib,gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).parent;PROJECT=ROOT.parent;SOURCE=PROJECT/'stage40_pointrend'
sys.path.insert(0,str(SOURCE))
from train import System,base,digest
from evaluate import helpers,refrows,compare
from author_modules import make_scope
def dump(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def status(state,**kw):
 obj=dict(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
 dump(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def wr(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def random_grid(seed):
 def select(u,p):
  n,_,h,w=u.shape;g=torch.Generator().manual_seed(seed+h*1000+w)
  ix=torch.randperm(h*w,generator=g)[:min(p,h*w)].to(u.device)[None].expand(n,-1)
  xy=torch.stack([(ix%w+.5)/w,(ix//w+.5)/h],dim=-1)
  return ix,xy
 return select
def transcounts(before,after,truth):
 dims=tuple(range(1,before.ndim))
 fixed=(before!=truth)&(after==truth);broken=(before==truth)&(after!=truth)
 return dict(corrected=fixed.sum(dims),damaged=broken.sum(dims),
  removed_fp=(fixed&~truth).sum(dims),removed_fn=(fixed&truth).sum(dims),
  added_fp=(broken&~truth).sum(dims),added_fn=(broken&truth).sum(dims),
  before_errors=(before!=truth).sum(dims),after_errors=(after!=truth).sum(dims))
@torch.inference_mode()
def main():
 base.setup();cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
 ids=json.loads((SOURCE/'identity_partition.json').read_text())['splits']['val']
 assert len(ids)==762
 ref=json.loads((SOURCE/'REFERENCE_MODELS.json').read_text())['models'][0]
 baseline=refrows(ref['val_rows'],ids)
 data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
 xs,ys=data
 mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
 std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
 un,_=make_scope(False);rd,_=make_scope(True)
 selectors=dict(uncertain=un['get_uncertain_point_coords_on_grid'],random=rd['get_uncertain_point_coords_on_grid'],
  random_alt41001=random_grid(41001),random_alt41002=random_grid(41002))
 modes=cfg['primary_modes']+cfg['secondary_modes']
 results={}
 for arm in cfg['checkpoints']:
  folder=SOURCE/arm/'results';ck=folder/f'{arm}_best_dice.pth';cksha=digest(ck)
  state=torch.load(ck,map_location='cpu')
  for rel,sha in state['config']['sources'].items():assert digest(PROJECT/rel)==sha,rel
  model=System(arm).cuda().eval();model.load_state_dict(state['model'],strict=True);model.requires_grad_(False)
  expected=state['metric'];epoch=state['epoch'];del state
  fixed=np.load(folder/f'{arm}_val_predictions.npz');assert np.array_equal(fixed['ids'],ids)
  head=model.probe.decoder.author_head;scope=head.forward.__globals__
  original_selector=scope['get_uncertain_point_coords_on_grid'];original_uncertainty=scope['calculate_uncertainty']
  rows={m:[] for m in modes};transitions={m:[] for m in selectors};stage_rows=[]
  captures={}
  def capture_uncertainty(logits):
   captures['before']=logits
   return original_uncertainty(logits)
  def capture_points(module,args,out):
   captures['points']=out
  hook=head.point_head.register_forward_hook(capture_points)
  with ThreadPoolExecutor(8) as pool:
   for begin in range(0,len(ids),64):
    x=torch.from_numpy(xs[begin:begin+64]).permute(0,3,1,2).contiguous().cuda().float()/255
    y=torch.from_numpy(ys[begin:begin+64]>=127.5).cuda();n=len(x)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     f1,f2,mid=model.prefix((x-mean)/std);f3,f4=model.suffix(mid)
     features={f'p{i}':a(f) for i,(a,f) in enumerate(zip(model.probe.adapter,[f1,f2,f3,f4]))}
     coarse=head.coarse_sem_seg_head.layers(features)
     binary=coarse[:,1:2]-coarse[:,0:1]
     direct=F.interpolate(binary,size=(256,256),mode='bilinear',align_corners=False)
     step=F.interpolate(F.interpolate(coarse,scale_factor=2,mode='bilinear',align_corners=False),scale_factor=2,mode='bilinear',align_corners=False)
     pred=dict(coarse_direct=direct.float().sigmoid()[:,0]>=.5,
      coarse_stepwise=(step[:,1:2]-step[:,0:1]).float().sigmoid()[:,0]>=.5)
     for mode,selector in selectors.items():
      captured_stages=[]
      def traced_selector(u,p):
       ix,xy=selector(u,p)
       captured_stages.append(dict(before=captures['before'],indices=ix,coords=xy))
       return ix,xy
      def capture_output(module,args,out):captured_stages[-1]['points']=out
      hook2=head.point_head.register_forward_hook(capture_output)
      scope['get_uncertain_point_coords_on_grid']=traced_selector
      scope['calculate_uncertainty']=capture_uncertainty
      out,_=head(features);hook2.remove()
      pred[mode]=(out[:,1:2]-out[:,0:1]).float().sigmoid()[:,0]>=.5
      assert len(captured_stages)==2
      for si,c in enumerate(captured_stages):
       before=c['before'];ix=c['indices'];pt=c['points']
       b=before.flatten(2).gather(2,ix[:,None].expand(-1,2,-1))
       b=(b[:,1]-b[:,0]).float().sigmoid()>=.5
       a=(pt[:,1]-pt[:,0]).float().sigmoid()>=.5
       truth=un['point_sample'](y[:,None].float(),c['coords'],mode='nearest',align_corners=False)[:,0]>=.5
       changes=transcounts(b,a,truth)
       conf=(pt[:,1]-pt[:,0]).float().abs().sigmoid()
       for j,sid in enumerate(ids[begin:begin+n]):
        stage_rows.append(dict(id=sid,mode=mode,stage=si+1,resolution=before.shape[-1],selected=ix.shape[1],
         **{k:int(v[j]) for k,v in changes.items()},
         after_wrong_confidence_sum=float(conf[j][a[j]!=truth[j]].sum())))
    native_mode='uncertain' if arm=='point_uncertain' else 'random'
    packed=np.packbits(pred[native_mode].cpu().numpy().reshape(n,-1),axis=1)
    assert np.array_equal(packed,fixed['predictions_packed'][begin:begin+n]),(arm,begin,'native replay')
    yn=y.cpu().numpy()
    for mode in modes:
     rr=list(pool.map(lambda z:base.boundary_row(*z),zip(pred[mode].cpu().numpy(),yn)))
     for sid,row in zip(ids[begin:begin+n],rr):row['id']=sid
     rows[mode].extend(rr)
     if mode in selectors:
      changes=transcounts(pred['coarse_stepwise'],pred[mode],y)
      for j,sid in enumerate(ids[begin:begin+n]):
       transitions[mode].append(dict(id=sid,**{k:int(v[j]) for k,v in changes.items()}))
    status('running',arm=arm,processed=begin+n,n=len(ids))
  scope['get_uncertain_point_coords_on_grid']=original_selector;scope['calculate_uncertainty']=original_uncertainty
  hook.remove()
  metrics={m:base.summarize_rows(r) for m,r in rows.items()}
  assert metrics[native_mode]['counts']==expected['counts']
  assert abs(metrics[native_mode]['macro']['dice']-expected['macro']['dice'])<1e-12
  comparisons={m:compare(rows['coarse_stepwise'],rows[m]) for m in modes if m!='coarse_stepwise'}
  baseline_comparisons={m:compare(baseline,rows[m]) for m in modes}
  change_sums={m:{k:sum(r[k] for r in rr) for k in rr[0] if k!='id'} for m,rr in transitions.items()}
  stage_sums={}
  for mode in selectors:
   stage_sums[mode]={}
   for s in [1,2]:
    rr=[r for r in stage_rows if r['mode']==mode and r['stage']==s]
    stage_sums[mode][str(s)]={k:sum(r[k] for r in rr) for k in rr[0] if k not in ['id','mode','stage','resolution']}
  outdir=ROOT/arm;outdir.mkdir(exist_ok=True)
  for m in modes:wr(outdir/f'{m}_per_image.csv',rows[m])
  for m,rr in transitions.items():wr(outdir/f'{m}_transitions.csv',rr)
  wr(outdir/'point_stage_per_image.csv',stage_rows)
  result=dict(checkpoint_sha256=cksha,best_epoch=epoch,native_replay_exact=True,metrics=metrics,
   inference_vs_stepwise=comparisons,vs_baseline=baseline_comparisons,transitions=change_sums,point_stages=stage_sums)
  dump(outdir/'summary.json',result);results[arm]=result;assert digest(ck)==cksha
  del model,head,scope,features,coarse,direct,step,pred,out,f1,f2,mid,f3,f4
  gc.collect();torch.cuda.empty_cache()
 dump(ROOT/'summary.json',dict(protocol=cfg,baseline=base.summarize_rows(baseline),arms=results,complete=True))
 dump(ROOT/'SOURCE_HASHES.json',{str(p):digest(p) for p in [ROOT/'diagnose.py',ROOT/'PROTOCOL.json',SOURCE/'author_modules.py']})
 dump(ROOT/'DONE.json',dict(complete=True,validation_only=True,n=762,training=False,test_read=False))
 status('complete',validation_only=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc()));status('failed');raise
