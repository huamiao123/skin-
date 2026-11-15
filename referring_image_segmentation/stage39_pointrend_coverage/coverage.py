from pathlib import Path
import sys,json,csv,datetime,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'stage37_lr_horizon100'))
from train import System,base,digest
from evaluate import helpers
sys.path.insert(0,str(ROOT))
from author_sampling import select_grid,uncertainty,PROVENANCE
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def status(state,**kw):
 obj=dict(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);dump(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
def wr(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
@torch.inference_mode()
def main():
 base.setup();cfg=json.loads((ROOT/'PROTOCOL.json').read_text());budgets=cfg['budgets']
 source=PROJECT/'stage37_lr_horizon100';ids=json.loads((source/'identity_partition.json').read_text())['splits']['val']
 ck=PROJECT/cfg['reference'];sha=digest(ck);state=torch.load(ck,map_location='cpu')
 for rel,expected in state['config']['sources'].items():assert digest(PROJECT/rel)==expected,rel
 expected=state['metric'];model=System('swin_control').cuda().eval();model.load_state_dict(state['model'],strict=True);model.requires_grad_(False);del state
 xs,ys=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
 captures={}
 def hook(m,args,out):captures['coarse']=out
 handle=model.probe.decoder.baseline.readout.register_forward_hook(hook)
 original_rows=[];step_rows=[];all_rows={str(p):[] for p in budgets};ideal_rows={str(p):[] for p in budgets}
 fixed=np.load(source/'swin_control/results/swin_control_val_predictions.npz')
 assert np.array_equal(fixed['ids'],ids)
 mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
 generators=[torch.Generator(device='cuda').manual_seed(s) for s in [39001,39002,39003]]
 totals=dict(total_errors=0,confident95_errors=0,confident99_errors=0)
 with ThreadPoolExecutor(8) as pool:
  for begin in range(0,len(ids),64):
   x=torch.from_numpy(xs[begin:begin+64]).permute(0,3,1,2).contiguous().cuda().float()/255
   y=torch.from_numpy(ys[begin:begin+64]>=127.5).cuda();n=len(x)
   with torch.autocast('cuda',dtype=torch.bfloat16):native=model((x-mean)/std)
   base_pred=native.float().sigmoid()[:,0]>=.5
   packed=np.packbits(base_pred.cpu().numpy().reshape(n,-1),axis=1)
   assert np.array_equal(packed,fixed['predictions_packed'][begin:begin+n])
   lo=captures['coarse'].float();two=torch.cat([torch.zeros_like(lo),lo],dim=1)
   assert torch.allclose(two.softmax(1)[:,1],lo[:,0].sigmoid(),rtol=1e-6,atol=1e-7)
   up128=F.interpolate(two,scale_factor=2,mode='bilinear',align_corners=False)
   up256=F.interpolate(up128,scale_factor=2,mode='bilinear',align_corners=False)
   pred=up256[:,1]>=up256[:,0];error=pred!=y;fp=pred&~y;fn=~pred&y
   confidence=(up256[:,1]-up256[:,0]).abs().sigmoid()
   totals['total_errors']+=int(error.sum());totals['confident95_errors']+=int((error&(confidence>=.95)).sum());totals['confident99_errors']+=int((error&(confidence>=.99)).sum())
   random_ranks=[torch.rand((n,256*256),device='cuda',generator=g).topk(max(budgets),dim=1).indices for g in generators]
   def metric_rows(pr):
    rr=list(pool.map(lambda z:base.boundary_row(*z),zip(pr.cpu().numpy(),y.cpu().numpy())))
    for sid,row in zip(ids[begin:begin+n],rr):row['id']=sid
    return rr
   original_rows.extend(metric_rows(base_pred));step_rows.extend(metric_rows(pred))
   for budget in budgets:
    i128,_=select_grid(uncertainty(up128),budget);i256,_=select_grid(uncertainty(up256),budget)
    mask=torch.zeros((n,256*256),device='cuda',dtype=torch.bool).scatter_(1,i256,True).view(n,256,256)
    m128=torch.zeros((n,128*128),device='cuda',dtype=torch.float32).scatter_(1,i128,1.).view(n,1,128,128)
    footprint=(F.interpolate(m128,scale_factor=2,mode='bilinear',align_corners=False)[:,0]>0)|mask
    ideal=torch.where(mask,y,pred)
    ideal_rows[str(budget)].extend(metric_rows(ideal))
    err=error.flatten(1);fpf=fp.flatten(1);fnf=fn.flatten(1)
    random_hits=[err.gather(1,idx[:,:budget]).sum(1).cpu().tolist() for idx in random_ranks]
    for j,sid in enumerate(ids[begin:begin+n]):
     all_rows[str(budget)].append(dict(id=sid,errors=int(error[j].sum()),fp=int(fp[j].sum()),fn=int(fn[j].sum()),
       selected=budget,selected_errors=int((error[j]&mask[j]).sum()),selected_fp=int((fp[j]&mask[j]).sum()),selected_fn=int((fn[j]&mask[j]).sum()),
       footprint_pixels=int(footprint[j].sum()),footprint_errors=int((footprint[j]&error[j]).sum()),
       random_selected_errors_mean=sum(h[j] for h in random_hits)/3))
   status('running',processed=begin+n,n=len(ids))
 native_metric=base.summarize_rows(original_rows);steps_metric=base.summarize_rows(step_rows)
 assert native_metric['counts']==expected['counts'] and abs(native_metric['macro']['dice']-expected['macro']['dice'])<1e-12
 result=dict(original=native_metric,stepwise_bilinear=steps_metric,confidence=totals,budgets={},checkpoint_sha256=sha)
 for budget in budgets:
  rows=all_rows[str(budget)];wr(ROOT/f'coverage_{budget}_per_image.csv',rows);wr(ROOT/f'ideal_{budget}_per_image.csv',ideal_rows[str(budget)])
  sums={k:sum(r[k] for r in rows) for k in rows[0] if k!='id'}
  ideal=base.summarize_rows(ideal_rows[str(budget)])
  hit=sums['selected_errors']/max(sums['errors'],1);rand=sums['random_selected_errors_mean']/max(sums['errors'],1)
  result['budgets'][str(budget)]=dict(totals=sums,error_coverage=hit,random_error_coverage=rand,enrichment=hit/max(rand,1e-15),
    fp_coverage=sums['selected_fp']/max(sums['fp'],1),fn_coverage=sums['selected_fn']/max(sums['fn'],1),
    footprint_error_coverage=sums['footprint_errors']/max(sums['errors'],1),ideal_fixed_set_metrics=ideal,
    ideal_gain_vs_stepwise_pp=100*(ideal['macro']['dice']-steps_metric['macro']['dice']))
 a=result['budgets'][str(cfg['primary_budget'])]
 result['pilot_screen_passed']=a['error_coverage']>=.1 and a['ideal_gain_vs_stepwise_pp']>=.5 and a['enrichment']>=2
 dump(ROOT/'AUTHOR_PROVENANCE.json',PROVENANCE);dump(ROOT/'summary.json',result)
 dump(ROOT/'DONE.json',dict(complete=True,validation_only=True,training=False,pilot_screen_passed=result['pilot_screen_passed']))
 assert digest(ck)==sha
 status('complete',pilot_screen_passed=result['pilot_screen_passed'])
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc()));status('failed');raise
