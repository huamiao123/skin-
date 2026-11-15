"""Read-only checkpoint diagnostics; waits for Stage27 training AND analysis to exit."""
import os, sys, time, json, pathlib, subprocess, datetime, traceback, csv
ROOT=pathlib.Path('/home/waas/mambavision_stage22')
PROJECT=ROOT/'source/referring_image_segmentation'
OUT=ROOT/'research/early_peak_20261110/calibration_audit'
OUT.mkdir(exist_ok=True)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name,obj):
 f=OUT/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(f)
def running(pid,start):
 try:
  s=pathlib.Path(f'/proc/{pid}/stat').read_text().split()
  return s[2]!='Z' and s[21]==start
 except FileNotFoundError:return False
def main():
 parent=22018
 p=pathlib.Path(f'/proc/{parent}/stat')
 start=p.read_text().split()[21] if p.exists() else None
 dump('status.json',dict(state='waiting_for_stage27_pipeline',pid=os.getpid(),parent_pid=parent,time=stamp()))
 while start and running(parent,start):time.sleep(20)
 stage=PROJECT/'stage27_coverage_supervision'
 if (stage/'pipeline_failure.json').exists():raise RuntimeError('Stage27 pipeline failed; diagnostic not started.')
 if not (stage/'results/DONE.json').exists():raise RuntimeError('Stage27 is not complete; diagnostic not started.')
 import numpy as np
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(PROJECT))
 from stage22_partial_finetune.train import System, setup, tensor_hash
 from stage26_coverage_probes.run import prepare_data
 setup();data,_=prepare_data();results=[]
 for stage_name in ['stage22_partial_finetune','stage24_seed43_confirmation']:
  for criterion in ['best_loss','best_dice','last']:
   tag=stage_name+'_'+criterion
   dump('status.json',dict(state='evaluating',tag=tag,pid=os.getpid(),time=stamp()))
   checkpoint=PROJECT/stage_name/'results'/('finetune34_'+criterion+'.pth')
   saved=torch.load(checkpoint,map_location='cpu')
   model=System(False).cuda();model.load_state_dict(saved['model'],strict=True)
   model.requires_grad_(False);model.eval();before=tensor_hash(model)
   hist=json.loads((PROJECT/stage_name/'results/history.json').read_text())
   reference=(min(hist,key=lambda r:r['validation']['finetune34']['loss']) if criterion=='best_loss' else max(hist,key=lambda r:r['validation']['finetune34']['macro']['dice']) if criterion=='best_dice' else hist[-1])
   bins_n=np.zeros(15);bins_p=np.zeros(15);bins_y=np.zeros(15);rows=[]
   counts=np.zeros(4,dtype=np.int64)
   with torch.inference_mode():
    for begin in range(0,762,64):
     idx=torch.arange(begin,min(begin+64,762),device='cuda');images,gt=data.batch('val',idx)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      f1,f2,mid=model.prefix(images);f3,f4=model.suffix(mid.to(torch.bfloat16))
      logits=model.probe([f1.to(torch.bfloat16),f2.to(torch.bfloat16),f3,f4])
     z=logits.float();prob=z.sigmoid();truth=gt>=.5;pred=prob>=.5
     bce=F.binary_cross_entropy_with_logits(z,gt.float(),reduction='none').mean((1,2,3))
     softdice=1-(2*(prob*gt).sum((1,2,3))+1)/(prob.sum((1,2,3))+gt.sum((1,2,3))+1)
     nll=F.binary_cross_entropy_with_logits(z,truth.float(),reduction='none').mean((1,2,3))
     brier=((prob-truth.float())**2).mean((1,2,3))
     batch_counts=torch.stack([(pred&truth).sum((1,2,3)),(pred&~truth).sum((1,2,3)),(~pred&truth).sum((1,2,3)),(~pred&~truth).sum((1,2,3))],1).cpu().numpy()
     counts+=batch_counts.sum(0)
     pnp=prob[:,0].cpu().numpy();ynp=truth[:,0].cpu().numpy()
     stats=torch.stack([bce,softdice,nll,brier],1).cpu().numpy()
     for j,(pp,yy) in enumerate(zip(pnp,ynp)):
      pp=pp.flatten().astype(np.float64);yy=yy.flatten();label=pp>=.5;wrong=label!=yy
      confidence=np.where(label,pp,1-pp)
      b=np.minimum((pp*15).astype(int),14);bn=np.bincount(b,minlength=15)
      bp=np.bincount(b,weights=pp,minlength=15);by=np.bincount(b,weights=yy,minlength=15)
      bins_n+=bn;bins_p+=bp;bins_y+=by
      tp,fp,fn,tn=map(int,batch_counts[j]);wrong_count=int(wrong.sum())
      rows.append(dict(id=data.ids['val'][begin+j],tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/max(2*tp+fp+fn,1),bce_soft=float(stats[j,0]),soft_dice_loss=float(stats[j,1]),nll_binary=float(stats[j,2]),brier_binary=float(stats[j,3]),ece15_foreground_probability=float(np.abs(bp-by).sum()/len(pp)),wrong_pixels=wrong_count,wrong_confidence_sum=float(confidence[wrong].sum()),wrong_confidence_gt99=int((wrong&(confidence>.99)).sum()),wrong_confidence_gt999=int((wrong&(confidence>.999)).sum())))
   ref=reference['validation']['finetune34']
   assert counts.tolist()==[ref['counts'][k] for k in ['tp','fp','fn','tn']],(tag,counts,ref['counts'])
   total_loss=float(np.mean([r['bce_soft']+r['soft_dice_loss'] for r in rows]))
   assert abs(total_loss-ref['loss'])<2e-6,(tag,total_loss,ref['loss'])
   assert before==tensor_hash(model)
   wrong=sum(r['wrong_pixels'] for r in rows)
   report=dict(tag=tag,checkpoint=str(checkpoint),epoch=reference['epoch'],source_model_unchanged=True,original_counts_exact=True,original_loss_reproduced=total_loss,macro_dice=float(np.mean([r['dice'] for r in rows])),bce_soft=float(np.mean([r['bce_soft'] for r in rows])),soft_dice_loss=float(np.mean([r['soft_dice_loss'] for r in rows])),nll_binary=float(np.mean([r['nll_binary'] for r in rows])),brier_binary=float(np.mean([r['brier_binary'] for r in rows])),ece15_pooled_foreground_probability=float(np.abs(bins_p-bins_y).sum()/bins_n.sum()),ece15_macro_foreground_probability=float(np.mean([r['ece15_foreground_probability'] for r in rows])),error_pixel_fraction=wrong/(762*256*256),mean_confidence_on_wrong_pixels=sum(r['wrong_confidence_sum'] for r in rows)/wrong,wrong_confidence_gt99_fraction=sum(r['wrong_confidence_gt99'] for r in rows)/wrong,wrong_confidence_gt999_fraction=sum(r['wrong_confidence_gt999'] for r in rows)/wrong,bins=dict(count=bins_n.tolist(),prob_sum=bins_p.tolist(),positive_count=bins_y.tolist()))
   results.append(report);dump('results.json',results)
   with (OUT/(tag+'_per_image.csv')).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   print(tag,json.dumps({k:v for k,v in report.items() if k not in ['bins','checkpoint']}),flush=True)
   del model,saved;torch.cuda.empty_cache()
 dump('DONE.json',dict(time=stamp(),n_checkpoints=len(results),test_used=False,training_performed=False))
 dump('status.json',dict(state='complete',time=stamp()))
if __name__=='__main__':
 try:main()
 except Exception:
  dump('failure.json',dict(time=stamp(),traceback=traceback.format_exc()));raise
