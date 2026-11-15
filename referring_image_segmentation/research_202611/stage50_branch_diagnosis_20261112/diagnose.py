from pathlib import Path
import sys,json,gc,datetime
import numpy as np
import torch
R=Path('/home/waas/mambavision_stage22');P=R/'source/referring_image_segmentation';S=P/'stage50_mamba_spatial_adapter'
D=R/'research/stage50_branch_diagnosis_20261112'
sys.path.insert(0,str(P));sys.path.insert(0,str(S))
from train import System,base,digest,dump
from evaluate import helpers,write_rows,compare
@torch.no_grad()
def main():
 base.setup();ids=json.loads((S/'identity_partition.json').read_text())['splits']['val']
 data=helpers.load_data(ids,P/'data/isic2018/val/images',P/'data/isic2018/val/masks')
 report=dict(validation_only=True,test_read=False,n=len(ids),arms={},interpretation='Fixed-checkpoint interventions are conditional contribution checks, not retraining ablations.')
 for arm in ['cab','conv_control']:
  ck=S/arm/'results'/f'{arm}_best_dice.pth';sha=digest(ck);state=torch.load(ck,map_location='cpu')
  m=System(arm).cuda().eval();mlp=m.encoder.model.levels[2].blocks[3].mlp;a=mlp.spatial_adapter
  init=base.cpu_tree(a.cab.state_dict());m.load_state_dict(state['model'],strict=True);m.requires_grad_(False)
  trained=base.cpu_tree(a.state_dict());learn={}
  for k,v in a.cab.state_dict().items():
   z=v.detach().cpu().double();ref=init[k].double()
   learn[k]=dict(norm=float(z.norm()),change_norm=float((z-ref).norm()),
       relative_change=float((z-ref).norm()/ref.norm().clamp_min(1e-30)))
  stats=[];stash={}
  def capture_raw(module,inputs,out):stash['raw']=out.detach()
  def capture_delta(module,inputs,out):
   def real(v):return a.reverse(v,14,28,28)[:,:,:16,:16].float().flatten(1)
   d=real(out);raw=real(stash.pop('raw'));inp=real(inputs[0])
   for x,y,z in zip(d,raw,inp):
    stats.append([float(x.norm()/y.norm().clamp_min(1e-20)),float(x.norm()/z.norm().clamp_min(1e-20)),
      float((x.abs()>0).float().mean())])
  h1=mlp.fc2.register_forward_hook(capture_raw);h2=a.register_forward_hook(capture_delta)
  metric,rows=helpers.evaluate(m,data,ids,base)
  h1.remove();h2.remove()
  prior=json.loads((S/arm/'results'/f'{arm}_validation_metrics.json').read_text())
  assert metric['counts']==prior['counts'] and abs(metric['macro']['dice']-prior['macro']['dice'])<1e-12
  stats=np.asarray(stats);assert len(stats)==762
  res=dict(checkpoint=str(ck),sha256=sha,epoch=state['epoch'],trained=metric,
    scale_abs_mean=float(a.scale.abs().mean()),scale_abs_max=float(a.scale.abs().max()),
    branch_relative_to_mlp_median=float(np.median(stats[:,0])),
    branch_relative_to_mlp_p95=float(np.quantile(stats[:,0],.95)),
    branch_relative_to_norm2_input_median=float(np.median(stats[:,1])),
    layer_parameter_changes=learn)
  a.scale.zero_();byp,br=helpers.evaluate(m,data,ids,base)
  write_rows(D/f'{arm}_bypass_validation.csv',br)
  a.load_state_dict(trained);a.cab.load_state_dict(init)
  reinit,rr=helpers.evaluate(m,data,ids,base)
  write_rows(D/f'{arm}_reinit_features_validation.csv',rr)
  a.load_state_dict(trained);assert digest(ck)==sha
  res.update(bypass=byp,reinitialized_features_keep_trained_scale=reinit,
    trained_to_bypass=compare(rows,br),trained_to_reinit_features=compare(rows,rr))
  report['arms'][arm]=res;dump(D/'RESULTS.json',report)
  print(arm,json.dumps({k:v for k,v in res.items() if k not in ['layer_parameter_changes','trained','bypass','reinitialized_features_keep_trained_scale']}),
    'dice',metric['macro']['dice'],byp['macro']['dice'],reinit['macro']['dice'],flush=True)
  del m,a,mlp,state,trained,init;gc.collect();torch.cuda.empty_cache()
 report['complete']=True;report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 dump(D/'RESULTS.json',report)
if __name__=='__main__':main()
