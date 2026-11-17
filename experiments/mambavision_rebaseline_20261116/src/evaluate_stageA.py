from pathlib import Path
import sys,json,csv,gc,hashlib,datetime,concurrent.futures
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evaluations/stageA_20261117';sys.path.insert(0,str(ROOT/'src'))
from model import PROJECT,SegmentationSystem,load_pretrained_encoder
from train_mb64 import reference,base,atomic_json,evaluate
from stage1_visual.analyze import metrics
from evaluation import summarize_rows
ARMS=['A-64-noaux','B-512-noaux','C-512-aux'];R=PROJECT.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootstrap(a,b):
 d=np.asarray(b)-np.asarray(a);g=np.random.default_rng(42)
 boot=[float(d[g.integers(0,len(d),len(d))].mean()) for _ in range(2000)]
 return dict(delta=float(d.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
  improved=int((d>0).sum()),worsened=int((d<0).sum()),note='Exploratory single-seed image intervals; not patient or seed uncertainty')
def profile(args):
 sid,prob,target=args;pred=prob>=.5;r=metrics(pred,target);ga=int(target.sum());pa=int(pred.sum())
 dist=edt(target);bgdist=edt(~target)
 r.update(id=sid,GT_area_px=ga,Pred_area_px=pa,Pred_to_GT_area_ratio=pa/max(ga,1),
  FN_interior_gt_distance_gt3=int((~pred&target&(dist>3)).sum()),
  FN_interior_gt_distance_gt8=int((~pred&target&(dist>8)).sum()),
  FN_boundary_band=int((~pred&target&(dist<=3)).sum()),
  FP_far_from_GT_gt8=int((pred&~target&(bgdist>8)).sum()),
  severe_undercoverage_flag=int(r['recall']<.5),mean_foreground_probability=float(prob[target].mean()) if ga else 0.,
  mean_background_probability=float(prob[~target].mean()) if (~target).any() else 0.,
  source_dataset='ISIC2018-derived',patient_id='unavailable',seed=42)
 return r
def stratify(rows,cuts):
 result={}
 for group in range(4):
  rs=[r for r in rows if int(np.searchsorted(cuts,r['GT_area_px'],side='right'))==group]
  if not rs:continue
  ratios=[r['Pred_to_GT_area_ratio'] for r in rs]
  result['Q'+str(group+1)]=dict(n=len(rs),macro_dice=float(np.mean([r['dice'] for r in rs])),
   pooled_dice=2*sum(r['tp'] for r in rs)/max(1,sum(2*r['tp']+r['fp']+r['fn'] for r in rs)),
   ratio_mean=float(np.mean(ratios)),ratio_quantiles=np.quantile(ratios,[.25,.5,.75,.95]).tolist(),
   FN=sum(r['fn'] for r in rs),FP=sum(r['fp'] for r in rs),
   FN_deep8=sum(r['FN_interior_gt_distance_gt8'] for r in rs),
   FP_far8=sum(r['FP_far_from_GT_gt8'] for r in rs),
   severe_undercoverage=sum(r['severe_undercoverage_flag'] for r in rs))
 return result
def main():
 base.setup();torch.use_deterministic_algorithms(True,warn_only=True)
 OUT.mkdir(exist_ok=True)
 partition_path=PROJECT/'stage45_upernet_decoder/identity_partition.json'
 part=json.loads(partition_path.read_text());assert part['counts']==dict(train=1886,val=762,test=867)
 manifest_path=R/'local_audit_records/test_assets/test_manifest.json'
 assert sha(manifest_path)==part['audit_sha256']['test_manifest.json']
 manifest={r['id']:r for r in json.loads(manifest_path.read_text())}
 records=[]
 for arm in ARMS:
  folder=ROOT/'results/formal_mb64'/arm;h=json.loads((folder/'history.json').read_text())
  assert len(h)==100 and json.loads((folder/'status.json').read_text())['state']=='complete'
  best=max(h,key=lambda r:r['validation']['macro']['dice'])
  records.append(dict(arm=arm,checkpoint=str(folder/'best_dice.pth'),sha256=sha(folder/'best_dice.pth'),
   selected_epoch=best['epoch'],validation_original=best['validation'],
   training_gpu=json.loads((folder/'config.json').read_text())['device']))
 prereg=dict(records=records,selection='pre-existing max validation macroDice earliest tie',threshold=.5,
  input_size=256,physical_eval_batch=64,partition_sha256=sha(partition_path),test_manifest_sha256=sha(manifest_path),
  test_n=867,test_previously_exposed=True,patient_disjointness_verified=False,test_used_for_selection=False,
  actual_gpu=torch.cuda.get_device_name(0),B_cross_gpu_resume=True)
 path=OUT/'FIXED_EVALUATION_PROTOCOL.json'
 if path.exists():assert json.loads(path.read_text())==prereg
 else:atomic_json(path,prereg)
 data,_=reference.load_data();vc=reference.cache_new(762)
 targets=(data.masks['val'][:,0].cpu().numpy()>=127.5)
 cuts=np.quantile(targets.sum((1,2)),[.25,.5,.75]);atomic_json(OUT/'AREA_CUTS.json',dict(validation_area_quartiles_px=cuts.tolist(),fixed_for_all_models=True))
 # Validate every test input before any test forward.
 testids=part['splits']['test']
 for sid in testids:
  for folder,key in [('images','image_256_sha256'),('masks','mask_256_sha256')]:
   assert sha(R/'heldout_data/test'/folder/(sid+'.png'))==manifest[sid][key],sid
 atomic_json(OUT/'TEST_INPUT_VERIFIED.json',dict(passed=True,n=867,all_image_mask_sha256_match=True))
 thresholds=np.arange(.2,.801,.05);summaries={};paired={}
 for rec in records:
  arm=rec['arm'];dest=OUT/arm;dest.mkdir(exist_ok=True);state=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)
  assert state['epoch']==rec['selected_epoch'];model=SegmentationSystem(load_pretrained_encoder(),arm,42).cuda()
  model.load_state_dict(state['model'],strict=True);del state;model.requires_grad_(False).eval()
  reference.fill_cache(model,data,'val',vc)
  probs=[]
  with torch.no_grad():
   for st in range(0,762,64):
    ix=torch.arange(st,min(762,st+64),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
     output=model.from_cache(vc['features'][0][ix],vc['features'][1][ix],vc['mid'][ix],(256,256))['main']
    probs.append(output.float().sigmoid()[:,0].cpu().numpy())
  prob=np.concatenate(probs);np.save(dest/'validation_probabilities.npy',prob.astype(np.float16))
  with concurrent.futures.ThreadPoolExecutor(8) as pool:
   rows=list(pool.map(profile,zip(part['splits']['val'],prob,targets)))
  val=summarize_rows(rows);csvout(dest/'validation_per_image.csv',rows)
  original=rec['validation_original']
  replay=dict(original_counts=original['counts'],actual_counts=val['counts'],
   counts_exact=original['counts']==val['counts'],macro_dice_delta=val['macro']['dice']-original['macro']['dice'],
   caveat='A trained on3090; B interrupted/resumed on4090; C on4090. Hardware and non-deterministic ops can cause replay differences.')
  atomic_json(dest/'VALIDATION_REPLAY.json',replay)
  sweep=[]
  for t in thresholds:
   pr=prob>=t;tp=(pr&targets).sum((1,2));fp=(pr&~targets).sum((1,2));fn=(~pr&targets).sum((1,2))
   dice=np.divide(2*tp,2*tp+fp+fn,out=np.ones(762),where=(2*tp+fp+fn)>0)
   ir=tp/max(1,int(tp.sum())) if False else None
   sweep.append(dict(threshold=float(round(t,2)),macro_dice=float(dice.mean()),pooled_dice=float(2*tp.sum()/max(1,2*tp.sum()+fp.sum()+fn.sum())),
    FP=int(fp.sum()),FN=int(fn.sum()),area_groups={str(g+1):dict(macro_dice=float(dice[np.searchsorted(cuts,targets.sum((1,2)),side='right')==g].mean()),
      FP=int(fp[np.searchsorted(cuts,targets.sum((1,2)),side='right')==g].sum()),FN=int(fn[np.searchsorted(cuts,targets.sum((1,2)),side='right')==g].sum())) for g in range(4)}))
  atomic_json(dest/'VALIDATION_THRESHOLD_SWEEP.json',dict(validation_only=True,candidates=sweep,best=max(sweep,key=lambda x:x['macro_dice']),test_calibrated=False))
  # Test uses the fixed0.5 threshold, regardless of threshold sweep.
  testrows=[]
  with torch.no_grad(),concurrent.futures.ThreadPoolExecutor(8) as pool:
   for st in range(0,867,64):
    ids=testids[st:st+64]
    xs=np.stack([np.asarray(Image.open(R/'heldout_data/test/images'/(sid+'.png')).convert('RGB')) for sid in ids])
    ys=np.stack([np.asarray(Image.open(R/'heldout_data/test/masks'/(sid+'.png')).convert('L'))>=127.5 for sid in ids])
    im=torch.from_numpy(xs).permute(0,3,1,2).contiguous().cuda().float()/255.
    with torch.autocast('cuda',dtype=torch.bfloat16):
     logit=model((im-data.mean)/data.std)['main']
    probability=logit.float().sigmoid()[:,0].cpu().numpy()
    testrows+=list(pool.map(profile,zip(ids,probability,ys)))
  test=summarize_rows(testrows);csvout(dest/'test_per_image.csv',testrows)
  summary=dict(arm=arm,checkpoint=rec,validation=val,test=test,validation_area_groups=stratify(rows,cuts),
   test_area_groups_using_validation_cuts=stratify(testrows,cuts),complete=True,threshold=.5,
   test_used_for_selection=False,patient_disjointness_verified=False)
  atomic_json(dest/'METRICS.json',summary);summaries[arm]=summary;paired[arm]=[r['dice'] for r in rows]
  assert sha(rec['checkpoint'])==rec['sha256']
  atomic_json(OUT/'SUMMARY.json',dict(complete=False,models=summaries))
  print('RESULT',arm,'val',val['macro']['dice'],'test',test['macro']['dice'],'replay',replay['macro_dice_delta'],flush=True)
  del model,prob,probs,rows,testrows;gc.collect();torch.cuda.empty_cache()
 pairs={a+'_vs_A':bootstrap(paired[ARMS[0]],paired[a]) for a in ARMS[1:]}
 pairs['C_vs_B']=bootstrap(paired[ARMS[1]],paired[ARMS[2]])
 atomic_json(OUT/'SUMMARY.json',dict(complete=True,models=summaries,validation_paired_comparisons=pairs,
  test_n_per_model=867,validation_n=762,seed=42,test_exposed=True))
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;atomic_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
