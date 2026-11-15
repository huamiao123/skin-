from pathlib import Path
import sys,json,gc,csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy import ndimage
ROOT=Path(__file__).parent
P=Path('/home/waas/mambavision_stage22/source/referring_image_segmentation/stage36_topconf_skip_fusion')
sys.path.insert(0,str(P))
from train import System,base,digest
from evaluate import helpers
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def spatial(item):
 sid,y,preds=item;dist=ndimage.distance_transform_edt(~y);out={}
 for arm,pred in preds.items():
  fp=pred&~y;fn=(~pred)&y
  labels,n=ndimage.label(pred,structure=np.ones((3,3),dtype=np.uint8))
  sizes=np.bincount(labels.ravel());hit=np.unique(labels[y&pred]);nohit=np.ones(len(sizes),dtype=bool);nohit[0]=False;nohit[hit]=False
  out[arm]=dict(id=sid,gt_pixels=int(y.sum()),pred_pixels=int(pred.sum()),fp=int(fp.sum()),fn=int(fn.sum()),
    fp_within3=int((fp&(dist<=3)).sum()),fp_3to10=int((fp&(dist>3)&(dist<=10)).sum()),fp_beyond10=int((fp&(dist>10)).sum()),
    wholly_disjoint_component_pixels=int(sizes[nohit].sum()),wholly_disjoint_component_count=int(nohit.sum()))
 return out
@torch.inference_mode()
def main():
 base.setup()
 protocol=json.loads((ROOT/'PROTOCOL.json').read_text());arms=protocol['arms'];ths=protocol['thresholds']
 ids=json.loads((P/'identity_partition.json').read_text())['splits']['val']
 xs,ys=helpers.load_data(ids,P.parent/'data/isic2018/val/images',P.parent/'data/isic2018/val/masks')
 masks=ys>=127.5
 pred={}
 result={}
 mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
 for arm in arms:
  done=json.loads((P/arm/'results/summary.json').read_text());ck=Path(done['selection']['checkpoint'])
  assert digest(ck)==done['selection']['sha256']
  m=System(arm).cuda().eval();state=torch.load(ck,map_location='cpu');m.load_state_dict(state['model'],strict=True);del state
  scores={t:[] for t in ths};counts={t:np.zeros(3,dtype=np.int64) for t in ths}
  predictions=[];nll=0.;brier=0.;n_pixels=0
  bins_n=torch.zeros(15,device='cpu',dtype=torch.float64);bins_p=bins_n.clone();bins_y=bins_n.clone()
  for begin in range(0,len(ids),64):
   x=torch.from_numpy(xs[begin:begin+64]).permute(0,3,1,2).contiguous().cuda().float()/255
   y=torch.from_numpy(masks[begin:begin+64]).cuda();yf=y.float()
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=m((x-mean)/std)[:,0]
   logits=logits.float();probs=logits.sigmoid()
   nll+=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,yf,reduction='sum'))
   brier+=float((probs-yf).square().sum());n_pixels+=y.numel()
   bi=(probs*15).long().clamp(max=14).flatten().cpu()
   bins_n+=torch.bincount(bi,minlength=15);bins_p+=torch.bincount(bi,weights=probs.double().flatten().cpu(),minlength=15);bins_y+=torch.bincount(bi,weights=yf.double().flatten().cpu(),minlength=15)
   for t in ths:
    pr=probs>=t
    tp=(pr&y).sum((1,2)).double();fp=(pr&~y).sum((1,2)).double();fn=(~pr&y).sum((1,2)).double()
    dice=2*tp/(2*tp+fp+fn).clamp_min(1)
    scores[t].extend(dice.cpu().tolist());counts[t]+=np.array([int(tp.sum()),int(fp.sum()),int(fn.sum())])
    if t==.5:predictions.append(pr.cpu().numpy())
  pred[arm]=np.concatenate(predictions)
  official=done['validation'][arm]
  assert abs(np.mean(scores[.5])-official['macro']['dice'])<1e-10
  assert counts[.5].tolist()==[official['counts'][k] for k in ['tp','fp','fn']]
  table=[dict(threshold=t,macro_dice=float(np.mean(scores[t])),pooled_dice=float(2*counts[t][0]/(2*counts[t][0]+counts[t][1]+counts[t][2])),tp=int(counts[t][0]),fp=int(counts[t][1]),fn=int(counts[t][2])) for t in ths]
  ece=float((bins_p-bins_y).abs().sum()/n_pixels)
  result[arm]=dict(checkpoint_sha256=digest(ck),epoch=done['selection']['epoch'],fixed_threshold_replay_exact=True,
   thresholds=table,best_on_scanned_validation=max(table,key=lambda q:q['macro_dice']),nll=nll/n_pixels,brier=brier/n_pixels,ece15=ece)
  dump(ROOT/'threshold_and_calibration.json',result)
  print('INFERENCE_COMPLETE',arm,json.dumps(result[arm]['best_on_scanned_validation']),flush=True)
  del m,probs,logits,x,y,yf;gc.collect();torch.cuda.empty_cache()
 with ThreadPoolExecutor(8) as pool:
  rows=list(pool.map(spatial,[(sid,masks[i],{a:pred[a][i] for a in arms}) for i,sid in enumerate(ids)]))
 totals={}
 for arm in arms:
  records=[r[arm] for r in rows]
  with (ROOT/(arm+'_spatial_errors.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  totals[arm]={k:sum(r[k] for r in records) for k in records[0] if k!='id'}
 dump(ROOT/'spatial_totals.json',totals)
 dump(ROOT/'DONE.json',dict(complete=True,validation_n=len(ids),test_read=False,trained=False))
 print('SPATIAL',json.dumps(totals),flush=True)
if __name__=='__main__':main()
