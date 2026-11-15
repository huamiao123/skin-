from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
OUT=ROOT/'diagnosis';OUT.mkdir(exist_ok=True)
ids=json.loads((ROOT/'identity_partition.json').read_text())['splits']['val']
gt=np.stack([np.asarray(Image.open(P/'data/isic2018/val/masks'/f'{sid}.png').convert('L'))>=127.5 for sid in ids])
assert gt.shape==(762,256,256)
def read_preds(path):
 z=np.load(path);assert list(z['ids'])==ids
 return np.unpackbits(z['predictions_packed'],axis=1)[:,:256*256].reshape(-1,256,256).astype(bool)
def profile(pred,target,sid):
 fp=pred&~target;fn=target&~pred
 distance=ndi.distance_transform_edt(~target);inside=ndi.distance_transform_edt(target)
 lab,n=ndi.label(pred,np.ones((3,3),dtype=int));disconnected=np.zeros_like(pred)
 for k in range(1,n+1):
  comp=lab==k
  if not (comp&target).any():disconnected|=comp
 row=dict(id=sid,gt_area=int(target.sum()),pred_area=int(pred.sum()),
 tp=int((pred&target).sum()),fp=int(fp.sum()),fn=int(fn.sum()),
 fp_near3=int((fp&(distance<=3)).sum()),fp_far3=int((fp&(distance>3)).sum()),
 disconnected_fp=int(disconnected.sum()),connected_fp=int((fp&~disconnected).sum()),
 fn_boundary3=int((fn&(inside<=3)).sum()),fn_deep3=int((fn&(inside>3)).sum()))
 row.update(dice=2*row['tp']/max(row['pred_area']+row['gt_area'],1),
 recall=row['tp']/max(row['gt_area'],1),precision=row['tp']/max(row['pred_area'],1),
 far_fp_over_gt=row['fp_far3']/max(row['gt_area'],1))
 return row
paths={
 'uper64':P/'stage45_upernet_decoder/uper64/results/uper64_val_predictions.npz',
 'position_bias':P/'stage47_original_window_swin/position_bias/results/position_bias_val_predictions.npz',
 'shifted_windows':P/'stage47_original_window_swin/shifted_windows/results/shifted_windows_val_predictions.npz'}
allrows={};preds={};summary={'validation_only':True,'n':762,'rules':{
'near_boundary':'FP distance to GT <=3 pixels; FN distance to GT exterior <=3 pixels; exploratory fixed pixel band, not official boundary-F1',
'disconnected':'8-connected predicted components with zero GT overlap; no minimum-size threshold',
'limitations':'Connectivity and distance categories are proxies, not clinical/semantic cause labels; no hair annotation used.'},'models':{}}
for arm,path in paths.items():
 pred=read_preds(path);preds[arm]=pred
 rows=[profile(a,b,sid) for a,b,sid in zip(pred,gt,ids)];allrows[arm]=rows
 with (OUT/f'{arm}_error_profiles.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 sums={k:sum(row[k] for row in rows) for k in ['tp','fp','fn','fp_near3','fp_far3','disconnected_fp','connected_fp','fn_boundary3','fn_deep3']}
 sums.update(macro_dice=float(np.mean([a['dice'] for a in rows])),severe_undercoverage=sum(a['recall']<.5 for a in rows),
 disconnected_fp_fraction=sums['disconnected_fp']/sums['fp'],far_fp_fraction=sums['fp_far3']/sums['fp'],deep_fn_fraction=sums['fn_deep3']/sums['fn'])
 summary['models'][arm]=sums
 assert sum(sums[x] for x in ['fp_near3','fp_far3'])==sums['fp']
 assert sum(sums[x] for x in ['fn_boundary3','fn_deep3'])==sums['fn']
baseline=allrows['uper64']
groups={'far_fp':sorted(range(len(ids)),key=lambda i:baseline[i]['far_fp_over_gt'],reverse=True)[:6],
'undercoverage':sorted(range(len(ids)),key=lambda i:baseline[i]['recall'])[:6]}
summary['selected_cases']={k:[ids[i] for i in ix] for k,ix in groups.items()}
for group,ix in groups.items():
 fig,axes=plt.subplots(2,3,figsize=(12,8))
 for ax,i in zip(axes.flat,ix):
  im=Image.open(P/'data/isic2018/val/images'/f'{ids[i]}.png').convert('RGB');ax.imshow(im)
  ax.contour(gt[i],levels=[.5],colors=['cyan'],linewidths=1)
  ax.contour(preds['uper64'][i],levels=[.5],colors=['red'],linewidths=1)
  rr=baseline[i];ax.set_title(f"{ids[i]}  D={rr['dice']:.3f}\nP={rr['precision']:.3f} R={rr['recall']:.3f}",fontsize=10);ax.axis('off')
 fig.suptitle(f'Validation UPer64: {group}; cyan=GT, red=prediction',fontsize=14);fig.tight_layout()
 fig.savefig(OUT/f'{group}.png',dpi=130)
 fig.savefig(OUT/f'{group}_preview.jpg',dpi=75,pil_kwargs={'quality':65});plt.close(fig)
summary['source_sha256']={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths.values()}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
