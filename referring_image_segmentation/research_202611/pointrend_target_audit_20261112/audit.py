from pathlib import Path
import sys,json,hashlib,ast,datetime
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).parent
PROJECT=ROOT.parent.parent/'source/referring_image_segmentation'
sys.path.insert(0,str(PROJECT/'stage42_frozen_pointrend'))
from train import System,setup,load_data,base,digest
from author_modules import make_scope
@torch.no_grad()
def main():
 setup();m=System().cuda().eval();data,_=load_data();scope,_=make_scope(False)
 total=dict(images=0,pixels=0,gray_pixels=0,threshold05_fg=0,old_hard_fg=0,disagreement=0,
  sampled_points=0,sampled_correct_fg=0,sampled_old_fg=0,sampled_disagreement=0)
 rows=[]
 for start in range(0,len(data.ids['train']),64):
  ids=torch.arange(start,min(start+64,len(data.ids['train'])),device='cuda')
  x,y=data.batch('train',ids)
  old=y[:,0].long();new=(y[:,0]>=.5).long()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   fine,coarse=m.frozen_features(x)
   xy=scope['get_uncertain_point_coords_with_randomness'](coarse,scope['calculate_uncertainty'],1024,3,.75)
   a=scope['point_sample'](old[:,None].float(),xy,mode='nearest',align_corners=False)[:,0].long()
   b=scope['point_sample'](new[:,None].float(),xy,mode='nearest',align_corners=False)[:,0].long()
  assert bool((old<=new).all())
  total['images']+=len(ids);total['pixels']+=y.numel();total['gray_pixels']+=int(((y>0)&(y<1)).sum())
  total['threshold05_fg']+=int(new.sum());total['old_hard_fg']+=int(old.sum());total['disagreement']+=int((old!=new).sum())
  total['sampled_points']+=a.numel();total['sampled_correct_fg']+=int(b.sum());total['sampled_old_fg']+=int(a.sum())
  total['sampled_disagreement']+=int((a!=b).sum())
  for i,index in enumerate(ids.cpu().tolist()):
   rows.append(dict(id=data.ids['train'][index],pixels_disagree=int((old[i]!=new[i]).sum()),
    sampled_disagree=int((a[i]!=b[i]).sum())))
  print('PROCESSED',total['images'],flush=True)
 # Verify coarse-loss path uses integer point targets in joint models.
 source=PROJECT/'stage40_pointrend/author_modules.py'
 author=PROJECT.parent.parent/'research/pointrend_author_20261112/official_code/projects/PointRend/point_rend/semantic_seg.py'
 assert 'targets[:,0].long()' in source.read_text()
 assert 'self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets)' in author.read_text()
 source42=PROJECT/'stage42_frozen_pointrend/train.py'
 assert 'target[:,0].long()' in source42.read_text()
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False,training=False,
  training_masks=total,affected_foreground_fraction=total['disagreement']/total['threshold05_fg'],
  sampled_disagreement_fraction=total['sampled_disagreement']/total['sampled_points'],
  affected_sampled_fg_fraction=total['sampled_disagreement']/total['sampled_correct_fg'],
  sampled_geometry='unaugmented training images; fixed baseline uncertainty sampling1024,oversample3,importance.75; does not estimate every augmented epoch',
  code=dict(stage40_source_sha256=digest(source),stage42_source_sha256=digest(source42),author_semantic_sha256=digest(author)),
  findings=['Float mask.long truncates fractional positives instead of thresholding0.5',
   'Stage40 coarse loss receives truncated integer target, diverging from the original soft-label coarse BCE+Dice',
   'Stage42 backbone is fixed; wrong point CE targets remain, but cannot change baseline weights',
   'Historical measured scores remain records of executed code; prior claims about faithful PointRend adaptation must be qualified'],
  minimal_fix=['Point CE target=(original_target>=.5).long()',
   'For joint training preserve original soft target exclusively for existing coarse loss; do not feed the integer point target into original BCE+Dice',
   'Retest frozen baseline head-only corrected labels first; no new attention or LR changes'])
 (ROOT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
 import csv
 with (ROOT/'train_per_image.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
