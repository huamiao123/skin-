"""Fair zero-offset preference calibration on reused development val50/val100."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from project_paths import PHASE1

import numpy as np

from evaluation import full_metrics,repair_metrics
from train_local import CACHE,ROOT
from contour.geometry import mask_contours,polygon_area,build_prediction_context,compose_prediction_context,rasterize

ALPHAS=(0.0,0.02,0.05,0.1,0.2,0.5,1.0,2.0)
METHODS=('local','local_large','local8','medium32','global')
SPLIT=PHASE1/'assets/val_split_seed17.csv'

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();seed=args.seed
    with SPLIT.open(newline='') as f:roles={r['image_id']:r['role'] for r in csv.DictReader(f)}
    records=[r for r in json.loads((CACHE/'COMPLETE.json').read_text())['records'] if r['split']=='val']
    names=[name for name in METHODS if (ROOT/'models'/f'{name}_seed{seed}'/'DONE.json').exists()]
    if not names:raise RuntimeError('No trained score model')
    scores={name:np.load(ROOT/'models'/f'{name}_seed{seed}'/'candidate_scores.npy',mmap_mode='r') for name in names}
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('prediction','gt')}
    cand={name:np.load(CACHE/'candidates_R32'/(name+'.npy'),mmap_mode='r') for name in ('points','valid','gt_distance','has_contour')}
    rows=[]
    for n,rec in enumerate(records,1):
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(cand['points'][i]);valid=np.asarray(cand['valid'][i],bool)
        contours=mask_contours(pred)
        context=build_prediction_context(pred,max(contours,key=lambda c:abs(polygon_area(c)))) if contours else None
        role=roles[rec['image_id']]
        for name in names:
            value=np.asarray(scores[name][i],np.float32)
            for alpha in ALPHAS:
                if context is None:mask=pred;selected=None
                else:
                    penalty=alpha*np.abs(np.arange(-32,33,dtype=np.float32))
                    selected=np.argmax(np.where(valid,value-penalty[None,:],-np.inf),axis=1)
                    mask=compose_prediction_context(rasterize(point[np.arange(256),selected],pred.shape),context)
                intersection=np.count_nonzero(mask&gt)
                dice=2*intersection/(mask.sum()+gt.sum()) if mask.sum()+gt.sum() else 1.
                rows.append({'image_id':rec['image_id'],'role':role,'model':name,'alpha':alpha,'dice':dice,
                             'moved_fraction':float((selected!=32).mean()) if selected is not None else None})
        if n%25==0:print(f'{n}/{len(records)}',flush=True)
    out=ROOT/'results/transformer'/f'keep_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_image_alpha_curve.csv',rows)
    summary=[]
    for name in names:
        curve=[]
        for alpha in ALPHAS:
            group=[r for r in rows if r['model']==name and r['alpha']==alpha]
            cal=[r['dice'] for r in group if r['role']=='calibration50']
            ver=[r['dice'] for r in group if r['role']=='locked_verification100']
            curve.append({'seed':seed,'model':name,'alpha':alpha,'cal50_dice':float(np.mean(cal)),
                          'val100_dice':float(np.mean(ver)),'all150_dice':float(np.mean([r['dice'] for r in group])),
                          'mean_moved_fraction':float(np.mean([r['moved_fraction'] for r in group if r['moved_fraction'] is not None]))})
        best=max(curve,key=lambda r:r['cal50_dice'])
        summary.extend(curve)
        print('CHOSEN',name,best,flush=True)
    write_csv(out/'summary.csv',summary)
    metadata={'scope':'ISIC2017 val50 selection and val100 reused development verification; val100 was used in previous phase-1 and DP development, so not independent',
              'selection':'cal50 mean Dice; ties smallest alpha','alphas':ALPHAS,
              'models':names,'official_test_GT_opened':False,
              'prior_dp_grid_val150_access':True,'interpretation':'alpha adds learned-score penalty proportional to absolute offset; zero remains selectable'}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
