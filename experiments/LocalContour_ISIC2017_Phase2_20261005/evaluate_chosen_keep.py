"""Apply cal50-selected movement penalties to reused val100; full metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from project_paths import PHASE1

import numpy as np

from evaluation import full_metrics,repair_metrics
from train_local import CACHE,ROOT
from contour.geometry import (mask_contours,polygon_area,build_prediction_context,
                              compose_prediction_context,rasterize)

SPLIT=PHASE1/'assets/val_split_seed17.csv'

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();seed=a.seed
    table=read(ROOT/'results/transformer'/f'keep_seed{seed}'/'summary.csv')
    models=sorted({r['model'] for r in table})
    chosen={model:max((r for r in table if r['model']==model),key=lambda r:float(r['cal50_dice'])) for model in models}
    roles={r['image_id']:r['role'] for r in read(SPLIT)}
    records=[r for r in json.loads((CACHE/'COMPLETE.json').read_text())['records'] if r['split']=='val']
    scores={name:np.load(ROOT/'models'/f'{name}_seed{seed}'/'candidate_scores.npy',mmap_mode='r') for name in models}
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('prediction','gt')}
    cand={name:np.load(CACHE/'candidates_R32'/(name+'.npy'),mmap_mode='r') for name in ('points','valid','gt_distance','has_contour')}
    output=ROOT/'results/transformer'/f'chosen_keep_seed{seed}'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n,rec in enumerate(records,1):
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(cand['points'][i]);valid=np.asarray(cand['valid'][i],bool)
        dist=np.asarray(cand['gt_distance'][i],np.float32)
        contours=mask_contours(pred)
        context=build_prediction_context(pred,max(contours,key=lambda x:abs(polygon_area(x)))) if contours else None
        role=roles[rec['image_id']]
        rows.append({'image_id':rec['image_id'],'role':role,'seed':seed,'model':'CNN','selected_alpha':None,
                     **full_metrics(pred,gt)})
        for model in models:
            alpha=float(chosen[model]['alpha'])
            if context is None:mask=pred;indices=None
            else:
                values=np.asarray(scores[model][i],np.float32)
                penalty=alpha*np.abs(np.arange(-32,33,dtype=np.float32))
                indices=np.argmax(np.where(valid,values-penalty[None,:],-np.inf),axis=1)
                polygon=point[np.arange(256),indices]
                mask=compose_prediction_context(rasterize(polygon,pred.shape),context)
            row={'image_id':rec['image_id'],'role':role,'seed':seed,'model':model,'selected_alpha':alpha,
                 **full_metrics(mask,gt)}
            if indices is not None:
                row.update(repair_metrics(indices,valid,dist))
                row['mean_selected_gt_boundary_distance_proxy']=float(np.mean(dist[np.arange(256),indices]))
            rows.append(row)
        if n%25==0:print(f'{n}/{len(records)}',flush=True)
    write_csv(output/'per_image.csv',rows)
    summary={'seed':seed,'scope':'reused development val100 for evidence only; official test untouched',
             'selection':'each model alpha selected by cal50 mean Dice only',
             'models':{},'official_test_images_or_GT_opened':0}
    for name in ('CNN',*models):
        group=[r for r in rows if r['model']==name and r['role']=='locked_verification100']
        result={'n_images':len(group),'alpha':None if name=='CNN' else float(chosen[name]['alpha'])}
        for metric in ('dice','iou','bf1','hd95','assd','pixel_precision','pixel_recall','specificity'):
            result[metric]=float(np.mean([r[metric] for r in group]))
        for metric in ('repair_precision','repair_recall','keep_accuracy','wrong_move_rate','moved_fraction','mean_abs_offset','p95_abs_offset','selection_within_2px','mean_selected_gt_boundary_distance_proxy'):
            v=[r[metric] for r in group if r.get(metric) not in (None,'')]
            result[metric]=float(np.mean(v)) if v else None
        summary['models'][name]=result
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['models'],indent=2),flush=True)

if __name__=='__main__':main()
