"""Untuned direct candidate selection screen for attention range, validation only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import full_metrics,repair_metrics
from train_local import CACHE,ROOT
from contour.geometry import mask_contours,polygon_area,build_prediction_context,compose_prediction_context,rasterize

METHODS=('local','local_large','local8','global')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();seed=a.seed
    available=[name for name in METHODS if (ROOT/'models'/f'{name}_seed{seed}'/'DONE.json').exists()]
    if 'local8' not in available or 'global' not in available:raise RuntimeError('Both attention models required')
    scores={name:np.load(ROOT/'models'/f'{name}_seed{seed}'/'candidate_scores.npy',mmap_mode='r') for name in available}
    cache=json.loads((CACHE/'COMPLETE.json').read_text())
    records=[r for r in cache['records'] if r['split']=='val']
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('prediction','gt')}
    cand={name:np.load(CACHE/'candidates_R32'/(name+'.npy'),mmap_mode='r') for name in ('points','valid','gt_distance','has_contour')}
    rows=[]
    for n,rec in enumerate(records,1):
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(cand['points'][i]);valid=np.asarray(cand['valid'][i],bool)
        dist=np.asarray(cand['gt_distance'][i],np.float32)
        contours=mask_contours(pred)
        context=build_prediction_context(pred,max(contours,key=lambda c:abs(polygon_area(c)))) if contours else None
        rows.append({'image_id':rec['image_id'],'method':'CNN',**full_metrics(pred,gt)})
        for method in available:
            if context is None:mask=pred;indices=None
            else:
                values=np.asarray(scores[method][i],np.float32)
                indices=np.argmax(np.where(valid,values,-np.inf),axis=1)
                polygon=point[np.arange(len(indices)),indices]
                mask=compose_prediction_context(rasterize(polygon,pred.shape),context)
            row={'image_id':rec['image_id'],'method':method,**full_metrics(mask,gt)}
            if indices is not None:row.update(repair_metrics(indices,valid,dist))
            rows.append(row)
        if n%25==0:print(f'{n}/{len(records)}',flush=True)
    out=ROOT/'results/transformer'/f'quick_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    summary={'scope':'previously exposed 2017 val150 development readout; direct argmax, no DP or displacement calibration',
             'seed':seed,'methods':{method:{metric:float(np.mean([r[metric] for r in rows if r['method']==method])) for metric in ('dice','bf1','hd95','assd')}
                               for method in ('CNN',*available)},
             'test_images_opened':0}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
