"""Development-only dense R32 GT-assisted contour result; fixed original DP lambda."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from project_paths import PHASE1

import numpy as np

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.geometry import (build_prediction_context,closed_dp,compose_prediction_context,
                              distance_to_boundary,extract_geometry,rasterize,segmentation_metrics)
from radius_sensitivity import dense

ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    paths=sorted((SOURCE/'results/cases').glob('*.npz'))
    for n,p in enumerate(paths,1):
        with np.load(p) as z:
            pred=z['masks_G0'];gt=z['gt_mask'];q=z['boundary_probability'];role=str(z['role'])
        g0=segmentation_metrics(pred,gt)
        base=extract_geometry(pred,256,16)
        if base is None:
            oracle=g0
        else:
            geo=replace(base,search_radius=32.0)
            field=dense(q,geo)
            costs=np.where(field.valid,distance_to_boundary(field.points,gt),np.inf)
            indices,_=closed_dp(costs,field.offsets,0.05)
            context=build_prediction_context(pred,base.source_polygon)
            oracle=segmentation_metrics(compose_prediction_context(
                rasterize(field.points[np.arange(len(indices)),indices],pred.shape),context),gt)
        rows.append({'image_id':p.stem,'role':role,'g0_dice':g0['dice'],'g0_bf1':g0['bf1'],
                     'r32_dense_oracle_dice':oracle['dice'],'r32_dense_oracle_bf1':oracle['bf1']})
        if n%10==0:print(f'{n}/{len(paths)}',flush=True)
    out=ROOT/'results/candidates'
    with (out/'radius32_oracle_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    result={'n_images':len(rows),'r32_dense_oracle_dice_mean':float(np.mean([r['r32_dense_oracle_dice'] for r in rows])),
            'r32_dense_oracle_bf1_mean':float(np.mean([r['r32_dense_oracle_bf1'] for r in rows])),
            'g0_dice_mean':float(np.mean([r['g0_dice'] for r in rows])),
            'scope':'GT-assisted development diagnosis, no method performance claim',
            'dp_lambda':0.05,'lambda_selection':'historical R16 calibration, fixed before R32 readout'}
    (out/'radius32_oracle_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
