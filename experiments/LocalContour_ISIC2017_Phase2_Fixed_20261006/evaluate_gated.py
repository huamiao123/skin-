"""Cal50 gate threshold selection and reused-val100 gated contour evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from scipy.special import expit

from evaluation import full_metrics,repair_metrics
from train_local import CACHE,ROOT
from contour.geometry import (mask_contours,polygon_area,build_prediction_context,
                              compose_prediction_context,rasterize)

THRESHOLDS=(0.3,0.5,0.7,0.8,0.85,0.9,0.95,0.99)
MODELS=('local8_gate','global_gate')
SPLIT=PHASE1/'assets/val_split_seed17.csv'

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();seed=a.seed
    roles={r['image_id']:r['role'] for r in read(SPLIT)}
    records=[r for r in json.loads((CACHE/'COMPLETE.json').read_text())['records'] if r['split']=='val']
    scores={};gates={}
    for model in MODELS:
        folder=ROOT/'models'/f'{model}_seed{seed}'
        if json.loads((folder/'DONE.json').read_text())['status']!='PASS':raise RuntimeError(f'Model incomplete: {folder}')
        scores[model]=np.load(folder/'candidate_scores.npy',mmap_mode='r')
        gates[model]=np.load(folder/'gate_logits.npy',mmap_mode='r')
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('prediction','gt')}
    cand={name:np.load(CACHE/'candidates_R32'/(name+'.npy'),mmap_mode='r') for name in ('points','valid','gt_distance','has_contour')}
    raw=[]
    for n,rec in enumerate(records,1):
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(cand['points'][i]);valid=np.asarray(cand['valid'][i],bool)
        contours=mask_contours(pred)
        context=build_prediction_context(pred,max(contours,key=lambda x:abs(polygon_area(x)))) if contours else None
        role=roles[rec['image_id']]
        for model in MODELS:
            value=np.asarray(scores[model][i],np.float32)
            gate=expit(np.asarray(gates[model][i],np.float32))
            usable=valid.copy();usable[:,32]=False
            proposal=np.argmax(np.where(usable,value,-np.inf),axis=1)
            for threshold in THRESHOLDS:
                if context is None:mask=pred;selected=None
                else:
                    selected=np.where(gate>=threshold,proposal,32)
                    polygon=point[np.arange(256),selected]
                    mask=compose_prediction_context(rasterize(polygon,pred.shape),context)
                intersection=np.count_nonzero(mask&gt)
                dice=2*intersection/(mask.sum()+gt.sum()) if mask.sum()+gt.sum() else 1.
                raw.append({'image_id':rec['image_id'],'role':role,'model':model,
                            'threshold':threshold,'dice':dice,
                            'moved_fraction':float((selected!=32).mean()) if selected is not None else None})
        if n%25==0:print(f'{n}/{len(records)}',flush=True)
    selected={}
    for model in MODELS:
        curve=[]
        for threshold in THRESHOLDS:
            group=[r for r in raw if r['model']==model and r['threshold']==threshold]
            cal=[r['dice'] for r in group if r['role']=='calibration50']
            verification=[r['dice'] for r in group if r['role']=='locked_verification100']
            curve.append({'model':model,'seed':seed,'threshold':threshold,
                          'cal50_dice':float(np.mean(cal)),'val100_dice':float(np.mean(verification)),
                          'all150_dice':float(np.mean([r['dice'] for r in group]))})
        selected[model]=max(curve,key=lambda r:r['cal50_dice'])
        print('CHOSEN',selected[model],flush=True)
    out=ROOT/'results/transformer'/f'gated_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'threshold_curve.csv',raw)
    rows=[]
    for n,rec in enumerate(records,1):
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(cand['points'][i]);valid=np.asarray(cand['valid'][i],bool)
        dist=np.asarray(cand['gt_distance'][i],np.float32)
        contours=mask_contours(pred)
        context=build_prediction_context(pred,max(contours,key=lambda x:abs(polygon_area(x)))) if contours else None
        role=roles[rec['image_id']]
        rows.append({'image_id':rec['image_id'],'role':role,'seed':seed,'model':'CNN',
                     'threshold':None,**full_metrics(pred,gt)})
        for model in MODELS:
            value=np.asarray(scores[model][i],np.float32)
            gate=expit(np.asarray(gates[model][i],np.float32))
            usable=valid.copy();usable[:,32]=False
            proposed=np.argmax(np.where(usable,value,-np.inf),axis=1)
            threshold=selected[model]['threshold']
            if context is None:mask=pred;indices=None
            else:
                indices=np.where(gate>=threshold,proposed,32)
                polygon=point[np.arange(256),indices]
                mask=compose_prediction_context(rasterize(polygon,pred.shape),context)
            row={'image_id':rec['image_id'],'role':role,'seed':seed,'model':model,
                 'threshold':threshold,**full_metrics(mask,gt)}
            if indices is not None:row.update(repair_metrics(indices,valid,dist))
            rows.append(row)
    write_csv(out/'per_image.csv',rows)
    summary={'seed':seed,'scope':'val50 threshold selection and previously exposed val100 development verification; official test untouched',
             'threshold_selection':'cal50 Dice, exact ties smaller threshold',
             'models':{},'official_test_images_or_GT_opened':0}
    for model in ('CNN',*MODELS):
        group=[r for r in rows if r['role']=='locked_verification100' and r['model']==model]
        result={'n_images':len(group),'threshold':None if model=='CNN' else selected[model]['threshold']}
        for metric in ('dice','iou','bf1','hd95','assd','moved_fraction','repair_precision','repair_recall','keep_rate_on_initially_correct','move_rate_on_initially_correct','distance_increase_rate','correct_to_incorrect_rate'):
            vals=[float(r[metric]) for r in group if r.get(metric) not in (None,'')]
            result[metric]=float(np.mean(vals)) if vals else None
        summary['models'][model]=result
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['models'],indent=2),flush=True)

if __name__=='__main__':main()
