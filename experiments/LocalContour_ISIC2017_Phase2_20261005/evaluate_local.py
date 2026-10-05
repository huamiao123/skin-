"""Strong-local validation and independently tuned top-8 closed-DP baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from evaluation import full_metrics,repair_metrics,selected_mask
from local_model import parameter_count,StrongLocal
from train_local import CACHE,ROOT
from contour.geometry import (closed_dp,mask_contours,polygon_area,build_prediction_context,
                              compose_prediction_context,rasterize)

LAMBDAS=(0.0,0.002,0.01,0.05,0.2)
DISPLACEMENT=(0.0,0.02,0.1)
TOP_NONZERO=8

def read_maps():
    return {n:np.load(CACHE/(n+'.npy'),mmap_mode='r') for n in ('prediction','gt')},\
           {n:np.load(CACHE/'candidates_R32'/(n+'.npy'),mmap_mode='r') for n in ('points','valid','gt_distance','has_contour')}

def prepare_dp(scores,valid,displacement):
    rows=len(scores);off=np.arange(-32,33,dtype=np.float64)
    index=np.empty((rows,TOP_NONZERO+1),dtype=np.int64)
    index[:,0]=32
    for i in range(rows):
        candidates=np.flatnonzero(valid[i]);candidates=candidates[candidates!=32]
        ordered=sorted(candidates,key=lambda j:(-float(scores[i,j]),abs(j-32),j))
        if len(ordered)<TOP_NONZERO:ordered+=([32]*(TOP_NONZERO-len(ordered)))
        index[i,1:]=ordered[:TOP_NONZERO]
    gathered_scores=scores[np.arange(rows)[:,None],index]
    gathered_offsets=off[index]
    cost=-gathered_scores+np.max(gathered_scores,axis=1,keepdims=True)+displacement*np.abs(gathered_offsets)
    return index,cost,gathered_offsets

def choose_dp(scores,valid,lam,displacement):
    index,cost,offset=prepare_dp(scores,valid,displacement)
    path,_=closed_dp(cost,offset,lam)
    return index[np.arange(len(path)),path]

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--model',choices=('local','local_large','local8','medium32','global'),default='local')
    args=parser.parse_args();seed=args.seed;model_name=args.model
    output=(ROOT/'results/local'/f'seed{seed}' if model_name=='local'
            else ROOT/'results/transformer'/model_name/f'seed{seed}')
    output.mkdir(parents=True,exist_ok=True)
    model_folder=ROOT/'models'/f'{model_name}_seed{seed}'
    model_done=json.loads((model_folder/'DONE.json').read_text())
    scores=np.load(model_folder/'candidate_scores.npy',mmap_mode='r')
    label={'local':'StrongLocal','local_large':'LocalLarge','local8':'LocalTransformer',
           'medium32':'MediumTransformer','global':'GlobalTransformer'}[model_name]
    cache_done=json.loads((CACHE/'COMPLETE.json').read_text())
    records=[r for r in cache_done['records'] if r['split']=='val']
    maps,candidates=read_maps()
    if len(records)!=150:raise RuntimeError('Expected 150 validation images')
    prepared=[]
    for rec in records:
        i=rec['index'];pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool)
        point=np.asarray(candidates['points'][i],np.float32)
        valid=np.asarray(candidates['valid'][i],bool)
        sample=np.asarray(scores[i],np.float32)
        if candidates['has_contour'][i]:
            contours=mask_contours(pred)
            source=max(contours,key=lambda p:abs(polygon_area(p)))
            context=build_prediction_context(pred,source)
            dp_inputs={disp:prepare_dp(sample,valid,disp) for disp in DISPLACEMENT}
        else:context=None;dp_inputs={}
        prepared.append({'record':rec,'index':i,'pred':pred,'gt':gt,'points':point,
                         'valid':valid,'sample':sample,'context':context,'dp_inputs':dp_inputs})
    rows=[];curve=[]
    for lam in LAMBDAS:
        for disp in DISPLACEMENT:
            dice=[];bf1=[]
            for case in prepared:
                pred,gt=case['pred'],case['gt']
                if case['context'] is None:mask=pred
                else:
                    index,cost,offset=case['dp_inputs'][disp]
                    path,_=closed_dp(cost,offset,lam)
                    selected=index[np.arange(len(path)),path]
                    polygon=case['points'][np.arange(len(selected)),selected]
                    mask=compose_prediction_context(rasterize(polygon,pred.shape),case['context'])
                intersection=np.count_nonzero(mask&gt)
                dice.append(2*intersection/(mask.sum()+gt.sum()) if mask.sum()+gt.sum() else 1.)
                bf1.append(float('nan'))
            result={'seed':seed,'smooth_lambda':lam,'displacement_penalty':disp,
                    'validation_dice':float(np.mean(dice)),'validation_bf1':'not_used_for_grid_selection'}
            curve.append(result);print(result,flush=True)
    write_csv(output/'dp_grid.csv',curve)
    best=max(curve,key=lambda r:r['validation_dice'])
    for case in prepared:
        rec=case['record'];i=case['index'];pred=case['pred'];gt=case['gt']
        point=case['points'];valid=case['valid']
        dist=np.asarray(candidates['gt_distance'][i],np.float32)
        sample=case['sample']
        methods={'CNN':(pred,None)}
        if case['context'] is not None:
            local=np.argmax(np.where(valid,sample,-np.inf),axis=1)
            index,cost,offset=case['dp_inputs'][best['displacement_penalty']]
            path,_=closed_dp(cost,offset,best['smooth_lambda'])
            dp=index[np.arange(len(path)),path]
            methods[label]=(compose_prediction_context(rasterize(point[np.arange(len(local)),local],pred.shape),case['context']),local)
            methods[label+'_DP']=(compose_prediction_context(rasterize(point[np.arange(len(dp)),dp],pred.shape),case['context']),dp)
        else:
            methods[label]=(pred,None);methods[label+'_DP']=(pred,None)
        for name,(mask,indices) in methods.items():
            row={'image_id':rec['image_id'],'seed':seed,'method':name,
                 'source_git_commit':model_done['source_git_commit'],**full_metrics(mask,gt)}
            if indices is not None:row.update(repair_metrics(indices,valid,dist))
            rows.append(row)
    write_csv(output/'per_image.csv',rows)
    summary={'seed':seed,'model':model_name,'scope':'ISIC2017 official validation previously exposed to public CNN and phase1; model selection only',
             'best_dp':best,'model_parameters':model_done['parameters'],'model_best_epoch':model_done['best_epoch'],
             'methods':{m:{metric:float(np.mean([r[metric] for r in rows if r['method']==m])) for metric in ('dice','iou','bf1','hd95','assd')}
                        for m in ('CNN',label,label+'_DP')},
             'test_images_opened':0,'dp_candidate_pruning':'zero plus eight highest learned nonzero scores for tractable exact closed DP',
             'dp_grid':{'lambda':LAMBDAS,'displacement':DISPLACEMENT},
             'source_git_commit':model_done['source_git_commit']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['methods'],indent=2),flush=True)

if __name__=='__main__':main()
