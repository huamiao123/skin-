"""Predeclared five failure categories; render only categories with real cases."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from PIL import Image,ImageDraw

from evaluation import selected_mask
from train_local import CACHE,ROOT
from contour.geometry import boundary_pixels

SEEDS=(17,23,42)

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def outline(rgb,mask,color):
    image=rgb.copy();image[boundary_pixels(mask)]=color
    return Image.fromarray(image)

def main():
    values={}
    for seed in SEEDS:
        for row in read(ROOT/'results/transformer'/f'chosen_keep_seed{seed}'/'per_image.csv'):
            if row['role']=='locked_verification100':
                values.setdefault(row['image_id'],{}).setdefault(row['model'],[]).append(float(row['dice']))
    oracle={r['image_id']:float(r['r32_dense_oracle_dice'])
            for r in read(ROOT/'results/candidates/radius32_oracle_per_image.csv')}
    cases={key:[] for key in ('A','B','C','D','E')}
    measures={}
    for image_id,v in values.items():
        cnn,global_,local=float(np.mean(v['CNN'])),float(np.mean(v['global'])),float(np.mean(v['local_large']))
        ora=oracle[image_id]
        measures[image_id]={'CNN':cnn,'global':global_,'local_large':local,'oracle':ora}
        if cnn>=.9 and abs(global_-cnn)<=.001 and abs(local-cnn)<=.005:cases['A'].append((cnn,image_id))
        if global_-cnn>=.005 and global_-local>=.005:cases['B'].append((global_-local,image_id))
        if global_-cnn>=.005 and local-cnn>=.005:cases['C'].append((min(global_-cnn,local-cnn),image_id))
        if ora-cnn>=.05 and global_<=cnn+.001 and local<=cnn+.001:cases['D'].append((ora-cnn,image_id))
        if ora-cnn<=.01:cases['E'].append((-ora+cnn,image_id))
    selected={key:[i for _,i in sorted(items,key=lambda x:(-x[0],x[1]))[:5]] for key,items in cases.items()}
    out=ROOT/'results/phase2_summary/case_panels';out.mkdir(parents=True,exist_ok=True)
    cache=json.loads((CACHE/'COMPLETE.json').read_text())
    indices={r['image_id']:r['index'] for r in cache['records'] if r['split']=='val'}
    pred=np.load(CACHE/'prediction.npy',mmap_mode='r')
    gt=np.load(CACHE/'gt.npy',mmap_mode='r')
    points=np.load(CACHE/'candidates_R32/points.npy',mmap_mode='r')
    valid=np.load(CACHE/'candidates_R32/valid.npy',mmap_mode='r')
    scores={name:np.load(ROOT/'models'/f'{name}_seed17/candidate_scores.npy',mmap_mode='r') for name in ('local_large','global')}
    keep_rows=read(ROOT/'results/transformer/keep_seed17/summary.csv')
    alpha={model:float(max((r for r in keep_rows if r['model']==model),key=lambda r:float(r['cal50_dice']))['alpha'])
           for model in scores}
    for key,ids in selected.items():
        folder=out/key;folder.mkdir(exist_ok=True)
        for image_id in ids:
            i=indices[image_id]
            with np.load(PHASE1/'results/cases'/(image_id+'.npz')) as source:
                rgb=np.asarray(source['input_rgb'],np.uint8)
            if rgb.shape!=(256,256,3):raise RuntimeError('Panel RGB shape differs')
            original=np.asarray(pred[i],bool);truth=np.asarray(gt[i],bool)
            panes=[outline(rgb,truth,(0,255,0)),outline(rgb,original,(0,180,255))]
            for model,color in [('local_large',(255,0,200)),('global',(255,0,0))]:
                values=np.asarray(scores[model][i],np.float32)
                adjusted=values-alpha[model]*np.abs(np.arange(-32,33,dtype=np.float32))[None,:]
                indices_selected=np.argmax(np.where(valid[i],adjusted,-np.inf),axis=1)
                mask=selected_mask(original,np.asarray(points[i]),indices_selected)
                panes.append(outline(rgb,mask,color))
            canvas=Image.new('RGB',(4*256,296),'white')
            for j,pane in enumerate(panes):canvas.paste(pane,(j*256,40))
            draw=ImageDraw.Draw(canvas)
            m=measures[image_id]
            labels=[f'{image_id} GT','CNN mean '+format(m['CNN'],'.3f'),
                    'Local mean '+format(m['local_large'],'.3f'),'Global mean '+format(m['global'],'.3f')]
            for j,label in enumerate(labels):draw.text((j*256+5,8),label,fill='black')
            canvas.save(folder/(image_id+'.png'))
    record={'scope':'reused 2017 val100 development cases, seed17 masks shown; category membership uses three-seed mean Dice',
            'thresholds':{'A':'CNN>=.9 and abs(global-CNN)<=.001 and abs(local-CNN)<=.005',
                          'B':'global-CNN>=.005 and global-local_large>=.005',
                          'C':'global-CNN>=.005 and local_large-CNN>=.005',
                          'D':'oracle-CNN>=.05 and global<=CNN+.001 and local_large<=CNN+.001',
                          'E':'GT-assisted oracle-CNN<=.01'},
            'counts':{key:len(items) for key,items in cases.items()},'selected_ids':selected,
            'reason_fewer_than_five':'There are fewer real cases under fixed criteria; no cases fabricated.',
            'oracle_GTagreement':'GT-assisted R32 oracle Dice is case metadata, not a displayed deployable mask',
            'official_test_scored':False}
    (out/'selection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record['counts'],indent=2),flush=True)

if __name__=='__main__':main()
