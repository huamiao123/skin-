"""Correct the cached GT slice and its EDT labels without repeating CNN inference."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
from project_paths import PHASE1,ISIC2017

import numpy as np
from scipy import ndimage

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.data import load_mask
from contour.geometry import boundary_pixels

ROOT=Path(__file__).resolve().parent
DATA=ISIC2017
CACHE=ROOT/'cache'
RADIUS=32

def main():
    meta=json.loads((CACHE/'COMPLETE.json').read_text())
    target=CACHE/f'candidates_R{RADIUS}'
    candidate_meta=json.loads((target/'COMPLETE.json').read_text())
    if meta['status']!='PASS' or candidate_meta['status']!='PASS':raise RuntimeError('Source cache incomplete')
    n=len(meta['records'])
    gt=np.load(CACHE/'gt.npy',mmap_mode='r+')
    points=np.load(target/'points.npy',mmap_mode='r')
    valid=np.load(target/'valid.npy',mmap_mode='r')
    distances=np.load(target/'gt_distance.npy',mmap_mode='r+')
    if n!=2150 or int(gt[0].sum())!=0:raise RuntimeError('Unexpected cache state')
    # Invalid seed-17 model was trained on all-zero GT. Preserve a text audit,
    # then remove its weights and score arrays so it cannot be selected.
    invalid=ROOT/'models/local_seed17'
    log=ROOT/'local_seed17.log'
    record={'time_utc':datetime.now(timezone.utc).isoformat(),
            'cause':'load_mask returns [1,256,256]; code used [0,0] instead of [0], broadcasting the first GT row over the image',
            'effect':'train and val GT caches all zero; seed17 local model trained with invalid labels; no result usable',
            'action':'rebuild all 2150 GT arrays and GT-distance labels; preserve CNN feature, RGB, probability, boundary, candidate points/valid exactly; delete invalid model weights/scores and retrain all seeds',
            'invalid_seed17_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None,
            'official_test_bytes_opened':False}
    (ROOT/'protocols/invalid_local_training_repair.json').write_text(json.dumps(record,indent=2))
    if invalid.exists():shutil.rmtree(invalid)
    gt_sums=[]
    for i,rec in enumerate(meta['records']):
        mask=DATA/rec['split']/'masks'/(rec['image_id']+'_segmentation.png')
        image=load_mask({'mask_path':str(mask),'split':rec['split']})[0].numpy().astype(np.uint8)
        if image.shape!=(256,256) or not (0<int(image.sum())<256*256):
            raise RuntimeError(f'Invalid GT {rec["image_id"]}')
        gt[i]=image;gt_sums.append(int(image.sum()))
        border=boundary_pixels(image.astype(bool))
        edt=ndimage.distance_transform_edt(~border)
        point=np.asarray(points[i],dtype=np.float32)
        allowable=np.asarray(valid[i],dtype=bool)
        dist=ndimage.map_coordinates(edt,[point[...,1],point[...,0]],order=1,mode='nearest',prefilter=False)
        distances[i]=np.where(allowable,dist,1000.0).astype(np.float16)
        if (i+1)%100==0:print(f'{i+1}/{n}',flush=True)
    gt.flush();distances.flush()
    sample=np.asarray(distances[:min(100,n)])
    if not np.any(sample<4):raise RuntimeError('Repaired candidate target has no near-boundary samples')
    meta['gt_repaired_utc']=datetime.now(timezone.utc).isoformat()
    meta['gt_shape_verified_for_all']=n
    meta['gt_pixel_sum_min']=min(gt_sums);meta['gt_pixel_sum_max']=max(gt_sums)
    (CACHE/'COMPLETE.json').write_text(json.dumps(meta,indent=2))
    candidate_meta['gt_distance_repaired_utc']=meta['gt_repaired_utc']
    candidate_meta['gt_distance_label_source']='repaired complete 2D official GT mask'
    (target/'COMPLETE.json').write_text(json.dumps(candidate_meta,indent=2))
    print(json.dumps({'status':'PASS','GT_images':n,'min_GT_area':min(gt_sums),
                      'max_GT_area':max(gt_sums),'sample_near_boundary_fraction':float((sample<4).mean())},indent=2),flush=True)

if __name__=='__main__':main()
