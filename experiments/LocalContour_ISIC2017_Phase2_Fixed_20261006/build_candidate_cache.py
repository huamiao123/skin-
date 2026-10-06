"""GT-free dense candidate coordinates and training-only GT distance labels."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from scipy import ndimage

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.geometry import boundary_pixels,extract_geometry
from radius_sensitivity import dense

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--radius',type=int,choices=(16,32),required=True)
    args=parser.parse_args();radius=args.radius
    done=json.loads((CACHE/'COMPLETE.json').read_text())
    if done['status']!='PASS' or len(done['records'])!=2150:raise RuntimeError('Frozen feature cache incomplete')
    n=len(done['records']);k=2*radius+1;out=CACHE/f'candidates_R{radius}'
    out.mkdir(exist_ok=True)
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('prediction','gt','boundary')}
    dest={
      'points':np.lib.format.open_memmap(out/'points.npy',mode='w+',dtype=np.float32,shape=(n,256,k,2)),
      'source_points':np.lib.format.open_memmap(out/'source_points.npy',mode='w+',dtype=np.float32,shape=(n,256,2)),
      'normals':np.lib.format.open_memmap(out/'normals.npy',mode='w+',dtype=np.float32,shape=(n,256,2)),
      'valid':np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype=np.uint8,shape=(n,256,k)),
      'gt_distance':np.lib.format.open_memmap(out/'gt_distance.npy',mode='w+',dtype=np.float16,shape=(n,256,k)),
      'has_contour':np.lib.format.open_memmap(out/'has_contour.npy',mode='w+',dtype=np.uint8,shape=(n,)),
    }
    coordinates=np.arange(-radius,radius+1,dtype=np.float32)
    for i,rec in enumerate(done['records']):
        pred=maps['prediction'][i].astype(bool);gt=maps['gt'][i].astype(bool);q=maps['boundary'][i,0].astype(np.float32)
        if not (0<int(gt.sum())<256*256):
            raise RuntimeError(f'GT cache has invalid 2D lesion mask for {rec["image_id"]}')
        geo=extract_geometry(pred,256,16)
        if geo is None:
            dest['points'][i]=0;dest['source_points'][i]=0;dest['normals'][i]=0
            dest['valid'][i]=0;dest['gt_distance'][i]=1000;dest['has_contour'][i]=0
            continue
        if radius==32:geo=replace(geo,search_radius=32.0)
        field=dense(q,geo)
        gt_border=boundary_pixels(gt)
        edt=ndimage.distance_transform_edt(~gt_border) if gt_border.any() else np.full(gt.shape,1000.0)
        distance=ndimage.map_coordinates(edt,[field.points[...,1],field.points[...,0]],order=1,mode='nearest',prefilter=False)
        distance=np.where(field.valid,distance,1000.0)
        dest['points'][i]=field.points.astype(np.float32)
        dest['source_points'][i]=geo.points.astype(np.float32)
        dest['normals'][i]=geo.normals.astype(np.float32)
        dest['valid'][i]=field.valid.astype(np.uint8)
        dest['gt_distance'][i]=distance.astype(np.float16)
        dest['has_contour'][i]=1
        if (i+1)%100==0:print(f'{i+1}/{n}',flush=True)
    for m in dest.values():m.flush()
    result={'status':'PASS','radius':radius,'nodes':256,'candidate_count':k,
            'records':n,'test_image_or_gt_opened':False,
            'training_label':'GT nearest boundary pixel EDT, bilinearly sampled; invalid candidate distance=1000',
            'gt_full_shape_verified_for_all':n,
            'gt_distance_label_source':'complete 2D official GT mask',
            'coordinates':'derived solely from frozen CNN mask, frozen boundary map, and fixed geometry; GT loaded only for label distance',
            'temporary_cache':True}
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
