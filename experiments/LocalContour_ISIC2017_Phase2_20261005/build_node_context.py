"""Compact node inputs for attention-range comparisons; no GT in features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'

def main():
    done=json.loads((CACHE/'COMPLETE.json').read_text())
    candidates=json.loads((CACHE/'candidates_R32/COMPLETE.json').read_text())
    if done['status']!='PASS' or candidates['status']!='PASS':raise RuntimeError('Cache incomplete')
    n=len(done['records'])
    maps={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ('features','rgb','probability','boundary')}
    source=np.load(CACHE/'candidates_R32/source_points.npy',mmap_mode='r')
    normals=np.load(CACHE/'candidates_R32/normals.npy',mmap_mode='r')
    has=np.load(CACHE/'candidates_R32/has_contour.npy',mmap_mode='r')
    result=np.lib.format.open_memmap(CACHE/'node_context.npy',mode='w+',dtype=np.float16,shape=(n,256,44))
    phase=np.arange(256)*2*np.pi/256
    phase_features=np.stack((np.sin(phase),np.cos(phase)),axis=-1)
    for i in range(n):
        if not has[i]:result[i]=0;continue
        p=np.asarray(source[i],np.float32);normal=np.asarray(normals[i],np.float32)
        field=np.concatenate([np.asarray(maps[name][i],np.float32) for name in ('features','rgb','probability','boundary')],axis=0)
        sampled=np.stack([ndimage.map_coordinates(channel,[p[:,1],p[:,0]],order=1,mode='nearest',prefilter=False)
                          for channel in field],axis=-1)
        curvature=np.linalg.norm(np.roll(normal,-1,axis=0)-np.roll(normal,1,axis=0),axis=-1,keepdims=True)
        additional=np.concatenate((p/255.0,normal,curvature,phase_features),axis=-1)
        token=np.concatenate((sampled,additional),axis=-1)
        if token.shape!=(256,44) or not np.isfinite(token).all():raise RuntimeError(f'Invalid node features {i}')
        result[i]=token.astype(np.float16)
        if (i+1)%100==0:print(f'{i+1}/{n}',flush=True)
    result.flush()
    output={'status':'PASS','shape':[n,256,44],'GT_used_in_features':False,
            'source_maps':'frozen CNN decoder32, normalized RGB3, CNN probability1, frozen boundary probability1',
            'geometry':'source point xy2, normal2, curvature1, circular index sin/cos2',
            'test_images_opened':0,'temporary_cache':True}
    (CACHE/'node_context_COMPLETE.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
