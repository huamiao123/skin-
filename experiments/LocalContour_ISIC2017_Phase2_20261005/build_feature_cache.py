"""Temporary train/val-only frozen CNN cache for the phase-2 selector comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from project_paths import PHASE1,ISIC2017

import numpy as np
import torch

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.data import load_image,load_mask
from contour.models import FrozenMSGUNet,segmentation_mask
from tools.train_boundary import load_selected_head

ROOT=Path(__file__).resolve().parent
DATA=ISIC2017
CACHE=ROOT/'cache'

def split_ids(split):return (ROOT/'splits'/f'{split}.txt').read_text().splitlines()

def main():
    if (CACHE/'COMPLETE.json').exists():
        print('Cache already complete');return
    ids=[(split,name) for split in ('train','val') for name in split_ids(split)]
    if len(ids)!=2150:raise RuntimeError('Incorrect train+val count')
    CACHE.mkdir(exist_ok=True)
    n=len(ids)
    files={
      'features':(np.float16,(n,32,256,256)),
      'rgb':(np.float16,(n,3,256,256)),
      'probability':(np.float16,(n,1,256,256)),
      'boundary':(np.float16,(n,1,256,256)),
      'prediction':(np.uint8,(n,256,256)),
      'gt':(np.uint8,(n,256,256)),
    }
    maps={name:np.lib.format.open_memmap(CACHE/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,(dtype,shape) in files.items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cnn=FrozenMSGUNet(SOURCE/'third_party/msgu_net').to(device)
    head=load_selected_head(root=SOURCE,device=device)
    before=cnn.state_hash()
    for i,(split,name) in enumerate(ids):
        image=DATA/split/'images'/(name+'.jpg')
        mask=DATA/split/'masks'/(name+'_segmentation.png')
        if not image.is_file() or not mask.is_file():raise RuntimeError(f'Missing {name}')
        rgb=load_image(image)
        gt=load_mask({'image_id':name,'mask_path':str(mask),'split':split})[0].numpy().astype(np.uint8)
        if gt.shape!=(256,256) or not (0<int(gt.sum())<256*256):
            raise RuntimeError(f'GT shape/content check failed for {name}')
        with torch.inference_mode():
            output=cnn.forward_with_features(rgb[None].to(device))
            boundary=torch.sigmoid(head(output['features']))
        maps['features'][i]=output['features'][0].half().cpu().numpy()
        maps['rgb'][i]=rgb.half().numpy()
        maps['probability'][i]=output['probability'][0].half().cpu().numpy()
        maps['boundary'][i]=boundary[0].half().cpu().numpy()
        maps['prediction'][i]=output['mask'][0,0].byte().cpu().numpy()
        maps['gt'][i]=gt
        if (i+1)%100==0:print(f'{i+1}/{n}',flush=True)
    for value in maps.values():value.flush()
    if cnn.state_hash()!=before:raise RuntimeError('Frozen CNN state changed')
    complete={'status':'PASS','records':[{'index':i,'split':s,'image_id':name} for i,(s,name) in enumerate(ids)],
              'cnn_sha256':cnn.architecture_audit()['weight_sha256'],'cnn_state_hash':before,
              'test_image_or_gt_opened':False,'purpose':'temporary frozen features, remove after models trained'}
    (CACHE/'COMPLETE.json').write_text(json.dumps(complete,indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
