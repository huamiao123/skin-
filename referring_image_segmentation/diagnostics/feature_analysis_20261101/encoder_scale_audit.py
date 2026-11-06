"""Resolution-aware checks for the trained 16x16 encoder router."""
from collections import defaultdict
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
from diagnose import (BATCH, OUT, PROJECT, S19, ImageData, Segmentor,
                      load_trainable, region_targets, save_json, setup, vector)


def main():
    setup();data=ImageData(42);ids=data.ids['val'];n=len(ids)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    model=Segmentor('region_scan').cuda().eval()
    ck=torch.load(S19/'results/region_scan/best_dice.pth',map_location='cpu')
    load_trainable(model,ck['model'])
    assert ck['epoch']==86 and model.encoder.frozen_state_hash()==ck['frozen_hash']
    columns=defaultdict(list);soft_confusion=np.zeros((3,3),dtype=np.float64)
    windows=((slice(0,14),slice(0,14)),(slice(0,14),slice(14,16)),
             (slice(14,16),slice(0,14)),(slice(14,16),slice(14,16)))
    print('ENCODER 16x16 SCALE AUDIT',flush=True)
    with torch.no_grad():
      for start in range(0,n,BATCH):
        end=min(n,start+BATCH);image,gt=data.batch('val',torch.arange(start,end,device='cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):_,route=model.encoder(image)
        t=region_targets(gt);hard=route.argmax(1);bd=t[:,1];selected=(hard==1)
        k=selected.sum((1,2));ordered=bd.flatten(1).sort(dim=1,descending=True).values
        oracle=(ordered*(torch.arange(256,device='cuda')[None,:] < k[:,None])).sum(1)
        actual=(bd*selected).sum((1,2));mass=bd.sum((1,2))
        foreground=F.avg_pool2d((gt>=.5).float(),16)[:,0]
        values=dict(foreground_fraction=(gt>=.5).float().mean((1,2,3)),
          boundary_area_units=mass,boundary_assigned_cells=k,
          boundary_captured_area_units=actual,boundary_oracle_equal_budget_units=oracle,
          boundary_touching_cells=(bd>0).sum((1,2)),
          boundary_majority_cells=(bd>.5).sum((1,2)),
          foreground_touching_cells=(foreground>0).sum((1,2)),
          foreground_majority_cells=(foreground>.5).sum((1,2)))
        for w,(ys,xs) in enumerate(windows):
            values[f'window_{w}_boundary_area_units']=bd[:,ys,xs].sum((1,2))
            values[f'window_{w}_boundary_assigned_cells']=selected[:,ys,xs].sum((1,2))
        for key,item in values.items():columns[key].append(vector(item))
        for k0 in range(3):
            for k1 in range(3):soft_confusion[k0,k1]+=float((t[:,k0]*(hard==k1)).sum().cpu())
        if start%256==0:print(f'  {end}/{n}',flush=True)
    v={k:np.concatenate(x) for k,x in columns.items()};small=v['foreground_fraction']<.1
    masses=v['boundary_area_units'];actual=v['boundary_captured_area_units'];oracle=v['boundary_oracle_equal_budget_units']
    assigned=v['boundary_assigned_cells'];assert (oracle+1e-4>=actual).all()
    win=[]
    for w,area in enumerate((196,28,28,4)):
        true=v[f'window_{w}_boundary_area_units']>0
        selected=v[f'window_{w}_boundary_assigned_cells']>0
        win.append(dict(valid_cells=area,images_with_true_boundary=int(true.sum()),
           images_with_true_boundary_but_no_boundary_route=int((true&~selected).sum()),
           fraction_conditional_on_true_boundary=float((true&~selected).sum()/true.sum()) if true.sum() else None))
    summary=dict(n=n,seed=42,checkpoint_epoch=86,grid=[16,16],
       ground_truth_morphology='5x5 width-two boundary pooled by exact 16x16 area fraction',
       ground_truth_soft_area_by_predicted_argmax=soft_confusion.tolist(),
       ground_truth_soft_area_confusion_row_normalized=(soft_confusion/soft_confusion.sum(1,keepdims=True)).tolist(),
       boundary=dict(true_area_fraction=float(masses.sum()/(n*256)),
         assigned_patch_fraction=float(assigned.sum()/(n*256)),
         actual_recall_area=float(actual.sum()/masses.sum()),
         actual_area_purity=float(actual.sum()/assigned.sum()),
         oracle_recall_equal_assigned_cells_per_image=float(oracle.sum()/masses.sum()),
         oracle_area_purity_equal_assigned_cells_per_image=float(oracle.sum()/assigned.sum()),
         fraction_images_with_no_boundary_majority_patch=float((v['boundary_majority_cells']==0).mean())),
       small_lesions_lt10pct=dict(n=int(small.sum()),
         no_foreground_majority_cell=int((v['foreground_majority_cells'][small]==0).sum()),
         at_most_one_foreground_majority_cell=int((v['foreground_majority_cells'][small]<=1).sum()),
         at_most_four_foreground_majority_cells=int((v['foreground_majority_cells'][small]<=4).sum()),
         median_foreground_majority_cells=float(np.median(v['foreground_majority_cells'][small])),
         median_foreground_touching_cells=float(np.median(v['foreground_touching_cells'][small]))),
       window_boundary_route=win,
       notes=['Oracle uses GT and is an upper bound only for this fixed number of routed cells, not an implementable model.',
        'Area purity is limited by coarse 16x16 cells and a thin morphological boundary.',
        'An empty predicted window is counted as a miss only if GT boundary actually intersects that window.',
        'This is the same validation set used to choose the checkpoint.'])
    save_json('encoder_scale_audit.json',summary)
    with (OUT/'encoder_scale_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id']+list(v));w.writeheader()
        for i,identifier in enumerate(ids):w.writerow({'id':identifier,**{k:float(x[i]) for k,x in v.items()}})
    print('FINISHED',json.dumps(dict(boundary=summary['boundary'],small=summary['small_lesions_lt10pct'],windows=win)),flush=True)


if __name__=='__main__':main()
