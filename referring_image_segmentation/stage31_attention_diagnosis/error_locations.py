from pathlib import Path
import sys,json,csv
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
from stage28_drop_path.train import dump
OUT=ROOT/'error_locations';OUT.mkdir(exist_ok=True)
part=json.loads((PROJECT/'stage30_channel_fusion/identity_partition.json').read_text())
ids=part['splits']['val']
def read_gt(sid):
    return np.asarray(Image.open(PROJECT/'data/isic2018/val/masks'/(sid+'.png')).convert('L'))>=127.5
with ThreadPoolExecutor(8) as pool:gt=np.stack(list(pool.map(read_gt,ids)))
with ThreadPoolExecutor(8) as pool:
    inside=np.stack(list(pool.map(distance_transform_edt,gt)))
    outside=np.stack(list(pool.map(distance_transform_edt,~gt)))
summary={};predictions={}
for name in ['control','local_fusion','anchor_fusion','eca_channel','cross_channel','cross_channel_scale']:
    stage='stage29_context_fusion' if name in ['control','local_fusion','anchor_fusion'] else 'stage30_channel_fusion'
    d=np.load(PROJECT/stage/'results'/f'{name}_val_predictions.npz')
    assert list(d['ids'])==ids
    pred=np.unpackbits(d['predictions_packed'],axis=1,count=256*256).reshape(-1,256,256).astype(bool)
    predictions[name]=pred
    fn=gt&~pred;fp=~gt&pred
    area=gt.sum((1,2))
    tp=(gt&pred).sum((1,2));fns=fn.sum((1,2));fps=fp.sum((1,2))
    rows=[dict(id=sid,gt_area=int(area[i]),tp=int(tp[i]),fn=int(fns[i]),fp=int(fps[i]),
            recall=float(tp[i]/max(area[i],1))) for i,sid in enumerate(ids)]
    details={}
    for distance in [3,5,10]:
        fnfar=(fn&(inside>distance)).sum((1,2))
        fpfar=(fp&(outside>distance)).sum((1,2))
        for i,row in enumerate(rows):
            row[f'fn_deeper_than_{distance}px']=int(fnfar[i]);row[f'fp_farther_than_{distance}px']=int(fpfar[i])
        details[str(distance)]=dict(fn_pixels=int(fnfar.sum()),fp_pixels=int(fpfar.sum()),
              fraction_of_fn=float(fnfar.sum()/max(fn.sum(),1)),fraction_of_fp=float(fpfar.sum()/max(fp.sum(),1)),
              deep_fn_fraction_in_severe_cases=float(fnfar[tp/area<.5].sum()/max(fns[tp/area<.5].sum(),1)))
    summary[name]=dict(n=len(ids),fn=int(fn.sum()),fp=int(fp.sum()),distance_bands=details,
                      severe_undercoverage_ids=[r['id'] for r in rows if r['recall']<.5])
    with (OUT/f'{name}_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
comparisons={}
local=predictions['local_fusion']
for name,pred in predictions.items():
    if name=='local_fusion':continue
    recovered=~local&pred&gt;lost=local&~pred&gt
    added=~local&pred&~gt;removed=local&~pred&~gt
    comparisons[name]=dict(recovered_fn=int(recovered.sum()),new_fn=int(lost.sum()),
          added_fp=int(added.sum()),removed_fp=int(removed.sum()),
          recovered_fn_deep5=int((recovered&(inside>5)).sum()),new_fn_deep5=int((lost&(inside>5)).sum()),
          added_fp_far5=int((added&(outside>5)).sum()),removed_fp_far5=int((removed&(outside>5)).sum()))
sets=[set(v['severe_undercoverage_ids']) for v in summary.values()]
dump(OUT/'summary.json',dict(models=summary,vs_local_fusion=comparisons,
    all_six_severe_intersection=sorted(set.intersection(*sets)),
    note='Validation-only pixel distance to ground-truth region at fixed256 resolution; not a visibility annotation or lesion-identity diagnosis. Thresholds3/5/10 reported together. No test read.'))
print('ERROR_LOCATIONS_COMPLETE',flush=True)
