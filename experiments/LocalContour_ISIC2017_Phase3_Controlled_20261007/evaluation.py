"""Common contour raster, segmentation and repair metrics for phase 2."""
from __future__ import annotations

import sys
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from scipy import ndimage

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.geometry import (boundary_pixels,build_prediction_context,
                              compose_prediction_context,mask_contours,polygon_area,rasterize,
                              segmentation_metrics)

def selected_mask(pred,points,indices):
    pred=np.asarray(pred,bool)
    contours=mask_contours(pred)
    if not contours:return pred.copy()
    source=max(contours,key=lambda p:abs(polygon_area(p)))
    context=build_prediction_context(pred,source)
    path=points[np.arange(len(indices)),indices]
    return compose_prediction_context(rasterize(path,pred.shape),context)

def full_metrics(pred,gt):
    pred=np.asarray(pred,bool);gt=np.asarray(gt,bool)
    result=segmentation_metrics(pred,gt)
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum())
    fn=int((~pred&gt).sum());tn=int((~pred&~gt).sum())
    result.update(pixel_precision=tp/(tp+fp) if tp+fp else 1.,
                  pixel_recall=tp/(tp+fn) if tp+fn else 1.,
                  specificity=tn/(tn+fp) if tn+fp else 1.)
    pb=boundary_pixels(pred);gb=boundary_pixels(gt)
    if pb.any() and gb.any():
        a=ndimage.distance_transform_edt(~gb)[pb]
        b=ndimage.distance_transform_edt(~pb)[gb]
        result['hd95']=float(max(np.percentile(a,95),np.percentile(b,95)))
        result['assd']=float((a.sum()+b.sum())/(len(a)+len(b)))
    elif not pb.any() and not gb.any():result.update(hd95=0.,assd=0.)
    else:result.update(hd95=float(np.hypot(*pred.shape)),assd=float(np.hypot(*pred.shape)))
    return result

def repair_metrics(indices,valid,gt_distance):
    if not len(indices):return {k:None for k in ('repair_precision','repair_recall','keep_rate_on_initially_correct','move_rate_on_initially_correct','distance_increase_rate','correct_to_incorrect_rate','moved_fraction','mean_abs_offset','p95_abs_offset','selection_within_2px')}
    rows=np.arange(len(indices))
    d=np.where(valid,gt_distance,np.inf)
    d0=d[:,32];chosen=d[rows,indices];best=d.min(axis=1)
    moved=indices!=32;improved=moved&(chosen<d0-1e-3)
    fixable=(d0>2)&(best<d0-1e-3)
    correct=d0<=2
    values=np.abs(indices-32)
    return {'repair_precision':float(improved.sum()/moved.sum()) if moved.any() else None,
            'repair_recall':float((improved&fixable).sum()/fixable.sum()) if fixable.any() else None,
            'keep_rate_on_initially_correct':float((~moved&correct).sum()/correct.sum()) if correct.any() else None,
            'move_rate_on_initially_correct':float((moved&correct).sum()/correct.sum()) if correct.any() else None,
            'distance_increase_rate':float((moved&(chosen>d0+1e-3)).sum()/moved.sum()) if moved.any() else None,
            'correct_to_incorrect_rate':float((moved&correct&(chosen>2)).sum()/correct.sum()) if correct.any() else None,
            'moved_fraction':float(moved.mean()),'mean_abs_offset':float(values.mean()),
            'p95_abs_offset':float(np.percentile(values,95)),
            'selection_within_2px':float((chosen<=2).mean())}
