"""Summarize fixed-weight interventions with paired image bootstrap."""
import csv
import numpy as np
from diagnose import OUT, save_json


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def dice(row,prefix=''):
    tp,fp,fn=(int(row[prefix+k]) for k in ('tp','fp','fn'))
    den=2*tp+fp+fn
    return 2*tp/den if den else 1.


def main():
    encoder=read('encoder_per_image.csv');decoder=read('decoder_per_image.csv')
    assert len(encoder)==len(decoder)==808 and [r['id'] for r in encoder]==[r['id'] for r in decoder]
    areas=np.array([float(r['target_area']) for r in encoder])
    results={}
    for name,rows in [('encoder',encoder),('decoder',decoder)]:
        orig=np.array([dice(r) for r in rows]);counter=np.array([dice(r,'cf_') for r in rows]);delta=orig-counter
        rng=np.random.default_rng(42)
        ci=np.quantile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)],[.025,.975])
        strata={}
        for label,lo,hi in [('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)]:
            keep=(areas>=lo)&(areas<hi)
            strata[label]=dict(n=int(keep.sum()),original_macro_dice=float(orig[keep].mean()),
               counterfactual_macro_dice=float(counter[keep].mean()),delta=float(delta[keep].mean()))
        results[name]=dict(original_macro_dice=float(orig.mean()),counterfactual_macro_dice=float(counter.mean()),
          original_minus_counterfactual=float(delta.mean()),image_bootstrap_95=ci.tolist(),
          better=int((delta>0).sum()),worse=int((delta<0).sum()),ties=int((delta==0).sum()),strata=strata)
    save_json('counterfactual_summary.json',dict(n=808,seed=42,n_bootstraps=2000,models=results,
      limitations=['Inference-only interventions may be out of distribution.',
       'Bootstrap is image-level and post-selection, not across seeds or patients.']))
    print(results)


if __name__=='__main__':main()
