"""Paired per-image test comparison for each completed dataset."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

root=Path(__file__).resolve().parent
reference=root.parent / 'official_full_uper512_fcn_20261120'
def read_test(folder):
    record=json.loads((folder/'final_test.json').read_text())
    path=folder/'final_test_per_image.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['per_image_sha256']
    fit=json.loads((folder/'fit_complete.json').read_text())
    assert (record['model_sha256'],record['epoch'])==(fit['sha256'],fit['best_epoch'])
    with path.open() as f:
        rows=list(csv.DictReader(f))
    per_image={r['image_id']:float(r['foreground_dice']) for r in rows}
    assert len(per_image)==len(rows)==record['n'] and abs(statistics.mean(per_image.values())-record['test_macro_dice'])<1e-10
    return record,per_image,fit
def run():
    results=[]
    for name in ('ISIC2018','PH2','Waterloo','HAM10000'):
        if not (root/name/'final_test.json').exists():continue
        base,b,_=read_test(reference/name)
        trial,t,fit=read_test(root/name)
        assert set(b)==set(t) and base['split_sha256']==trial['split_sha256'] and base['threshold']==trial['threshold']==.5
        orig=json.loads((reference/name/'config.json').read_text())
        new=json.loads((root/name/'config.json').read_text())
        assert new.pop('method')['name']=='native_preprojection_branch_readout' and new==orig
        delta=np.array([100*(t[k]-b[k]) for k in b])
        rng=np.random.default_rng(42)
        ci=np.percentile(rng.choice(delta,size=(5000,len(b)),replace=True).mean(1),[2.5,97.5])
        results.append(dict(dataset=name,test_n=len(b),baseline_test_dice=base['test_macro_dice'],branch_test_dice=trial['test_macro_dice'],
                delta_pp=float(delta.mean()),paired_image_ci95_pp=ci.tolist(),best_epoch=fit['best_epoch'],epochs=fit['epochs']))
    if results:
        temp=root/'test_comparison.json.new'
        temp.write_text(json.dumps(results,indent=2)+'\n');temp.replace(root/'test_comparison.json')
    print(json.dumps(results),flush=True)
if __name__=='__main__':run()

