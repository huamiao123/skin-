import csv,json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
def read(d,f):return json.loads((d/f).read_text())
def fetch(arm):
    d=root/arm/'ISIC2017'
    t=read(d,'final_test.json');fit=read(d,'fit_complete.json')
    p=d/'final_test_per_image.csv'
    assert t['n']==600 and t['threshold']==.5 and t['epoch']==fit['best_epoch'] and t['model_sha256']==fit['sha256']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==t['per_image_sha256']
    with p.open() as h:rows=list(csv.DictReader(h))
    assert len(rows)==600 and len({r['image_id'] for r in rows})==600
    sc={r['image_id']:float(r['foreground_dice']) for r in rows}
    assert abs(np.mean(list(sc.values()))-t['test_macro_dice'])<1e-10
    c=read(d,'config.json');c.pop('method')
    return c,sc,t,fit
a,x,ta,fa=fetch('baseline');b,y,tb,fb=fetch('native_two_branch')
assert a==b and set(x)==set(y)
with (root/'split.csv').open() as h:assert set(x)=={r['image_id'] for r in csv.DictReader(h) if r['split']=='test'}
delta=np.array([100*(y[k]-x[k]) for k in x]);ci=np.percentile(np.random.default_rng(42).choice(delta,(5000,len(delta))).mean(1),[2.5,97.5])
out=dict(test_n=600,baseline=ta['test_macro_dice'],native_two_branch=tb['test_macro_dice'],delta_pp=delta.mean(),paired_image_95ci_pp=ci.tolist(),baseline_best_epoch=fa['best_epoch'],branch_best_epoch=fb['best_epoch'])
p=root/'comparison.json.new';p.write_text(json.dumps(out,indent=2));p.replace(root/'comparison.json')
print(out,flush=True)

