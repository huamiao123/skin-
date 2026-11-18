import csv,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
arms=["f3","uper64","f4"]
scores={}
summaries={}
for arm in arms:
    summaries[arm]=json.loads((root/arm/"complete.json").read_text())["best"]
    with (root/arm/"best_validation.csv").open() as handle:
        rows=list(csv.DictReader(handle))
    scores[arm]={r["id"]:float(r["dice"]) for r in rows}
    assert len(scores[arm])==len(rows)==762
    assert abs(np.mean(list(scores[arm].values()))-summaries[arm]["val_macro_dice"])<1e-10
keys=sorted(scores["f3"])
rng=np.random.default_rng(42)
samples=rng.integers(len(keys),size=(5000,len(keys)))
comparisons={}
for control in ("uper64","f4"):
    assert keys==sorted(scores[control])
    diff=np.array([scores["f3"][k]-scores[control][k] for k in keys])*100
    comparisons[control]={"f3_minus_control_pp":diff.mean(),"image_bootstrap_95pct_pp":np.quantile(diff[samples].mean(1),[.025,.975]).tolist(),"images_better":int((diff>0).sum()),"images_worse":int((diff<0).sum())}
report={"best_by_validation":summaries,"paired_validation":comparisons,"validation_only":True,"seed":42,"resampling_unit":"image; patient groups not established","bootstrap_samples":5000}
dest=root/"comparison.json"
tmp=dest.with_suffix(".tmp")
tmp.write_text(json.dumps(report,indent=2)+"\n")
tmp.replace(dest)
