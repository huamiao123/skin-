"""Evaluate the validation-selected 16/8 Swin + local fusion against archived aligned UPer64 predictions."""
import csv
import fcntl
import sys
from pathlib import Path
import numpy as np
import torch
from fit_local import ROOT,WINDOW_BASE,PROJECT,build,setup,read,sha,fixed,train
sys.path.insert(0,"/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119/shallow_bypass_20261119")
from test_bypass import test_inputs,score,GROUPS
from score_sdi import paired

def main():
    if (ROOT/"test_result.json").exists():return
    protocol=read(ROOT/"protocol.json")
    assert protocol["arms"]==["local16"] and protocol["encoder_windows"]==[16,8]
    ref=read(WINDOW_BASE/"locked_before_test.json")["models"]["window16_8"]
    local=read(ROOT/"local16/replay.json")
    partition=PROJECT/"stage45_upernet_decoder/identity_partition.json"
    checkpoint=ROOT/"local16/best.pth"
    table=WINDOW_BASE/"test_per_image.csv"
    assert sha(checkpoint)==local["sha256"] and sha(table)==protocol["aligned_reference_row_sha"]
    fixed(ROOT/"locked_before_test.json",dict(models={"uper16":ref,"local16":local},threshold=.5,
        partition_sha=sha(partition),groups_sha=sha(GROUPS),reference_table_sha=sha(table),
        reference_source="previously verified 16/8 UPer64 per-image Dice",
        protocol_sha=sha(ROOT/"protocol.json"),score_sha=sha(Path(__file__))))
    spec=read(partition);ids=spec["splits"]["test"];cases=read(GROUPS)["cases"]["test"]
    with table.open() as f:prior=list(csv.DictReader(f))
    assert len(set(ids))==867 and ids==[r["id"] for r in cases]==[r["id"] for r in prior]
    a=np.array([float(r["window16_8"]) for r in prior])
    assert abs(a.mean()-read(WINDOW_BASE/"test_result.json")["all"]["macro_dice"]["window16_8"])<1e-12
    x,y=test_inputs(spec);setup(42)
    model=build("local16");ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["epoch"]==local["epoch"];model.load_state_dict(ck["model"]);del ck
    b=score(model,x,y);assert np.isfinite(b).all() and b.shape==(867,)
    result={}
    for label,sel in [("all",np.ones(len(ids),dtype=bool)),("filtered",np.array([r["retain"] for r in cases]))]:
        g=np.array([r["group"] for r in cases])[sel]
        result[label]=dict(n=int(sel.sum()),dice={"uper16":float(a[sel].mean()),"local16":float(b[sel].mean())},
                           paired={"local16_minus_uper16":paired(a[sel],b[sel],g)})
    assert result["all"]["n"]==867 and result["filtered"]["n"]==814
    train.write_rows(ROOT/"test_per_image.csv",[dict(id=k,uper16=float(a[i]),local16=float(b[i])) for i,k in enumerate(ids)])
    fixed(ROOT/"test_result.json",dict(reference_rows_checked=867,trained_arms=["local16"],test_inference_arms=["local16"],**result))

if __name__=="__main__":
    with (ROOT/"score.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
