"""Locked 2x3 comparison of plain UPer64, product SDI and inference-only weak-region gain."""
import csv
import fcntl
import sys
from pathlib import Path
import numpy as np
import torch
from fit_aligned import ROOT, PRIOR, ALIGNED, original_build, build_aligned, fixed, read, setup, sha, original
sys.path.insert(0,str(PRIOR/"shallow_bypass_20261119"))
from test_bypass import test_inputs, GROUPS, score as score_plain
sys.path.insert(0,str(PRIOR/"feature_maps_20261119"))
from weak_gain import predict
from score_sdi import paired

def csv_rows(path):
    with path.open() as f: return list(csv.DictReader(f))

def dice(prob, truth):
    assert torch.isfinite(prob).all()
    mask=prob>=.5
    den=mask.sum((1,2))+truth.sum((1,2))
    return torch.where(den>0,2*(mask & truth).sum((1,2)).double()/den,1.).cpu().numpy()

@torch.inference_mode()
def score_product(model,x,y):
    native=[];weak=[]
    mu=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    for image,target in zip(x.split(64),y.split(64)):
        img=(image.cuda().float()/255.-mu)/std
        with torch.autocast("cuda",dtype=torch.bfloat16):
            f1,f2,mid=model.prefix(img)
            feats=[f1,f2,*model.suffix(mid)]
            prob=model.probe(feats,(256,256))["main"][:,0].float().sigmoid()
        truth=target.cuda()
        native.append(dice(prob,truth))
        weak.append(dice(predict(model,feats,"weak2"),truth))
    return np.concatenate(native),np.concatenate(weak)

def main():
    if (ROOT/"test_result.json").exists():return
    comp=read(ROOT/"complete.json");hist=read(ROOT/"history.json")
    assert len(hist)==comp["epochs"]==100 and max(hist,key=lambda r:r["val_macro_dice"])==comp["best"]
    assert abs(comp["replay"]-comp["best"]["val_macro_dice"])<1e-10
    priorlock=read(PRIOR/"runs/locked_before_test.json")["models"]
    plainlock=read(ALIGNED/"locked_before_test.json")["models"]
    models={
      "plain14":(original.REF/"best.pth",plainlock["native14_7"]),
      "plain16":(ALIGNED/"best.pth",plainlock["window16_8"]),
      "sdi14":(PRIOR/"runs/product_42/best.pth",priorlock["product_42"]),
      "sdi16":(ROOT/"best.pth",dict(epoch=comp["best"]["epoch"],sha256=comp["sha256"]))}
    for path,info in models.values():assert sha(path)==info["sha256"]
    partition=original.train.PROJECT/"stage45_upernet_decoder/identity_partition.json"
    fixed(ROOT/"locked_before_test.json",dict(checkpoints={k:dict(epoch=v[1]["epoch"],sha256=v[1]["sha256"]) for k,v in models.items()},
        threshold=.5,weak2_uses_same_sdi_checkpoint=True,protocol_sha=sha(ROOT/"protocol.json"),script_sha=sha(Path(__file__)),
        historical_gain_sha=sha(PRIOR/"feature_maps_20261119/weak_gain.py"),partition_sha=sha(partition),groups_sha=sha(GROUPS)))
    part=read(partition);ids=part["splits"]["test"];cases=read(GROUPS)["cases"]["test"]
    native_old=csv_rows(ALIGNED/"test_per_image.csv")
    gain_old=csv_rows(PRIOR/"feature_maps_20261119/gain_test_per_image.csv")
    assert len(set(ids))==867 and ids==[r["id"] for r in cases]==[r["id"] for r in native_old]==[r["id"] for r in gain_old]
    x,y=test_inputs(part)
    scores={}
    for name,(path,meta) in models.items():
        setup(42)
        builder=build_aligned if name.endswith("16") else original_build
        net=builder("add" if name.startswith("plain") else "product",42).eval()
        ck=torch.load(path,map_location="cpu",weights_only=True)
        assert ck["epoch"]==meta["epoch"]
        net.load_state_dict(ck["model"]);del ck
        if name.startswith("plain"):scores[name]=score_plain(net,x,y)
        else:scores[name],scores[name.replace("sdi","weak")]=score_product(net,x,y)
        del net
        torch.cuda.empty_cache()
    for tag,table,col in [("plain14",native_old,"native14_7"),("plain16",native_old,"window16_8"),
                          ("sdi14",gain_old,"native"),("weak14",gain_old,"weak2")]:
        assert np.allclose(scores[tag],[float(r[col]) for r in table],rtol=0,atol=1e-12),tag
    comparisons=[("plain16","sdi16"),("plain16","weak16"),("sdi16","weak16"),("sdi14","sdi16"),("weak14","weak16")]
    result={}
    for label,keep in [("all",np.ones(len(ids),dtype=bool)),("filtered",np.array([r["retain"] for r in cases]))]:
        g=np.array([r["group"] for r in cases])[keep];v={k:a[keep] for k,a in scores.items()}
        result[label]=dict(n=int(keep.sum()),dice={k:float(a.mean()) for k,a in v.items()},
            paired={b+"_minus_"+a:paired(v[a],v[b],g) for a,b in comparisons},
            interaction={typ:paired(v[typ+"14"]-v["plain14"],v[typ+"16"]-v["plain16"],g) for typ in ("sdi","weak")})
    assert (result["all"]["n"],result["filtered"]["n"])==(867,814)
    original.train.write_rows(ROOT/"test_per_image.csv",[dict(id=s,**{k:float(a[i]) for k,a in scores.items()}) for i,s in enumerate(ids)])
    fixed(ROOT/"test_result.json",dict(verified_historical_arms=4,verified_rows_per_arm=len(ids),**result))

if __name__=="__main__":
    with (ROOT/"score.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
