"""Window-only check on the original seed-42 MambaVision-S and UPer64 training loop."""
import csv
import fcntl
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
RUN = Path("/tmp/mv_s_uper64_window_16_8_20261119")
REFERENCE = Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/uper64_feature_audit_20261119") / "runs/add"
sys.path.insert(0, "/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119")
from fit_sdi import fixed, read, sha, setup, train
from score_sdi import GROUPS, summarize
from feature_decoder import build as native_build

def build(arm):
    model = native_build("add", seed=42)
    late = model.encoder.model.levels[2:4]
    assert [stage.window_size for stage in late] == [14, 7]
    if arm == "window16_8":
        for level, width in zip(late, (16, 8)):
            level.window_size = width
    elif arm != "native14_7":
        raise ValueError(arm)
    return model

def preflight():
    cfg = read(REFERENCE / "config.json")
    for file, digest in cfg["source_hashes"].items():
        assert sha(Path(file)) == digest, file
    setup(42)
    native = build("native14_7").eval()
    setup(42)
    candidate = build("window16_8").eval()
    a, b = native.state_dict(), candidate.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert [n for n,p in native.named_parameters() if p.requires_grad] == [n for n,p in candidate.named_parameters() if p.requires_grad]
    geometries = {}
    for name,model,expected in [("native",native,[[4,196],[4,49]]),("aligned",candidate,[[1,256],[1,64]])]:
        spans = []
        hooks = [level.blocks[0].register_forward_pre_hook(lambda module, args: spans.append(list(args[0].shape[:2]))) for level in model.encoder.model.levels[2:4]]
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
            out = model(torch.zeros(1,3,256,256,device="cuda"))["main"]
        for h in hooks: h.remove()
        assert list(out.shape) == [1,1,256,256] and torch.isfinite(out).all() and spans == expected
        model.train()
        assert all(not level.training for level in model.encoder.model.levels)
        geometries[name] = spans
    fixed(WORK / "preflight.json",dict(state_dict_tensors_equal=len(a),trainable_names_equal=True,actual_windows_tokens=geometries,source_hashes_verified=True))
    fixed(WORK / "protocol.json",dict(reference=str(REFERENCE),reference_config_sha=sha(REFERENCE/"config.json"),
          reference_weight_sha=sha(REFERENCE/"best.pth"),runner_sha=sha(Path(__file__)),window_control=[14,7],
          window_candidate=[16,8],seed=42,epochs=100,batch=64,input=[256,256],selection="earliest maximum validation per-image foreground Dice",
          train=1886,validation=762,test=867,sensitivity=814,test_threshold=0.5,initialization="Same ImageNet-1K weights; same fresh native UPer64",
          training_policy="As in reference, train stages 3 and 4 with encoder eval, no stage-output LN, no auxiliary head"))
          
def evaluate():
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    import gc
    cases=read(GROUPS)["cases"]["test"]
    parent=train.PROJECT.parent.parent
    spec=train.PROJECT/"stage45_upernet_decoder/identity_partition.json"
    split=read(spec)
    ids=split["splits"]["test"]
    assert ids==[r["id"] for r in cases] and len(ids)==867 and sum(r["retain"] for r in cases)==814
    dirs={"native14_7":REFERENCE,"window16_8":WORK}
    locked={}
    for arm,loc in dirs.items():
        record=read(loc/"complete.json")["best"]
        locked[arm]=dict(epoch=record["epoch"],seed=42,sha256=sha(loc/"best.pth"))
    fixed(WORK/"locked_before_test.json",dict(models=locked,threshold=.5,split_sha=sha(spec),groups_sha=sha(GROUPS)))
    manifest_file=parent/"local_audit_records/test_assets/test_manifest.json"
    assert sha(manifest_file)==split["audit_sha256"]["test_manifest.json"]
    manifest={r["id"]:r for r in read(manifest_file)}
    def get(sid):
        base=parent/"heldout_data/test"
        img=base/"images"/(sid+".png")
        gt=base/"masks"/(sid+".png")
        assert sha(img)==manifest[sid]["image_256_sha256"] and sha(gt)==manifest[sid]["mask_256_sha256"]
        with Image.open(img) as a, Image.open(gt) as b:
            return np.array(a.convert("RGB")),np.array(b.convert("L"))>=128
    with ThreadPoolExecutor(8) as pool: records=list(pool.map(get,ids))
    xs=torch.from_numpy(np.stack([r[0] for r in records])).permute(0,3,1,2).contiguous()
    ys=torch.from_numpy(np.stack([r[1] for r in records]));del records
    mu=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    results={}
    for arm,loc in dirs.items():
        setup(42)
        model=build(arm).eval()
        ck=torch.load(loc/"best.pth",map_location="cpu",weights_only=True)
        assert ck["epoch"]==locked[arm]["epoch"]
        model.load_state_dict(ck["model"]);del ck
        chunks=[]
        with torch.inference_mode():
            for img,truth in zip(xs.split(64),ys.split(64)):
                rgb=(img.cuda().float()/255.-mu)/std
                with torch.autocast("cuda",dtype=torch.bfloat16): logits=model(rgb)["main"][:,0].float()
                assert torch.isfinite(logits).all()
                mask=logits.sigmoid()>=.5
                truth=truth.cuda()
                den=mask.sum((1,2))+truth.sum((1,2))
                chunks.append(torch.where(den>0,2*(mask&truth).sum((1,2)).double()/den,1.).cpu().numpy())
        results[arm]=np.concatenate(chunks)
        del model;gc.collect();torch.cuda.empty_cache()
    historic=Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119")/"runs/test_per_image.csv"
    if not historic.is_file():
        historic=Path("/tmp/mv_s_uper64_sdi_formal_20261119/test_per_image.csv")
    with historic.open() as fh: prior=list(csv.DictReader(fh))
    assert ids==[r["id"] for r in prior]
    assert np.allclose(results["native14_7"],[float(r["add_42"]) for r in prior],rtol=0,atol=1e-12)
    table=[dict(id=k,**{n:float(v[i]) for n,v in results.items()}) for i,k in enumerate(ids)]
    train.write_rows(WORK/"test_per_image.csv",table)
    fixed(WORK/"test_result.json",dict(verified_native_rows=len(table),**summarize(results,cases,locked)))

def main():
    RUN.mkdir(exist_ok=True)
    with (WORK/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/"test_result.json").exists(): return
        preflight()
        torch.cuda.empty_cache()
        train.build=build
        train.train("window16_8",RUN)
        for name in ("config.json","first_update.json","best.pth","best_validation.csv","history.json","complete.json"):
            destination=WORK/name
            tmp=destination.with_name(name+".partial")
            shutil.copyfile(RUN/name,tmp);os.replace(tmp,destination)
        setup(42)
        replay_model = build("window16_8").eval()
        saved = torch.load(WORK/"best.pth",map_location="cpu",weights_only=True)
        replay_model.load_state_dict(saved["model"])
        data,_ = train.reference.load_data()
        valids = data.ids["val"]
        cache = train.reference.cache_new(len(valids))
        train.reference.fill_cache(replay_model,data,"val",cache)
        _,rows = train.validate(replay_model,cache,valids,None)
        with (WORK/"best_validation.csv").open() as fh: original=list(csv.DictReader(fh))
        assert [r["id"] for r in original]==[r["id"] for r in rows] and len(rows)==762
        assert all(abs(float(a["dice"])-float(b["dice"]))<1e-10 for a,b in zip(original,rows))
        fixed(WORK/"validation_replay.json",dict(epoch=saved["epoch"],matching_rows=len(rows)))
        del data,cache,saved,replay_model
        torch.cuda.empty_cache()
        evaluate()
        print("Window comparison complete",flush=True)

if __name__=="__main__":
    main()
