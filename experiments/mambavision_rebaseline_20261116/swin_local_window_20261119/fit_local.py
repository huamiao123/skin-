"""Run the repository's Stage29 Swin + local fusion using the matched current UPer64 recipe."""
import csv
import fcntl
import gc
import os
from pathlib import Path
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parent
SCRATCH=Path("/tmp/mv_swin_local_window_20261119")
PROJECT=Path("/home/waas/mambavision_stage22/source/referring_image_segmentation")
HIST=PROJECT/"stage29_context_fusion"
WINDOW_BASE=Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/window_alignment_20261119")
sys.path.insert(0,"/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119")
from fit_sdi import train,setup,sha,read,fixed
from stage29_context_fusion import train as old

class LocalSystem(old.System):
    def __init__(self,arm):
        super().__init__("local_fusion")
        if arm != "local16":
            raise ValueError(arm)
        levels=self.encoder.model.levels[2:4]
        assert [s.window_size for s in levels]==[14,7]
        levels[0].window_size,levels[1].window_size=16,8

    def from_cache(self,f1,f2,mid,output_size):
        assert tuple(output_size)==(256,256)
        return {"main":self.probe([f1,f2,*self.suffix(mid)])}

    def forward(self,images):
        return {"main":super().forward(images)}

def build(arm):
    return LocalSystem(arm).cuda()

def preflight():
    history = read(ROOT/"scope_before_change.json")
    for fname,digest in history["source_files"].items():
        assert sha(PROJECT/fname)==digest, fname
    assert read(ROOT/"preflight.json")["initial_weights_match_stage29"]
    cfg=read(WINDOW_BASE/"config.json")
    for fname,digest in cfg["source_hashes"].items():
        assert sha(Path(fname))==digest,fname
    assert sha(old.base.HF/"model.safetensors")==cfg["encoder_weight_sha256"]
    fixed(ROOT/"protocol.json",dict(seed=42,arms=["local16"],encoder_windows=[16,8],
        training_images=1886,validation_images=762,test_images=867,group_filtered=814,epochs=100,
        cosine_horizon=100,effective_and_physical_batch=64,threshold=.5,
        model=history["model"],decoder=history["decoder"],selection=history["selection"],
        comparison="Aligned Swin + local fusion vs previously verified 16/8 UPer64 per-image values",
        source_files=history["source_files"],fit_sha=sha(Path(__file__)),
        aligned_reference_row_sha=sha(WINDOW_BASE/"test_per_image.csv"),
        reference_cfg_sha=sha(WINDOW_BASE/"config.json")))

def fit():
    preflight();train.build=build
    arm="local16"
    dst=ROOT/arm;dst.mkdir(exist_ok=True)
    if (dst/"replay.json").exists():return
    loc=SCRATCH/arm;loc.mkdir(exist_ok=True)
    train.train(arm,loc)
    for n in ("best.pth","best_validation.csv","config.json","first_update.json","history.json","complete.json"):
        tmp=dst/(n+".partial")
        shutil.copyfile(loc/n,tmp);os.replace(tmp,dst/n)
    hist=read(dst/"history.json")
    setup(42);model=build(arm).eval()
    state=torch.load(dst/"best.pth",map_location="cpu",weights_only=True)
    assert len(hist)==100 and state["epoch"]==max(hist,key=lambda r:r["val_macro_dice"])["epoch"]
    model.load_state_dict(state["model"])
    data,_=train.reference.load_data();ids=data.ids["val"]
    cache=train.reference.cache_new(len(ids));train.reference.fill_cache(model,data,"val",cache)
    _,rows=train.validate(model,cache,ids,None)
    with (dst/"best_validation.csv").open() as f:expected=list(csv.DictReader(f))
    assert len(rows)==len(expected)==762
    assert all(a["id"]==b["id"] and abs(float(a["dice"])-float(b["dice"]))<1e-10 for a,b in zip(rows,expected))
    fixed(dst/"replay.json",dict(epoch=state["epoch"],sha256=sha(dst/"best.pth"),matching_dev_rows=len(rows)))
    del state,model,cache,data
    gc.collect();torch.cuda.empty_cache()

if __name__=="__main__":
    with (ROOT/"fit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fit()
