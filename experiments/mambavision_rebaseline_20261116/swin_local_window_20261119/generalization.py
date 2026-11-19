"""Read-only, aligned-window generalization check on unaugmented train and development images."""
import csv
import fcntl
import gc
from pathlib import Path
import sys
import torch
import numpy as np
from fit_local import ROOT,build as local_build,setup,train,fixed,sha,read
sys.path.insert(0,"/home/featurize/work/mambavision_seg_rebaseline_20261116/window_alignment_20261119")
from compare_windows import build as native_build
sys.path.insert(0,"/home/featurize/work/mambavision_seg_rebaseline_20261116/sdi_window_alignment_20261119")
from fit_aligned import build_aligned
MODELS={
    "UPer64":(lambda:native_build("window16_8"),Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/window_alignment_20261119"),Path("/tmp/mv_s_uper64_window_16_8_20261119")),
    "SDI":(lambda:build_aligned("product",42),Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/sdi_window_alignment_20261119"),Path("/tmp/mv_sdi_window_16_8_20261119")),
    "SwinLocal":(lambda:local_build("local16"),ROOT/"local16",Path("/tmp/mv_swin_local_window_20261119/local16"))
}

@torch.inference_mode()
def evaluate(model,data,split):
    out=[]
    for begin in range(0,len(data.ids[split]),64):
        ix=torch.arange(begin,min(begin+64,len(data.ids[split])),device="cuda")
        img,g=data.batch(split,ix,None)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(img)["main"][:,0].float()
        pred=logits.sigmoid()>=.5;gt=g[:,0]>=.5
        assert torch.isfinite(logits).all()
        den=pred.sum((1,2))+gt.sum((1,2))
        out.append(torch.where(den>0,2*(pred&gt).sum((1,2)).double()/den,1.).cpu().numpy())
    return float(np.concatenate(out).mean())

def main():
    if (ROOT/"generalization.json").exists():return
    assert (ROOT/"test_result.json").is_file()
    setup(42);data,_=train.reference.load_data()
    assert len(data.ids["train"])==1886 and len(data.ids["val"])==762
    result={}
    for name,(builder,work,scratch) in MODELS.items():
        model=builder().eval()
        assert [stage.window_size for stage in model.encoder.model.levels[2:4]]==[16,8]
        hist=read(work/"history.json")
        assert len(hist)==100
        record={}
        for tag,path in [("best",work/"best.pth"),("last",scratch/"latest.pth")]:
            state=torch.load(str(path),map_location="cpu",weights_only=False,mmap=True)
            epoch=state["epoch"] if tag=="best" else len(state["history"])
            assert epoch==(max(hist,key=lambda r:r["val_macro_dice"])["epoch"] if tag=="best" else 100)
            model.load_state_dict(state["model"]);del state
            t=evaluate(model,data,"train")
            v=evaluate(model,data,"val")
            assert abs(v-hist[epoch-1]["val_macro_dice"])<1e-10,(name,tag,v,hist[epoch-1]["val_macro_dice"])
            record[tag]=dict(epoch=epoch,train_mean=t,development_mean=v,gap_pp=100*(t-v),checkpoint_sha=sha(path))
        result[name]=dict(decoder_parameters=sum(p.numel() for p in model.probe.parameters()),**record)
        del model;gc.collect();torch.cuda.empty_cache()
    fixed(ROOT/"generalization.json",dict(scores=result,train=1886,development=762,test_images_used=0,
        windows=[16,8],normal_image_orientation=True,metric="per-image foreground Dice with threshold 0.5, common eval mode; best vs epoch 100",
        selection="checkpoints already locked by prior experiment; one seed; development used in best epoch selection",
        source_sha=sha(Path(__file__))))

if __name__=="__main__":
    with (ROOT/"generalization.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
