"""Retrain the existing product SDI with aligned encoder windows, retaining the original fit loop."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import torch

ROOT=Path(__file__).resolve().parent
SCRATCH=Path("/tmp/mv_sdi_window_16_8_20261119")
PRIOR=Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119")
ALIGNED=Path("/home/featurize/work/mambavision_seg_rebaseline_20261116/window_alignment_20261119")
sys.path.insert(0,str(PRIOR))
import fit_sdi as original
from fit_sdi import fixed, read, sha, setup
original_build=original.build

def build_aligned(arm,seed):
    model=original_build(arm,seed)
    levels=model.encoder.model.levels[2:4]
    assert [stage.window_size for stage in levels]==[14,7]
    for stage,width in zip(levels,(16,8)):
        stage.window_size=width
    return model

def main():
    with (ROOT/"fit.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=read(PRIOR/"runs/product_42/config.json")
        assert sha(PRIOR/"fit_sdi.py")==raw["training_sha"]
        assert sha(PRIOR.parent/"smoke_sdi.py")==raw["adapter_sha"]
        assert sha(PRIOR.parent/"verified_sources/sdi.py")==raw["source_sha"]
        fixed(ROOT/"protocol.json",dict(seed=42,windows=[16,8],epochs=100,effective_batch=64,
            train=1886,validation=762,test=867,filtered=814,input=[256,256],
            model="MambaVision-S ImageNet initialization, native UPer64 plus original four-scale product SDI at F2",
            selection="earliest highest validation mean per-image foreground Dice of unmodified SDI",
            training="original fit_sdi.fit, train stages 3/4 in encoder eval; unchanged BCE + soft Dice",
            inference=["original product SDI (gain 1)","same weights with original fixed weak2 rule"],threshold=.5,
            paired_comparisons=["aligned SDI minus aligned UPer64","aligned weak2 minus aligned UPer64",
            "aligned weak2 minus aligned SDI","window effect for each SDI readout","difference of window effects for SDI versus UPer64"],
            case_level_interval="5000 resamples of prior image groups; single seed and previously exposed test",
            fit_sha=sha(Path(__file__)),ref_config_sha=sha(ALIGNED/"config.json"),prior_product=raw,
            weak_gain_sha=sha(PRIOR/"feature_maps_20261119/weak_gain.py"),
            plain_aligned_sha=sha(ALIGNED/"best.pth")))
        if (ROOT/"complete.json").exists():return
        setup(42)
        product=build_aligned("product",42).eval()
        plain=build_aligned("add",42).eval()
        inputs=torch.arange(3*256*256,device="cuda").reshape(1,3,256,256).float().remainder(251)/125.-1
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            a=product(inputs)["main"];b=plain(inputs)["main"]
            assert torch.equal(a,b) and torch.isfinite(a).all()
        head_params=sum(v.numel() for v in product.probe.parameters())
        assert head_params==raw["head_parameters"]==1240961
        fixed(ROOT/"preflight.json",dict(initial_logits_match_aligned_native=True,head_parameters=head_params,
              added_vs_plain=head_params-1089089,windows=[x.window_size for x in product.encoder.model.levels[2:4]],
              historical_training_and_sdi_sources_exact=True))
        del product,plain,a,b
        torch.cuda.empty_cache()
        # The original loop checks the initial minibatch loss against its designated plain reference.
        original.REF=ALIGNED
        original.build=build_aligned
        original.run_dir=lambda arm,seed: SCRATCH
        original.fit("product",42)
        for name in ("best.pth","best_validation.csv","history.json","first_update.json","config.json","complete.json"):
            tmp=ROOT/(name+".partial")
            shutil.copyfile(SCRATCH/name,tmp);os.replace(tmp,ROOT/name)
        assert sha(ROOT/"best.pth")==read(ROOT/"complete.json")["sha256"]

if __name__=="__main__":main()
