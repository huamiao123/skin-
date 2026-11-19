"""Frozen, per-token linear readouts of lesion occupancy. Does not alter the segmentation model."""
import importlib
from pathlib import Path
import csv
import fcntl
import math
import numpy as np
import torch
from torch.nn import functional as F
from compare_windows import WORK, REFERENCE, build, fixed, read, setup, sha, train

STAGES = ((2,3,16),(3,2,8))
NAMES = ("ssm","non_ssm","joint","projected")

def select_view(pre, post, name):
    half = pre.shape[-1]//2
    if name == "ssm":
        return pre[..., :half]
    if name == "non_ssm":
        return pre[..., half:]
    return pre if name == "joint" else post

def main():
    if (WORK/"branch_readout.json").exists():
        return
    fixed(WORK/"branch_readout_protocol.json",dict(checkpoint_sha=sha(REFERENCE/"best.pth"),
        stage_blocks=[[3,3],[4,2]],block_indices="zero-based",stage_numbers="one-based",
        model_windows=[14,7],mode="frozen eval with normal bfloat16 forward",
        target="fraction of foreground in the corresponding input area, mask cutoff 0.5",
        train="32 sampled spatial cells per image without replacement, all 1886 training images, natural image frequency",
        validation="all cells of all 762 development images, regular orientation, no augmentation",
        estimator="ordinary least squares with intercept, float64 sufficient statistics, pseudoinverse rcond=1e-12 on Gram matrix",
        features=list(NAMES),score="occupancy RMSE; image-level SSE/number of cells; all readouts use same positions",
        fixed_random_seed=42,test_images_used=0,script_sha=sha(Path(__file__))))
    setup(42)
    torch.set_num_threads(4)
    model=build("native14_7").eval()
    saved=torch.load(REFERENCE/"best.pth",map_location="cpu",weights_only=True)
    model.load_state_dict(saved["model"]);del saved
    data,_=train.reference.load_data()
    assert [len(data.ids[k]) for k in ("train","val")]==[1886,762]
    captured={}
    hooks=[]
    def capture(stage):
        def hook(module,inputs,output):
            captured[stage]=(inputs[0].detach(),output.detach())
        return hook
    for stage,block,_ in STAGES:
        proj=model.encoder.model.levels[stage].blocks[block].mixer.out_proj
        hooks.append(proj.register_forward_hook(capture(stage)))
    reverse=importlib.import_module(type(model.encoder.model.levels[2]).__module__).window_reverse
    rng=np.random.default_rng(42)
    grams={};rhs={};weights={};rank={};errors={}
    positions={}
    for stage,_,size in STAGES:
        positions[stage]=np.stack([rng.choice(size*size,32,replace=False) for _ in data.ids["train"]])
    with torch.inference_mode():
        for split in ("train","val"):
            for start in range(0,len(data.ids[split]),32):
                ids=torch.arange(start,min(start+32,len(data.ids[split])),device="cuda")
                images,mask=data.batch(split,ids,None)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    _,_,mid=model.prefix(images)
                    model.suffix(mid)
                for stage,_,size in STAGES:
                    ws=model.encoder.model.levels[stage].window_size
                    hp=math.ceil(size/ws)*ws
                    def restore(t):
                        q=reverse(t,ws,hp,hp)[:,:,:size,:size].permute(0,2,3,1)
                        return q.reshape(len(ids),size*size,-1)
                    pre,post=map(restore,captured.pop(stage))
                    target=F.adaptive_avg_pool2d((mask>=.5).float(),(size,size)).reshape(len(ids),-1)
                    if split=="train":
                        pos=torch.from_numpy(positions[stage][start:start+len(ids)]).to("cuda")
                        batch_ids=torch.arange(len(ids),device="cuda")[:,None]
                        pre,post,target=pre[batch_ids,pos],post[batch_ids,pos],target[batch_ids,pos]
                    y=target.double().cpu()
                    pre=pre.float().cpu();post=post.float().cpu()
                    for name in NAMES:
                        key=(stage,name)
                        x=select_view(pre,post,name).reshape(y.numel(),-1).double()
                        x=torch.cat((x,x.new_ones((len(x),1))),dim=1)
                        if split=="train":
                            if key not in grams:
                                grams[key]=x.new_zeros((x.shape[1],x.shape[1]))
                                rhs[key]=x.new_zeros(x.shape[1])
                            grams[key].addmm_(x.T,x)
                            rhs[key].addmv_(x.T,y.ravel())
                        else:
                            preds=(x@weights[key]).reshape_as(y)
                            err=(preds-y).square().mean(dim=1)
                            errors.setdefault(key,[]).extend(err.tolist())
            if split=="train":
                for key,a in grams.items():
                    v,u=torch.linalg.eigh(a)
                    keep=v>v[-1]*1e-12
                    rank[key]=int(keep.sum())
                    weights[key]=u[:,keep]@((u[:,keep].T@rhs[key])/v[keep])
                print("Readouts fitted on training images",flush=True)
    for h in hooks: h.remove()
    result={}
    for stage,_,size in STAGES:
        result["stage"+str(stage+1)]={n:dict(rmse=float(np.sqrt(np.mean(errors[(stage,n)]))),
            n_images=len(errors[(stage,n)]),cells_per_image=size*size,fit_rank_with_intercept=rank[(stage,n)]) for n in NAMES}
    rows=[dict(id=sid,**{"stage"+str(s+1)+"_"+n:errors[(s,n)][i] for s,_,_ in STAGES for n in NAMES})
          for i,sid in enumerate(data.ids["val"])]
    train.write_rows(WORK/"branch_readout_per_image.csv",rows)
    fixed(WORK/"branch_readout.json",dict(development_readout=result,complete=True))
    print(result,flush=True)

if __name__=="__main__":
    with (WORK/"probe.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
