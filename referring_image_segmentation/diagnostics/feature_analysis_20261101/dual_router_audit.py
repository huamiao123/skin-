"""Inspect the older DINOv3+VMamba HiRAM routers without changing artifacts."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
S16=PROJECT/'stage16_hiram_dual_seed42'
OUT=HERE/'outputs'
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(S16))
from common import setup  # noqa: E402
from models import RegionProbe,tensor_hash  # noqa: E402
from stage4_fusion.models import EncoderPair  # noqa: E402
from stage3_confirmation.data import ImageData  # noqa: E402


def main():
    setup();data=ImageData(42);n=len(data.ids['val'])
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=EncoderPair().cuda().eval()
    hashes={name:tensor_hash(getattr(encoder,name)) for name in ('dinov3','vmamba')}
    assert hashes==json.loads((S16/'preflight.json').read_text())['encoder_hashes']
    nets={};counts={};route={}
    for name in ('region_only','hiram'):
        model=RegionProbe(name).cuda().eval()
        ck=torch.load(S16/'results'/name/'best_dice.pth',map_location='cpu')
        model.load_state_dict(ck['model'],strict=True)
        assert ck['encoder_hashes']==hashes
        nets[name]=model;counts[name]=[];route[name]=np.zeros((2,3,3),dtype=np.float64)
    print('DUAL ENCODER ROUTE AUDIT',flush=True)
    with torch.no_grad():
      for start in range(0,n,64):
        end=min(n,start+64);x,y=data.batch('val',torch.arange(start,end,device='cuda'))
        gt=y>=.5
        hits=F.conv2d(gt.float(),torch.ones((1,1,5,5),device='cuda'),padding=2)
        boundary=(hits>0)^(hits==25);inside=gt&~boundary;bg=~boundary&~inside
        target=F.avg_pool2d(torch.cat((bg.float(),boundary.float(),inside.float()),1),16)
        with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(x)
        for name,model in nets.items():
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,regions=model(features,return_regions=True)
            pred=logits.float().sigmoid()>=.5
            v=[(pred & gt).sum((1,2,3)),(pred & ~gt).sum((1,2,3)),
               (~pred & gt).sum((1,2,3)),(~pred & ~gt).sum((1,2,3))]
            counts[name].append(torch.stack(v,1).cpu().numpy())
            for branch,logits in enumerate(regions):
                cls=logits.argmax(1)
                for true in range(3):
                    for predicted in range(3):
                        route[name][branch,true,predicted]+=float((target[:,true]*(cls==predicted)).sum().cpu())
        if start%256==0:print(f'  {end}/{n}',flush=True)
    res={}
    for name in nets:
        recorded=np.load(S16/'results'/name/'best_dice/counts.npy')
        assert np.array_equal(np.concatenate(counts[name]),recorded),name
        confusion=route[name]
        total=confusion.sum(axis=(1,2))
        res[name]=dict(all_808_per_image_counts_match=True,
            predicted_hard_area_fraction=[(confusion[b].sum(0)/total[b]).tolist() for b in range(2)],
            true_area_fraction=[(confusion[b].sum(1)/total[b]).tolist() for b in range(2)],
            true_soft_area_by_predicted_argmax=confusion.tolist())
    dest=OUT/'dual_router_audit.json'
    dest.write_text(json.dumps(dict(n=n,seed=42,encoder_hashes=hashes,classes=['background','boundary','interior'],
      branches=['DINOv3','VMamba'],models=res),ensure_ascii=False,indent=2)+'\n')
    print('FINISHED',json.dumps({k:v['predicted_hard_area_fraction'] for k,v in res.items()}),flush=True)


if __name__=='__main__':main()
