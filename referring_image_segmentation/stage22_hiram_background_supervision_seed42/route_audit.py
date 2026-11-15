"""Audit both trained routers on all validation images with area-aware labels."""
from common import *
from models import NAMES, DecoderProbe, FrozenEncoder, region_targets, tensor_hash
from stage3_confirmation.data import ImageData
from torch.nn import functional as F


@torch.no_grad()
def area_targets(mask):
    fg=(mask>=.5).float()
    count=F.conv2d(fg,torch.ones((1,1,5,5),device=fg.device),padding=2)
    boundary=((count>0)^(count==25)).float()
    interior=fg*(1-boundary)
    return F.adaptive_avg_pool2d(torch.cat((1-boundary-interior,boundary,interior),1),(32,32))


def ratio(num,den):
    return float(num/den) if den else None


def main():
    setup()
    done=json.loads((ROOT/'results/DONE.json').read_text())
    assert done['epochs']==300 and all(x==9000 for x in done['updates'].values())
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    n=len(data.ids['val']);assert n==808
    encoder=FrozenEncoder().cuda().eval()
    assert tensor_hash(encoder)==done['encoder_hashes']['mambavision']
    models={}
    acc={}
    for name in NAMES:
        model=DecoderProbe('hiram').cuda().eval()
        ck=torch.load(ROOT/'results'/name/'best_dice.pth',map_location='cpu')
        model.load_state_dict(ck['model'],strict=True)
        models[name]=model
        acc[name]=[dict(confusion=torch.zeros((3,3),device='cuda',dtype=torch.float64),
                        weighted=torch.zeros(3,device='cuda',dtype=torch.float64),
                        soft=torch.zeros(3,device='cuda',dtype=torch.float64),
                        hard=torch.zeros(3,device='cuda',dtype=torch.float64),
                        entropy=torch.zeros((),device='cuda',dtype=torch.float64)) for _ in range(2)]
    target_area=torch.zeros(3,device='cuda',dtype=torch.float64)
    target_bilinear=torch.zeros(3,device='cuda',dtype=torch.float64)
    counts={name:[] for name in NAMES}
    with torch.no_grad():
        for start in range(0,n,64):
            end=min(start+64,n)
            image,gt=data.batch('val',torch.arange(start,end,device='cuda'))
            area=area_targets(gt)
            bd,inside,bg=region_targets(gt)
            bilinear=torch.cat((bg,bd,inside),1)
            target_area+=area.sum((0,2,3)).double()
            target_bilinear+=bilinear.sum((0,2,3)).double()
            with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(image)
            for name,model in models.items():
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,regions=model(features,return_regions=True)
                pred=logits.float().sigmoid()>=.5;truth=gt>=.5
                vals=[(pred&truth).sum((1,2,3)),(pred&~truth).sum((1,2,3)),
                      (~pred&truth).sum((1,2,3)),(~pred&~truth).sum((1,2,3))]
                counts[name].append(torch.stack(vals,1).cpu().numpy())
                for i,logp in enumerate(regions):
                    p=logp.float().softmax(1)
                    hard=F.one_hot(logp.argmax(1),3).permute(0,3,1,2).float()
                    a=acc[name][i]
                    a['confusion']+=torch.einsum('bkhw,bjhw->kj',area,hard).double()
                    a['weighted']+=(p*area).sum((0,2,3)).double()
                    a['soft']+=p.sum((0,2,3)).double()
                    a['hard']+=hard.sum((0,2,3)).double()
                    a['entropy']+=(-(p*p.clamp_min(1e-12).log()).sum(1)).sum().double()
            if start%256==0:print('ROUTE',end,'/',n,flush=True)
    total=n*32*32
    area_np=target_area.cpu().numpy();bilinear_np=target_bilinear.cpu().numpy()
    results={}
    for name in NAMES:
        actual=np.concatenate(counts[name]).astype(np.int64)
        saved=np.load(ROOT/'results'/name/'best_dice/counts.npy')
        assert np.array_equal(actual,saved),(name,'per-image count mismatch')
        results[name]={}
        for i,branch in enumerate(('coarse','skip')):
            a=acc[name][i]
            mat=a['confusion'].cpu().numpy()
            soft=a['soft'].cpu().numpy();hard=a['hard'].cpu().numpy();weighted=a['weighted'].cpu().numpy()
            results[name][branch]=dict(hard_fraction=(hard/total).tolist(),
              mean_soft_probability=(soft/total).tolist(),mean_entropy_nats=float(a['entropy'].item()/total),
              soft_target_by_hard_route=(mat/total).tolist(),
              class_recall=[ratio(mat[k,k],area_np[k]) for k in range(3)],
              class_precision=[ratio(mat[k,k],hard[k]) for k in range(3)],
              class_target_weighted_mean_probability=[ratio(weighted[k],area_np[k]) for k in range(3)])
    assert tensor_hash(encoder)==done['encoder_hashes']['mambavision']
    dump(ROOT/'analysis/route_summary.json',dict(n=n,classes=['background','boundary','interior'],
        target_area_fraction=(area_np/total).tolist(),training_bilinear_target_fraction=(bilinear_np/total).tolist(),
        evaluation='deterministic argmax, ground truth only for posthoc metrics',
        area_definition='width2 binary morphology at 256, area-pool to 32',models=results))
    print('ROUTE AUDIT COMPLETE',json.dumps({n:results[n]['coarse']['hard_fraction'] for n in NAMES}),flush=True)


if __name__=='__main__':main()
