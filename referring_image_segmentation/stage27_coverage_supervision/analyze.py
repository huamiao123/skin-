"""Report the prespecified paired comparison and the trained auxiliary readout."""
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train import System,setup,dump,boundary_row
from coverage_aux import forward_training,occupancy_target
sys.path.insert(0,str(ROOT.parent))
from stage3_confirmation.data import ImageData
from evaluation import summarize_rows
from concurrent.futures import ThreadPoolExecutor

def main():
    setup();r=ROOT/'results'
    assert (r/'DONE.json').exists()
    summary=json.loads((r/'summary.json').read_text())
    rows=list(csv.DictReader((r/'per_image.csv').open()))
    by={name:{x['id']:x for x in rows if x['model']==name and x['criterion']=='best_dice'} for name in ('control','coverage_aux')}
    ids=list(by['control']);assert ids==list(by['coverage_aux'])
    areas=np.array([int(by['control'][sid]['tp'])+int(by['control'][sid]['fn']) for sid in ids])
    cuts=np.quantile(areas,[.25,.5,.75]);paired=[]
    for sid,area in zip(ids,areas):
        a,b=by['control'][sid],by['coverage_aux'][sid]
        paired.append(dict(id=sid,area_quartile=int(np.searchsorted(cuts,area,side='right'))+1,
            control_dice=float(a['dice']),aux_dice=float(b['dice']),
            delta=float(b['dice'])-float(a['dice']),
            control_recall=float(a['recall']),aux_recall=float(b['recall'])))
    delta=np.array([x['delta'] for x in paired])
    result=dict(overall=summary,paired_n=len(paired),mean_dice_delta=float(delta.mean()),
        improved=int((delta>0).sum()),worsened=int((delta<0).sum()),
        gains_over_5pp=int((delta>.05).sum()),losses_over_5pp=int((delta<-.05).sum()),
        severe_undercoverage={n:sum(float(x['recall'])<.5 for x in by[n].values()) for n in by},
        recurring_cases=[x for x in paired if x['id'] in ('305','581','791')],
        area_quartiles={str(q):dict(n=sum(x['area_quartile']==q for x in paired),
            mean_delta=float(np.mean([x['delta'] for x in paired if x['area_quartile']==q]))) for q in range(1,5)},
        archived_stage22_finetune_macro_dice=json.loads((ROOT.parent/'stage22_partial_finetune/results/summary.json').read_text())['finetune34/best_dice']['macro']['dice'],
        interpretation='One seed on development validation. Auxiliary readout below is diagnostic, not used by the deployed primary output. No novelty or test claim.',test_evaluated=False)
    with (r/'paired_diagnosis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    dump(r/'analysis.json',result)
    # Only the already selected candidate checkpoint is used; no separate auxiliary selection.
    data=ImageData(42)
    part=json.loads((r/'active_partition.json').read_text())
    pos={sid:i for i,sid in enumerate(data.ids['val'])}
    idx=torch.tensor([pos[sid] for sid in part['splits']['val']],device='cuda')
    data.images['val']=data.images['val'][idx];data.masks['val']=data.masks['val'][idx];data.ids['val']=part['splits']['val']
    model=System(True).cuda();state=torch.load(r/'coverage_aux_best_dice.pth',map_location='cpu')
    model.load_state_dict(state['model']);model.requires_grad_(False);model.eval()
    logits=[];truth=[];primary=[]
    with torch.no_grad():
        for begin in range(0,len(data.ids['val']),64):
            ix=torch.arange(begin,min(begin+64,len(data.ids['val'])),device='cuda')
            images,gt=data.batch('val',ix)
            model._capture_coverage=True;model._coverage_logits=None
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f1,f2,mid=model.prefix(images);f3,f4=model.suffix(mid.to(torch.bfloat16))
                output=model.probe([f1.to(torch.bfloat16),f2.to(torch.bfloat16),f3,f4])
            logits.append(model._coverage_logits.float())
            primary.append(output.float().sigmoid()[:,0].cpu().numpy())
            model._capture_coverage=False;model._coverage_logits=None
            truth.append((gt>=.5).float())
    all_logits=torch.cat(logits);gt=torch.cat(truth)
    ref=np.load(r/'coverage_aux_probabilities.npy')
    assert np.array_equal(np.concatenate(primary)>=.5,ref>=.5)
    p8=all_logits.sigmoid();t8=occupancy_target(gt)
    coarse_mse=(p8-t8).square().mean((1,2,3)).cpu().numpy()
    probs=F.interpolate(all_logits,size=(256,256),mode='bilinear',align_corners=False).sigmoid()[:,0].cpu().numpy()
    g=gt[:,0].cpu().numpy()>=.5
    with ThreadPoolExecutor(8) as pool:
        auxrows=list(pool.map(lambda pair:boundary_row(pair[0]>=.5,pair[1]),zip(probs,g)))
    for sid,row,mse in zip(data.ids['val'],auxrows,coarse_mse):row.update(id=sid,coarse_mse=float(mse))
    auxmetric=summarize_rows(auxrows);auxmetric.update(coarse_mse=float(coarse_mse.mean()),
        recurring_cases=[x for x in auxrows if x['id'] in ('305','581','791')],checkpoint_epoch=state['epoch'])
    dump(r/'auxiliary_readout_diagnostic.json',auxmetric)
    np.save(r/'auxiliary_coarse_probabilities.npy',p8.cpu().numpy())
    dump(ROOT/'ANALYSIS_DONE.json',dict(complete=True,mean_macro_dice_delta=result['mean_dice_delta'],test_evaluated=False))
    print(json.dumps(dict(delta=result['mean_dice_delta'],recurring_cases=result['recurring_cases'])),flush=True)
if __name__=='__main__':main()
