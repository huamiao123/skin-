"""Read-only inference interventions and visual diagnostics for stages 17/19."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
S19 = PROJECT / 'stage19_region_mambavision_encoder_seed42'
S17 = PROJECT / 'stage17_mambavision_decoder_hiram_seed42'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(S19))
from common import setup, sha  # noqa: E402
from models import Segmentor, region_targets  # noqa: E402
from train import load_trainable  # noqa: E402
from stage3_confirmation.data import ImageData  # noqa: E402
from evaluation import aggregate_counts  # noqa: E402
import stage17_mambavision_decoder_hiram_seed42.models as d17  # noqa: E402

CLASSES = ('background', 'boundary', 'interior')
OUT = HERE / 'outputs'
BATCH = 64


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def relative(a, b):
    a, b = a.float().flatten(1), b.float().flatten(1)
    den = a.norm(dim=1).clamp_min(1e-12)
    return ((a - b).norm(dim=1) / den).cpu().numpy()


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(1), b.float().flatten(1), dim=1).cpu().numpy()


def linear_cka(a,b):
    # Spatially centered linear CKA: describes spatial representation
    # similarity despite differing channel bases; it is not causal redundancy.
    x=a.float().flatten(2).transpose(1,2);y=b.float().flatten(2).transpose(1,2)
    x=x-x.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
    xy=torch.bmm(x.transpose(1,2),y).square().sum((1,2))
    xx=torch.bmm(x.transpose(1,2),x).square().sum((1,2))
    yy=torch.bmm(y.transpose(1,2),y).square().sum((1,2))
    return (xy/(xx*yy).sqrt().clamp_min(1e-12)).cpu().numpy()


def vector(a):
    return a.detach().float().cpu().numpy()


def counts(prob, truth):
    pred = prob >= .5
    gt = truth >= .5
    return torch.stack(((pred & gt).sum((1,2,3)), (pred & ~gt).sum((1,2,3)),
                       (~pred & gt).sum((1,2,3)), (~pred & ~gt).sum((1,2,3))), 1).cpu().numpy()


def desc(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                p05=float(np.quantile(v,.05)), p95=float(np.quantile(v,.95)))


def morphology_soft(mask, side):
    fg = (mask >= .5).float()
    hits = F.conv2d(fg, torch.ones((1,1,5,5),device=fg.device), padding=2)
    boundary = ((hits > 0) ^ (hits == 25)).float()
    interior = fg*(1-boundary)
    return F.adaptive_avg_pool2d(torch.cat((1-boundary-interior,boundary,interior),1), (side,side))


def route_values(logits, targets):
    p = logits.float().softmax(1)
    hard = p.argmax(1)
    assert p.shape == targets.shape
    # Each target is the area fraction of that class in the patch, not a
    # majority-class label. Thus thin boundaries are represented faithfully.
    soft_mass = targets.sum((2,3))
    hard_mass = torch.stack([(hard == k).sum((1,2)) for k in range(3)],1)
    captured = torch.stack([(targets[:,k]*(hard == k)).sum((1,2)) for k in range(3)],1)
    weighted_prob = (p*targets).sum((2,3))
    entropy = -(p*p.clamp_min(1e-12).log()).sum(1).mean((1,2))
    return dict(prob=p, hard=hard, soft_mass=soft_mass, hard_mass=hard_mass,
                captured=captured, weighted_prob=weighted_prob, entropy=entropy)


def summarize_route(acc):
    area = np.concatenate(acc['target'],0).sum(0)
    predicted = np.concatenate(acc['hard'],0).sum(0)
    captured = np.concatenate(acc['captured'],0).sum(0)
    weighted = np.concatenate(acc['weighted'],0).sum(0)
    total = float(area.sum())
    return dict(n=len(np.concatenate(acc['entropy'])),
                mean_entropy_nats=desc(np.concatenate(acc['entropy'])),
                classes={name:dict(target_area_fraction=float(area[k]/total),
                  predicted_hard_patch_fraction=float(predicted[k]/total),
                  fraction_true_area_in_assigned_patches=float(captured[k]/area[k]) if area[k] else None,
                  fraction_assigned_patch_area_truly_class=float(captured[k]/predicted[k]) if predicted[k] else None,
                  target_area_weighted_mean_probability=float(weighted[k]/area[k]) if area[k] else None)
                  for k,name in enumerate(CLASSES)})


def append_route(acc, v):
    for out, item in (('target','soft_mass'),('hard','hard_mass'),('captured','captured'),
                      ('weighted','weighted_prob'),('entropy','entropy')):
        acc[out].append(vector(v[item]))


def case_choices(ids):
    records = defaultdict(dict)
    with (S19/'analysis/per_image.csv').open() as f:
        for row in csv.DictReader(f):
            if row['model'] in ('baseline','region_scan'):
                records[row['id']][row['model']] = float(row['dice'])
                records[row['id']]['area'] = float(row['foreground_fraction'])
    assert len(records) == len(ids) == 808
    order = sorted(ids, key=lambda i:records[i]['region_scan']-records[i]['baseline'])
    small = sorted((i for i in ids if records[i]['area'] < .1),
                   key=lambda i:records[i]['region_scan']-records[i]['baseline'])
    choices = [('largest_regression_1',order[0]),('largest_regression_2',order[1]),
               ('largest_improvement_1',order[-1]),('largest_improvement_2',order[-2]),
               ('near_median',order[len(order)//2]),('small_lesion_regression',small[0])]
    # A duplicate can occur if the largest regression is already a small lesion.
    seen = set(); selected = []
    for role, identifier in choices:
        if identifier not in seen:
            selected.append(dict(role=role,id=identifier,baseline_dice=records[identifier]['baseline'],
              region_scan_dice=records[identifier]['region_scan'],area=records[identifier]['area']))
            seen.add(identifier)
    return selected


def make_plot(row, title, filename):
    fig, axes = plt.subplots(3,3,figsize=(13.5,12))
    fig.suptitle(title, fontsize=12)
    image = row['image']
    gt = row['gt']
    cmap = ListedColormap(['#3268be','#f6b32f','#47b074'])
    panels = [
      (image,'Image; GT contour in green',None,None,None),
      (row['baseline'],'Baseline: lesion probability', 'magma',0,1),
      (row['current'],'Current: lesion probability','magma',0,1),
      (row['counterfactual'],'Same weights: intervention probability','magma',0,1),
      (row['route'],'Router: blue=bg, yellow=edge, green=inside',cmap,0,2),
      (row['boundary_prob'],'Router: predicted edge probability','magma',0,1),
      (row['feature_energy'],'Module channel RMS (not probability)','viridis',None,None),
      (row['module_delta'],'Intervention difference: module RMS','inferno',None,None),
      (row['decoder_delta'],'Intervention difference: decoder RMS','inferno',None,None),
    ]
    for ax,(arr,lab,cm,vmin,vmax) in zip(axes.flat,panels):
        im=ax.imshow(arr,cmap=cm,vmin=vmin,vmax=vmax,interpolation='nearest',extent=(0,256,256,0))
        if gt.min() != gt.max(): ax.contour(gt,levels=[.5],colors=['#00ff00'],linewidths=.7,origin='upper',extent=(0,256,256,0))
        ax.set_title(lab,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        if cm not in (None,cmap):fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    fig.tight_layout();fig.savefig(filename,dpi=125);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'cases_encoder').mkdir(exist_ok=True)
    (OUT/'cases_decoder').mkdir(exist_ok=True)
    # CJK labels are optional; DejaVu fallback remains legible in the numeric
    # files if no Chinese font is installed.
    for font in ('Noto Sans CJK JP','WenQuanYi Zen Hei','DejaVu Sans'):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams['font.family']=font;break
    setup()
    assert torch.cuda.is_available()
    data = ImageData(42)
    manifest = json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())
    assert data.audit == manifest['members']
    ids = data.ids['val'];n=len(ids)
    cases = case_choices(ids); wanted = {c['id']:c for c in cases}
    save_json('cases.json',cases)
    s19_probs=np.load(S19/'results/region_scan/best_dice/probabilities.npy',mmap_mode='r')
    baseline_probs=np.load(S17/'results/paired_baseline/best_dice/probabilities.npy',mmap_mode='r')
    d17_probs=np.load(S17/'results/hiram/best_dice/probabilities.npy',mmap_mode='r')
    assert all(a.shape==(n,256,256) for a in (s19_probs,baseline_probs,d17_probs))
    assert all(json.loads(p.read_text()) == ids for p in (
      S19/'results/region_scan/best_dice/ids.json', S17/'results/paired_baseline/best_dice/ids.json',
      S17/'results/hiram/best_dice/ids.json'))

    model=Segmentor('region_scan').cuda().eval()
    ck=torch.load(S19/'results/region_scan/best_dice.pth',map_location='cpu')
    load_trainable(model,ck['model'])
    assert ck['epoch']==86 and model.encoder.frozen_state_hash()==ck['frozen_hash']
    taps={}
    def hook(name):
        def capture(_m,_i,o):taps[name]=o.detach()
        return capture
    h1=model.encoder.mixer.register_forward_hook(hook('mixer'))
    h2=model.probe.decoder.blocks[0].register_forward_hook(hook('decoder'))
    route_acc=defaultdict(list); stats=defaultdict(list); saved_cases={}; actual_counts=[]; cf_counts=[]
    valid_boxes=((slice(0,14),slice(0,14)),(slice(0,14),slice(14,16)),
                 (slice(14,16),slice(0,14)),(slice(14,16),slice(14,16)))
    window_hist=np.zeros((4,3,5),dtype=np.int64)
    repro_max=0.;repro_sum=0.;repro_n=0;repro_binary=0
    prior_norm=float(model.encoder.mixer.region_prior.float().norm().cpu())
    print('ENCODER DIAGNOSTIC',flush=True)
    with torch.no_grad():
        for start in range(0,n,BATCH):
            end=min(n,start+BATCH); x,y=data.batch('val',torch.arange(start,end,device='cuda'))
            model.encoder.mixer.mode='region_scan'
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fs,r=model.encoder(x); output=model.probe(fs)
            m1=model.encoder.mixer.window_map(taps['mixer']); d1=taps['decoder']
            y1=output.float().sigmoid();route=route_values(r,region_targets(y))
            model.encoder.mixer.mode='ordered_prior'
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fs0,r0=model.encoder(x);output0=model.probe(fs0)
            m0=model.encoder.mixer.window_map(taps['mixer']);d0=taps['decoder'];y0=output0.float().sigmoid()
            assert torch.equal(r,r0), 'routing input changed before the intervention'
            assert torch.equal(fs[0],fs0[0]) and torch.equal(fs[1],fs0[1])
            append_route(route_acc,route)
            for label,a,b in [('mixer',m1,m0),('encoder_16',fs[2],fs0[2]),
                              ('encoder_8',fs[3],fs0[3]),('decoder_64',d1,d0)]:
                stats[label+'_relative_l2'].append(relative(a,b));stats[label+'_cosine'].append(cosine(a,b))
            stats['mean_probability_abs_change'].append(vector((y1-y0).abs().mean((1,2,3))))
            stats['binary_changed_fraction'].append(vector(((y1>=.5)!=(y0>=.5)).float().mean((1,2,3))))
            stats['target_area'].append(vector((y>=.5).float().mean((1,2,3))))
            actual_counts.append(counts(y1,y));cf_counts.append(counts(y0,y))
            hard=route['hard']
            for w,(ys,xs) in enumerate(valid_boxes):
                for k in range(3):
                    cnt=(hard[:,ys,xs]==k).sum((1,2)).cpu().numpy()
                    window_hist[w,k] += np.bincount(np.digitize(cnt,[1,2,5,17]),minlength=5)
            old=np.asarray(s19_probs[start:end]);cur=vector(y1[:,0]);err=np.abs(cur-old)
            repro_max=max(repro_max,float(err.max()));repro_sum+=float(err.sum());repro_n+=err.size
            repro_binary+=int(((cur>=.5)!=(old>=.5)).sum())
            for j in range(end-start):
                identifier=ids[start+j]
                if identifier in wanted:
                    selected=dict(image=vector(data.images['val'][start+j].permute(1,2,0)).astype('uint8'),
                      gt=vector((y[j,0]>=.5).float()),baseline=np.asarray(baseline_probs[start+j]).copy(),
                      current=cur[j],counterfactual=vector(y0[j,0]),route=vector(hard[j]),
                      boundary_prob=vector(route['prob'][j,1]),
                      feature_energy=vector(m1[j].float().square().mean(0).sqrt()),
                      module_delta=vector((m1[j].float()-m0[j].float()).square().mean(0).sqrt()),
                      decoder_delta=vector((d1[j].float()-d0[j].float()).square().mean(0).sqrt()))
                    saved_cases[identifier]=selected
            if start%256==0:print(f'  encoder {end}/{n}',flush=True)
    h1.remove();h2.remove()
    assert set(saved_cases)==set(wanted)
    for identifier,row in saved_cases.items():
        c=wanted[identifier]
        np.savez_compressed(OUT/'cases_encoder'/f'{c["role"]}_{identifier}.npz',**row)
        make_plot(row,f'{c["role"]}, id={identifier}; baseline {c["baseline_dice"]:.3f}, scan {c["region_scan_dice"]:.3f}',
                  OUT/'cases_encoder'/f'{c["role"]}_{identifier}.png')
    actual=np.concatenate(actual_counts);cf=np.concatenate(cf_counts)
    saved_counts=np.load(S19/'results/region_scan/best_dice/counts.npy')
    assert np.array_equal(actual,saved_counts), 'stage19 per-image counts do not reproduce'
    assert repro_max==0.0 and repro_binary==0, 'stage19 saved probability image does not reproduce'
    actual_metrics=aggregate_counts(*actual.T); cf_metrics=aggregate_counts(*cf.T)
    summary=dict(encoder=dict(checkpoint_epoch=ck['epoch'], region_prior_l2=prior_norm,
      reproduction=dict(max_abs_probability_error=repro_max,mean_abs_probability_error=repro_sum/repro_n,
                        binary_pixel_differences=repro_binary,total_pixels=repro_n,
                        all_per_image_counts_match=True,saved_probability_consistent=True),
      original=actual_metrics,counterfactual_ordered_scan=cf_metrics,
      feature_change={k:desc(np.concatenate(v)) for k,v in stats.items()},
      route=summarize_route(route_acc),
      valid_window_counts=[196,28,28,4],
      window_class_group_size_histogram=dict(bins=['0','1','2-4','5-16','17+'],
         counts_by_window_and_class=window_hist.tolist())))
    with (OUT/'encoder_per_image.csv').open('w',newline='') as f:
        keys=list(stats); writer=csv.DictWriter(f,fieldnames=['id']+keys+['tp','fp','fn','tn','cf_tp','cf_fp','cf_fn','cf_tn'])
        writer.writeheader();cols={k:np.concatenate(v) for k,v in stats.items()}
        for i,identifier in enumerate(ids):
            row={'id':identifier,**{k:float(v[i]) for k,v in cols.items()}}
            row.update(dict(zip(['tp','fp','fn','tn'],map(int,actual[i]))))
            row.update(dict(zip(['cf_tp','cf_fp','cf_fn','cf_tn'],map(int,cf[i]))));writer.writerow(row)
    # The next diagnostic uses the unmodified pretrained features from the same
    # verified encoder, while loading the independently trained stage17 decoder.
    model.encoder.mixer.mode='native'
    probe=d17.DecoderProbe('hiram').cuda().eval()
    dck=torch.load(S17/'results/hiram/best_dice.pth',map_location='cpu')
    probe.load_state_dict(dck['model'],strict=True)
    assert dck['epoch']==164
    taps={};dh=probe.decoder.blocks[0].register_forward_hook(hook('decoder'))
    original=d17.batched_hiram; intercepted={}; bypass=[False]
    def intercept(module,a,b,stochastic):
        outa,outb,regions=original(module,a,b,stochastic)
        intercepted.update(a=a.detach(),b=b.detach(),outa=outa.detach(),outb=outb.detach())
        return (a,b,regions) if bypass[0] else (outa,outb,regions)
    d17.batched_hiram=intercept
    droute=[defaultdict(list),defaultdict(list)];dstats=defaultdict(list);dcases={};ac=[];bc=[]
    drepro_max=0.;drepro_sum=0.;drepro_n=0;drepro_binary=0
    print('DECODER HIRAM DIAGNOSTIC',flush=True)
    try:
      with torch.no_grad():
        for start in range(0,n,BATCH):
            end=min(n,start+BATCH);x,y=data.batch('val',torch.arange(start,end,device='cuda'))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f,_=model.encoder(x);bypass[0]=False;out,regions=probe(f,return_regions=True)
            dcur=taps['decoder'];p=out.float().sigmoid()
            a,b,oa,ob=(intercepted[k] for k in ('a','b','outa','outb'))
            target=morphology_soft(y,32)
            routes=[route_values(r,target) for r in regions]
            for collector,rv in zip(droute,routes):append_route(collector,rv)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                bypass[0]=True;out0,_=probe(f,return_regions=True)
            dcf=taps['decoder'];p0=out0.float().sigmoid()
            for key,u,v in [('branch_coarse',oa,a),('branch_skip',ob,b),('branch_sum',oa.float()+ob.float(),a.float()+b.float()),
                            ('decoder_64',dcur,dcf)]:
                dstats[key+'_relative_l2'].append(relative(u,v));dstats[key+'_cosine'].append(cosine(u,v))
            dstats['residual_cross_branch_cosine'].append(cosine(oa.float()-a.float(),ob.float()-b.float()))
            dstats['input_branch_spatial_linear_cka'].append(linear_cka(a,b))
            dstats['mean_probability_abs_change'].append(vector((p-p0).abs().mean((1,2,3))))
            dstats['binary_changed_fraction'].append(vector(((p>=.5)!=(p0>=.5)).float().mean((1,2,3))))
            ac.append(counts(p,y));bc.append(counts(p0,y))
            old=np.asarray(d17_probs[start:end]);cur=vector(p[:,0]);err=np.abs(cur-old)
            drepro_max=max(drepro_max,float(err.max()));drepro_sum+=float(err.sum());drepro_n+=err.size
            drepro_binary+=int(((cur>=.5)!=(old>=.5)).sum())
            for j in range(end-start):
                identifier=ids[start+j]
                if identifier in wanted:
                    dcases[identifier]=dict(image=vector(data.images['val'][start+j].permute(1,2,0)).astype('uint8'),
                      gt=vector((y[j,0]>=.5).float()),baseline=np.asarray(baseline_probs[start+j]).copy(),
                      current=cur[j],counterfactual=vector(p0[j,0]),route=vector(routes[0]['hard'][j]),
                      boundary_prob=vector(routes[0]['prob'][j,1]),
                      feature_energy=vector(oa[j].float().square().mean(0).sqrt()),
                      module_delta=vector((oa[j].float()+ob[j].float()-a[j].float()-b[j].float()).square().mean(0).sqrt()),
                      decoder_delta=vector((dcur[j].float()-dcf[j].float()).square().mean(0).sqrt()))
            if start%256==0:print(f'  decoder {end}/{n}',flush=True)
    finally:
        d17.batched_hiram=original;dh.remove()
    assert set(dcases)==set(wanted)
    for identifier,row in dcases.items():
        c=wanted[identifier]
        np.savez_compressed(OUT/'cases_decoder'/f'{c["role"]}_{identifier}.npz',**row)
        make_plot(row,f'{c["role"]}, id={identifier}; decoder HiRAM',OUT/'cases_decoder'/f'{c["role"]}_{identifier}.png')
    ac=np.concatenate(ac);bc=np.concatenate(bc)
    saved_counts=np.load(S17/'results/hiram/best_dice/counts.npy')
    assert np.array_equal(ac,saved_counts), 'stage17 HiRAM per-image counts do not reproduce'
    old_path=S17/'results/hiram/best_dice/probabilities.npy'
    summary['decoder']=dict(checkpoint_epoch=dck['epoch'],
      reproduction=dict(max_abs_probability_error=drepro_max,mean_abs_probability_error=drepro_sum/drepro_n,
                        binary_pixel_differences=drepro_binary,total_pixels=drepro_n,
                        all_per_image_counts_match=True,saved_probability_consistent=drepro_binary==0 and drepro_max<1e-6,
                        saved_probability_allocated_bytes=old_path.stat().st_blocks*512,
                        saved_probability_logical_bytes=old_path.stat().st_size,
                        note='Existing stage17 hiram probability artifact contains sparse zeroed areas. Metrics below were regenerated from the verified checkpoint; all 808 TP/FP/FN/TN records match the historical count file.'),
      original=aggregate_counts(*ac.T),counterfactual_hiram_bypassed=aggregate_counts(*bc.T),
      feature_change={k:desc(np.concatenate(v)) for k,v in dstats.items()},
      coarse_route=summarize_route(droute[0]),skip_route=summarize_route(droute[1]))
    with (OUT/'decoder_per_image.csv').open('w',newline='') as f:
        keys=list(dstats);writer=csv.DictWriter(f,fieldnames=['id']+keys+['tp','fp','fn','tn','cf_tp','cf_fp','cf_fn','cf_tn'])
        writer.writeheader();cols={k:np.concatenate(v) for k,v in dstats.items()}
        for i,identifier in enumerate(ids):
            row={'id':identifier,**{k:float(v[i]) for k,v in cols.items()}}
            row.update(dict(zip(['tp','fp','fn','tn'],map(int,ac[i]))))
            row.update(dict(zip(['cf_tp','cf_fp','cf_fn','cf_tn'],map(int,bc[i]))));writer.writerow(row)
    summary['audit']=dict(n=n,seed=42,split='legacy ISIC2018 development validation',
      precision='BF16 autocast',batch=BATCH,device=torch.cuda.get_device_name(),
      official_encoder_hash=model.encoder.frozen_state_hash(),
      source_sha256={str(p):sha(p) for p in [HERE/'diagnose.py',S19/'models.py',S17/'models.py',
        S19/'results/region_scan/best_dice.pth',S17/'results/hiram/best_dice.pth']},
      notes=['One seed; models and checkpoints selected on this same development validation set.',
        'Interventions change inference paths while retaining trained weights and may be out of distribution.',
        'Region area uses 256-pixel morphology and area pooling, not hard patch-majority labels.',
        'Feature RMS is not attention, class attribution, or segmentation probability.',
        'No raw images or checkpoints have been uploaded.'])
    assert summary['audit']['official_encoder_hash']==ck['frozen_hash']
    save_json('diagnostic_summary.json',summary)
    print('FINISHED',json.dumps(dict(encoder_original=summary['encoder']['original']['macro']['dice'],
       encoder_counterfactual=summary['encoder']['counterfactual_ordered_scan']['macro']['dice'],
       decoder_original=summary['decoder']['original']['macro']['dice'],
       decoder_counterfactual=summary['decoder']['counterfactual_hiram_bypassed']['macro']['dice'])),flush=True)


if __name__=='__main__':main()
