"""Complete metrics and paired error-repair diagnostics; no selector retuning."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from evaluation import summarize_rows
from evaluate_runs import paired
from stage1_visual.analyze import metrics
from models import NAMES


def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    base=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((base/'DONE.json').read_text());budget=2 if args.smoke else 300
    assert done['epochs']==budget and done['optimizer_updates_per_model']==budget*30
    assert done['smoke_only']==args.smoke and done['encoders_unchanged'] and done['gradient_checks_passed']
    if not args.smoke:
        prov=json.loads((ROOT/'launch_provenance.json').read_text())
        for section in ('source_sha256','historical_sha256'):
            for path,digest in prov[section].items():assert sha(path)==digest,path
    ids=json.loads((ROOT.parent/'stage1_visual/results/dinov3/ids.json').read_text())
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (ROOT.parent/'data/isic2018/val/masks').iterdir()}
    manifest=json.loads((ROOT.parent/'stage1_visual/data_manifest.json').read_text())
    for m in manifest['members']:
        if m['split']=='val':assert sha(masks[m['id']])==m['mask_sha256']
    gt=np.stack([np.asarray(Image.open(masks[s]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128 for s in ids])
    history=json.loads((base/'history.json').read_text())
    assert [r['epoch'] for r in history]==list(range(1,budget+1))
    summaries={};all_rows={};comparison={};flat=[]
    for criterion in ('best_dice','best_loss','last'):
        probabilities={};records={}
        for name in NAMES:
            path=base/name/criterion
            assert json.loads((path/'ids.json').read_text())==ids
            record=json.loads((path/'metrics.json').read_text());records[name]=record
            selected=(max(history,key=lambda r:r['models'][name]['val']['macro']['dice']) if criterion=='best_dice'
                else min(history,key=lambda r:r['models'][name]['val']['loss']) if criterion=='best_loss' else history[-1])
            assert record['selected_epoch']==selected['epoch'] and record['final']==selected['models'][name]['val']
            probabilities[name]=np.load(path/'probabilities.npy',mmap_mode='r')
            assert probabilities[name].shape==(808,256,256)
        def sample(i):
            pp={n:p[i] for n,p in probabilities.items()}
            for d in ('light','swin'):pp['mean_'+d]=(pp['dinov3_'+d]+pp['vmamba_'+d])*.5
            assert all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in pp.values())
            return dict(id=ids[i],foreground_fraction=float(gt[i].mean()),models={n:metrics(p>=.5,gt[i]) for n,p in pp.items()})
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(ids))))
        all_rows[criterion]=rows;summaries[criterion]={}
        for name in rows[0]['models']:
            full=[dict(id=r['id'],foreground_fraction=r['foreground_fraction'],**r['models'][name]) for r in rows]
            s=summarize_rows(full);summaries[criterion][name]=s
            if name in NAMES:
                assert s['counts']==records[name]['final']['counts']
                for mode in ('macro','pooled'):
                    for key in ('dice','iou','precision','recall'):assert abs(s[mode][key]-records[name]['final'][mode][key])<1e-12
                s['selected_epoch']=records[name]['selected_epoch']
                s['probability_sha256']=sha(base/name/criterion/'probabilities.npy')
                s['checkpoint_sha256']=sha(base/name/(criterion+'.pth'))
                binary=np.load(base/name/criterion/'masks.npy',mmap_mode='r')
                assert all(np.array_equal(binary[i],probabilities[name][i]>=.5) for i in range(len(ids)))
                dump(base/name/criterion/'per_image_full.json',full);dump(base/name/criterion/'full_metrics.json',s)
            flat.append(dict(criterion=criterion,model=name,selected_epoch=s.get('selected_epoch','paired'),
                **{f'macro_{k}':v for k,v in s['macro'].items()},**{f'pooled_{k}':v for k,v in s['pooled'].items()},
                **{f'prec_at_{k}':v for k,v in s['prec_at'].items()},threshold_jaccard_065=s['threshold_jaccard_065']))
        dump(out/f'per_image_{criterion}.json',rows)
        comparison[criterion]={}
        for d in ('light','swin'):
            for ref in ('dinov3','vmamba','mean'):
                key=f'fusion_{d}_minus_{ref}_{d}'
                comparison[criterion][key]=paired(rows,f'fusion_{d}',f'{ref}_{d}')
        delta=np.array([(r['models']['fusion_swin']['dice']-r['models']['dinov3_swin']['dice'])-
                       (r['models']['fusion_light']['dice']-r['models']['dinov3_light']['dice']) for r in rows])
        rng=np.random.default_rng(42)
        boot=[float(delta[rng.integers(len(delta),size=len(delta))].mean()) for _ in range(2000)]
        comparison[criterion]['interaction_swin_gain_minus_light_gain']=dict(macro_dice_delta=float(delta.mean()),
            exploratory_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist())
        print('EVALUATED',criterion,{n:s['macro']['dice'] for n,s in summaries[criterion].items()},flush=True)
    rows=all_rows['best_dice'];diagnostics={};strata=[]
    primary={n:np.load(base/n/'best_dice/probabilities.npy',mmap_mode='r') for n in NAMES}
    bands=np.stack([distance_transform_edt(~(g ^ binary_erosion(g)))<=2 for g in gt])
    for d in ('light','swin'):
        a=primary['dinov3_'+d]>=.5;b=primary['vmamba_'+d]>=.5;c=primary['fusion_'+d]>=.5
        ae=a!=gt;be=b!=gt;ce=c!=gt
        codes=ae.astype(np.uint8)*2+ce.astype(np.uint8)
        np.savez_compressed(out/f'dino_fusion_errors_{d}.npz',ids=np.array(ids),categories=codes)
        pixel_records=[]
        for i in range(len(ids)):
            complementary=ae[i]&~be[i]
            pixel_records.append(dict(id=ids[i],dino_errors=int(ae[i].sum()),corrected=int((ae[i]&~ce[i]).sum()),
                introduced=int((~ae[i]&ce[i]).sum()),vmamba_only_correct=int(complementary.sum()),
                vmamba_only_correct_recovered=int((complementary&~ce[i]).sum()),
                dice_delta=rows[i]['models']['fusion_'+d]['dice']-rows[i]['models']['dinov3_'+d]['dice']))
        dump(out/f'error_repair_per_image_{d}.json',pixel_records)
        diagnostic={}
        for label,region in [('all',np.ones_like(gt)),('foreground',gt),('boundary_band_2px',bands)]:
            complementary=ae&~be&region;corrected=int((ae&~ce&region).sum());introduced=int((~ae&ce&region).sum())
            denominator=int(complementary.sum());original_errors=int((ae&region).sum())
            diagnostic[label]=dict(dino_errors=original_errors,corrected=corrected,introduced=introduced,
                net_errors_removed=corrected-introduced,original_error_repair_rate=corrected/original_errors if original_errors else None,
                vmamba_only_correct=denominator,vmamba_only_correct_recovered=int((complementary&~ce).sum()),
                complementary_repair_rate=float((complementary&~ce).sum())/denominator if denominator else None)
        diagnostics[d]=diagnostic
        for variable,groups in [('dino_dice',[('lt08',0,.8),('08to09',.8,.9),('ge09',.9,1.01)]),
                                ('foreground_fraction',[('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)])]:
            for label,lo,hi in groups:
                subset=[r for r in rows if lo <= (r['models']['dinov3_'+d]['dice'] if variable=='dino_dice' else r[variable]) < hi]
                if subset:
                    strata.append(dict(decoder=d,variable=variable,group=label,n=len(subset),
                        models={n:summarize_rows([r['models'][n+'_'+d] for r in subset]) for n in ('dinov3','vmamba','fusion','mean')}))
        # Show largest gains and regressions using the predeclared primary selector.
        order=np.argsort([r['dice_delta'] for r in pixel_records]);indices=list(order[:3])+list(order[-3:][::-1])
        fig,axes=plt.subplots(6,5,figsize=(12,14))
        for row_index,i in enumerate(indices):
            img=Image.open(ROOT.parent/'data/isic2018/val/images'/f'{ids[i]}.png').convert('RGB').resize((256,256))
            for ax,value,title in zip(axes[row_index],[img,gt[i],a[i],b[i],c[i]],['Image','GT','DINO','VMamba','Fusion']):
                ax.imshow(value,cmap='gray');ax.axis('off');ax.set_title(title)
            axes[row_index,0].set_title(f'{ids[i]} delta={pixel_records[i]["dice_delta"]:+.3f}')
        fig.tight_layout();fig.savefig(out/f'gain_and_regression_{d}.png',dpi=130);plt.close(fig)
    dump(out/'strata.json',strata)
    config=json.loads((base/'config.json').read_text());bench=json.loads((ROOT/'preflight.json').read_text())
    result=dict(seed=42,smoke_only=args.smoke,evaluation_split='legacy ISIC2018 development validation, not independent test',
        primary_criterion='best_dice',models=summaries,comparisons=comparison,error_repair=diagnostics,
        error_code_order=['both_correct','dino_correct_fusion_wrong','dino_wrong_fusion_correct','both_wrong'],
        parameters=config['parameters'],inference=bench['inference'],inference_note=bench['inference_note'],
        caveat='One seed; no capacity-matched control. Repair of DINO errors alone cannot exclude added capacity as an explanation.')
    dump(out/'summary.json',result)
    with (out/'all_runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# 第四轮双编码器融合结果'+('（两轮预检，非正式结果）' if args.smoke else ''),'',
        'ISIC2018 1886/808开发划分，seed42，256×256，主选模为最高验证mean Dice。非独立测试。','',
        '| Model | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,s in summaries['best_dice'].items():
        vals=[s['macro']['dice'],s['pooled']['dice'],s['macro']['iou'],s['pooled']['iou'],s['macro']['boundary_f1'],s['macro']['hd95_pixels']]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','## 融合收益','', '| Comparison | Mean Dice delta | Exploratory 95% interval | Better / worse / ties |','|---|---:|---|---|']
    for name,c in comparison['best_dice'].items():
        if 'dice' not in c:continue
        v=c['dice'];lines.append(f"| {name} | {v['mean_delta']:.6f} | {v['bootstrap95']} | {v['positive']}/{v['negative']}/{v['ties']} |")
    lines+=['','## 原有错误是否被纠正','', '| Decoder | DINO wrong pixels | Corrected | New errors | VMamba-only correct recovery |','|---|---:|---:|---:|---:|']
    for d,s in diagnostics.items():
        a=s['all'];lines.append(f"| {d} | {a['dino_errors']} | {a['corrected']} | {a['introduced']} | {a['complementary_repair_rate']:.6f} |")
    lines+=['','各组完整Precision/Recall、Prec@X、Threshold Jaccard、HD95归一化和三种选模结果见 all_runs.csv/summary.json。',
        '逐图记录、难例和面积分组、纠错类别图、改善及退步示例均已保存。',
        '该结果仍混合了特征来源和额外容量的影响；需要容量对照及跨seed确认后才可归因。开发集bootstrap不等于跨seed显著性。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,all_checkpoints_verified=True,
        counts_recomputed=True,data_manifest_match=True,original_artifacts_match=not args.smoke))
    print('ANALYSIS PASSED',flush=True)


if __name__=='__main__':main()
