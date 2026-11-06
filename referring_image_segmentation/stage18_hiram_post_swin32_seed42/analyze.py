"""Compare the new post-Swin probes to immutable stage-17 predictions."""
from common import *
import csv
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from scipy.ndimage import convolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import summarize_rows
from stage1_visual.analyze import metrics

OLD = PROJECT / 'stage17_mambavision_decoder_hiram_seed42'
SOURCES = {
    'baseline': (OLD, 'paired_baseline'),
    'pre_region': (OLD, 'region_only'),
    'pre_hiram': (OLD, 'hiram'),
    'post_region': (ROOT, 'region_only'),
    'post_hiram': (ROOT, 'hiram'),
}
PAIRS = [('post_hiram','pre_hiram'), ('post_hiram','post_region'),
         ('post_region','pre_region'), ('post_hiram','baseline')]


def main():
    for directory in (OLD, ROOT):
        done=json.loads((directory/'results/DONE.json').read_text())
        assert done['seed']==42 and done['epochs']==300 and done['encoder_unchanged']
        assert all(x==9000 for x in done['updates'].values())
        ev=json.loads((directory/'results/evaluation.json').read_text())
        assert ev['passed'] and ev['n_val']==808
    outputs={name:directory/'results'/key/'best_dice' for name,(directory,key) in SOURCES.items()}
    ids=json.loads((outputs['baseline']/'ids.json').read_text())
    assert len(ids)==808 and all(json.loads((p/'ids.json').read_text())==ids for p in outputs.values())
    probabilities={n:np.load(p/'probabilities.npy',mmap_mode='r') for n,p in outputs.items()}
    assert all(p.shape==(808,256,256) for p in probabilities.values())
    maskdir=PROJECT/'data/isic2018/val/masks'
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in maskdir.iterdir() if p.is_file()}
    def sample(i):
        gt=np.asarray(Image.open(masks[ids[i]]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        pred={n:p[i]>=.5 for n,p in probabilities.items()}
        vals={n:metrics(v,gt) for n,v in pred.items()}
        err={n:v!=gt for n,v in pred.items()}
        count=convolve(gt.astype(np.uint8),np.ones((5,5),dtype=np.uint8),mode='constant',cval=0)
        boundary=(count>0)&(count<25)
        zones=dict(boundary=boundary,interior=gt&~boundary,background=~gt&~boundary)
        deltas={}
        for new,old in PAIRS:
            corrected=err[old]&~err[new]
            introduced=~err[old]&err[new]
            deltas[f'{new}_minus_{old}']=dict(corrected=int(corrected.sum()),introduced=int(introduced.sum()),
                zones={key:dict(corrected=int((z&corrected).sum()),introduced=int((z&introduced).sum()))
                       for key,z in zones.items()})
        return dict(id=ids[i],foreground_fraction=float(gt.mean()),models=vals,errors=deltas)
    with ThreadPoolExecutor(6) as pool:rows=list(pool.map(sample,range(808)))
    summaries={}
    evaluations={d:json.loads((d/'results/evaluation.json').read_text()) for d in (OLD,ROOT)}
    for name,(directory,key) in SOURCES.items():
        s=summarize_rows([r['models'][name] for r in rows])
        ref=evaluations[directory]['results'][key]['best_dice']
        assert s['counts']==ref['metrics']['counts']
        assert abs(s['macro']['dice']-ref['metrics']['macro']['dice'])<1e-12
        assert abs(s['macro']['iou']-ref['metrics']['macro']['iou'])<1e-12
        s['selected_epoch']=ref['selected_epoch']
        summaries[name]=s
    comparisons={}
    for new,old in PAIRS:
        key=f'{new}_minus_{old}'
        delta=np.array([r['models'][new]['dice']-r['models'][old]['dice'] for r in rows])
        rng=np.random.default_rng(42)
        ci=np.percentile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)],[2.5,97.5]).tolist()
        corrected=sum(r['errors'][key]['corrected'] for r in rows)
        introduced=sum(r['errors'][key]['introduced'] for r in rows)
        order=np.argsort(delta)
        zone_totals={z:{field:sum(r['errors'][key]['zones'][z][field] for r in rows)
                        for field in ('corrected','introduced')}
                     for z in ('boundary','interior','background')}
        comparisons[key]=dict(macro_dice_delta=float(delta.mean()),bootstrap95=ci,
            better=int((delta>0).sum()),worse=int((delta<0).sum()),ties=int((delta==0).sum()),
            corrected_pixels=corrected,introduced_errors=introduced,net_errors_removed=corrected-introduced,
            by_morphological_region=zone_totals,
            largest_regressions=[dict(id=ids[i],delta_dice=float(delta[i])) for i in order[:10]],
            largest_improvements=[dict(id=ids[i],delta_dice=float(delta[i])) for i in order[-10:][::-1]])
    strata=[]
    for label,lo,hi in [('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)]:
        subset=[r for r in rows if lo<=r['foreground_fraction']<hi]
        strata.append(dict(group=label,n=len(subset),macro_dice={n:float(np.mean([r['models'][n]['dice'] for r in subset])) for n in SOURCES}))
    HERE=ROOT/'analysis'
    HERE.mkdir(exist_ok=True)
    with (HERE/'per_image.csv').open('w',newline='') as f:
        fields=['id','foreground_fraction','model','dice','iou','precision','recall','boundary_f1','hd95_pixels','hd95_diagonal','tp','fp','fn','tn']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            for name in SOURCES:writer.writerow(dict(id=row['id'],foreground_fraction=row['foreground_fraction'],model=name,**row['models'][name]))
    histories={d:json.loads((d/'results/history.json').read_text()) for d in (OLD,ROOT)}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,(directory,key) in SOURCES.items():
        history=histories[directory]
        axes[0].plot([r['epoch'] for r in history],[r['models'][key]['val']['macro']['dice'] for r in history],label=name)
        axes[1].plot([r['epoch'] for r in history],[r['models'][key]['val']['loss'] for r in history],label=name)
    for ax,title in zip(axes,['Validation macro Dice','Validation segmentation loss']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(HERE/'learning_curves.png',dpi=140);plt.close(fig)
    bench={d:json.loads((d/'analysis/benchmark.json').read_text()) for d in (OLD,ROOT)}
    benchmark={n:bench[d]['models'][key] for n,(d,key) in SOURCES.items()}
    result=dict(seed=42,split='ISIC2018 1886/808 development validation',
        selection='maximum validation macro Dice, earliest exact tie',models=summaries,
        comparisons=comparisons,foreground_area_strata=strata,benchmark=benchmark,
        post_parameters=json.loads((ROOT/'preflight.json').read_text())['models'],
        post_train_seconds=json.loads((ROOT/'results/DONE.json').read_text())['active_training_seconds'],
        limitations=['single seed','repeated development validation used for checkpoint selection',
            'not an independent or verified patient-disjoint test',
            'post-Swin first input and regional-supervision input differ from pre-Swin',
            'different placement is not an isolated causal intervention',
            'paired bootstrap resamples images, not patients or seeds'])
    dump(HERE/'summary.json',result)
    lines=['# HiRAM 后置 Swin32：ISIC2018 seed 42','',
           '历史三组和新两组均为 300 epochs、9000 更新；开发验证集不等于独立测试集。','',
           '| 模型 | 最佳轮 | Macro Dice | Pooled Dice | Macro IoU | Boundary F1 | HD95 px | 完整推理 ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['selected_epoch']} | {s['macro']['dice']:.6f} | {s['pooled']['dice']:.6f} | {s['macro']['iou']:.6f} | {s['macro']['boundary_f1']:.6f} | {s['macro']['hd95_pixels']:.3f} | {benchmark[name]['mean_ms']:.2f} |")
    for new,old in PAIRS:
        key=f'{new}_minus_{old}';c=comparisons[key]
        lines += ['',f"{key}: Macro Dice {c['macro_dice_delta']*100:+.3f} 个百分点；逐图 bootstrap 95% 区间 {c['bootstrap95']}（小数尺度）；纠正 {c['corrected_pixels']}、新增 {c['introduced_errors']} 个错误像素。"]
    lines += ['', '两种位置的第一路输入及区域监督输入也不同，不能把差异归结为纯位置效应。单 seed 且使用开发验证集选模，不能据此宣称跨种子稳定或外部泛化。',
              '', '![学习曲线](learning_curves.png)']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('ANALYSIS COMPLETE',json.dumps({n:s['macro']['dice'] for n,s in summaries.items()}),flush=True)


if __name__=='__main__':main()
