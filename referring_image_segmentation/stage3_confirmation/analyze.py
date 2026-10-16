"""Independent probability evaluation and across-seed summary for all selectors."""
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from evaluation import summarize_rows
from stage1_visual.analyze import metrics

SEEDS=(43,44,45)
DECODERS=('light','swin')
CRITERIA=('best_dice','best_loss','last')


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def flatten(s):
    return {**{f'macro_{k}':v for k,v in s['macro'].items()},
            **{f'pooled_{k}':v for k,v in s['pooled'].items()},
            **{f'prec_at_{k}':v for k,v in s['prec_at'].items()},
            'threshold_jaccard_065':s['threshold_jaccard_065']}


def moments(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),values=list(values))


def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    provenance=json.loads((ROOT/'launch_provenance.json').read_text())
    for filename,expected in provenance['source_sha256'].items():assert sha256(Path(filename))==expected,filename
    manifest=json.loads((ROOT.parent/'stage1_visual/data_manifest.json').read_text())
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (ROOT.parent/'data/isic2018/val/masks').iterdir() if p.is_file()}
    for m in manifest['members']:
        if m['split']=='val':assert sha256(masks[m['id']])==m['mask_sha256']
    reference_ids=json.loads((ROOT.parent/'stage1_visual/results/dinov3/ids.json').read_text())
    gt=[np.asarray(Image.open(masks[sid]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128 for sid in reference_ids]
    all_runs=[];run_rows={};histories={}
    for seed in SEEDS:
        base=ROOT/'results'/f'seed{seed}'
        done=json.loads((base/'DONE.json').read_text())
        assert done['epochs']==300 and done['optimizer_updates_per_decoder']==9000 and not done['smoke_only']
        assert done['encoder_unchanged'] and done['gradient_checks_passed']
        history=json.loads((base/'history.json').read_text());histories[seed]=history
        assert [r['epoch'] for r in history]==list(range(1,301))
        config=json.loads((base/'config.json').read_text())
        for decoder in DECODERS:
            assert np.isfinite([r['models'][decoder]['train_loss'] for r in history]).all()
            for criterion in CRITERIA:
                path=base/decoder/criterion
                record=json.loads((path/'metrics.json').read_text())
                selected=max(history,key=lambda r:r['models'][decoder]['val']['macro']['dice']) if criterion=='best_dice' else min(history,key=lambda r:r['models'][decoder]['val']['loss']) if criterion=='best_loss' else history[-1]
                assert record['selected_epoch']==selected['epoch']
                assert record['final']==selected['models'][decoder]['val']
                ids=json.loads((path/'ids.json').read_text());assert ids==reference_ids
                probs=np.load(path/'probabilities.npy',mmap_mode='r')
                binary=np.load(path/'masks.npy',mmap_mode='r')
                assert probs.shape==binary.shape==(808,256,256)
                def sample(i):
                    p=probs[i];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                    pred=p>=.5;assert np.array_equal(pred,binary[i])
                    return dict(id=ids[i],foreground_fraction=float(gt[i].mean()),**metrics(pred,gt[i]))
                with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(ids))))
                summary=summarize_rows(rows)
                assert summary['counts']==record['final']['counts']
                for mode in ('macro','pooled'):
                    for key in ('dice','iou','precision','recall'):
                        assert abs(summary[mode][key]-record['final'][mode][key])<1e-12,(seed,decoder,criterion,mode,key)
                assert np.all(np.diff(list(summary['prec_at'].values()))<=0)
                summary.update(seed=seed,decoder=decoder,criterion=criterion,selected_epoch=record['selected_epoch'],
                    checkpoint_sha256=sha256(base/decoder/f'{criterion}.pth'),probability_sha256=sha256(path/'probabilities.npy'),
                    parameters=config['initial'][decoder])
                dump(path/'per_image_full.json',rows);dump(path/'full_metrics.json',summary)
                all_runs.append(summary);run_rows[(seed,decoder,criterion)]=rows
                print('EVALUATED',seed,decoder,criterion,summary['macro']['dice'],flush=True)
    groups={};paired={}
    for criterion in CRITERIA:
        groups[criterion]={}
        for decoder in DECODERS:
            records=[r for r in all_runs if r['criterion']==criterion and r['decoder']==decoder]
            fs=[flatten(r) for r in records]
            groups[criterion][decoder]={key:moments([r[key] for r in fs]) for key in fs[0]}
        paired[criterion]={}
        for key in groups[criterion]['light']:
            delta=[b-a for a,b in zip(groups[criterion]['light'][key]['values'],groups[criterion]['swin'][key]['values'])]
            paired[criterion][key]=moments(delta)
        for seed in SEEDS:
            light=run_rows[(seed,'light',criterion)];swin=run_rows[(seed,'swin',criterion)]
            assert [r['id'] for r in light]==[r['id'] for r in swin]
            rows=[dict(id=a['id'],foreground_fraction=a['foreground_fraction'],
                **{f'{k}_delta':b[k]-a[k] for k in ('dice','iou','precision','recall','boundary_f1','hd95_pixels')}) for a,b in zip(light,swin)]
            dump(out/f'paired_seed{seed}_{criterion}.json',rows)
    strata=[]
    for (seed,decoder,criterion),rows in run_rows.items():
        for label,lo,hi in [('small',0,.1),('medium',.1,.3),('large',.3,1.01)]:
            subset=[r for r in rows if lo<=r['foreground_fraction']<hi]
            s=summarize_rows(subset)
            strata.append(dict(seed=seed,decoder=decoder,criterion=criterion,group=label,n=len(subset),**flatten(s)))
    dump(out/'area_strata.json',strata)
    summary=dict(seeds=list(SEEDS),primary_criterion='best_dice',groups=groups,paired_swin_minus_light=paired,
        evaluation_split='development validation; reused for design and checkpoint selection; not independent test',
        dispersion='sample standard deviation across 3 training seeds, ddof=1; images are not independent seed replicates',
        runs=all_runs)
    dump(out/'summary.json',summary)
    flat=[dict(seed=r['seed'],decoder=r['decoder'],criterion=r['criterion'],epoch=r['selected_epoch'],**flatten(r)) for r in all_runs]
    with (out/'all_runs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    group_flat=[]
    for criterion,g in groups.items():
        for decoder,s in g.items():
            group_flat.append(dict(criterion=criterion,decoder=decoder,**{f'{k}_{stat}':v[stat] for k,v in s.items() for stat in ('mean','std')}))
    with (out/'across_seeds.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(group_flat[0]));writer.writeheader();writer.writerows(group_flat)
    columns=[('macro_dice','Mean Dice'),('pooled_dice','Pooled Dice'),('macro_iou','mIoU'),('pooled_iou','oIoU'),('macro_boundary_f1','Boundary F1'),('macro_hd95_pixels','HD95 px')]
    lines=['# 第三轮：三 seed 与选模规则对照','',
        'DINOv3-S/16 冻结编码器；Light vs Swin；seeds 43/44/45；各 300 epochs / 9000 updates。',
        '1886/808 旧开发划分，256×256，固定概率阈值 .5。不是独立测试。± 为三个训练 seeds 的样本标准差（ddof=1）。',
        '', '## 主结果：按 mean Dice 选模','', '| Decoder | '+' | '.join(label for _,label in columns)+' |', '|---|'+'---:|'*len(columns)]
    def row(decoder,criterion):
        return '| '+decoder+' | '+' | '.join(f"{groups[criterion][decoder][key]['mean']:.6f} ± {groups[criterion][decoder][key]['std']:.6f}" for key,_ in columns)+' |'
    lines += [row(d,'best_dice') for d in DECODERS]
    lines += ['', '## 各 seed 的主结果','', '| Seed | Decoder | Selected epoch | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in all_runs:
        if r['criterion']=='best_dice':
            f=flatten(r);lines.append(f"| {r['seed']} | {r['decoder']} | {r['selected_epoch']} | "+' | '.join(f'{f[k]:.6f}' for k,_ in columns)+' |')
    lines += ['', '## 主结果的 Swin−Light 配对差','', '| Metric | Seed 43 | Seed 44 | Seed 45 | Mean delta ± SD |','|---|---:|---:|---:|---:|']
    for key,label in columns:
        r=paired['best_dice'][key]
        lines.append('| '+label+' | '+' | '.join(f'{v:+.6f}' for v in r['values'])+f" | {r['mean']:+.6f} ± {r['std']:.6f} |")
    lines += ['', 'HD95 的负差表示改善。只用 3 个 seeds 的方向与离散程度描述训练随机性，不作患者级或独立测试显著性推断。',
              '', '## 次要选模对照']
    for criterion in ('best_loss','last'):
        lines += ['',f'### {criterion}','', '| Decoder | '+' | '.join(label for _,label in columns)+' |','|---|'+'---:|'*len(columns)]
        lines += [row(d,criterion) for d in DECODERS]
    extra=[('macro_precision','Mean Precision'),('macro_recall','Mean Recall'),('pooled_precision','Pooled Precision'),('pooled_recall','Pooled Recall'),
           *[(f'prec_at_{t}',f'P@{t}') for t in ('0.5','0.6','0.7','0.8','0.9')],('threshold_jaccard_065','Threshold Jaccard'),('macro_hd95_diagonal','Normalized HD95')]
    lines += ['', '## 完整补充指标','', '| Criterion | Decoder | '+' | '.join(label for _,label in extra)+' |','|---|---|'+'---:|'*len(extra)]
    for criterion in CRITERIA:
        for decoder in DECODERS:
            s=groups[criterion][decoder]
            lines.append(f'| {criterion} | {decoder} | '+' | '.join(f"{s[k]['mean']:.6f} ± {s[k]['std']:.6f}" for k,_ in extra)+' |')
    lines += ['', '全部逐 seed/逐 checkpoint 指标：all_runs.csv；完整均值与标准差：across_seeds.csv；逐图记录在对应 results/seed*/decoder/criterion/per_image_full.json。',
              '', 'Light/Swin 的可训练参数为 246849/506465，Decoder 参数为 148033/407649；未作参数量匹配。',
              '', '## 学习曲线','', '![Learning curves](learning_curves.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True)
    for col,seed in enumerate(SEEDS):
        h=histories[seed]
        for decoder,color in [('light','#3974b5'),('swin','#d17c26')]:
            axes[0,col].plot([r['epoch'] for r in h],[r['models'][decoder]['val']['loss'] for r in h],color=color,label=decoder,alpha=.8)
            axes[1,col].plot([r['epoch'] for r in h],[r['models'][decoder]['val']['macro']['dice'] for r in h],color=color,label=decoder,alpha=.8)
            selected=next(r['selected_epoch'] for r in all_runs if r['seed']==seed and r['decoder']==decoder and r['criterion']=='best_dice')
            axes[1,col].axvline(selected,color=color,linestyle=':',alpha=.7)
        axes[0,col].set_title(f'Seed {seed}');axes[0,col].legend()
        axes[0,col].set_ylabel('Validation loss');axes[1,col].set_ylabel('Mean image Dice');axes[1,col].set_xlabel('Epoch')
    fig.suptitle('Frozen DINOv3: decoder comparison; dotted lines = primary Dice-selected checkpoint')
    fig.tight_layout();fig.savefig(out/'learning_curves.png');fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    for filename,expected in provenance['historical_sha256'].items():assert sha256(Path(filename))==expected,filename
    validation=dict(passed=True,seeds=list(SEEDS),runs=6,epochs_per_run=300,updates_per_run=9000,
        evaluated_checkpoints=18,probability_exports=18,per_export_samples=808,
        checkpoint_selection_verified=True,probability_mask_metric_consistency=True,
        sources_unchanged=True,historical_artifacts_unchanged=True,figures_complete=True)
    dump(ROOT/'validation.json',validation)
    print('ANALYSIS_COMPLETE',json.dumps(validation),flush=True)


if __name__=='__main__':main()
