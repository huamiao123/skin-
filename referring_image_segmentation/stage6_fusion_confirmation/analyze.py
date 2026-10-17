"""Independent full metrics and seed-paired aggregation; old artifacts read-only."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import ROOT, PROJECT, NAMES, ALL_NAMES, SEEDS
from stage5_selective_fusion.diagnostics import context, dump, sha, BINS, REGIONS
from stage1_visual.analyze import metrics
from evaluation import summarize_rows
from evaluate_runs import paired

CRITERIA=('best_dice','best_loss','last')
PAIRS=(('fusion_concat','dinov3'),('fusion_concat','capacity_concat'),('capacity_concat','dinov3'))


def flatten(s):
    return {**{f'macro_{k}':v for k,v in s['macro'].items()},
            **{f'pooled_{k}':v for k,v in s['pooled'].items()},
            **{f'prec_at_{k}':v for k,v in s['prec_at'].items()},
            'threshold_jaccard_065':s['threshold_jaccard_065']}


def csv_file(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def repair(ids,gt,regions,dino,candidate):
    columns=('pixels','dino_wrong','dino_correct','corrected','introduced','candidate_wrong')
    joint=np.zeros((5,3,6),dtype=np.int64);rows=[]
    def record(values):
        s={k:int(v) for k,v in zip(columns,values)}
        s.update(net_errors_removed=s['corrected']-s['introduced'],
            repair_rate=s['corrected']/s['dino_wrong'] if s['dino_wrong'] else None,
            damage_rate=s['introduced']/s['dino_correct'] if s['dino_correct'] else None)
        assert s['dino_wrong']-s['candidate_wrong']==s['net_errors_removed']
        return s
    for i,sid in enumerate(ids):
        de=(dino[i]>=.5)!=gt[i];ce=(candidate[i]>=.5)!=gt[i]
        bins=np.clip(np.searchsorted([0.,.1,.4,.6,.9,1.],dino[i],side='right')-1,0,4)
        index=bins*3+regions[i]
        masks=(np.ones_like(de),de,~de,de&~ce,~de&ce,ce)
        for c,m in enumerate(masks):joint[:,:,c]+=np.bincount(index[m],minlength=15).reshape(5,3)
        rows.append(dict(id=sid,**record([m.sum() for m in masks])))
    return dict(all=record(joint.sum((0,1))),
        probability={n:record(joint[i].sum(0)) for i,n in enumerate(BINS)},
        regions={n:record(joint[:,i].sum(0)) for i,n in enumerate(REGIONS)}),rows


def analyze_seed(seed,smoke):
    base=ROOT/('smoke' if smoke else 'results')/f'seed{seed}'
    out=ROOT/('smoke_analysis' if smoke else 'analysis')/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    budget=2 if smoke else 300
    done=json.loads((base/'DONE.json').read_text())
    assert done['epochs']==budget and done['optimizer_updates_per_model']==budget*30 and done['smoke_only']==smoke
    assert done['encoders_unchanged'] and done['gradient_checks_passed']
    hist=json.loads((base/'history.json').read_text());assert [r['epoch'] for r in hist]==list(range(1,budget+1))
    oldbase=PROJECT/f'stage3_confirmation/results/seed{seed}'
    oldhist=json.loads((oldbase/'history.json').read_text())
    ids,gt,regions=context();summaries={};comparisons={};repairs={};strata=[];flat=[]
    audit=json.loads((ROOT/'preflight.json').read_text())['reused_baselines'][str(seed)]
    for criterion in CRITERIA:
        probs={};records={};locations={}
        for n in ALL_NAMES:
            path=(oldbase/'swin' if n=='dinov3' else base/n)/criterion;locations[n]=path
            assert json.loads((path/'ids.json').read_text())==ids
            record=json.loads((path/'metrics.json').read_text());records[n]=record
            history,key=(oldhist,'swin') if n=='dinov3' else (hist,n)
            select=max(history,key=lambda r:r['models'][key]['val']['macro']['dice']) if criterion=='best_dice' else min(history,key=lambda r:r['models'][key]['val']['loss']) if criterion=='best_loss' else history[-1]
            assert record['selected_epoch']==select['epoch'] and record['final']==select['models'][key]['val']
            probs[n]=np.load(path/'probabilities.npy',mmap_mode='r');binary=np.load(path/'masks.npy',mmap_mode='r')
            assert probs[n].shape==binary.shape==gt.shape
            assert all(np.array_equal(binary[i],probs[n][i]>=.5) for i in range(len(ids)))
            if n=='dinov3':
                assert sha(path/'probabilities.npy')==audit['checkpoints'][criterion]['probabilities_sha256']
                assert sha(path.parent/f'{criterion}.pth')==audit['checkpoints'][criterion]['checkpoint_sha256']
        def sample(i):
            assert all(np.isfinite(p[i]).all() and p[i].min()>=0 and p[i].max()<=1 for p in probs.values())
            return dict(id=ids[i],foreground_fraction=float(gt[i].mean()),models={n:metrics(p[i]>=.5,gt[i]) for n,p in probs.items()})
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(ids))))
        summaries[criterion]={};repairs[criterion]={}
        for n in ALL_NAMES:
            per=[dict(id=r['id'],foreground_fraction=r['foreground_fraction'],**r['models'][n]) for r in rows]
            summary=summarize_rows(per);expected=records[n]['final']
            assert summary['counts']==expected['counts']
            for mode in ('macro','pooled'):
                for key in ('dice','iou','precision','recall'):assert abs(summary[mode][key]-expected[mode][key])<1e-12
            summary.update(selected_epoch=records[n]['selected_epoch'],source=str(locations[n]),
                probability_sha256=sha(locations[n]/'probabilities.npy'),checkpoint_sha256=sha(locations[n].parent/f'{criterion}.pth'))
            summaries[criterion][n]=summary
            dest=out/n/criterion;dest.mkdir(parents=True,exist_ok=True)
            dump(dest/'per_image_full.json',per);dump(dest/'full_metrics.json',summary)
            flat.append(dict(seed=seed,model=n,criterion=criterion,selected_epoch=summary['selected_epoch'],**flatten(summary)))
            if n in NAMES:
                diagnostic,per_errors=repair(ids,gt,regions,probs['dinov3'],probs[n])
                assert diagnostic['all']['candidate_wrong']==summary['counts']['fp']+summary['counts']['fn']
                repairs[criterion][n]=diagnostic;dump(out/f'errors_{n}_{criterion}.json',per_errors)
                if criterion=='best_dice':
                    codes=((probs['dinov3']>=.5)!=gt).astype(np.uint8)*2+((probs[n]>=.5)!=gt).astype(np.uint8)
                    np.savez_compressed(out/f'dino_vs_{n}_errors.npz',ids=np.array(ids),categories=codes)
        comparisons[criterion]={f'{a}_minus_{b}':paired(rows,a,b) for a,b in PAIRS}
        dump(out/f'per_image_{criterion}.json',rows)
        for variable,groups in [('dino_dice',[('lt08',0,.8),('08to09',.8,.9),('ge09',.9,1.01)]),
                                ('foreground_fraction',[('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)])]:
            for label,lo,hi in groups:
                subset=[r for r in rows if lo<=(r['models']['dinov3']['dice'] if variable=='dino_dice' else r[variable])<hi]
                if subset:strata.append(dict(criterion=criterion,variable=variable,group=label,n=len(subset),
                    models={n:summarize_rows([r['models'][n] for r in subset]) for n in ALL_NAMES}))
        print('EVALUATED',seed,criterion,{n:s['macro']['dice'] for n,s in summaries[criterion].items()},flush=True)
    dump(out/'summary.json',dict(seed=seed,smoke_only=smoke,models=summaries,comparisons=comparisons,error_repair=repairs,
        caveat='Development validation. Smoke uses only two epochs for new probes versus completed baseline; smoke comparisons are invalid for model ranking.'))
    dump(out/'strata.json',strata);csv_file(out/'all_runs.csv',flat)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for n in ALL_NAMES:
        history,key=(oldhist,'swin') if n=='dinov3' else (hist,n)
        for ax,metric in zip(axes,('dice','loss')):
            ax.plot([r['epoch'] for r in history],[r['models'][key]['val']['macro']['dice'] if metric=='dice' else r['models'][key]['val']['loss'] for r in history],label=n)
    for ax,label in zip(axes,('Validation mean Dice','Validation loss')):ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=140);plt.close(fig)
    dump(out/'validation.json',dict(passed=True,seed=seed,smoke_only=smoke,checkpoint_selection_verified=True,
        probabilities_masks_counts_match=True,all_metrics_recomputed=True,reused_baseline_unchanged=True))


def aggregate(smoke):
    seeds=(43,) if smoke else SEEDS
    out=ROOT/('smoke_analysis' if smoke else 'analysis');by_seed={}
    for seed in seeds:
        assert json.loads((out/f'seed{seed}/validation.json').read_text())['passed']
        by_seed[str(seed)]=json.loads((out/f'seed{seed}/summary.json').read_text())
    def stats(values):
        return dict(values=[float(v) for v in values],mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None)
    models={};pairs={};flat=[];repair_stats={}
    for criterion in CRITERIA:
        models[criterion]={};pairs[criterion]={};repair_stats[criterion]={}
        for n in ALL_NAMES:
            rows=[flatten(by_seed[str(s)]['models'][criterion][n]) for s in seeds]
            models[criterion][n]={key:stats([r[key] for r in rows]) for key in rows[0]}
            for seed,row in zip(seeds,rows):flat.append(dict(seed=seed,model=n,criterion=criterion,
                selected_epoch=by_seed[str(seed)]['models'][criterion][n]['selected_epoch'],**row))
        for a,b in PAIRS:
            aa=models[criterion][a];bb=models[criterion][b]
            pairs[criterion][f'{a}_minus_{b}']={key:stats(np.array(aa[key]['values'])-np.array(bb[key]['values'])) for key in aa}
        for n in NAMES:
            rs=[by_seed[str(s)]['error_repair'][criterion][n]['all'] for s in seeds]
            repair_stats[criterion][n]={k:stats([r[k] for r in rs]) for k in rs[0]}
    pre=json.loads((ROOT/'preflight.json').read_text())
    result=dict(seeds=list(seeds),smoke_only=smoke,primary_criterion='best_dice',aggregate=models,paired_seed_deltas=pairs,
        error_repair=repair_stats,parameters=pre['parameters'],inference=pre['inference'],inference_note=pre['inference_note'],
        by_seed={str(s):str(out/f'seed{s}/summary.json') for s in seeds},
        seed42_exploratory_sources=[str(PROJECT/'stage4_fusion/analysis/summary.json'),str(PROJECT/'stage5_selective_fusion/analysis/summary.json')],
        caveat='Seeds43/44/45 were previously used in decoder selection. Three paired training seeds, not 2424 independent images. No independent test claim. Seed42 excluded from primary aggregate.')
    dump(out/'summary.json',result);csv_file(out/'all_runs.csv',flat)
    lines=['# 第六轮：融合与容量的配对种子复验'+('（流程预检，不能用于排名）' if smoke else ''),'',
        'ISIC2018 1886/808开发划分，256×256；DINO基线校验后只读复用第三轮；六个新训练。最高验证mean Dice选模，±为训练seed样本标准差。','',
        '| Model | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px |','|---|---:|---:|---:|---:|---:|---:|']
    keys=('macro_dice','pooled_dice','macro_iou','pooled_iou','macro_boundary_f1','macro_hd95_pixels')
    for n in ALL_NAMES:
        cells=[]
        for key in keys:
            q=models['best_dice'][n][key];cells.append(f"{q['mean']:.6f}"+(f" ± {q['std']:.6f}" if q['std'] is not None else ''))
        lines.append('| '+n+' | '+' | '.join(cells)+' |')
    lines+=['','## 同seed主比较','', '| Comparison | Seed deltas (43/44/45) | Mean delta | Sample SD |','|---|---|---:|---:|']
    for pair,p in pairs['best_dice'].items():
        q=p['macro_dice'];lines.append(f"| {pair} | {q['values']} | {q['mean']:.6f} | {q['std']} |")
    lines+=['','## 参数与完整推理','', '| Model | Trainable | Total | Batch1 ms | Batch64 images/s |','|---|---:|---:|---:|---:|']
    for n in ALL_NAMES:
        p=pre['parameters'][n];t=pre['inference'][n]
        lines.append(f"| {n} | {p['trainable']} | {p['total']} | {1000*t['1']['seconds_per_batch']:.3f} | {t['64']['images_per_second']:.2f} |")
    lines+=['','全部macro/pooled Precision/Recall、Prec@X、Threshold Jaccard和HD95归一化见all_runs.csv与summary.json；三个checkpoint规则全部保留。',
        '纠正/新增/净减少错误及其分母、分区和逐图配对记录见各seed目录。不同seed的像素计数是同一批图上的重复，不视为新的独立病例。',
        '容量匹配仅针对可训练参数；此开发集及这些DINO seeds已有历史使用，不代表独立测试泛化。seed42单列为探索背景，没有混入主均值。',
        '如为smoke，旧DINO训练300轮而新组只有2轮；表格只验证分析流程，不得作性能结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'validation.json',dict(passed=True,smoke_only=smoke,seeds=list(seeds),models_per_seed=3,
        evaluated_checkpoints=len(seeds)*9,full_metrics_passed=True,paired_aggregation_verified=True))
    print('AGGREGATE PASSED',seeds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--smoke',action='store_true');p.add_argument('--aggregate',action='store_true');args=p.parse_args()
    if args.seed is not None:analyze_seed(args.seed,args.smoke)
    if args.aggregate:aggregate(args.smoke)
    assert args.seed is not None or args.aggregate
