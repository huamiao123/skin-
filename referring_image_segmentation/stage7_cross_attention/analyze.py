"""Independent complete metrics, capacity/selection comparisons and error repair."""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import ROOT,NAMES
from stage5_selective_fusion.diagnostics import PROJECT,context,pixel_diagnostics,dump,sha
from references import references,REFERENCE_PATHS
from evaluation import summarize_rows
from evaluate_runs import paired
from stage1_visual.analyze import metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    base=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((base/'DONE.json').read_text());budget=2 if args.smoke else 300
    assert done['epochs']==budget and done['optimizer_updates_per_model']==budget*30 and done['smoke_only']==args.smoke
    assert done['encoders_unchanged'] and done['gradient_checks_passed']
    if not args.smoke:
        provenance=json.loads((ROOT/'launch_provenance.json').read_text())
        for section in ('source_sha256','historical_sha256'):
            for path,digest in provenance[section].items():assert sha(path)==digest,path
    ids,gt,regions=context();history=json.loads((base/'history.json').read_text())
    assert [r['epoch'] for r in history]==list(range(1,budget+1))
    old=json.loads((PROJECT/'stage5_selective_fusion/analysis/summary.json').read_text())
    summaries={};comparisons={};repairs={};strata=[];flat=[]
    pairs=[('window_cross_attention',n) for n in ('local_conv','fusion','static_residual','dynamic_gate','capacity_concat','mean','dinov3')]+[('local_conv','dinov3'),('local_conv','fusion')]
    for criterion in ('best_dice','best_loss','last'):
        ref=references(criterion,ids);probs=dict(ref);records={}
        for n in NAMES:
            path=base/n/criterion;record=json.loads((path/'metrics.json').read_text());records[n]=record
            assert json.loads((path/'ids.json').read_text())==ids
            selected=(max(history,key=lambda r:r['models'][n]['val']['macro']['dice']) if criterion=='best_dice'
                      else min(history,key=lambda r:r['models'][n]['val']['loss']) if criterion=='best_loss' else history[-1])
            assert record['selected_epoch']==selected['epoch'] and record['final']==selected['models'][n]['val']
            probs[n]=np.load(path/'probabilities.npy',mmap_mode='r');binary=np.load(path/'masks.npy',mmap_mode='r')
            assert probs[n].shape==binary.shape==(808,256,256)
            assert all(np.array_equal(binary[i],probs[n][i]>=.5) for i in range(len(ids)))
        def sample(i):
            assert all(np.isfinite(p[i]).all() and p[i].min()>=0 and p[i].max()<=1 for p in probs.values())
            return dict(id=ids[i],foreground_fraction=float(gt[i].mean()),models={n:metrics(p[i]>=.5,gt[i]) for n,p in probs.items()})
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(sample,range(len(ids))))
        summaries[criterion]={};repairs[criterion]={}
        for n in probs:
            per=[dict(id=r['id'],foreground_fraction=r['foreground_fraction'],**r['models'][n]) for r in rows]
            summary=summarize_rows(per);summaries[criterion][n]=summary
            if n in REFERENCE_PATHS:summary['selected_epoch']=json.loads((REFERENCE_PATHS[n]/criterion/'metrics.json').read_text())['selected_epoch']
            expected=records[n]['final'] if n in NAMES else old['models'][criterion][n]
            assert summary['counts']==expected['counts']
            for mode in ('macro','pooled'):
                for key in ('dice','iou','precision','recall'):assert abs(summary[mode][key]-expected[mode][key])<1e-12,(n,criterion,mode,key)
            if n in NAMES:
                summary['selected_epoch']=records[n]['selected_epoch']
                summary['probability_sha256']=sha(base/n/criterion/'probabilities.npy')
                summary['checkpoint_sha256']=sha(base/n/(criterion+'.pth'))
                dump(base/n/criterion/'per_image_full.json',per);dump(base/n/criterion/'full_metrics.json',summary)
                diagnostic,per_errors=pixel_diagnostics(ids,gt,regions,ref['dinov3'],ref['vmamba'],probs[n])
                assert diagnostic['all']['candidate_wrong']==summary['counts']['fp']+summary['counts']['fn']
                repairs[criterion][n]=diagnostic
                dump(out/f'errors_{n}_{criterion}.json',per_errors)
                if criterion=='best_dice':
                    codes=((ref['dinov3']>=.5)!=gt).astype(np.uint8)*2+((probs[n]>=.5)!=gt).astype(np.uint8)
                    np.savez_compressed(out/f'dino_vs_{n}_errors.npz',ids=np.array(ids),categories=codes)
            flat.append(dict(model=n,criterion=criterion,selected_epoch=summary.get('selected_epoch','probability_mean'),
                **{f'macro_{k}':v for k,v in summary['macro'].items()},**{f'pooled_{k}':v for k,v in summary['pooled'].items()},
                **{f'prec_at_{k}':v for k,v in summary['prec_at'].items()},threshold_jaccard_065=summary['threshold_jaccard_065']))
        comparisons[criterion]={f'{a}_minus_{b}':paired(rows,a,b) for a,b in pairs}
        dump(out/f'per_image_{criterion}.json',rows)
        for variable,groups in [('dino_dice',[('lt08',0,.8),('08to09',.8,.9),('ge09',.9,1.01)]),
                                ('foreground_fraction',[('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)])]:
            for label,lo,hi in groups:
                subset=[r for r in rows if lo<=(r['models']['dinov3']['dice'] if variable=='dino_dice' else r[variable])<hi]
                if subset:
                    strata.append(dict(criterion=criterion,variable=variable,group=label,n=len(subset),
                        models={n:summarize_rows([r['models'][n] for r in subset]) for n in probs}))
        print('EVALUATED',criterion,{n:s['macro']['dice'] for n,s in summaries[criterion].items()},flush=True)
    config=json.loads((base/'config.json').read_text());bench=json.loads((ROOT/'preflight.json').read_text())
    summary=dict(seed=42,smoke_only=args.smoke,primary_criterion='best_dice',models=summaries,comparisons=comparisons,
        error_repair=repairs,parameters=config['parameters'],inference=bench['inference'],inference_note=bench['inference_note'],
        caveat='Single seed on reused development validation. Cross and local conv have approximately matched trainable parameters, not identical receptive fields/FLOPs. Both use the same fixed .1 residual; historical residual/fusion parameterizations differ.',
        error_code_order=['both_correct','dino_correct_candidate_wrong','dino_wrong_candidate_correct','both_wrong'])
    dump(out/'summary.json',summary);dump(out/'strata.json',strata)
    with (out/'all_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for n in NAMES:
        axes[0].plot([r['epoch'] for r in history],[r['models'][n]['val']['macro']['dice'] for r in history],label=n)
        axes[1].plot([r['epoch'] for r in history],[r['models'][n]['val']['loss'] for r in history],label=n)
    axes[0].set_ylabel('Validation mean Dice');axes[1].set_ylabel('Validation loss')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=140);plt.close(fig)
    lines=['# 第七轮：窗口Cross-Attention与局部卷积对照'+('（两轮流程预检，不能与正式结果比较）' if args.smoke else ''),'',
        'Seed42；ISIC2018旧1886/808开发划分，非独立测试；主checkpoint最高验证mean Dice，其他规则一并报告。','',
        '| 模型 | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 px |','|---|---:|---:|---:|---:|---:|---:|']
    for n,s in summaries['best_dice'].items():
        values=[s['macro']['dice'],s['pooled']['dice'],s['macro']['iou'],s['pooled']['iou'],s['macro']['boundary_f1'],s['macro']['hd95_pixels']]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','## 关键配对比较','', '| 比较 | Mean Dice差 | 开发集探索性95%区间 | 改善/退步/持平 |','|---|---:|---|---|']
    for n,c in comparisons['best_dice'].items():
        d=c['dice'];lines.append(f"| {n} | {d['mean_delta']:.6f} | {d['bootstrap95']} | {d['positive']}/{d['negative']}/{d['ties']} |")
    lines+=['','## 相对DINO的纠错和破坏','', '| 模型 | 纠正 | 新增 | 净减少 | 修复率 | 破坏率 |','|---|---:|---:|---:|---:|---:|']
    for n,s in repairs['best_dice'].items():
        d=s['all'];lines.append(f"| {n} | {d['corrected']} | {d['introduced']} | {d['net_errors_removed']} | {d['repair_rate']:.4%} | {d['damage_rate']:.4%} |")
    lines+=['','## 参数与完整推理','', '| 模型 | 可训练参数 | 总参数 | batch1 ms | batch64 images/s |','|---|---:|---:|---:|---:|']
    for n,p in config['parameters'].items():
        t=bench['inference'][n];lines.append(f"| {n} | {p['trainable']} | {p['total']} | {t['1']['seconds_per_batch']*1000:.3f} | {t['64']['images_per_second']:.2f} |")
    lines+=['', '全部Precision/Recall、Prec@X、Threshold Jaccard、HD95归一化、三种checkpoint结果见all_runs.csv/summary.json。',
        '前景概率分组、边界/内部/背景、两单分支正确性分组、逐图修复和退步均已保存。不得只凭新增错误减少或attention热图认定机制有效。',
        'Cross与LocalConv近似匹配可训练参数且同残差系数，但感受野与初始函数不同；主比较Cross对LocalConv/Concat。开发集单seed结果不证明稳定性或独立泛化。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'validation.json',dict(passed=True,smoke_only=args.smoke,checkpoint_selection_verified=True,
        predictions_recomputed=True,stage4_stage5_reference_metrics_match=True,data_manifest_match=True,
        original_artifacts_match=not args.smoke))
    print('ANALYSIS PASSED',flush=True)


if __name__=='__main__':main()
