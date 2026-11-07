"""Full per-image metrics and paired comparisons against MambaVision + Swin."""
from hgm_model import *
import csv
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import summarize_rows
from stage1_visual.analyze import metrics


def main():
    local=Path(__file__).resolve().parent
    out=local/'results'
    done=json.loads((out/'DONE.json').read_text())
    evaluation=json.loads((out/'evaluation.json').read_text())
    assert done['epochs']==300 and done['updates']==9000 and evaluation['n_val']==808
    baseline=PROJECT/'stage20_decoder_fusion_seed42/results/paired_baseline/best_dice'
    baseline_done=json.loads((PROJECT/'stage20_decoder_fusion_seed42/results/DONE.json').read_text())
    assert baseline_done['epochs']==300 and baseline_done['encoder_hash']==done['encoder_hash']
    ids=json.loads((out/'best_dice/ids.json').read_text())
    assert ids==json.loads((baseline/'ids.json').read_text())
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p
           for p in (PROJECT/'data/isic2018/val/masks').iterdir() if p.is_file()}
    probs={'hgm_bmd':np.load(out/'best_dice/probabilities.npy',mmap_mode='r'),
           'swin':np.load(baseline/'probabilities.npy',mmap_mode='r')}
    def sample(i):
        sid=ids[i]
        gt=np.asarray(Image.open(masks[sid]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        predictions={n:probs[n][i]>=.5 for n in probs}
        vals={n:metrics(predictions[n],gt) for n in probs}
        base_error=predictions['swin']!=gt
        other_error=predictions['hgm_bmd']!=gt
        return dict(id=sid,foreground_fraction=float(gt.mean()),models=vals,
                    corrected=int((base_error & ~other_error).sum()),
                    introduced=int((~base_error & other_error).sum()))
    with ThreadPoolExecutor(6) as pool:
        rows=list(pool.map(sample,range(len(ids))))
    summaries={n:summarize_rows([r['models'][n] for r in rows]) for n in probs}
    expected=evaluation['results']['best_dice']['metrics']
    assert summaries['hgm_bmd']['counts']==expected['counts']
    assert abs(summaries['hgm_bmd']['macro']['dice']-expected['macro']['dice'])<1e-12
    expected_base=json.loads((baseline/'metrics.json').read_text())['metrics']
    assert summaries['swin']['counts']==expected_base['counts']
    summaries['hgm_bmd']['selected_epoch']=evaluation['results']['best_dice']['selected_epoch']
    summaries['swin']['selected_epoch']=json.loads((baseline/'metrics.json').read_text())['selected_epoch']
    delta=np.array([r['models']['hgm_bmd']['dice']-r['models']['swin']['dice'] for r in rows])
    rng=np.random.default_rng(42)
    ci=np.percentile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)],[2.5,97.5]).tolist()
    order=np.argsort(delta)
    comparison=dict(macro_dice_delta=float(delta.mean()),bootstrap95=ci,better=int((delta>0).sum()),
        worse=int((delta<0).sum()),ties=int((delta==0).sum()),
        corrected_pixels=sum(r['corrected'] for r in rows),introduced_pixels=sum(r['introduced'] for r in rows),
        largest_improvements=[dict(id=ids[i],delta_dice=float(delta[i])) for i in order[-10:][::-1]],
        largest_regressions=[dict(id=ids[i],delta_dice=float(delta[i])) for i in order[:10]])
    strata=[]
    for label,lo,hi in [('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)]:
        subset=[r for r in rows if lo<=r['foreground_fraction']<hi]
        strata.append(dict(group=label,n=len(subset),macro_dice={n:float(np.mean([r['models'][n]['dice'] for r in subset])) for n in probs}))
    analysis=local/'analysis'
    analysis.mkdir(exist_ok=True)
    with (analysis/'per_image.csv').open('w',newline='') as f:
        fields=['id','foreground_fraction','model','dice','iou','precision','recall','boundary_f1','hd95_pixels','hd95_diagonal','tp','fp','fn','tn']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            for n in probs:writer.writerow(dict(id=row['id'],foreground_fraction=row['foreground_fraction'],model=n,**row['models'][n]))
    history=json.loads((out/'history.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,values,title in zip(axes.flat,
        [[r['train_loss'] for r in history],[r['val']['loss'] for r in history],
         [r['val']['macro']['dice'] for r in history],[r['val']['pooled']['dice'] for r in history]],
        ['Train loss','Validation loss','Validation macro Dice','Validation pooled Dice']):
        ax.plot([r['epoch'] for r in history],values)
        ax.axvline(summaries['hgm_bmd']['selected_epoch'],color='orange',linestyle='--')
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(analysis/'learning_curves.png',dpi=140);plt.close(fig)
    benchmark=json.loads((analysis/'benchmark.json').read_text())
    result=dict(seed=42,split='legacy ISIC2018 1886/808 development validation',
        models=summaries,comparison_hgm_bmd_minus_swin=comparison,strata=strata,
        parameters=json.loads((local/'preflight.json').read_text()),benchmark=benchmark,
        limitations=['single seed and development validation used for selection',
          'partial Swin plus HGM BMD adaptation, not the complete HGM architecture',
          'decoder parameters, topology and normalization differ from Swin',
          'FP32 Mamba 1.2.2 versus upstream dependency 2.2.4','no independent external test'])
    dump(analysis/'summary.json',result)
    lines=['# MambaVision-S + 适配版 HGM BMD Decoder：ISIC2018 seed 42','',
           '两组均为 BF16、300 epochs、9000 updates；验证集为用于选模的开发集。', '',
           '| 模型 | 最佳轮 | Macro Dice | Pooled Dice | Macro IoU | Boundary F1 | HD95 px |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n,s in summaries.items():
        lines.append(f"| {n} | {s['selected_epoch']} | {s['macro']['dice']:.6f} | {s['pooled']['dice']:.6f} | {s['macro']['iou']:.6f} | {s['macro']['boundary_f1']:.6f} | {s['macro']['hd95_pixels']:.3f} |")
    lines += ['',f"HGM BMD 适配版相对同机 Swin 的 Macro Dice 差 {comparison['macro_dice_delta']*100:+.3f} 个百分点；逐图配对 bootstrap 95% 区间 {ci}（小数尺度）。",
              f"纠正 {comparison['corrected_pixels']} 个像素，新增 {comparison['introduced_pixels']} 个错误像素。",'',
              '这是部分 Swin 与 HGM BMD 的适配版，使用本地 Mamba 1.2.2，不能称为完整 HGM 复现。',
              'HGM BMD 适配版约 105 万可训练参数，而 Swin probe 约 50 万；拓扑与归一化不同，不能把差异仅归因于扫描或视作等参数实验。',
              '原 HGM 的 BDFM、图网络并未迁移；适配版改变 Decoder 的多个内部组件。',
              '单种子且反复使用开发集，不能宣称稳定或外部泛化；完整指标、逐图分数和耗时见 summary.json。',
              '', '![学习曲线](learning_curves.png)']
    (analysis/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('ANALYSIS COMPLETE',json.dumps({n:s['macro']['dice'] for n,s in summaries.items()}),flush=True)


if __name__=='__main__':main()
