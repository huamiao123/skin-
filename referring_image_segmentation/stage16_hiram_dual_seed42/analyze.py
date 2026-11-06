"""Full per-image region/boundary evaluation and paired error diagnostics."""
from common import *
import csv
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import summarize_rows
from stage1_visual.analyze import metrics

NAMES=('paired_baseline','region_only','hiram')


def main():
    base=ROOT/'results'
    done=json.loads((base/'DONE.json').read_text())
    evaluation=json.loads((base/'evaluation.json').read_text())
    assert done['epochs']==300 and evaluation['passed'] and evaluation['n_val']==808
    assert all(v==9000 for v in done['updates'].values())
    ids=json.loads((base/NAMES[0]/'best_dice/ids.json').read_text())
    for n in NAMES:
        assert json.loads((base/n/'best_dice/ids.json').read_text())==ids
    masks={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p
           for p in (PROJECT/'data/isic2018/val/masks').iterdir() if p.is_file()}
    probs={n:np.load(base/n/'best_dice/probabilities.npy',mmap_mode='r') for n in NAMES}
    def sample(i):
        sid=ids[i]
        gt=np.asarray(Image.open(masks[sid]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        predictions={n:probs[n][i]>=.5 for n in NAMES}
        vals={n:metrics(predictions[n],gt) for n in NAMES}
        base_err=predictions['paired_baseline']!=gt
        error={}
        for n in ('region_only','hiram'):
            err=predictions[n]!=gt
            error[n]=dict(corrected=int((base_err & ~err).sum()),introduced=int((~base_err & err).sum()))
        return dict(id=sid,foreground_fraction=float(gt.mean()),models=vals,errors=error)
    with ThreadPoolExecutor(6) as pool:
        rows=list(pool.map(sample,range(len(ids))))
    summaries={}
    for n in NAMES:
        s=summarize_rows([r['models'][n] for r in rows])
        expected=evaluation['results'][n]['best_dice']['metrics']
        assert s['counts']==expected['counts']
        assert abs(s['macro']['dice']-expected['macro']['dice'])<1e-12
        assert abs(s['macro']['iou']-expected['macro']['iou'])<1e-12
        s['selected_epoch']=evaluation['results'][n]['best_dice']['selected_epoch']
        summaries[n]=s
    comparisons={}
    for n in ('region_only','hiram'):
        delta=np.array([r['models'][n]['dice']-r['models']['paired_baseline']['dice'] for r in rows])
        rng=np.random.default_rng(42)
        ci=np.percentile([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)],[2.5,97.5]).tolist()
        corrected=sum(r['errors'][n]['corrected'] for r in rows)
        introduced=sum(r['errors'][n]['introduced'] for r in rows)
        comparisons[n]=dict(macro_dice_delta=float(delta.mean()),bootstrap95=ci,
            better=int((delta>0).sum()),worse=int((delta<0).sum()),ties=int((delta==0).sum()),
            corrected_pixels=corrected,introduced_errors=introduced,net_errors_removed=corrected-introduced)
    delta=np.array([r['models']['hiram']['dice']-r['models']['region_only']['dice'] for r in rows])
    comparisons['hiram_minus_region_only']=dict(macro_dice_delta=float(delta.mean()),
        better=int((delta>0).sum()),worse=int((delta<0).sum()),ties=int((delta==0).sum()))
    strata=[]
    for label,lo,hi in [('lt10pct',0,.1),('10to30pct',.1,.3),('ge30pct',.3,1.01)]:
        subset=[r for r in rows if lo<=r['foreground_fraction']<hi]
        strata.append(dict(group=label,n=len(subset),macro_dice={n:float(np.mean([r['models'][n]['dice'] for r in subset])) for n in NAMES}))
    history=json.loads((base/'history.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for n in NAMES:
        axes[0,0].plot([r['epoch'] for r in history],[r['models'][n]['train_seg'] for r in history],label=n)
        axes[0,1].plot([r['epoch'] for r in history],[r['models'][n]['val']['loss'] for r in history],label=n)
        axes[1,0].plot([r['epoch'] for r in history],[r['models'][n]['val']['macro']['dice'] for r in history],label=n)
        axes[1,1].plot([r['epoch'] for r in history],[r['models'][n]['val']['pooled']['dice'] for r in history],label=n)
    for ax,title in zip(axes.flat,['Train segmentation loss','Validation segmentation loss','Validation macro Dice','Validation pooled Dice']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'analysis/learning_curves.png',dpi=140);plt.close(fig)
    with (ROOT/'analysis/per_image.csv').open('w',newline='') as f:
        fields=['id','foreground_fraction','model','dice','iou','precision','recall','boundary_f1','hd95_pixels','hd95_diagonal','tp','fp','fn','tn']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            for n in NAMES:writer.writerow(dict(id=row['id'],foreground_fraction=row['foreground_fraction'],model=n,**row['models'][n]))
    bench=json.loads((ROOT/'analysis/benchmark.json').read_text())
    result=dict(seed=42,split='legacy ISIC2018 1886/808 development validation',selection='best validation macro Dice',
        models=summaries,comparisons=comparisons,strata=strata,parameters=json.loads((ROOT/'preflight.json').read_text())['models'],
        benchmark=bench,limitations=['single seed','development validation also used for checkpoint selection','paired baseline rerun due historical initialization hash mismatch',
        'adapted rather than full SAM-based HiRAM','16x16 boundary targets can disappear','added regional and interaction capacity'])
    dump(ROOT/'analysis/summary.json',result)
    lines=['# HiRAM 迁移：ISIC2018 seed 42','','全部模型为 300 epochs / 9000 updates；开发验证集不是独立测试。','',
           '| 模型 | 最佳轮 | Macro Dice | Pooled Dice | Macro IoU | Boundary F1 | HD95 px |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n,s in summaries.items():
        lines.append(f"| {n} | {s['selected_epoch']} | {s['macro']['dice']:.6f} | {s['pooled']['dice']:.6f} | {s['macro']['iou']:.6f} | {s['macro']['boundary_f1']:.6f} | {s['macro']['hd95_pixels']:.3f} |")
    for n in ('region_only','hiram'):
        c=comparisons[n]
        lines += ['',f"{n} 相对本机配对基线 Macro Dice 差 {c['macro_dice_delta']*100:+.3f} 个百分点；逐图配对 bootstrap 95% 区间 {c['bootstrap95']}（小数尺度）。纠正 {c['corrected_pixels']}、新增 {c['introduced_errors']} 个错误像素。"]
    lines += ['',f"HiRAM 相对仅区域监督的 Macro Dice 差 {comparisons['hiram_minus_region_only']['macro_dice_delta']*100:+.3f} 个百分点。",'',
        '历史 stage4 fusion_swin Macro Dice 为 0.895492；本机历史初始摘要无法复现，已重新训练配对基线。',
        '16×16 区域标签的双线性缩小会使 11 张训练图和 1 张验证图的边界标签消失。',
        '新增的区域监督、SSM、跨分支交互和参数量共同变化；B 与 C 比较仍不能完全排除容量贡献。',
        '单种子且反复使用开发集，不能宣称稳定提升或外部泛化。各项完整指标和推理速度见 summary.json。',
        '', '![学习曲线](learning_curves.png)']
    (ROOT/'analysis/REPORT.md').write_text('\n'.join(lines)+'\n')
    print('ANALYSIS COMPLETE',json.dumps({n:s['macro']['dice'] for n,s in summaries.items()}),flush=True)


if __name__=='__main__':main()
