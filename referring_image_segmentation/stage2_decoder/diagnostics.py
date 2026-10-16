"""Plots and target-area strata for the completed controlled decoder experiment."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent


def main():
    analysis=ROOT/'analysis'
    rows=json.loads((analysis/'per_image.json').read_text())
    summary=json.loads((analysis/'summary.json').read_text())
    figures=analysis/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.bbox':'tight'})
    # A common set of area thresholds for all methods, no sample filtering.
    bins=[('small (<10%)',0,.1),('medium (10-30%)',.1,.3),('large (>=30%)',.3,1.01)]
    records=[]
    for label,lo,hi in bins:
        subset=[r for r in rows if lo<=r['foreground_fraction']<hi]
        for name in summary['models']:
            record=dict(area_group=label,n=len(subset),model=name)
            for key in ('dice','iou','boundary_f1','hd95_pixels'):
                record[key]=float(np.mean([r['models'][name][key] for r in subset])) if subset else None
            records.append(record)
    with (analysis/'area_strata.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (analysis/'area_strata.json').write_text(json.dumps(records,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,encoder in enumerate(('dinov3','vmamba')):
        for stage,label,color in [('stage1_visual','Light','#3974b5'),('stage2_decoder','Swin','#d17c26')]:
            history=json.loads((PROJECT/stage/'results'/encoder/'history.json').read_text())
            done=json.loads((PROJECT/stage/'results'/encoder/'DONE.json').read_text())
            epochs=[r['epoch'] for r in history]
            axes[0,col].plot(epochs,[r['val']['loss'] for r in history],label=label,color=color,alpha=.8)
            axes[1,col].plot(epochs,[r['val']['macro']['dice'] for r in history],label=label,color=color,alpha=.8)
            for ax in axes[:,col]: ax.axvline(done['best_epoch'],color=color,linestyle=':',alpha=.7)
        axes[0,col].set_title(encoder)
        axes[0,col].set_ylabel('Validation loss')
        axes[1,col].set_ylabel('Mean image Dice')
        axes[1,col].set_xlabel('Epoch')
        axes[0,col].legend()
    fig.suptitle('Seed 42, development validation; dotted lines = minimum-loss checkpoint')
    fig.tight_layout();fig.savefig(figures/'learning_curves.png');fig.savefig(figures/'learning_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,encoder in zip(axes,('dinov3','vmamba')):
        light=np.array([r['models'][encoder+'_light']['dice'] for r in rows])
        swin=np.array([r['models'][encoder+'_swin']['dice'] for r in rows])
        points=ax.scatter(light,swin,c=[r['foreground_fraction'] for r in rows],s=9,cmap='viridis',alpha=.65)
        ax.plot([0,1],[0,1],'k--',linewidth=1)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Light Dice per image',ylabel='Swin Dice per image',title=encoder)
        ax.set_aspect('equal')
    fig.colorbar(points,ax=axes.tolist(),label='Ground-truth foreground fraction',shrink=.8)
    fig.savefig(figures/'paired_dice.png');fig.savefig(figures/'paired_dice.pdf');plt.close(fig)
    image_paths={p.stem:p for p in (PROJECT/'data/isic2018/val/images').iterdir() if p.is_file()}
    gt_paths={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (PROJECT/'data/isic2018/val/masks').iterdir() if p.is_file()}
    selected={}
    for encoder in ('dinov3','vmamba'):
        delta=np.array([r['models'][encoder+'_swin']['dice']-r['models'][encoder+'_light']['dice'] for r in rows])
        order=np.argsort(delta)
        indices=list(order[-2:][::-1])+list(order[:2])
        selected[encoder]=[dict(id=rows[i]['id'],dice_delta=float(delta[i]),selection='largest gains' if j<2 else 'largest losses') for j,i in enumerate(indices)]
        lp=np.load(PROJECT/'stage1_visual/results'/encoder/'probabilities.npy',mmap_mode='r')
        sp=np.load(ROOT/'results'/encoder/'probabilities.npy',mmap_mode='r')
        fig,axes=plt.subplots(4,4,figsize=(10,10))
        for j,i in enumerate(indices):
            sid=rows[i]['id']
            image=np.asarray(Image.open(image_paths[sid]).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            gt=np.asarray(Image.open(gt_paths[sid]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
            axes[j,0].imshow(image);axes[j,0].set_title(f'{sid}: delta {delta[i]:+.3f}')
            axes[j,1].imshow(gt,cmap='gray',vmin=0,vmax=1);axes[j,1].set_title('Ground truth')
            for col,prob,label in [(2,lp,'light'),(3,sp,'swin')]:
                axes[j,col].imshow(image)
                pred=prob[i]>=.5
                if pred.any() and not pred.all(): axes[j,col].contour(pred,levels=[.5],colors=['cyan'],linewidths=1)
                if gt.any() and not gt.all(): axes[j,col].contour(gt,levels=[.5],colors=['yellow'],linewidths=.7)
                axes[j,col].set_title(f"{label}: Dice {rows[i]['models'][encoder+'_'+label]['dice']:.3f}")
            for ax in axes[j]: ax.axis('off')
        fig.suptitle(f'{encoder}: two largest gains and losses (selected extremes)\nCyan = prediction; yellow = ground truth')
        fig.tight_layout();fig.savefig(figures/f'extremes_{encoder}.png');plt.close(fig)
    (analysis/'selected_examples.json').write_text(json.dumps(selected,indent=2)+'\n')
    lines=['# Decoder 对照诊断图', '', '所有图基于 seed 42 和已有开发验证集。',
        '', '![训练曲线](figures/learning_curves.png)', '', '![逐图 Dice](figures/paired_dice.png)',
        '', '## 按目标面积分组', '', '面积定义为 GT 前景像素占 256×256 的比例；固定分组 <10%、10–30%、>=30%。探索性分组不用于选模。',
        '', '| Group | N | Model | Mean Dice | mIoU | Boundary F1 | HD95 px |','|---|---:|---|---:|---:|---:|---:|']
    for r in records:
        if r['n']:
            lines.append(f"| {r['area_group']} | {r['n']} | {r['model']} | {r['dice']:.6f} | {r['iou']:.6f} | {r['boundary_f1']:.6f} | {r['hd95_pixels']:.3f} |")
    lines += ['', '## 改善/退步最大的病例', '', '下图有意选择 Dice 改善最大的两张与退步最大的两张，不能代表随机样本或整体分布。未人工标注毛发、反光等原因。',
              '', '![DINOv3](figures/extremes_dinov3.png)', '', '![VMamba](figures/extremes_vmamba.png)']
    (analysis/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print('Saved diagnostic plots and target-area strata:',analysis,flush=True)


if __name__=='__main__':main()
