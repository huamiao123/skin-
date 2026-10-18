"""Locked-checkpoint native evaluation, image-cluster comparisons and text diagnostics."""
from common import *
import argparse,csv
from collections import defaultdict
import numpy as np,torch
from models import build_probes,EncoderPair,tensor_hash
from data import RISData,FeatureCache
from engine import predict,score,restore,native_gt

def cluster_compare(rows,a,b):
    groups=defaultdict(list)
    for x,y in zip(rows[a],rows[b]):
        assert x['sample_id']==y['sample_id'] and x['image_id']==y['image_id']
        def iou(r):return r['tp']/(r['tp']+r['fp']+r['fn']) if r['tp']+r['fp']+r['fn'] else 1.
        groups[x['image_id']].append(iou(x)-iou(y))
    sums=np.array([sum(g) for g in groups.values()]);counts=np.array([len(g) for g in groups.values()]);rng=np.random.default_rng(42)
    boot=[]
    for _ in range(2000):
        ids=rng.integers(len(sums),size=len(sums));boot.append(float(sums[ids].sum()/counts[ids].sum()))
    return dict(mean_iou_delta=float(sums.sum()/counts.sum()),image_cluster_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
        independent_images=len(groups),expressions=int(counts.sum()),note='Exploratory paired image clusters, not independent expressions or training seeds.')

def switching(logits,data,split):
    groups=defaultdict(dict)
    for i,r in enumerate(data.rows[split][:len(logits)]):groups[r['image_id']].setdefault(r['ann_id'],i)
    rows=[]
    for image_id,anns in sorted(groups.items()):
        if len(anns)<2:continue
        ia,ib=[anns[k] for k in sorted(anns)[:2]];a,b=data.rows[split][ia],data.rows[split][ib]
        image=data.images[a['image_index']];pa=restore(logits[ia],image)>=0;pb=restore(logits[ib],image)>=0
        ga,gb=native_gt(a['ann_id']),native_gt(b['ann_id'])
        def iou(p,g):return float((p&g).sum()/max((p|g).sum(),1))
        rows.append(dict(image_id=image_id,sample_a=a['sample_id'],sample_b=b['sample_id'],expression_a=a['expression'],expression_b=b['expression'],
            correct_a=iou(pa,ga),wrong_target_a=iou(pa,gb),correct_b=iou(pb,gb),wrong_target_b=iou(pb,ga),
            changed_pixel_fraction=float((pa!=pb).mean()),both_prefer_correct=bool(iou(pa,ga)>iou(pa,gb) and iou(pb,gb)>iou(pb,ga))))
    summary=dict(images_with_two_targets=len(rows),mean_changed_pixel_fraction=float(np.mean([r['changed_pixel_fraction'] for r in rows])) if rows else None,
        both_targets_preferred_fraction=float(np.mean([r['both_prefer_correct'] for r in rows])) if rows else None,
        note='Deterministic first expression per first two annotation IDs of each image, not selected by scores. Correct target IoU is compared to the other real target.')
    return summary,rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    base=ROOT/('smoke' if args.smoke else 'results')/'seed42';out=ROOT/('smoke_analysis' if args.smoke else 'analysis');out.mkdir(exist_ok=True)
    done=json.loads((base/'DONE.json').read_text());assert done['passed'] and done['checkpoint_selection_locked']
    history=json.loads((base/'history.json').read_text());assert len(history)==(2 if args.smoke else 30)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    enc=EncoderPair().cuda();enc_hash={n:tensor_hash(getattr(enc,n)) for n in ('dinov3','vmamba')}
    data=RISData();splits=('val',) if args.smoke else ('val','testA','testB');cache=FeatureCache(enc,data,splits=splits)
    for s in splits:cache.encode(s,rows=data.rows[s][:128] if args.smoke else None)
    models=build_probes();all_metrics={};all_rows={};diagnostics={};flat=[]
    for n,p in models.items():
        p.cuda();state=torch.load(base/n/'best_miou.pth',map_location='cpu');p.load_state_dict(state['probe']);all_metrics[n]={};all_rows[n]={};diagnostics[n]={}
        selected=max(history,key=lambda x:x['models'][n]['val']['macro']['iou']);assert selected['epoch']==state['epoch']
        for s in splits:
            logits,loss=predict(p,cache,s,limit=128 if args.smoke else None);summary,rows=score(logits,data,s,full=True)
            if s=='val':
                assert summary['counts']==state['metrics']['counts'],(n,'Val counts differ')
                for mode in ('macro','pooled'):
                    for k in ('iou','dice','precision','recall'):assert abs(summary[mode][k]-state['metrics'][mode][k])<1e-12
                assert np.array_equal(logits,np.load(base/n/'best_miou_logits64.npy'))
            summary.update(selected_epoch=state['epoch'],loss=loss,checkpoint_sha256=sha(base/n/'best_miou.pth'))
            dest=out/n/s;dest.mkdir(parents=True,exist_ok=True);np.save(dest/'logits64.npy',logits)
            dump(dest/'metrics.json',summary);dump(dest/'per_expression.json',rows)
            diag,pairs=switching(logits,data,s);dump(dest/'same_image_pairs.json',pairs);diagnostics[n][s]=diag
            all_metrics[n][s]=summary;all_rows[n][s]=rows
            flat.append(dict(model=n,split=s,selected_epoch=state['epoch'],**{f'macro_{k}':v for k,v in summary['macro'].items()},
                **{f'pooled_{k}':v for k,v in summary['pooled'].items()},**{f'prec_at_{k}':v for k,v in summary['prec_at'].items()},
                image_balanced_mean_iou=summary['image_balanced_mean_iou']))
            print('EVALUATED',n,s,summary['macro']['iou'],summary['pooled']['iou'],flush=True)
        if n!='dinov3_fixed_text':
            logits,_=predict(p,cache,'val',limit=128 if args.smoke else None,fixed_intervention=True)
            intervention,rows=score(logits,data,'val',full=True)
            dump(out/n/'fixed_text_intervention.json',dict(metrics=intervention,note='Posthoc fixed text on trained real-text model; not a trained capacity control.'))
        p.cpu()
    comparisons={s:{f'{a}_minus_{b}':cluster_compare({n:all_rows[n][s] for n in NAMES},a,b)
        for a,b in [('fusion_text','dinov3_text'),('fusion_text','vmamba_text'),('dinov3_text','vmamba_text'),('dinov3_text','dinov3_fixed_text')]} for s in splits}
    # Show deterministic multi-target validation examples using native predictions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    groups=defaultdict(dict)
    limit=128 if args.smoke else len(data.rows['val'])
    for i,r in enumerate(data.rows['val'][:limit]):groups[r['image_id']].setdefault(r['ann_id'],i)
    examples=[(im,anns) for im,anns in sorted(groups.items()) if len(anns)>=2][:4]
    gallery=out/'examples';gallery.mkdir(exist_ok=True)
    for image_id,anns in examples:
        indices=[anns[k] for k in sorted(anns)[:2]];fig,axes=plt.subplots(2,6,figsize=(18,7))
        for rr,i in enumerate(indices):
            row=data.rows['val'][i];info=data.images[row['image_index']];image=Image.open(ASSETS/'images'/info['file_name']).convert('RGB')
            values=[image,native_gt(row['ann_id'])]+[restore(np.load(out/n/'val/logits64.npy',mmap_mode='r')[i],info)>=0 for n in NAMES]
            for ax,value,title in zip(axes[rr],values,['Image','GT']+list(NAMES)):
                ax.imshow(value,cmap='gray');ax.axis('off');ax.set_title(title)
            axes[rr,0].set_title(row['expression'],wrap=True)
        fig.tight_layout();fig.savefig(gallery/f'{image_id}.png',dpi=120);plt.close(fig)
    assert all(tensor_hash(getattr(enc,n))==h for n,h in enc_hash.items())
    result=dict(seed=42,smoke_only=args.smoke,models=all_metrics,comparisons=comparisons,same_image_switching=diagnostics,
        caveat='One exploratory seed. Frozen pretrained sources differ. CLIP text-only alignment baseline, not CRIS reproduction. Original images and multiple expressions share statistical clusters.')
    dump(out/'summary.json',result)
    with (out/'all_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# RefCOCO UNC：统一文本匹配基线'+('（流程预检，不能作为正式结果）' if args.smoke else ''),'',
        'Seed42；按原图逐表达validation mean IoU选模；testA/testB在四组选择锁定后评分。固定文本为独立从头训练的DINO对照。','',
        '| Model | Split | Mean IoU | Overall IoU | Mean Dice | Pooled Dice | Boundary F1 | HD95 px |','|---|---|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        for s,m in all_metrics[n].items():
            vals=[m['macro']['iou'],m['pooled']['iou'],m['macro']['dice'],m['pooled']['dice'],m['macro']['boundary_f1'],m['macro']['hd95_pixels']]
            lines.append('| '+n+' | '+s+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','全部Precision/Recall、Prec@X、HD95归一化、image-balanced mIoU见all_runs.csv和summary.json。',
        '配对区间按图像聚类；同图表达/目标切换是诊断，模型不是在所有情况下都能正确选择目标。所有源码/数据/权重来源见冻结协议和provenance。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'validation.json',dict(passed=True,seed=42,smoke_only=args.smoke,models=4,splits=list(splits),
        native_region_counts_match_training=True,logits_match_selection=True,full_metrics_passed=True,same_image_diagnostics_saved=True))
    print('EVALUATION PASSED',flush=True)

if __name__=='__main__':main()
