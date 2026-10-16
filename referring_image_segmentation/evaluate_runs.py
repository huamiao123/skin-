"""Re-score immutable probability exports with the common complete metric set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

from evaluation import summarize_rows
from stage1_visual.analyze import metrics

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def paired(rows, a, b):
    result = dict(direction=f'{a} minus {b}')
    for key in ('dice','iou','boundary_f1','hd95_pixels'):
        delta = np.array([r['models'][a][key]-r['models'][b][key] for r in rows])
        rng = np.random.default_rng(42)
        boot = np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
        result[key] = dict(mean_delta=float(delta.mean()),bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
                           positive=int((delta>0).sum()),negative=int((delta<0).sum()),ties=int((delta==0).sum()))
    result['note'] = 'Exploratory development-set image bootstrap; single seed; not patient-level or independent-test inference. Negative HD95 delta is better.'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['stage1_visual','stage2_decoder'],required=True)
    args=parser.parse_args()
    out = ROOT/'evaluations/stage1_visual' if args.stage=='stage1_visual' else ROOT/'stage2_decoder/analysis'
    out.mkdir(parents=True,exist_ok=True)
    runs={f'{n}_light':ROOT/'stage1_visual/results'/n for n in ('dinov3','vmamba')}
    if args.stage=='stage2_decoder':
        runs.update({f'{n}_swin':ROOT/'stage2_decoder/results'/n for n in ('dinov3','vmamba')})
    ids=None; probs={}; metadata={}
    for name,path in runs.items():
        done=json.loads((path/'DONE.json').read_text())
        assert done['epochs']==300 and not done['smoke_only'] and done['encoder_unchanged']
        current=json.loads((path/'ids.json').read_text())
        if ids is None: ids=current
        assert current==ids
        probs[name]=np.load(path/'probabilities.npy',mmap_mode='r')
        assert probs[name].shape==(808,256,256)
        cfg=json.loads((path/'config.json').read_text())
        metadata[name]=dict(path=str(path),best_epoch=done['best_epoch'],trainable_params=cfg['trainable_params'],
            total_params=cfg['total_params'],config=cfg,
            probability_sha256=sha256(path/'probabilities.npy'),checkpoint_sha256=sha256(path/'best.pth'))
    mask_paths={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (ROOT/'data/isic2018/val/masks').iterdir()}
    expected=json.loads((ROOT/'stage1_visual/data_manifest.json').read_text())
    for item in expected['members']:
        if item['split']=='val':
            assert sha256(mask_paths[item['id']])==item['mask_sha256']
    pair_names=[('dinov3_light','vmamba_light')]
    if args.stage=='stage2_decoder':
        pair_names += [('dinov3_swin','dinov3_light'),('vmamba_swin','vmamba_light'),('dinov3_swin','vmamba_swin')]
    error_maps={f'{a}_vs_{b}':np.empty((len(ids),256,256),dtype=np.uint8) for a,b in pair_names}

    def sample(i):
        gt=np.asarray(Image.open(mask_paths[ids[i]]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        pp={name:p[i] for name,p in probs.items()}
        assert all(np.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in pp.values())
        pp['mean_light']=(pp['dinov3_light']+pp['vmamba_light'])*.5
        if args.stage=='stage2_decoder': pp['mean_swin']=(pp['dinov3_swin']+pp['vmamba_swin'])*.5
        preds={name:p>=.5 for name,p in pp.items()}
        for a,b in pair_names:
            error_maps[f'{a}_vs_{b}'][i]=(preds[a]!=gt).astype(np.uint8)*2+(preds[b]!=gt).astype(np.uint8)
        return dict(id=ids[i],foreground_fraction=float(gt.mean()),models={name:metrics(p,gt) for name,p in preds.items()})

    with ThreadPoolExecutor(8) as pool: rows=list(pool.map(sample,range(len(ids))))
    summaries={name:summarize_rows([r['models'][name] for r in rows]) for name in rows[0]['models']}
    for name,path in runs.items():
        final=json.loads((path/'DONE.json').read_text())['final']
        assert summaries[name]['counts']==final['counts']
        for mode in ('macro','pooled'):
            for key in ('dice','iou','precision','recall'):
                assert abs(summaries[name][mode][key]-final[mode][key])<1e-12,(name,mode,key)
    # Historical scores must be numerically unchanged by the expanded reporting.
    old=json.loads((ROOT/'stage1_visual/analysis/summary.json').read_text())['models']
    for name,original in [('dinov3_light','dinov3'),('vmamba_light','vmamba'),('mean_light','mean_probability')]:
        for mode in ('macro','pooled'):
            for key,value in old[original][mode].items():
                assert abs(summaries[name][mode][key]-value)<1e-12,(name,mode,key)
    comparisons={f'{a}_minus_{b}':paired(rows,a,b) for a,b in pair_names}
    for suffix in ('light','swin'):
        if f'mean_{suffix}' in summaries:
            comparisons[f'mean_{suffix}_minus_dinov3_{suffix}']=paired(rows,f'mean_{suffix}',f'dinov3_{suffix}')
    errors={}
    for key,codes in error_maps.items():
        np.savez_compressed(out/f'errors_{key}.npz',ids=np.array(ids),categories=codes)
        errors[key]=np.bincount(codes.ravel(),minlength=4).tolist()
    report=dict(stage=args.stage,evaluation_split='ISIC2018 legacy development validation; single seed 42; not independent test',
        samples=len(ids),resolution=[256,256],threshold=.5,models=summaries,runs=metadata,paired=comparisons,
        error_counts=errors,error_order=['both_correct','first_correct_second_wrong','second_correct_first_wrong','both_wrong'],
        historical_metrics_match=True)
    save_json(out/'per_image.json',rows);save_json(out/'summary.json',report)
    flat=[]
    for name,s in summaries.items():
        flat.append(dict(model=name,**{f'macro_{k}':v for k,v in s['macro'].items()},
            **{f'pooled_{k}':v for k,v in s['pooled'].items()},**{f'prec_at_{k}':v for k,v in s['prec_at'].items()},
            threshold_jaccard_065=s['threshold_jaccard_065']))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# 完整指标与 Decoder 对照' if args.stage=='stage2_decoder' else '# 第一阶段补充完整指标',
        '', 'ISIC2018 旧开发划分，1886 train / 808 validation，256×256，seed 42。非独立测试。所有概率阈值固定 0.5。',
        '', '## 区域指标', '', '| Model | Mean Dice | Pooled Dice | mIoU | oIoU | Mean Precision | Mean Recall | Pooled Precision | Pooled Recall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        m,p=s['macro'],s['pooled']
        values=[m['dice'],p['dice'],m['iou'],p['iou'],m['precision'],m['recall'],p['precision'],p['recall']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## 样本成功率与边界', '', '| Model | P@.5 | P@.6 | P@.7 | P@.8 | P@.9 | Threshold Jaccard | Boundary F1 | HD95 px | HD95 / diagonal |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        values=list(s['prec_at'].values())+[s['threshold_jaccard_065'],s['macro']['boundary_f1'],s['macro']['hd95_pixels'],s['macro']['hd95_diagonal']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', 'Prec@X 使用 IoU >= X。Threshold Jaccard 在 IoU < .65 时记零。当前划分/分辨率与官方挑战不同。',
              '', '## 逐图配对差', '', '| Comparison | Mean Dice delta | Exploratory 95% interval | Better / worse / ties |', '|---|---:|---|---|']
    for name,c in comparisons.items():
        d=c['dice']
        lines.append(f"| {name} | {d['mean_delta']:.6f} | [{d['bootstrap95'][0]:.6f}, {d['bootstrap95'][1]:.6f}] | {d['positive']} / {d['negative']} / {d['ties']} |")
    lines += ['', '2000 次逐图配对 bootstrap 仅作开发集探索性分析；不能推断跨 seed 稳定性或患者级独立泛化。',
              '', '## 模型与选模', '', '| Model | Best epoch by validation loss | Trainable parameters | Total parameters |', '|---|---:|---:|---:|']
    for name,m in metadata.items():
        lines.append(f"| {name} | {m['best_epoch']} | {m['trainable_params']} | {m['total_params']} |")
    if args.stage=='stage2_decoder':
        lines += ['', '## 推理效率预检', '', '| Encoder | Decoder | Batch 1 ms | Batch 64 images/s |', '|---|---|---:|---:|']
        for name in ('dinov3','vmamba'):
            bench=json.loads((ROOT/f'stage2_decoder/preflight_{name}.json').read_text())
            for decoder,t in bench['inference'].items():
                lines.append(f"| {name} | {decoder} | {1000*t['1']['seconds_per_batch']:.3f} | {t['64']['images_per_second']:.2f} |")
        lines += ['', '同机 BF16 合成输入，完整 Encoder+Adapter+Decoder，3 次预热/10 次重复，不含读图与后处理；为预检测量。',
                  'Light 与 Swin 参数量不同，差异代表本次 Decoder 配置的整体影响。']
    lines += ['', '编码器来源不同，跨编码器结果不能全部归因于架构。',
              '第一阶段全部原有 macro/pooled/边界结果重算与历史值一致（误差 < 1e-12）。',
              '逐图记录见 per_image.json，机器可读汇总见 summary.json/summary.csv，误差图类别见 errors_*.npz。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name:{'macro':s['macro'],'pooled':s['pooled']} for name,s in summaries.items()},indent=2),flush=True)


if __name__=='__main__': main()
