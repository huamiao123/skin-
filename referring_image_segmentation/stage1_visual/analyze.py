"""Paired validation diagnostics, simple averaging and oracle upper-bound analysis."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data' / 'isic2018'


def metrics(pred, gt):
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt)); tn = int(np.count_nonzero(~pred & ~gt))
    dice = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    iou = tp/(tp+fp+fn) if tp+fp+fn else 1.
    precision = tp/(tp+fp) if tp+fp else float(not gt.any())
    recall = tp/(tp+fn) if tp+fn else 1.
    pb = pred ^ binary_erosion(pred, border_value=0)
    gb = gt ^ binary_erosion(gt, border_value=0)
    diagonal = np.hypot(*gt.shape)
    tolerance = .005*diagonal
    if not pb.any() or not gb.any():
        bf1 = float(not pb.any() and not gb.any())
        hd95 = 0. if bf1 else diagonal
    else:
        dp = distance_transform_edt(~pb)[gb]
        dg = distance_transform_edt(~gb)[pb]
        bp,br = (dg<=tolerance).mean(),(dp<=tolerance).mean()
        bf1 = 2*bp*br/(bp+br) if bp+br else 0.
        hd95 = float(np.percentile(np.concatenate([dp,dg]),95))
    return dict(dice=dice,iou=iou,precision=precision,recall=recall,boundary_f1=bf1,
                hd95_pixels=hd95,hd95_diagonal=hd95/diagonal,tp=tp,fp=fp,fn=fn,tn=tn)


def main():
    out = ROOT/'analysis';out.mkdir(exist_ok=True)
    runs = {name:ROOT/'results'/name for name in ['dinov3','vmamba']}
    for path in runs.values(): assert (path/'DONE.json').exists()
    ids = json.loads((runs['dinov3']/'ids.json').read_text())
    assert ids == json.loads((runs['vmamba']/'ids.json').read_text())
    probs = {name:np.load(path/'probabilities.npy',mmap_mode='r') for name,path in runs.items()}
    mask_paths = {p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in (DATA/'val/masks').iterdir()}
    error_maps = np.empty((len(ids),256,256),dtype=np.uint8)

    def sample(index):
        sample_id = ids[index]
        gt = np.asarray(Image.open(mask_paths[sample_id]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        a,b = probs['dinov3'][index]>=.5,probs['vmamba'][index]>=.5
        average = (probs['dinov3'][index]+probs['vmamba'][index])*.5>=.5
        ea,eb = a!=gt,b!=gt
        # 0 both correct; 1 DINO correct/VMamba wrong; 2 reverse; 3 both wrong.
        codes = ea.astype(np.uint8)*2+eb.astype(np.uint8)
        error_maps[index] = codes
        border = gt ^ binary_erosion(gt,border_value=0)
        band = distance_transform_edt(~border)<=2
        row = dict(id=sample_id,foreground_fraction=float(gt.mean()),
                   models={name:metrics(pred,gt) for name,pred in [('dinov3',a),('vmamba',b),('mean_probability',average)]},
                   errors={name:np.bincount(codes[region],minlength=4).tolist()
                           for name,region in [('all',np.ones_like(gt)),('foreground',gt),('boundary_band_2px',band)]})
        return row
    with ThreadPoolExecutor(8) as pool: rows=list(pool.map(sample,range(len(ids))))
    (out/'per_image.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(out/'error_categories.npz',ids=np.array(ids),categories=error_maps)
    keys=['dice','iou','precision','recall','boundary_f1','hd95_pixels','hd95_diagonal']
    summary={}
    for name in ['dinov3','vmamba','mean_probability']:
        macro={key:float(np.mean([r['models'][name][key] for r in rows])) for key in keys}
        counts={k:sum(r['models'][name][k] for r in rows) for k in ['tp','fp','fn','tn']}
        t,f,n=counts['tp'],counts['fp'],counts['fn']
        summary[name]=dict(macro=macro,pooled=dict(dice=2*t/(2*t+f+n),iou=t/(t+f+n)),counts=counts)
    delta=np.array([r['models']['dinov3']['dice']-r['models']['vmamba']['dice'] for r in rows])
    rng=np.random.default_rng(42)
    boot=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
    oracle=float(np.mean([max(r['models']['dinov3']['dice'],r['models']['vmamba']['dice']) for r in rows]))
    counts={domain:np.sum([r['errors'][domain] for r in rows],axis=0).tolist()
            for domain in ['all','foreground','boundary_band_2px']}
    report=dict(evaluation_split='development validation, exploratory; not independent test',
        models=summary,paired_macro_dice_delta_dino_minus_vmamba=float(delta.mean()),
        paired_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
        bootstrap_note='2000 paired image resamples, seed42; selection on this validation set; no patient grouping available',
        wins=dict(dinov3=int((delta>0).sum()),vmamba=int((delta<0).sum()),ties=int((delta==0).sum())),
        oracle_macro_dice=oracle,oracle_note='Select better model per image using GT; diagnostic upper bound, not a deployable result',
        error_counts=counts,error_code_order=['both_correct','dino_correct_vmamba_wrong','vmamba_correct_dino_wrong','both_wrong'],
        boundary_definition='256x256 grid, erosion boundary, F1 tolerance=.005*image diagonal; HD95 in pixels and divided by diagonal; one empty mask HD95=diagonal, both empty=0',
        historical_reference=dict(cnn_only_pooled_dice=.885923196227111,cnn_transformer_pooled_dice=.8922425556820504,
                                 note='Different architecture/preprocessing/precision; historical context only, not a matched encoder control'))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 第一阶段视觉分割对照结果','',
           'ISIC2018 1886/808旧开发划分；结果属于开发验证集，非独立测试。',
           '', '| Model | Pooled Dice | Pooled IoU | Macro Dice | Boundary F1 | HD95 (px) |',
           '|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f"| {name} | {s['pooled']['dice']:.6f} | {s['pooled']['iou']:.6f} | {s['macro']['dice']:.6f} | {s['macro']['boundary_f1']:.6f} | {s['macro']['hd95_pixels']:.3f} |")
    lines += ['',f'逐图 oracle macro Dice：{oracle:.6f}，使用真值，仅作诊断。',
              f'DINO−VMamba macro Dice差：{delta.mean():.6f}；探索性配对95%区间：{report["paired_bootstrap95"]}。',
              '', '该实验比较两种不同预训练来源的冻结表征；不能把全部差异归因于架构。',
              '困难病例原因尚无人工标签；不将误差图自动解释为毛发、反光或语义错误。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Deterministic examples: largest wins in either direction, plus largest shared errors.
    candidates=list(np.argsort(delta)[:6])+list(np.argsort(-delta)[:6])
    candidates+=list(np.argsort([max(r['models']['dinov3']['dice'],r['models']['vmamba']['dice']) for r in rows])[:6])
    palette=np.array([[235,235,235],[40,110,240],[245,160,40],[220,45,45]],dtype=np.uint8)
    gallery=out/'examples';gallery.mkdir(exist_ok=True)
    for index in dict.fromkeys(candidates):
        sample_id=ids[index]
        image_path=next((DATA/'val/images').glob(sample_id+'.*'))
        image=Image.open(image_path).convert('RGB').resize((256,256))
        gt=Image.open(mask_paths[sample_id]).convert('RGB').resize((256,256),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(256*5,256));panel.paste(image,(0,0));panel.paste(gt,(256,0))
        for col,name in [(2,'dinov3'),(3,'vmamba')]:
            mask=Image.fromarray(((probs[name][index]>=.5)*255).astype(np.uint8)).convert('RGB')
            panel.paste(mask,(256*col,0))
        panel.paste(Image.fromarray(palette[error_maps[index]]),(1024,0))
        panel.save(gallery/(sample_id+'.png'))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
