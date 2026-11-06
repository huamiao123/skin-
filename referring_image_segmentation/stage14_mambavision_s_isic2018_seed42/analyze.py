"""Locked-checkpoint comparison, numerical figures and balanced local case panels."""
from common import *
from evaluate import csv_rows
from PIL import Image,ImageDraw
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_gt(ids):
    root=PROJECT/'data/isic2018/val/masks'
    paths={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in root.iterdir() if p.is_file()}
    def one(sid):return np.asarray(Image.open(paths[sid]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
    with ThreadPoolExecutor(8) as pool:return np.stack(list(pool.map(one,ids)))

def plot(history,rows,epoch):
    out=ROOT/'analysis';epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    series=[('Training loss',[r['train_loss'] for r in history]),('Validation loss',[r['val']['loss'] for r in history]),
        ('Validation macro Dice',[r['val']['macro']['dice'] for r in history]),
        ('Validation pooled Dice',[r['val']['pooled']['dice'] for r in history]),('Learning rate',[r['lr'] for r in history]),
        ('Epoch seconds',[r['seconds'] for r in history])]
    for ax,(title,values) in zip(axes.flat,series):
        ax.plot(epochs,values);ax.axvline(epoch,color='tab:red',ls='--',label=f'best_dice epoch {epoch}')
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.suptitle('MambaVision-S + historical Swin | ISIC2018 development validation | seed42')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes[:2],('dice','iou')):
        values=np.array([r[key] for r in rows]);ax.hist(values,bins=np.linspace(0,1,26),color='tab:blue',alpha=.8)
        ax.axvline(values.mean(),color='tab:red',label=f'macro={values.mean():.4f}');ax.set_title(f'Per-image {key}');ax.set_xlim(0,1);ax.legend()
    groups=[('GT<10%',lambda x:x<.1),('GT10-30%',lambda x:.1<=x<.3),('GT>=30%',lambda x:x>=.3)]
    values=[[r['dice'] for r in rows if fn(r['foreground_fraction'])] for _,fn in groups]
    axes[2].boxplot(values,labels=[f'{name}\nn={len(v)}' for (name,_),v in zip(groups,values)])
    axes[2].set_title('Dice by fixed lesion-area groups');axes[2].set_ylim(0,1)
    fig.tight_layout();fig.savefig(out/'per_image_distributions.png',dpi=160);plt.close(fig)

def main():
    verify_frozen_sources();out=ROOT/'analysis'
    config=json.loads((ROOT/'config.json').read_text());done=json.loads((ROOT/'results/DONE.json').read_text())
    history=json.loads((ROOT/'results/history.json').read_text());models=json.loads((out/'metrics.json').read_text())
    rows=json.loads((out/'per_image_best_dice.json').read_text());ids=[r['id'] for r in rows]
    gt=load_gt(ids);newprob=np.load(ROOT/'results/best_dice/probabilities.npy',mmap_mode='r');newpred=newprob>=.5
    old=PROJECT/'stage4_fusion';old_report=json.loads((old/'analysis/summary.json').read_text())
    old_rows=json.loads((old/'analysis/per_image_best_dice.json').read_text());assert [r['id'] for r in old_rows]==ids
    baseline_names=('dinov3_swin','vmamba_swin');comparisons={};reference_probs={}
    paired_rows=[dict(id=r['id'],foreground_fraction=r['foreground_fraction'],mambavision_dice=r['dice'],mambavision_iou=r['iou']) for r in rows]
    for name in baseline_names:
        reference=old_report['models']['best_dice'][name]
        old_dice=np.array([r['models'][name]['dice'] for r in old_rows]);delta=np.array([r['dice'] for r in rows])-old_dice
        rng=np.random.default_rng(42);boot=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
        comparison=dict(reference=reference,macro_delta={k:models['best_dice']['macro'][k]-reference['macro'][k] for k in reference['macro']},
            pooled_delta={k:models['best_dice']['pooled'][k]-reference['pooled'][k] for k in reference['pooled']},
            wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
            paired_image_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
            interval_caveat='Exploratory selected development images; not patient-level or training-seed uncertainty')
        path=old/'results/seed42'/name/'best_dice/probabilities.npy'
        if path.exists():
            assert sha(path)==reference['probability_sha256']
            archived_ids=json.loads((path.parent/'ids.json').read_text());assert archived_ids==ids
            probabilities=np.load(path,mmap_mode='r');assert probabilities.shape==newprob.shape
            reference_probs[name]=probabilities
            corrected=[];introduced=[];codes=np.lib.format.open_memmap(out/f'error_categories_{name}.npy',mode='w+',dtype=np.uint8,shape=gt.shape)
            for i in range(808):
                old_error=(probabilities[i]>=.5)!=gt[i];new_error=newpred[i]!=gt[i]
                corrected.append(int((old_error&~new_error).sum()));introduced.append(int((~old_error&new_error).sum()))
                codes[i]=old_error.astype(np.uint8)*2+new_error.astype(np.uint8)
                assert int(old_error.sum())==old_rows[i]['models'][name]['fp']+old_rows[i]['models'][name]['fn']
            codes.flush();del codes
            comparison['pixel_errors']=dict(reference_wrong=sum(r['models'][name]['fp']+r['models'][name]['fn'] for r in old_rows),
                corrected=sum(corrected),introduced=sum(introduced),net_errors_reduced=sum(corrected)-sum(introduced),
                code_order=['both_correct','new_wrong_reference_correct','new_correct_reference_wrong','both_wrong'])
        else:
            comparison['pixel_errors']=None;comparison['pixel_comparison_unavailable_reason']='Historical probability array unavailable; no historical inference rerun'
        comparisons[name]=comparison
        for i,row in enumerate(paired_rows):
            row[name+'_dice']=float(old_dice[i]);row[name+'_dice_delta']=float(delta[i])
            if comparison['pixel_errors'] is not None:row[name+'_corrected_pixels']=corrected[i];row[name+'_introduced_pixels']=introduced[i]
    csv_rows(out/'historical_comparison_per_image.csv',paired_rows)
    area_groups={}
    for label,fn in [('lt_0.1',lambda x:x<.1),('0.1_to_0.3',lambda x:.1<=x<.3),('ge_0.3',lambda x:x>=.3)]:
        ix=[i for i,r in enumerate(rows) if fn(r['foreground_fraction'])]
        area_groups[label]=dict(n=len(ix),mambavision_macro={k:float(np.mean([rows[i][k] for i in ix])) for k in ('dice','iou','boundary_f1','hd95_pixels')},
            historical={name:{k:float(np.mean([old_rows[i]['models'][name][k] for i in ix])) for k in ('dice','iou','boundary_f1','hd95_pixels')} for name in baseline_names})
    order=np.array([r['mambavision_dice']-r['dinov3_swin_dice'] for r in paired_rows])
    cases=[]
    for group,indices in [('largest_improvement_vs_dino',np.argsort(-order)[:6]),('largest_regression_vs_dino',np.argsort(order)[:6]),
        ('lowest_mambavision_dice',np.argsort([r['dice'] for r in rows])[:6])]:
        for i in indices:cases.append(dict(group=group,index=int(i),id=ids[i],mambavision_dice=rows[i]['dice'],delta_vs_dino=float(order[i])))
    examples=out/'examples';examples.mkdir(exist_ok=True)
    image_paths={p.stem:p for p in (PROJECT/'data/isic2018/val/images').iterdir() if p.is_file()}
    for case in cases:
        i=case['index'];sid=ids[i]
        panel=Image.new('RGB',(256*5,288),'white');draw=ImageDraw.Draw(panel)
        panel.paste(Image.open(image_paths[sid]).convert('RGB').resize((256,256),Image.Resampling.BILINEAR),(0,32))
        panel.paste(Image.fromarray((gt[i]*255).astype(np.uint8)).convert('RGB'),(256,32))
        panel.paste(Image.fromarray((newpred[i]*255).astype(np.uint8)).convert('RGB'),(512,32))
        for col,name in enumerate(baseline_names,3):
            if name in reference_probs:panel.paste(Image.fromarray(((reference_probs[name][i]>=.5)*255).astype(np.uint8)).convert('RGB'),(256*col,32))
        for col,title in enumerate((f'Image {sid}','Ground truth',f'MambaVision Dice {rows[i]["dice"]:.3f}','DINOv3 best_dice','VMamba best_dice')):
            draw.text((col*256+5,8),title,fill='black')
        panel.save(examples/f'{case["group"]}_{sid}.png')
    dump(out/'cases.json',cases);plot(history,rows,models['best_dice']['selected_epoch'])
    inference=json.loads((out/'inference.json').read_text());pf=json.loads((ROOT/'preflight.json').read_text())
    summary=dict(seed=42,epochs=300,optimizer_updates=9000,encoder='NVIDIA MambaVision-S-1K',decoder='Historical SwinPixelDecoder',
        dataset='ISIC2018 legacy 1886 train / 808 development validation',evaluation_resolution='256x256',
        primary_criterion='best_dice',models=models,parameters=config['parameters'],pretrained=config['pretrained'],environment=config['environment'],
        training=dict(active_seconds=done['active_training_seconds'],epoch_seconds_sum=done['epoch_seconds_sum'],
            mean_epoch_seconds=float(np.mean([r['seconds'] for r in history])),median_epoch_seconds=float(np.median([r['seconds'] for r in history])),
            max_training_peak_gpu_gib=max(r['peak_gpu_gib'] for r in history),resumed=done['resumed'],encoder_unchanged=done['encoder_unchanged_including_bn_buffers']),
        inference=inference,preflight_memory=pf['memory'],comparisons=comparisons,lesion_area_groups=area_groups,
        caveat='Single seed, selected development validation, unverified patient independence, different pretrained architectures/sizes/data. Not official challenge test or equal-capacity mechanism ablation.',
        artifacts=dict(history='results/history.json',per_image='analysis/per_image.csv',config='config.json',learning_curves='analysis/learning_curves.png',
            local_only_cases='analysis/examples/',dense_outputs='results/{best_dice,best_loss,last}/',full_resume='results/latest.pth'))
    dump(out/'summary.json',summary)
    p=config['parameters'];best=models['best_dice'];dino=comparisons['dinov3_swin'];vmamba=comparisons['vmamba_swin']
    lines=['# MambaVision-S 冻结 Encoder：ISIC2018 seed42 实验报告','',
        '## 目的与结论范围','',
        '仅替换预训练视觉 Encoder，复用历史 Adapter、Swin Pixel Decoder、数据处理、损失和评价。单模型、单 seed42、300 完整 epochs / 9000 updates；1886/808 开发划分，非官方独立测试成绩。',
        '',f'主权重 best_dice 选中第 **{best["selected_epoch"]}** 轮；验证 macro Dice **{best["macro"]["dice"]*100:.4f}%**，pooled Dice **{best["pooled"]["dice"]*100:.4f}%**。',
        f'相对 DINOv3+Swin 的 macro Dice 差为 **{dino["macro_delta"]["dice"]*100:+.4f} 个百分点**；相对 VMamba+Swin 为 **{vmamba["macro_delta"]["dice"]*100:+.4f} 个百分点**。',
        '', '## 官方来源与权重验收','',
        f'- 模型：https://huggingface.co/nvidia/MambaVision-S-1K ，revision `{HF_REVISION}`。',
        f'- 官方仓库：https://github.com/NVlabs/MambaVision ，检查 commit `{GITHUB_REVISION}`；实际执行固定 HF 官方特征接口，代码未修改。',
        f'- 权重 SHA256 `{WEIGHT_SHA256}`，与固定 revision 的远端 LFS 身份匹配。',
        '- AutoModel loading_info 的 missing/unexpected/mismatched/error 均为空；进一步 strict load 并逐张量核对所有参数与缓冲区。',
        '- 官方架构 depths=[3,3,7,5]、dim96，四级为 [B,96,64,64]、[B,192,32,32]、[B,384,16,16]、[B,768,8,8]。',
        '- 递归 eval、requires_grad=False、no_grad；训练前后完整 Encoder state_dict 摘要一致，包含 BN 状态。未运行预训练分类头，保留并计入完整 Encoder 参数。',
        '- 独立 smoke 完成两轮、通过新进程恢复与 optimizer/scheduler/Python/NumPy/CPU/CUDA RNG 核验，未载入正式训练。',
        '', '## 架构与参数','',
        '原始 MambaVision 四级特征 → 历史 Conv1×1(bias=False)+GroupNorm8+GELU Adapter → 64通道金字塔 → 历史 SwinPixelDecoder → 1通道256×256 logits。每尺度2块、window8、shift0/4、heads4、MLP4，无新模块或额外loss。',
        '', '| 部分 | 参数量 |','|---|---:|',f'| 完整官方 Encoder（冻结） | {p["encoder"]} |',f'| Adapter | {p["adapter"]} |',
        f'| Swin Decoder（含输出层） | {p["decoder_including_head"]} |',f'| 其中输出层（上项已包含） | {p["segmentation_head"]} |',
        f'| 可训练总计 | {p["trainable"]} |',f'| 完整模型 | {p["total"]} |',
        '', '## 数据、环境与训练设置','',
        '数据清单与 stage1 原始全部图像/Mask/解码 SHA 逐项一致。1886训练、808验证，无过滤；原 ImageData 的 RGB bilinear resize、Mask nearest、ImageNet normalization 与 zero-padding 几何增强原样复用。H/V/rotation各p0.5；角度均匀0–360°；每轮42+epoch×100003控制顺序与几何。',
        '', 'AdamW lr0.001、wd0.01、betas(.9,.999)、eps1e-8；batch64，尾批30，无梯度累积，30更新/epoch；300轮无early stopping；Cosine T_max300、eta_min1e-5、无warmup，轮末step；BCE+逐图SoftDice(smooth1)，BF16 autocast/FP32 loss，clip1。',
        '', '实际环境：`'+json.dumps(config['environment'],ensure_ascii=False)+'`。依赖仅安装在本阶段 runtime，保留历史代码与环境。',
        '', '## 三种预先固定的 Checkpoint','',
        '| Selector | Epoch | Macro Dice | Pooled Dice | Macro IoU | Pooled IoU | BF1 | HD95 px |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for criterion,m in models.items():
        lines.append(f'| {criterion} | {m["selected_epoch"]} | {m["macro"]["dice"]:.6f} | {m["pooled"]["dice"]:.6f} | {m["macro"]["iou"]:.6f} | {m["pooled"]["iou"]:.6f} | {m["macro"]["boundary_f1"]:.6f} | {m["macro"]["hd95_pixels"]:.6f} |')
    lines += ['', '主表只用 best_dice，不按各指标重新挑轮。best_loss 按最低样本加权验证loss，last固定第300轮，精确并列取早。三种权重、恢复latest与逐像素输出均保留本地。',
        '', '## 主权重完整指标','', '| 指标 | 数值 |','|---|---:|']
    for group in ('macro','pooled'):
        for key,value in best[group].items():lines.append(f'| {group} {key} | {value:.9f} |')
    for threshold,value in best['prec_at'].items():lines.append(f'| Prec@IoU {threshold} | {value:.9f} |')
    lines += [f'| ISIC Threshold Jaccard 0.65 | {best["threshold_jaccard_065"]:.9f} |',
        '', '阈值probability>=0.5，256网格；所有空Mask规则、对角线0.5%边界容差及HD95定义直接复用 evaluation.py / stage1_visual.analyze.metrics。808张图TP/FP/FN/TN及Dice/IoU独立重算通过；完整逐图指标见per_image.csv。',
        '', '## 与同条件历史基线比较','',
        '| Model | Macro Dice | Pooled Dice | Macro IoU | Pooled IoU | BF1 | HD95 px |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('DINOv3-S/16 + Swin',dino['reference']),('VMamba-Tiny + Swin',vmamba['reference']),('MambaVision-S + Swin',best)]:
        lines.append(f'| {name} | {m["macro"]["dice"]:.6f} | {m["pooled"]["dice"]:.6f} | {m["macro"]["iou"]:.6f} | {m["pooled"]["iou"]:.6f} | {m["macro"]["boundary_f1"]:.6f} | {m["macro"]["hd95_pixels"]:.6f} |')
    for name,c in comparisons.items():
        lines+=['',f'{name}：改善/退步/并列图数 {c["wins"]}/{c["losses"]}/{c["ties"]}；macro Dice差 {c["macro_delta"]["dice"]*100:+.4f} 个百分点，图像配对95%区间 {[round(v*100,4) for v in c["paired_image_bootstrap95"]]} 个百分点。']
        if c['pixel_errors'] is not None:
            e=c['pixel_errors'];lines.append(f'核验历史概率SHA后统计：纠正 {e["corrected"]} 像素、新增错误 {e["introduced"]} 像素、净减少错误 {e["net_errors_reduced"]} 像素。')
        else:lines.append(c['pixel_comparison_unavailable_reason'])
    tuning_judgment = ('冻结基线的 macro Dice 已超过两个历史单分支，可把部分解冻列为下一轮候选；仍需结合边界和效率，先固定独立验证设计。' if dino['macro_delta']['dice']>0 and vmamba['macro_delta']['dice']>0 else '冻结基线尚未同时超过两个历史单分支，不建议仅凭模型规模就优先投入全量微调。若后续继续，先围绕失败面积组提出一个受控的部分解冻诊断。')
    lines += ['', '不重跑历史模型，不比较不同Decoder或ISIC2016。预训练规模、数据、参数量均不同，本轮不作等容量或架构机制归因。',
        '', '## 病灶面积与失败病例','', '| GT前景比例 | n | MambaVision Dice | DINO Dice | VMamba Dice | MambaVision BF1 |', '|---|---:|---:|---:|---:|---:|']
    for label,g in area_groups.items():lines.append(f'| {label} | {g["n"]} | {g["mambavision_macro"]["dice"]:.6f} | {g["historical"]["dinov3_swin"]["dice"]:.6f} | {g["historical"]["vmamba_swin"]["dice"]:.6f} | {g["mambavision_macro"]["boundary_f1"]:.6f} |')
    lines += ['', '面积组预先固定为<10%、10–30%、>=30%，仅作描述。cases.json 保留相对DINO最大改善6例、最大退步6例、MambaVision最低Dice6例；同一图可出现在多个选择组。病例图片位于本地 examples/，不自动公开。不据无人工标签的误差图归因于毛发、反光或某种扫描机制。',
        '', '## 效率与耗时','',
        f'完整 uncached batch1 推理平均 **{inference["mean_ms"]:.3f} ms**，中位 {inference["median_ms"]:.3f} ms；BF16，预热3/重复10、逐次CUDA同步，包含Encoder+Adapter+Swin+输出层，不含I/O及CPU预处理/后处理。独立进程峰值 allocated {inference["peak_allocated_gib"]:.3f} GiB，增量 {inference["incremental_peak_gib"]:.3f} GiB。',
        f'训练平均 {summary["training"]["mean_epoch_seconds"]:.3f} 秒/epoch，峰值 {summary["training"]["max_training_peak_gpu_gib"]:.3f} GiB；完整正式训练进程活动耗时 {done["active_training_seconds"]:.3f} 秒（{done["active_training_seconds"]/60:.2f}分钟，含初始化/验证缓存/断点写入，不含预检和最终分析）。',
        'Encoder、Decoder预检各自峰值与单批时间见preflight.json；无完整Selective Scan FLOPs支持，因此未将局部FLOPs当作完整模型FLOPs。历史速度来自历史设备/会话，仅背景参考。',
        '', '## 学习曲线与局限','', '![学习曲线](learning_curves.png)','', '![逐图分布与面积组](per_image_distributions.png)','',
        '仅seed42，选模与分析均使用既有开发验证集；无患者级独立性认证和新外部测试。图像bootstrap不能估计患者相关性、训练随机性或消除开发选模偏差。BF16及CUDA算子不承诺逐位确定性。',
        '', '## 是否值得微调','',
        tuning_judgment+'本轮不执行微调。即使冻结分数较低，也不能直接归因于Mamba扫描或Attention结构；后续另立协议并经用户确认，不把本轮改为全量微调。',
        '', '数据和源码核验、300轮/9000更新、冻结状态、三选模规则与报告一致性由独立validate.py最终检查；只有全部通过，监督入口才写complete。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(out/'analysis_validation.json',dict(passed=True,comparison_source='stage4 best_dice only',
        no_historical_encoder_rerun=True,historical_probability_hashes_checked=list(reference_probs),
        numerical_figures_written=True,balanced_local_cases_written=True,area_group_total=sum(g['n'] for g in area_groups.values())))
    print('ANALYSIS_AND_REPORT_WRITTEN',flush=True)

if __name__=='__main__':main()
