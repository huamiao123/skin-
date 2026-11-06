"""Post-training interpretation only; no training, selection or metric changes."""
from common import *

def main():
    out=ROOT/'analysis';summary=json.loads((out/'summary.json').read_text())
    history=json.loads((ROOT/'results/history.json').read_text())
    rows=json.loads((out/'per_image_best_dice.json').read_text())
    comparisons=summary['comparisons'];best=summary['models']['best_dice']
    dino=comparisons['dinov3_swin'];vm=comparisons['vmamba_swin']
    groups=summary['lesion_area_groups'];deltas={}
    for group,value in groups.items():
        deltas[group]={name:(value['mambavision_macro']['dice']-value['historical'][name]['dice'])*100 for name in ('dinov3_swin','vmamba_swin')}
    record=dict(post_training_only=True,no_new_experiment=True,no_checkpoint_reselection=True,
        selected_epoch=best['selected_epoch'],empty_predictions=sum(r['tp']+r['fp']==0 for r in rows),
        zero_dice_samples=sum(r['dice']==0 for r in rows),dice_below_0_8=sum(r['dice']<.8 for r in rows),
        lesion_group_dice_deltas_percentage_points=deltas,
        fp_delta_vs_dino=best['counts']['fp']-dino['reference']['counts']['fp'],
        fn_delta_vs_dino=best['counts']['fn']-dino['reference']['counts']['fn'],
        loss=dict(first_train=history[0]['train_loss'],last_train=history[-1]['train_loss'],
            minimum_val=min(r['val']['loss'] for r in history),minimum_val_epoch=min(history,key=lambda r:r['val']['loss'])['epoch'],last_val=history[-1]['val']['loss']))
    dump(out/'interpretation.json',record)
    report=out/'REPORT.md';text=report.read_text();marker='\n## 完成后的结果解读（不改变训练或选模）\n'
    if marker in text:text=text.split(marker)[0]
    lines=[marker,
        '总体结果：本次冻结 MambaVision-S 没有优于 DINOv3；与 VMamba 的 macro Dice 差较小，图像配对区间跨0，不把它描述成已证实的稳定差距。三个模型的预训练来源和总容量不同，不能据此归因于扫描或Attention。',
        '',
        f'相对 DINO 的分组差为：GT<10% **{deltas["lt_0.1"]["dinov3_swin"]:+.3f}**、GT10–30% **{deltas["0.1_to_0.3"]["dinov3_swin"]:+.3f}**、GT>=30% **{deltas["ge_0.3"]["dinov3_swin"]:+.3f} 个百分点**。因此平均下降并非所有面积组都下降；本轮较明显的短板在中、大面积组。这些是既定开发集描述，未认证病灶尺寸的因果效应。',
        '',
        f'相对 DINO，新增 FN 为 **{record["fn_delta_vs_dino"]}**，FP 差为 **{record["fp_delta_vs_dino"]}**，总错误像素增加264929；漏检增加是本次计数层面主要差异。Boundary F1比DINO高0.553个百分点，但HD95高1.920像素，说明两类边界指标没有一致改善。不能据单一Boundary F1声称边界全面更好。',
        '',
        '相对 VMamba，MambaVision 净减少20365个错误像素，却有略低的macro/pooled Dice。总错误数和前景归一化比值不同，逐图平均还给予每图相同权重，这种排序差异不代表指标计算错误；独立TP/FP/FN检查已通过。',
        '',
        f'失败分布：808张中 **{record["dice_below_0_8"]}** 张Dice<0.8，**{record["empty_predictions"]}** 张预测为空且Dice=0。编号193整灶漏检，DINO同图Dice为0.8319；编号275的MambaVision Dice为0.1931，而DINO约0.9769。相反，编号694相对DINO改善0.4844，但MambaVision本身Dice也只有0.7506，仍有误差。所有这些病例已在本地画册中保留，不只展示成功例。可见毛发等现象没有被当成经标注验证的失败原因。',
        '',
        f'学习曲线显示训练loss由{record["loss"]["first_train"]:.4f}降至{record["loss"]["last_train"]:.4f}，验证loss最低在第{record["loss"]["minimum_val_epoch"]}轮，为{record["loss"]["minimum_val"]:.4f}，第300轮为{record["loss"]["last_val"]:.4f}；有过拟合迹象。主权重仍严格按最高macro Dice第159轮，不能事后改用最低loss或不同指标的最佳轮。',
        '',
        '可能的解释仅作为待验证假设：监督ImageNet-1K预训练与DINOv3的预训练任务/数据规模不同；冻结表征对病灶目标的适配程度可能不同；相同轻量Decoder与不同Encoder特征的匹配程度也可能不同。当前只有Encoder对照，无法分离这些因素，更不能证明四方向扫描就能消除差距。',
        '',
        '关于下一步SS2D：这是用户在本轮训练期间提出的候选，未实施或追加训练。VMamba的SS2D使用四条扫描路线（https://github.com/MzeroMiko/VMamba）；MambaVision后两级包含窗口内的MambaVisionMixer及Attention（https://github.com/NVlabs/MambaVision）。应区分“只修改SSM扫描路线”与“整块替换为完整SS2D”：前者可考虑共享兼容权重并保留窗口/非SSM分支/Attention以减少变量；后者可能新增不兼容参数，必须核验权重迁移，不能将随机新增模块冻结后当作完整官方预训练。若需要微调，对照原模型也须使用相同微调范围；不能把扫描与解冻两项收益混为一谈。无论选择哪种，均另立协议，本轮保持原始官方基线。',
        '']
    report.write_text(text+'\n'.join(lines))
    dump(out/'interpretation_provenance.json',dict(post_training_only=True,script=str(ROOT/'interpret_results.py'),
        script_sha256=sha(ROOT/'interpret_results.py'),input_summary_sha256=sha(out/'summary.json'),
        input_history_sha256=sha(ROOT/'results/history.json'),report_sha256=sha(report),
        original_training_and_analysis_sources_unchanged=True))
    print('POST_TRAINING_INTERPRETATION_WRITTEN',flush=True)

if __name__=='__main__':main()
