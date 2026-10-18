# ISIC2016 医学文本引导分割

用户于2026-10-18要求停止RefCOCO，试验小规模ISIC图文分割。本阶段仅seed42：DINOv3、VMamba、Concat双编码器三主组，统一冻结CLIP文本编码器、Swin Decoder和文本像素匹配头；增加同初值DINO固定文本训练对照。

四组100轮及379张测试集的最终评估已于2026-10-18 10:22 UTC全部完成。源文件/数据哈希检查通过，GPU已释放，训练和smoke恢复权重已清理，仅保留四组最佳权重。最终结果见 [REPORT.md](analysis/REPORT.md)、[完整指标JSON](analysis/summary.json)和[结果解释](analysis/FINDINGS.md)。

| 模型 | 最佳轮次 | 验证 macro Dice | 测试 macro Dice | 测试 pooled Dice |
|---|---:|---:|---:|---:|
| DINOv3 + 文本 | 31 | 92.476% | 90.120% | 91.381% |
| VMamba + 文本 | 62 | 92.226% | 90.670% | 91.664% |
| 双编码器 + 文本 | 23 | 93.325% | 90.415% | 92.056% |
| DINOv3 + 固定文本 | 30 | 92.520% | 90.180% | 91.573% |

验证集最好的融合模型没有获得最高测试macro Dice；固定文本DINO略高于真实文本DINO。三组错配文本后macro Dice仅下降约0.06–0.13个百分点。单seed配对差值的图像bootstrap区间均包含0，未证明双编码器或逐图属性文本有稳定增益。完整macro/pooled IoU、Precision/Recall、Prec@X及边界指标均已保存。

原作者文本标注810/90/379条，按test优先去除1条训练重复及3个训练/测试重叠ID，实际806训练/90验证/379测试；原图内容核查也通过。图像与Mask来自ISIC2016官方ZIP。文本采用P3颜色/形状，统一将类别词skin melanoma改为skin lesion；不使用Mask生成的大小、数量、位置提示。详见[协议](PROTOCOL.md)和[data_manifest.json](data_manifest.json)。

固定100epochs、batch16、AdamW1e-4、5epochs warmup后cosine到1e-6，视觉和文本编码器冻结。依据原图尺寸验证macro Dice选模；测试只在所有选模锁定后评估。使用通用指标定义、逐图记录、固定/错配文本诊断及四组完整GPU推理延迟。

运行 `python launch.py` 会依次完成预检、两轮中断恢复检查、正式训练、最终评价。状态见[run_status.json](run_status.json)，训练见 `results/seed42/status.json`、`train.log`；最终完整结果见 `analysis/REPORT.md` 与 `analysis/summary.json`。

本阶段只加载通用预训练Encoder，不加载旧ISIC或RefCOCO分割权重。此前ISIC2018与本数据跨年份可能有交集，因此本轮测试是本次训练的留出评估，不称整个研究流程的全新外部验证。单seed不能证明跨种子稳定性；不将模板化属性提示实验等同于复杂自然语言多目标RIS。

训练期间保留最佳权重与一个组合latest；训练和评价成功后仅保留四组最佳权重。源图像不重复复制，CLIP权重复用stage8资产，已校验解压的ZIP不长期保留。
