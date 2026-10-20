# ISIC2016：词级Cross-Attention在Swin Decoder之前融合

严格按用户要求，文本仅在Decoder之前使用：冻结DINO -> Adapter -> 词级Cross-Attention -> Swin -> 普通Mask Head。最后不再做图文点积匹配。

新训练同容量的固定文本与逐图属性文本两组，均seed42、100轮、batch16，806/90/379划分与原增强保持一致。Encoder冻结，公共视觉初值与历史纯视觉一致。两组新增模块同初值、独立参数和优化器。

运行 `python -u launch.py`。进度run_status.json/train.log；最终analysis/REPORT.md和summary.json。旧实验和数据只读复用；MosMed保持暂停。

## 已完成：seed42，两组各100轮

新固定文本组选中第21轮，新逐图文本组选中第25轮。原尺寸验证分数重放一致，10张新旧模型/划分的Dice表经逐图TP/FP/FN独立复算通过。

| 模型 | 测试macro Dice | 测试pooled Dice |
|---|---:|---:|
| predecoder_fixed_text | 90.368% | 91.884% |
| predecoder_image_text | 90.310% | 91.627% |
| E0_visual | 90.306% | 91.729% |
| E1_fixed_text | 90.180% | 91.573% |
| E2_image_text | 90.120% | 91.381% |

新逐图-新固定：-0.059个百分点，图像配对95%区间[-0.371, 0.261]个百分点。新逐图-旧晚期逐图：+0.190个百分点；区间跨0，单seed不证明稳定收益。

两种新方案均约比对应旧方案高0.19个百分点；逐图文本仍未优于同容量固定文本。相对无文本，新逐图的macro Dice几乎一致，但其他指标并不完全相同。全部结果不只以最高Dice概括。

保留2个新最佳权重，恢复及smoke权重已清理；旧文件没有改动，MosMed仍暂停。完整解释见analysis/FINDINGS.md，图表见analysis/learning_curves.png及fixed_id_prediction_panel.png。
