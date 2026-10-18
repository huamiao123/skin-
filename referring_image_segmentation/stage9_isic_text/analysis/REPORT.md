# ISIC2016 医学文本引导分割：seed 42

独立于已停止的 RefCOCO 实验。三视觉分支均冻结，P3 仅颜色/形状；类别词统一为 skin lesion。
本次为去重后的作者划分，图像级无精确重复，不宣称患者级独立。最佳权重仅由验证集原尺寸 macro Dice 选择。

| Model | Epoch | Test macro Dice | Test pooled Dice | Test macro IoU | Boundary F1 | HD95 px |
|---|---:|---:|---:|---:|---:|---:|
| dinov3_text | 31 | 0.901201 | 0.913811 | 0.833418 | 0.326793 | 77.462 |
| vmamba_text | 62 | 0.906703 | 0.916644 | 0.839071 | 0.347010 | 65.116 |
| fusion_text | 23 | 0.904147 | 0.920557 | 0.839633 | 0.353156 | 71.173 |
| dinov3_fixed_text | 30 | 0.901800 | 0.915731 | 0.834555 | 0.334443 | 75.645 |

固定文本训练对照只与 DINOv3 架构匹配；三主组的临时固定/错配文本结果是诊断，不等同于额外训练的对照。
完整 macro/pooled Dice/IoU/Precision/Recall、Prec@0.5/0.6/0.7/0.8/0.9、threshold Jaccard、BF1、HD95 及配对差值见 summary.json。
单 seed 不能估计训练随机性；图像 bootstrap 区间也不代表患者级泛化。
