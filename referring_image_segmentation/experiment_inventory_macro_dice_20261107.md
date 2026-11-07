# 实验汇总：逐图平均 Dice

范围为当前 referring_image_segmentation 项目中可核验的正式结果；不包括 smoke、未完成训练或使用真值的 oracle。每张图先计算 Dice，再等权求平均。
ISIC2018 使用旧划分 1886 训练/808 开发验证，256×256；验证集参与选模，不是独立测试。种子、选模与数据集不同的表格不作直接排名。

## MambaVision-S 系列：ISIC2018，seed42

| 阶段 | 实验 | 最佳轮 | 逐图平均 Dice |
|---|---|---:|---:|
| 14 | 原始 MambaVision-S + Swin | 159 | 88.360% |
| 15 | MambaVision 四方向 SSM 扫描 + Swin | 116 | 88.371% |
| 17 | MambaVision + Swin，HiRAM 轮配对基线 | 123 | 88.440% |
| 17 | 32×32 跳接前：仅区域监督 | 155 | 88.561% |
| 17 | 32×32 跳接前：HiRAM 适配 | 164 | 88.511% |
| 18 | 32×32 Swin 后：仅区域监督 | 176 | 88.204% |
| 18 | 32×32 Swin 后：HiRAM 适配 | 55 | 88.547% |
| 19 | Encoder 内原顺序 + 区域先验 | 39 | 88.250% |
| 19 | Encoder 内区域分组扫描 + 区域先验 | 86 | 88.551% |
| 20 | MambaVision + Swin，AXIS 轮配对基线 | 127 | 88.338% |
| 20 | 32×32 跳接：简单门控 | 32 | 88.153% |
| 20 | 32×32 跳接：AXIS-Bridge | 159 | 88.440% |
| 20 | MambaVision + SegMAN Decoder 适配版 | 105 | 88.094% |
| 21 | MambaVision + 部分 Swin + HGM BMD 适配版 | 116 | 88.239% |

四方向实验只替换 SSM 扫描与合并，不是完整 VMamba SS2D 替换；HiRAM/SegMAN/HGM 均为适配实验。阶段 17、20 为各自重新训练的配对基线；阶段 21 使用阶段 20 的配对基线。

## DINOv3、VMamba 和双编码器：ISIC2018，按最高验证逐图 Dice 选模

| 模型 | seed42 | seed43 | seed44 | seed45 |
|---|---:|---:|---:|---:|
| DINOv3 + Light | 88.695% | 88.611% | 88.583% | 88.700% |
| DINOv3 + Swin | 89.185% | 89.129% | 89.251% | 88.987% |
| VMamba + Light | 87.872% | — | — | — |
| VMamba + Swin | 88.516% | — | — | — |
| DINOv3 + VMamba + Light（拼接） | 88.911% | — | — | — |
| DINOv3 + VMamba + Swin（拼接） | 89.549% | 89.441% | 89.424% | 89.334% |
| DINOv3 双 Adapter 容量对照 + Swin | 89.272% | 89.179% | 89.337% | 89.172% |

seed42 来自第四/五轮，43–45 来自第三/六轮；破折号代表未运行该组合。

| 其他 seed42 实验 | 逐图平均 Dice |
|---|---:|
| Light，两单模型概率平均（非新训练） | 88.978% |
| Swin，两单模型概率平均（非新训练） | 89.449% |
| 双 Encoder 静态残差 | 89.483% |
| 双 Encoder 动态门控 | 89.413% |
| 双 Encoder 局部卷积对照 | 89.451% |
| 双 Encoder 窗口 Cross-Attention | 89.374% |
| 双 Encoder + Swin，HiRAM 轮重新训练基线（第 38 轮） | 89.611% |
| 双 Encoder，仅区域监督（第 132 轮） | 89.215% |
| 双 Encoder + HiRAM 适配（第 68 轮） | 89.385% |

阶段16使用 DINOv3 + VMamba 双编码器，不是 MambaVision；其基线为重新训练，不能当成第四轮同一次运行。

## 早期相同数据的视觉对照：按最低验证 Loss 选模，seed42

| 实验 | 逐图平均 Dice |
|---|---:|
| DINOv3 + Light | 88.219% |
| VMamba + Light | 87.556% |
| DINOv3 + Swin | 88.358% |
| VMamba + Swin | 87.574% |
| Light，两单模型概率平均（非新训练） | 88.701% |
| Swin，两单模型概率平均（非新训练） | 88.775% |

以上为阶段1/2旧的 best-loss 口径，不与 best-dice 口径混为一次相同运行。

## ISIC2016 文本实验：seed42，379 张测试图逐图平均

| 模型 | 100轮 | 300轮 |
|---|---:|---:|
| DINOv3，无文本 | 90.306% | 90.248% |
| DINOv3，Decoder 后固定文本 | 90.180% | 90.771% |
| DINOv3，Decoder 后逐图属性文本 | 90.120% | 90.433% |
| DINOv3，Decoder 前固定文本 | 90.368% | 90.311% |
| DINOv3，Decoder 前逐图属性文本 | 90.310% | 90.415% |
| VMamba + 逐图属性文本 | 90.670% | — |
| DINOv3 + VMamba + 逐图属性文本 | 90.415% | — |

100轮与300轮分别训练，不是续训。ISIC2016为 806/90/379，原尺寸测试结果；测试集此前已用于开发，不是全新外部验证，不能和 ISIC2018 分数直接排名。

## MosMedData+：seed42，216 张验证图逐图平均

| 实验 | 最佳轮 | 逐图平均 Dice |
|---|---:|---:|
| DINOv3 + Swin，无文本 | 45 | 47.570% |
| DINOv3 + Swin，固定文本 | 32 | 45.467% |
| DINOv3 + Swin，逐图医学文本 | 36 | 44.693% |

该数据为肺部感染病灶；seed43按要求中止、seed44未启动，正式测试集评估未完成。

## 不当作正式结果的尝试

- RefCOCO 图文训练按要求停止，未完成正式训练和测试。
- HGM 已完成部分 Swin + BMD 适配训练，但没有复现完整 HGM；MCADS、Glasses 未完成本项目正式训练。
- 旧 work 中 EGE-UNet、U-Net、PTU、Swin 和 CNN+Transformer 等的历史主报告使用或混有不同统计口径；未逐项核验出可比的逐图 Dice，不将其数值冒充本表结果。

## 每张图的原始记录

- [MambaVision + AXIS 三组逐图 CSV](stage20_decoder_fusion_seed42/analysis/per_image.csv)
- [MambaVision + SegMAN 与 Swin 逐图 CSV](stage20_decoder_fusion_seed42/segman_adaptation/analysis/per_image.csv)
- [MambaVision + HGM BMD 与 Swin 逐图 CSV](stage21_hgm_bmd_seed42/analysis/per_image.csv)
- [MambaVision + HiRAM 逐图 CSV](stage17_mambavision_decoder_hiram_seed42/analysis/per_image.csv)
- [MambaVision 编码器区域扫描逐图 CSV](stage19_region_mambavision_encoder_seed42/analysis/per_image.csv)
- [DINOv3/VMamba 第四轮逐图 JSON](stage4_fusion/analysis/per_image_best_dice.json)
- [ISIC2016 的 300轮无文本模型逐图 JSON](stage13_isic_300/analysis/E0_visual/test_per_image.json)；其他模型记录位于对应模型目录。
