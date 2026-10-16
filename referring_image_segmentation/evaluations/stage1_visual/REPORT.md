# 第一阶段补充完整指标

ISIC2018 旧开发划分，1886 train / 808 validation，256×256，seed 42。非独立测试。所有概率阈值固定 0.5。

## 区域指标

| Model | Mean Dice | Pooled Dice | mIoU | oIoU | Mean Precision | Mean Recall | Pooled Precision | Pooled Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov3_light | 0.882189 | 0.896165 | 0.805338 | 0.811865 | 0.883452 | 0.915594 | 0.889439 | 0.902993 |
| vmamba_light | 0.875560 | 0.879327 | 0.796599 | 0.784641 | 0.887311 | 0.901777 | 0.869022 | 0.889879 |
| mean_light | 0.887011 | 0.896047 | 0.812423 | 0.811671 | 0.893946 | 0.913101 | 0.891223 | 0.900923 |

## 样本成功率与边界

| Model | P@.5 | P@.6 | P@.7 | P@.8 | P@.9 | Threshold Jaccard | Boundary F1 | HD95 px | HD95 / diagonal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov3_light | 0.950495 | 0.909653 | 0.847772 | 0.669554 | 0.247525 | 0.756313 | 0.276200 | 16.092272 | 0.044449 |
| vmamba_light | 0.936881 | 0.893564 | 0.811881 | 0.626238 | 0.274752 | 0.731094 | 0.303801 | 18.065202 | 0.049899 |
| mean_light | 0.955446 | 0.919554 | 0.855198 | 0.688119 | 0.273515 | 0.761619 | 0.299603 | 15.689983 | 0.043338 |

Prec@X 使用 IoU >= X。Threshold Jaccard 在 IoU < .65 时记零。当前划分/分辨率与官方挑战不同。

## 逐图配对差

| Comparison | Mean Dice delta | Exploratory 95% interval | Better / worse / ties |
|---|---:|---|---|
| dinov3_light_minus_vmamba_light | 0.006629 | [0.001276, 0.011876] | 409 / 399 / 0 |
| mean_light_minus_dinov3_light | 0.004823 | [0.001990, 0.007707] | 492 / 316 / 0 |

2000 次逐图配对 bootstrap 仅作开发集探索性分析；不能推断跨 seed 稳定性或患者级独立泛化。

## 模型与选模

| Model | Best epoch by validation loss | Trainable parameters | Total parameters |
|---|---:|---:|---:|
| dinov3_light | 12 | 246849 | 21833793 |
| vmamba_light | 20 | 240705 | 29719233 |

编码器来源不同，跨编码器结果不能全部归因于架构。
第一阶段全部原有 macro/pooled/边界结果重算与历史值一致（误差 < 1e-12）。
逐图记录见 per_image.json，机器可读汇总见 summary.json/summary.csv，误差图类别见 errors_*.npz。
