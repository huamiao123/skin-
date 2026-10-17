# 固定权重门控诊断

使用dynamic主checkpoint，不重训、不重新选模。零门控使用当前共同训练后的DINO路径，不能当作原始DINO baseline。

| Gate | Mean Dice | Pooled Dice | Boundary F1 | HD95 px | 净纠错像素 |
|---|---:|---:|---:|---:|---:|
| learned | 0.894130 | 0.901826 | 0.357169 | 15.076 | 71661 |
| zero | 0.854202 | 0.881049 | 0.198008 | 17.400 | -651014 |
| spatial_mean | 0.889218 | 0.899341 | 0.314900 | 15.222 | -65898 |
| shift | 0.885804 | 0.895748 | 0.299955 | 15.576 | -175482 |

全部统一指标及每尺度gate分布见summary.json；逐图门控均值/空间标准差见per_image_gate_statistics.json。
