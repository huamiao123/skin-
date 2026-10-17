# 固定权重门控诊断

使用dynamic主checkpoint，不重训、不重新选模。零门控使用当前共同训练后的DINO路径，不能当作原始DINO baseline。

| Gate | Mean Dice | Pooled Dice | Boundary F1 | HD95 px | 净纠错像素 |
|---|---:|---:|---:|---:|---:|
| learned | 0.860283 | 0.870413 | 0.236342 | 22.283 | -955804 |
| zero | 0.850217 | 0.866055 | 0.231569 | 22.244 | -1065339 |
| spatial_mean | 0.863076 | 0.870734 | 0.252662 | 21.060 | -936802 |
| shift | 0.863524 | 0.870711 | 0.256136 | 21.033 | -931885 |

全部统一指标及每尺度gate分布见summary.json；逐图门控均值/空间标准差见per_image_gate_statistics.json。
