# RefCOCO UNC：统一文本匹配基线（流程预检，不能作为正式结果）

Seed42；按原图逐表达validation mean IoU选模；testA/testB在四组选择锁定后评分。固定文本为独立从头训练的DINO对照。

| Model | Split | Mean IoU | Overall IoU | Mean Dice | Pooled Dice | Boundary F1 | HD95 px |
|---|---|---:|---:|---:|---:|---:|---:|
| dinov3_text | val | 0.101485 | 0.120222 | 0.170222 | 0.214640 | 0.038279 | 347.412689 |
| vmamba_text | val | 0.076810 | 0.103956 | 0.132627 | 0.188334 | 0.061015 | 307.139650 |
| fusion_text | val | 0.083025 | 0.111704 | 0.142983 | 0.200961 | 0.039987 | 352.571426 |
| dinov3_fixed_text | val | 0.019279 | 0.018709 | 0.033670 | 0.036730 | 0.009177 | 440.761072 |

全部Precision/Recall、Prec@X、HD95归一化、image-balanced mIoU见all_runs.csv和summary.json。
配对区间按图像聚类；同图表达/目标切换是诊断，模型不是在所有情况下都能正确选择目标。所有源码/数据/权重来源见冻结协议和provenance。
