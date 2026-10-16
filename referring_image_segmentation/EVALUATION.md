# 统一评估协议（2026-10-16）

用户要求后续每轮同时报告逐样本平均和整体统计。统一实现为 `evaluation.py`。

- 对当前 256×256 网格，概率 >= 0.5 为前景，灰度 GT >= 128 为前景。
- 每张图 Dice = 2TP/(2TP+FP+FN)，IoU = TP/(TP+FP+FN)。macro 为所有样本等权平均；pooled 为先累加 TP/FP/FN 再计算。均只计算前景，不包含背景类别平均。
- 同时报告 macro/pooled 像素 Precision、Recall。
- Prec@X 为逐图 IoU >= X 的样本比例，X=0.5/0.6/0.7/0.8/0.9，与像素 Precision 区分。
- ISIC Threshold Jaccard：逐图 IoU < 0.65 记零，否则保留原值，然后逐图平均。当前使用旧开发划分和 256 网格，不能当作官方挑战测试分数。
- Boundary F1/HD95 完全沿用 `stage1_visual/analyze.py:metrics`：侵蚀边界、容差为对角线 0.5%，HD95 为双向边界距离合并后的 95 分位数。报告逐图平均；HD95 同时保留像素和对角线归一化。
- 双方空 mask：Dice/IoU/Precision/Recall/Boundary F1=1，HD95=0；预测空且 GT 非空：Precision=0；GT 空时 Recall=1；仅一方空时 Boundary F1=0、HD95=对角线。
- 每轮保存 ids、概率图、二值 mask、逐图指标、汇总表、参数量与运行配置。训练日志每 epoch 记录区域指标，边界指标在最佳权重最终导出后计算。
- Decoder 对照沿用验证 loss 最低选 checkpoint，完全相同取较早 epoch；同时报告全部指标，不按不同指标挑选不同 checkpoint。
- 新一轮主要分析 macro Dice；macro IoU、pooled Dice/IoU 与边界指标一起报告。阶段一结果是已有探索性证据，补指标不改变其权重或选模记录。
- 当前只有 seed 42；逐图配对 bootstrap 区间只描述该开发集上的探索性样本不确定性，不表示跨 seed、患者级或独立测试结论。
- 后续 RIS 按基准规定的图像—表达样本计算 mIoU/oIoU 和 Prec@X，使用该基准指定的分辨率与空目标规则，并在运行前记录差异。

参考：
- [ISIC2018 官方指标](https://challenge.isic-archive.com/landing/2018/45/)
- [Latent-VG ICCV2025 指标](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Latent_Expression_Generation_for_Referring_Image_Segmentation_and_Grounding_ICCV_2025_paper.pdf)
