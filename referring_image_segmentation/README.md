# 图文指代分割项目

## 第四轮：双编码器特征融合（2026-10-17）

根据第一阶段的错误互补与概率平均结果，新增 `stage4_fusion/`：seed42 下重新训练 DINOv3、VMamba、DINOv3+VMamba × Light/Swin 六组。双编码器冻结，四尺度各自 Adapter 后拼接，用 1×1 卷积融合；全部统一按最高验证 mean Dice 选模，同时保存最低 loss 与最后一轮。完整设置见 `stage4_fusion/PROTOCOL.md`。

后台入口 `python stage4_fusion/launch.py` 先完成真实数据恢复/导出/分析预检，再执行300轮和最终完整分析。六组各300轮/9000次更新及18组checkpoint导出已完成，原有实验文件、数据和独立指标重算均通过校验。完整结果见 [第四轮报告](stage4_fusion/analysis/REPORT.md)、[全部指标表](stage4_fusion/analysis/all_runs.csv)和[机器可读汇总](stage4_fusion/analysis/summary.json)。

| Encoder / 融合方式 | Light Mean Dice | Light Pooled Dice | Swin Mean Dice | Swin Pooled Dice |
|---|---:|---:|---:|---:|
| DINOv3 | 88.695% | 89.467% | 89.185% | 90.113% |
| VMamba | 87.872% | 88.492% | 88.516% | 88.961% |
| DINOv3 + VMamba 特征拼接 | 88.911% | 89.530% | 89.549% | 90.381% |
| 两单模型概率平均 | 88.978% | 89.623% | 89.449% | 89.957% |

Mean Dice为逐图平均，Pooled Dice为汇总像素后计算。相对同Decoder的DINOv3，拼接的Mean Dice分别提高0.216/0.365个百分点；相对概率平均，Light略低，Swin只高0.101个百分点。Swin融合纠正679485个DINO错误像素，同时新增614397个错误，净减少65088个。两种Decoder下融合对DINO的Mean Dice开发集bootstrap区间均包含零；单seed且无容量匹配，不能据此宣称稳定提升或证明收益来自VMamba特有信息。

本轮不与旧seed42的最低loss结果或第三轮三个seed均值混合比较。后续依据结果设计容量匹配及多seed确认；文本编码器尚未加入。当前设备实测为24GB RTX4090，预检和正式训练峰值分配显存约10.41GiB，完整推理效率与各组参数量见 `stage4_fusion/preflight.json`。

本目录用于后续的图文指代分割（Referring Image Segmentation, RSI）研究。当前已完成第一轮冻结视觉表征对照、第二轮 Swin Pixel Decoder 对照，以及第三轮 DINOv3＋Light/Swin 的三个新 seeds 确认；文本编码器、图文对齐和指代分割实验尚未在此目录开展。

## 目录

- `stage1_visual/`：DINOv3-S/16 与 VMamba-T 冻结编码器、共同轻量分割 Decoder 的完整实验。包含源码、预训练权重、配置、日志、最佳/最终 Probe 权重、逐图预测和误差互补分析。
- `stage1_visual/PROTOCOL.md`：已执行的训练与评估口径。
- `stage1_visual/analysis/REPORT.md`：第一阶段结果摘要；`analysis/summary.json` 是详细数值。
- `stage1_visual/run_status.json`：训练及分析的完成状态。
- `EVALUATION.md` / `evaluation.py`：后续所有实验统一报告 macro/pooled Dice、IoU、Precision、Recall、Prec@X、Threshold Jaccard 和边界指标的定义与实现。
- `evaluations/stage1_visual/REPORT.md`：从第一轮已保存概率图重算的完整指标，原始结果和权重保留。
- `stage2_decoder/PROTOCOL.md`：第二轮 Swin Pixel Decoder 对照的冻结协议。
- `stage2_decoder/run_status.json`：第二轮训练及最终分析状态；完成后结果在 `stage2_decoder/analysis/REPORT.md`。
- `stage2_decoder/analysis/FINDINGS.md`：第二轮结果解读、面积分组和选模指标影响；`DIAGNOSTICS.md` 包含学习曲线及病例对照图。
- `stage3_confirmation/PROTOCOL.md`：DINOv3＋Light/Swin 的 seeds 43/44/45 确认，主规则为最高验证 mean Dice，保留 best loss 与第 300 轮作统一次要对照。
- `stage3_confirmation/run_status.json`：第三轮状态；完整结果写入 `stage3_confirmation/analysis/REPORT.md`。
- `stage3_confirmation/analysis/FINDINGS.md`：第三轮主结果、选模敏感性、面积分组及后续研究判断。
- `data/isic2018`：指向既有 ISIC2018 图像/掩码划分的数据链接，供阶段一代码使用。原数据与旧实验共享。
- `MIGRATION.json`：迁移前后路径及核心结果文件的 SHA256 校验。

## 第一阶段结果

单 seed 42、256×256、ISIC2018 旧划分（1886 训练 / 808 开发验证）、冻结编码器，各训练 300 epoch。DINOv3 的 pooled Dice 为 0.896165，VMamba 为 0.879327；逐图错误互补分析已完成。这些是开发验证集结果，不能视为独立测试成绩。

历史实验结果、权重和配置保持原样，避免改变实验记录。旧路径 `/home/featurize/ris_visual_stage1_20261016` 是指向 `stage1_visual/` 的兼容链接；原 `work/experiments/ris_visual_stage1_20261016/` 的文件链接也已更新。后续新实验应使用本项目目录的路径。

## 第二轮运行

第二轮仅替换 Pixel Decoder，复用第一轮冻结 Encoder 和 Adapter 定义，保持 seed 42、输入、增强、优化器 batch、300 epoch 预算和验证 loss 选模口径。Swin Decoder 含 407649 参数，Light Decoder 含 148033 参数；这是两种 Decoder 配置的对照，未作参数量匹配。

在 `stage2_decoder/` 下运行 `python launch.py`，先自动核对两组预检记录，再启动正式训练和最终分析；可从 latest checkpoint 恢复。真实数据两轮预检权重与正式训练独立。第一轮补充指标可在项目根目录运行 `python evaluate_runs.py --stage stage1_visual` 重现。

第二轮已完成：两组各 300 epochs / 9000 updates，完整指标、逐图预测、边界分析与完成校验均已保存。按预先规定的最小验证 loss 选模，DINOv3+Swin 的 mean/pooled Dice 为 0.883576/0.890662，VMamba+Swin 为 0.875742/0.883819。DINOv3 的 Boundary F1 从 0.276200 提升至 0.346577；区域结果存在精确率/召回率和目标面积的取舍。

后期验证 loss 与二值 Dice 的趋势并不一致，第 300 轮 DINOv3+Swin 的 mean Dice 为 0.890411，仅作为选模诊断记录。下一步确认实验需要先统一选模规则，再验证多个 seeds；不能从各指标的不同峰值中挑选结果。

## 第三轮运行

第三轮使用未参与筛选的 seeds 43/44/45，每个 seed 分别训练 Light/Swin 各 300 轮。同一 seed 共享冻结 DINOv3 前向特征，两个 Probe 参数和优化器独立。主结果按验证 mean Dice 最高选模，同时完整导出最低验证 loss 和最后一轮的结果；报告跨 seed 均值、样本标准差和逐 seed 配对差。

在 `stage3_confirmation/` 下运行 `python launch.py`，会检查初始化/数据增强预检和真实数据断点恢复预检，再监督训练、导出 18 组 checkpoint 预测、独立重算完整指标并自动生成图表。`run_status.json` 为 complete 且 `validation.json` 为 passed 才表示本轮全部完成。

第三轮已完成且全部校验通过。按预定最高 mean Dice 选模，三个 seeds 的结果为：

| DINOv3 Decoder | Mean Dice | Pooled Dice | mIoU | oIoU | Boundary F1 | HD95 (px) |
|---|---:|---:|---:|---:|---:|---:|
| Light | 0.886314 ± 0.000614 | 0.897020 ± 0.002504 | 0.811351 ± 0.000107 | 0.813276 ± 0.004114 | 0.301777 ± 0.015373 | 15.817114 ± 0.317335 |
| Swin | 0.891227 ± 0.001321 | 0.899283 ± 0.000930 | 0.819644 ± 0.001311 | 0.816999 ± 0.001536 | 0.340208 ± 0.009175 | 15.178963 ± 0.099409 |

± 为三个训练 seeds 的样本标准差。Mean Dice、mIoU、Boundary F1 和 HD95 在三个 seeds 上方向一致地改善；Pooled Dice 在 seed 43 略降。最后一轮的 Boundary F1 优势未保持，详见第三轮 FINDINGS.md。当前支持保留 DINOv3＋Swin 作为后续视觉候选、Light 作为对照；结论仍限于已有开发划分。

## GitHub 发布说明

本页面描述本地完整实验。GitHub保存源码和数值存档；权重、大型预测数组、数据及源图拼图保留本地。发布范围、文件校验清单与复现前提见 [PUBLISHING.md](PUBLISHING.md)。
