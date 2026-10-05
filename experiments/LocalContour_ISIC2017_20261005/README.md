# 局部候选—整体轮廓：ISIC2017 简短机制验证

本轮结论为 **`GENERATOR_INADEQUATE`：当前候选生成器覆盖不足，先改进候选，再决定是否训练 Transformer**。它保留了“候选已包含正确答案、整体判断可能有帮助”的研究空间；本轮没有验证出可部署新方法，也没有证明这条大方向无望。

固定 256 个节点、搜索半径 16 输入像素、闭合 DP λ=0.05。150 张官方 val 按 seed17 预先拆分为 calibration50 / locked100；五组共 750 行，保留所有图像。以下是 locked100 的逐图宏平均（BF1 容差为 2 输入像素）：

| 组 | Dice | BF1 |
| --- | ---: | ---: |
| G0 | 0.853069 | 0.513837 |
| G1 | 0.853082 | 0.513374 |
| G2 | 0.841146 | 0.499581 |
| G3 | 0.849274 | 0.508378 |
| G4 | 0.888135 | 0.671347 |

G0 为原 CNN；G1 为零偏移重建；G2 为局部评分；G3 为闭合动态规划；G4 用真实标注提供固定候选的距离成本，并通过相同 DP / 平滑 / 栅格化流程选择。**G4 仅供离线诊断，不可部署，也不是严格 Dice 上界；GT 没有参与候选生成或移动。**

搜索带覆盖 90.64%，但候选覆盖有效法线仅 65.35%，低于预登记的 80%；已有候选覆盖时，局部错选比例 29.66%。G4−G0 的 Dice 为 +3.5066 个百分点，配对图像 bootstrap 95% 区间为 [2.8955, 4.1997] 个百分点；G3 未超过 G0。当前接口留下离线纠错空间，但候选覆盖不足阻止直接进入 Transformer 消歧训练。

本实验固定公开 CNN，仅训练 705 参数边界评分头，10 epoch；未训练 Transformer。CNN 上游原训练使用过官方 train / val 合并后的 70:30 划分。因此本轮是开发诊断，locked100 不能称为独立测试集，bootstrap 区间也仅作探索性解释；官方 test 未评分。未验证连续长缺口与清晰不规则细节的实际能力。

首版轮廓栅格化造成明显零位重建损失。`results_v1_representation_loss/` 和 `protocols/source_v1_representation_loss/` 保留原始结果与源码；`results/` 是修复后的正式本轮读数。修复只调整所有组共用的、无 GT 的组件保留/拼接，150 张图的候选坐标、G0、各组选择、原始栅格化逐字节与 v1 相同；没有重新训练 CNN / 边界头、没有重新选择候选参数或 λ、没有再次推理。`results/representation_repair_acceptance.json`、`lambda_curve_provenance.json` 和 `code_provenance.json` 保存验收及来源。修复后 G1−G0 Dice 约 +0.0013 个百分点，不能把重建变化计作整体消歧收益。

- [一页结论](results/conclusion_1page.txt)、[本轮报告](results/report.txt)、[主表](results/main_table.csv)、[决定与门槛](results/decision.json)、[完整运行验收](results/RUN_DONE.json)。
- [工程测试记录](engineering_cpu_tests.json)：52 项测试通过；[独立核验](independent_review.json)、[补充审阅](external_review.txt)。
- `case_panels/`：两张 seed17 随机病例、最低 G4−G0 增益病例和最低搜索带覆盖病例；来源及选择规则见 [selection.json](case_panels/selection.json)。
- `contour/`、`tools/`、`tests/`：本项目编写的实现、机械修复与工程测试。
- `protocols/`、`preregistration.json`：任务书、预登记、历史源码快照。
- `assets/`：2150 张官方 train/val 图像身份、150 张 val 拆分及公开 CNN 审计；不含完整图像或 mask 数据集。
- `boundary_head/`：config、10 epoch 训练日志、DONE；权重和特征缓存仅在本地。
- `results_v1_representation_loss/`、`diagnostics/`：首版数值结果及校准侧的表示损失归因。
- `cleanup/`：RSI/EGE 清理摘要及完整性核验；旧权重已删除，仅保留元数据。

公开 CNN 上游：[A1maan/msgu-net](https://github.com/A1maan/msgu-net)。来源 commit `c4b16bf372067eb163a44f16104473383ac7d718`；ISIC2017 checkpoint SHA256 `3a7fb064383cc68faf33b76561379b69a84141cf51e94679dc31e3a3e66a515d`。上游实现与 checkpoint 没有复制到本仓库；`third_party/msgu_net/source_manifest.json` 提供固定 URL 与逐文件 SHA，复现需另行从上游获取。完整复现还需数据路径、公开 CNN 及本地边界头权重或按相同协议重新训练边界头。

官方数据来源：[ISIC Challenge 数据页](https://challenge.isic-archive.com/data/#2017)。本目录发布数值成果及少量公开 ISIC2017 诊断 overlay，不发布完整数据集、GT NPZ、特征数组，也不包含 IMA / PH2 病例图片。
