# 图文指代分割项目

## 第十三轮：ISIC2016 五组统一 300 轮（2026-10-20）

[源码与配置](stage13_isic_300/README.md) · [冻结协议](stage13_isic_300/PROTOCOL.md) · [完整结果](stage13_isic_300/analysis/REPORT.md) · [结果解释](stage13_isic_300/analysis/FINDINGS.md) · [完成校验](stage13_isic_300/validation.json)。

五组均从头初始化可训练部分，完成 seed42、300 epochs / 15300 次更新，使用完整 300 轮 cosine 调度；没有续接旧 100 轮分割权重。ISIC2016 划分保持 806/90/379，输入 256、batch16、AdamW 1e-4、5 轮 warmup，DINOv3 与 CLIP 冻结，按原尺寸验证集 macro Dice 选模。

| 模型 | 最佳轮 | 测试 macro Dice | 测试 pooled Dice | 测试 macro IoU | 测试 pooled IoU |
|---|---:|---:|---:|---:|---:|
| 无文本 | 21 | 90.248% | 91.667% | 83.712% | 84.615% |
| 后置融合＋固定文本 | 157 | 90.771% | 91.724% | 84.159% | 84.713% |
| 后置融合＋逐图文本 | 89 | 90.433% | 91.683% | 83.725% | 84.644% |
| 前置交互＋固定文本 | 21 | 90.311% | 91.754% | 83.813% | 84.765% |
| 前置交互＋逐图文本 | 30 | 90.415% | 91.690% | 83.707% | 84.655% |

macro 为逐图平均，pooled 为先汇总像素计数再计算。前置逐图文本相对固定文本仅高 0.104 个百分点，图像配对区间跨 0；后置逐图文本低 0.338 个百分点。当前没有证明逐图属性文本的稳定额外收益，也没有证明前置交互优于后置融合。前后方案还同时改变词级/句子表示与输出头，不是单纯的融合位置消融。单 seed、测试集已用于开发、患者独立性未验证，不能据此宣称独立泛化。完整 Precision/Recall、Prec@X、边界指标、逐图数值和 100/300 轮对照均已保存。

## 第十一、十二轮：100 轮文本对照与前置交互

[stage11_isic_text_controls](stage11_isic_text_controls/README.md) 新增无文本对照，并核验复用第九轮固定/逐图文本结果；[stage12_predecoder_text](stage12_predecoder_text/README.md) 新训练固定/逐图文本两组，在 Adapter 之后、Swin 之前做词级 Cross-Attention，之后使用普通 Mask Head。两轮均仅 seed42，100 epochs，原始结果与历史文件保持不变。

## 第十轮：MosMedData+ 暂停与低 Dice 核查

[源码、协议与暂停状态](stage10_mosmed_text/README.md) · [低 Dice 只读核查](stage10_mosmed_text/diagnostics/low_dice_20261020/FINDINGS.md)。该任务分割肺部感染病灶；修复可追溯来源组跨集合问题后，划分为 2004/216/509。seed42 三组完成 100 轮；seed43 在第 60 轮部分更新后按用户要求停止，seed44 未启动，未进行正式测试集评估。

验证 macro Dice 为无文本 47.570%、固定文本 45.467%、逐图文本 44.693%，对应 pooled Dice 为 67.298%/66.098%/65.758%。独立指标复算、掩码缩放和几何检查通过，但小病灶漏检及训练/验证差距明显。来源组不重叠不代表已验证所有患者身份；文本来源及 Mask 辅助质检限制见协议。保留暂停记录，不自动恢复其他 seed。

## 第九轮：ISIC2016 医学文本引导分割（2026-10-18）

用户要求停止RefCOCO并先试小规模ISIC文本分割。新增 [stage9_isic_text](stage9_isic_text/README.md)：官方ISIC2016图像/Mask与MedVLSM P3颜色/形状文本，去重后806训练/90验证/379测试；统一类别词为skin lesion。保留DINOv3、VMamba、Concat三视觉分支与DINO固定文本训练对照，视觉/文本Encoder冻结，仅seed42，100epochs、batch16，验证原尺寸macro Dice选模。源码/协议/预检和运行状态分别保存在第九轮目录。

四组训练和测试已完成，[完整结果](stage9_isic_text/analysis/summary.json)与[解释](stage9_isic_text/analysis/FINDINGS.md)已保存。测试macro Dice依次90.120%、90.670%、90.415%、90.180%；pooled Dice依次91.381%、91.664%、92.056%、91.573%。真实文本未超过DINO固定文本对照，错配属性文本影响较小；单seed图像配对区间均包含0，不能据验证集93.325%宣称稳定图文增益。完成后仅保留四组最佳权重。

第八轮RefCOCO已按用户要求停止，四组已取得的最佳权重、日志与恢复记录保留，未完成正式30轮及最终测试。第九轮只读取其冻结CLIP权重资产，不使用RefCOCO训练出的分割参数。

## 第八轮：RefCOCO 图文指代分割（2026-10-18）

用户已确定进入RefCOCO UNC、只跑seed42。新增 [stage8_text_baselines](stage8_text_baselines/README.md)：DINOv3、VMamba、原Concat三种视觉分支，统一冻结CLIP ViT-B/32文本Encoder、Swin像素特征Decoder和可训练文本—像素余弦匹配头。额外从头训练一组DINO固定文本对照检验语言收益。所有历史分割权重不用于新任务初始化。

数据已经落实19994图/50000目标/142210真实表达，标注使用固定镜像保留的COCO polygon/RLE和原文本字段；原站连接失败及镜像来源已披露。原尺寸mask、图像和数据成员哈希完成，详见 [数据清单](stage8_text_baselines/data_manifest.json)。训练256×256等比缩放填充，padding排除loss，关闭翻转/旋转；batch16，30epochs，每epoch全部120624表达。按原尺寸逐表达validation mean IoU选模，锁定后评估testA/testB。

冻结配置见 [PROTOCOL.md](stage8_text_baselines/PROTOCOL.md)，当前流程见 [run_status.json](stage8_text_baselines/run_status.json)。CLIP句子缓存、初始化/梯度、padding排除、原尺寸几何、完整GPU缓存，以及短程中断恢复/指标重算预检均已通过。四组模型曾开始正式后台训练，现已按用户要求停止；当前没有已完成的RefCOCO正式结果。UNC原划分另发现一张验证图与训练图字节一致（9条表达），原计划的锁定checkpoint后敏感性分析随任务停止，见第八轮README。

## 第七轮：DINOv3 / VMamba Cross-Attention（2026-10-18）

已建立 `stage7_cross_attention/`，见 [冻结协议](stage7_cross_attention/PROTOCOL.md)、[模型源码](stage7_cross_attention/models.py)、[预检](stage7_cross_attention/preflight.json) 和 [后台状态](stage7_cross_attention/run_status.json)。本轮以seed42测试局部空间检索：DINO提供Query，VMamba提供Key/Value，四尺度64通道、8×8非重叠窗口、4heads和相对位置偏置。保留原Swin Pixel Decoder，使用固定残差 `D + 0.1 * message`。

新增两组300epochs：window_cross_attention和近似容量匹配的local_conv。后者为两路归一化特征拼接后的1×1/深度7×7/1×1卷积；两组可训练参数670321/670401，只差80，但不声称精确匹配感受野或FLOPs。Encoder冻结、batch64、AdamW .001、Cosine T_max300和其余数据/增强/loss/选模规则与前几轮一致。

当前机器是48GB RTX4090；旧六个模型的18组checkpoint在该机器重新前向与历史概率逐像素一致。注意力数学、梯度、窗口边界、分支参与和共同初始化预检通过。监督入口 `python stage7_cross_attention/launch.py` 先执行真实数据两轮恢复/完整指标/空间检索干预预检，再后台正式训练。只在正式训练、完整评估、干预及历史哈希核验全部通过后写complete。

本轮两组300轮和全部评估已经完成，见 [完整结果](stage7_cross_attention/analysis/REPORT.md) 与 [空间检索诊断](stage7_cross_attention/analysis/attention_audit/REPORT.md)。该版本Cross未超过近似容量匹配局部卷积或原Concat，且新增错误超过纠正错误。固定权重的窗口均值V/同位置V干预仅用于诊断，不替代独立训练对照；原源码、权重和报告保持冻结。

## 第六轮：融合与容量的配对种子复验（2026-10-17）

`stage6_fusion_confirmation/` 已于2026-10-17 16:02 UTC全部完成，见 [完整结果](stage6_fusion_confirmation/analysis/REPORT.md)、[全部指标](stage6_fusion_confirmation/analysis/all_runs.csv)、[冻结协议](stage6_fusion_confirmation/PROTOCOL.md)、[完成校验](stage6_fusion_confirmation/validation.json) 和 [复用预检](stage6_fusion_confirmation/preflight.json)。矩阵为DINOv3单分支、DINO双Adapter容量对照、DINOv3+VMamba Concat，均使用原Swin Decoder，seeds43/44/45。

第三轮三个DINO+Swin训练通过源文件、数据、初始化、配置、选模核验；当前GPU重算九个checkpoint，概率逐像素与历史一致，因此只读复用。新增六个训练：两个融合/容量组×三个seeds，每组300epochs/9000updates，batch64、冻结Encoder、最高验证mean Dice选模，另保存最低loss和最后一轮。seed42仅作探索背景，不混入主汇总；43/44/45及开发集此前已经使用过，不能称为全新独立测试。

监督入口 `python stage6_fusion_confirmation/launch.py` 已完成两轮真实数据恢复、导出、全指标和汇总预检，以及三个seed的正式训练。同seed共享冻结前向、独立参数和优化器；当前24GB RTX4090峰值分配约10.39GiB，batch64。九个模型与seed组合、27组checkpoint完整评估通过，源文件及历史文件哈希未变。逐seed结果在 `analysis/seed{seed}/`，主汇总如下（均值±训练seed样本标准差）：

| 模型 | Mean Dice | Pooled Dice | mIoU | Pooled IoU | Boundary F1 | HD95 px |
|---|---:|---:|---:|---:|---:|---:|
| DINOv3 | 0.891227 ± 0.001321 | 0.899283 ± 0.000930 | 0.819644 ± 0.001311 | 0.816999 ± 0.001536 | 0.340208 ± 0.009175 | 15.178963 ± 0.099409 |
| 双Adapter容量对照 | 0.892293 ± 0.000931 | 0.900196 ± 0.002658 | 0.821139 ± 0.001137 | 0.818513 ± 0.004394 | 0.337193 ± 0.008722 | 15.080887 ± 0.219909 |
| DINOv3+VMamba Concat | 0.893995 ± 0.000577 | 0.898871 ± 0.000941 | 0.823562 ± 0.001173 | 0.816318 ± 0.001553 | 0.365576 ± 0.015479 | 15.403120 ± 0.104561 |

主选模规则下，Concat在三个seed的mean Dice均超过DINO和容量对照，平均差分别为+0.277/+0.170个百分点；Boundary F1优于DINO，但pooled Dice均值略低、HD95更高。Concat平均每seed纠正704230个错误像素，新增737999个，净增加33770个；误差净减少并不稳定。完整单图延迟DINO/容量/Concat约11.00/11.71/27.81ms。容量对照只近似匹配可训练参数。结论支持部分指标上的额外收益，不能宣称全面提升、稳定减少错误或独立测试泛化；最低loss和最后一轮也完整保留，不混用选模规则。

图文阶段准备单独保存在 [text_stage_preparation](text_stage_preparation/README.md)：已整理样本接口、文本有效性检查、分组划分与评价草案，并生成本地数据清单审计。已检查的23份元数据表尚无验证过的指代表达字段；旧多标注者reference_id不能当作自然语言。当前未启动文本训练。

## 第五轮：容量与选择性纠错（2026-10-17）

`stage5_selective_fusion/` 已全部完成并通过校验：[冻结协议](stage5_selective_fusion/PROTOCOL.md)、[运行状态](stage5_selective_fusion/run_status.json)、[结果报告](stage5_selective_fusion/analysis/REPORT.md)。P0使用第四轮现有预测完成前景概率、单模型正确性、边界/内部/背景分层统计，见 [P0报告](stage5_selective_fusion/p0/REPORT.md)。

本轮仅新增三组seed42训练：DINO双Adapter容量对照、DINO+VMamba静态残差融合、DINO+VMamba动态空间门控。均为冻结Encoder、原Swin Decoder、256输入、batch64、300epochs、最高验证mean Dice选模；最低loss和最后一轮同步保留。静态/动态初始输出完全相同，初始alpha=.1，动态门控只多512个参数。容量组638305个可训练参数，相比第四轮Concat的632161相差0.972%。

监督入口为 `python stage5_selective_fusion/launch.py`。三个模型均完成300轮，结果在 `stage5_selective_fusion/analysis/`，包括全部指标、逐图纠错/新增错误、容量与动态对静态的比较，以及固定checkpoint的gate置零/空间均值/空间平移诊断。动态门控没有超过原Concat或静态残差，也没有减少相对原Concat的新增错误；这不证明VMamba无效。该轮属于seed42开发集机制探索，后续配对复验见第六轮。

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

本目录用于图文指代分割（Referring Image Segmentation, RSI）研究。第一至七轮为视觉实验；第八轮 RefCOCO 已停止，第九至十三轮包含医学属性文本引导分割、对照及前置交互实验。ISIC 的属性提示结果不能等同于复杂多目标自然语言指代理解。

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
