# 2026-10-20 实验发布说明

本目录存档截至第十三轮的源码、冻结协议、日志和数值结果。第十三轮五组 seed42 实验均从头训练 300 轮并完成原尺寸评估；第十一、十二轮为 100 轮无文本/固定文本/逐图文本和 Decoder 前交互对照。第十轮 MosMed 按用户要求暂停，正式 seed42 完成、seed43 部分完成、seed44 未启动，没有正式测试集结果。第七至九轮的原始记录继续保留。ISIC 颜色/形状属性提示不能视为复杂自然语言多目标指代理解的证明。

## 阅读入口

- [第十三轮：五组统一 300 轮](stage13_isic_300/analysis/REPORT.md)、[结果解释与预算对照](stage13_isic_300/analysis/FINDINGS.md)及[独立指标复核](stage13_isic_300/analysis/independent_report_check.json)。
- [第十二轮：Decoder 前词级交互](stage12_predecoder_text/README.md)与[第十一轮：100 轮文本对照](stage11_isic_text_controls/README.md)。
- [第十轮：MosMed 暂停说明](stage10_mosmed_text/README.md)、[暂停记录](stage10_mosmed_text/PAUSED_BY_USER.json)及[低 Dice 核查](stage10_mosmed_text/diagnostics/low_dice_20261020/FINDINGS.md)。
- [第十至十三轮依赖恢复说明](publication/REPRODUCE_STAGE10_13.md)及[本次发布检查](publication/update_20261020.json)。
- [项目总览](README.md)与[统一指标定义](EVALUATION.md)。
- [第一轮：DINOv3 / VMamba 视觉表征](stage1_visual/analysis/REPORT.md)。
- [第一轮补充完整指标](evaluations/stage1_visual/REPORT.md)。
- [第二轮：Swin Decoder 对照](stage2_decoder/analysis/REPORT.md)及[结果解读](stage2_decoder/analysis/FINDINGS.md)。
- [第三轮：三个新 seeds 的完整结果](stage3_confirmation/analysis/REPORT.md)及[结果解读](stage3_confirmation/analysis/FINDINGS.md)。
- [第三轮逐 seed、逐 checkpoint 表格](stage3_confirmation/analysis/all_runs.csv)。
- [第四轮：DINOv3 + VMamba、Light/Swin完整结果](stage4_fusion/analysis/REPORT.md)及[全部指标表](stage4_fusion/analysis/all_runs.csv)。
- [第四轮冻结协议](stage4_fusion/PROTOCOL.md)及[逐图纠错与汇总](stage4_fusion/analysis/summary.json)。
- [第五轮：容量、静态残差与动态门控](stage5_selective_fusion/analysis/REPORT.md)、[原有错误分层分析](stage5_selective_fusion/p0/REPORT.md)及[门控诊断](stage5_selective_fusion/analysis/gate_audit/REPORT.md)。
- [第六轮：融合与容量的三seed配对复验](stage6_fusion_confirmation/analysis/REPORT.md)、[全部指标](stage6_fusion_confirmation/analysis/all_runs.csv)及[冻结协议](stage6_fusion_confirmation/PROTOCOL.md)。
- [第七轮：Window Cross-Attention](stage7_cross_attention/analysis/REPORT.md)及[空间检索干预](stage7_cross_attention/analysis/attention_audit/REPORT.md)。
- [第八轮：RefCOCO源码与停止说明](stage8_text_baselines/README.md)及[停止记录](stage8_text_baselines/STOPPED_BY_USER.json)。
- [第九轮：ISIC文本实验](stage9_isic_text/README.md)、[完整结果](stage9_isic_text/analysis/summary.json)、[指标CSV](stage9_isic_text/analysis/all_models.csv)及[结果解释](stage9_isic_text/analysis/FINDINGS.md)。
- [图文数据准备](text_stage_preparation/README.md)及[数据接口和评价草案](text_stage_preparation/DATA_PROTOCOL.md)。

第三轮完成 seeds 43/44/45 × Light/Swin 共六次 300 轮训练，三种选模规则共导出 18 组结果。主结果按预先规定的最高验证 mean Dice 选模，并同时报告逐图平均与 pooled 指标。所有结果均来自 1886/808 开发划分，不是独立测试成绩。

第四轮完成 seed42、DINOv3/VMamba/双编码器拼接 × Light/Swin 六组300轮训练，统一主选模为最高mean Dice，另保存最低loss和最后一轮。全部18组的完整指标与逐图记录已存档；开发验证集单seed筛选尚不能替代容量匹配和跨seed确认。`smoke/`及`smoke_analysis/`仅为两轮恢复与导出预检，非正式结果。

第五轮完成seed42三个容量/残差/门控模型各300轮、九组checkpoint评估和门控干预。第六轮复用经过配置/初始化/源文件核验、且当前机器重算概率逐像素相同的第三轮三个DINO+Swin基线；新增容量对照及Concat各三个seed训练，九个模型与seed组合共27组checkpoint完整评估均通过。seed42只作探索背景，不混入seeds43/44/45主汇总。这些seed和开发划分曾用于前期研究，不能据此宣称全新独立验证。

图文准备目录保留早期审计及后续更正记录。后来已从MedVLSM作者仓库取得ISIC2016文本提示，原810/90/379记录按test优先去重后为806/90/379。第九轮只用P3颜色/形状，并统一将类别词skin melanoma改为skin lesion；GT生成的大小、数量、位置字段不进入模型。原始第三方文本JSON未重新分发，固定来源、版本、SHA与[恢复说明](publication/REPRODUCE_STAGE9.md)已提交。

第九轮DINO/VMamba/Concat/DINO固定文本的测试macro Dice为90.120%/90.670%/90.415%/90.180%，pooled Dice为91.381%/91.664%/92.056%/91.573%。固定文本对照与错配诊断尚未支持稳定文本收益，单seed不能估计跨seed稳定性；历史ISIC开发数据可能跨年份重叠，不称本轮为整个研究流程的全新外部验证。

## 发布内容与本地完整实验的区别

Git 保存自研源码、协议、数据成员清单、配置、日志、状态、校验结果、逐图数值、汇总表与纯数值图。遵循仓库已有的权重发布约定，模型权重不进入 Git 或 Release。

完整实验仍位于 `/home/featurize/referring_image_segmentation/`。以下文件保留在该机器，路径、大小和 SHA256 记录于 [source_inventory.json](publication/source_inventory.json)：

- 训练/预训练权重、优化器断点。
- 逐像素概率图、二值预测、误差数组（NPY/NPZ）；逐图指标 JSON 已提交。
- 含ISIC/COCO原始图像的示例拼图；报告中相关历史图片引用仅在本地完整目录可用。
- 上游 VMamba / VM-UNet 源码副本、安装包和 CUDA 扩展；固定来源与 Git SHA 见 [third_party.json](publication/third_party.json)。
- 原始数据目录链接不提交，数据划分和成员 SHA 见 `stage1_visual/data_manifest.json`。
- 第八/九轮 `assets/` 中的原图、Mask、第三方文本、特征/句子缓存与CLIP权重，以及第八轮 `runtime/`。
- 第十至十三轮 `assets/`、MosMed 原始文本表及转换出的第三方文本 JSON、上游加载器副本；来源 URL、版本、成员哈希与获取脚本已保存。含原图的诊断拼图继续只在本地保存，纯数值曲线可公开。

2026-10-18用户要求清理不用的权重。旧清单不代表文件现在仍存在：已删除的次要checkpoint及数组在新清单标记`local_deleted`，保留原摘要供追溯。第九轮在完成评估后只保留四组最佳权重；第八轮中止，保留已产生的最佳权重和恢复副本。详见[存储保留说明](STORAGE_RETENTION.md)及第九轮retention.json。体量较大的资产/依赖目录在清单中按目录排除，数据成员哈希由各阶段data_manifest/text_provenance保存，不把未重新逐文件审计的目录写成已校验。

缓存、嵌套Git元数据、运行锁和PID文件不属于发布快照。已完成实验源码、冻结协议和数值结果保持原字节；README和事后解读反映最新状态。清单同时记录源文件与发布文件SHA256。纯数值学习曲线允许提交，含源病例图像的拼图保留本地。

## 校验与复现

从仓库根目录运行：

```bash
python referring_image_segmentation/publication/verify_archive.py
```

该命令只核对已发布文件的大小和 SHA256，不训练、不评分、不要求本地权重或数据。运行时生成的 `validation.json` 是原实验环境的完成记录，不能理解成克隆仓库后已经重新完成训练校验。

复现训练需要另行恢复合法取得的数据、预训练权重、固定版本上游实现和运行环境。DINOv3 与 VMamba 权重的来源和 SHA 记录在 `stage1_visual/provenance.json`；Python/PyTorch/timm/CUDA 版本也在该文件中。VMamba 使用的 selective-scan 扩展依赖记录在同文件与 `dependencies.log`。

历史代码和provenance保留原绝对路径，尤其第一轮迁移前路径 `/home/featurize/ris_visual_stage1_20261016`。原机器通过兼容链接继续访问。跨机器复现应恢复路径映射，或在新实验目录显式记录调整；不能改写历史证据后声称字节级一致。阶段三至七启动器校验历史权重及源码，恢复依赖前不能直接复跑。部分次要权重/预测已按用户要求删除，重算对应次要分析需要恢复原文件。各轮硬件/软件信息见preflight.json；第七至九轮使用48GB RTX4090。第八/九轮需要CLIP文本权重与附加依赖，详见第八轮requirements-local.txt及安装日志；第九轮使用其冻结CLIP资产，完全不使用第八轮已训练的分割权重。

本轮不改变其他历史实验的结论或文件。完整指标及选模敏感性的解释以各阶段报告为准。

## 2026-10-20 新增实验

第十三轮保持 ISIC2016 的 806/90/379 数据划分、输入、增强、batch16、AdamW 1e-4 和 5 轮 warmup，仅将统一预算/调度长度改为 300 轮。五组没有加载旧分割权重，各完成 15300 次更新；原尺寸验证最佳权重统一用于测试，逐图计数独立复核通过。不能将不同总长度 cosine 下的前 100 轮理解为相同训练轨迹。

后置逐图文本相对固定文本的测试 macro Dice 为 -0.338 个百分点；前置逐图为 +0.104 个百分点且图像配对区间跨 0。单 seed、开发过程中已经使用的测试集、未验证患者身份限制结论范围。前后方案还改变词级/句子表示及输出头，不能作为只改变融合位置的严格消融。

第十一至十三轮分别保留 1、2、5 个新最佳权重，其组合恢复及 smoke 权重已按各自 retention.json 在评估通过后清理。第十一轮 E1/E2 只读复用第九轮结果。第十轮未完成整个计划，暂停恢复点保留，不在本次发布中恢复、重训或清理。
