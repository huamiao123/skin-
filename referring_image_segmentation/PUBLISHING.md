# 2026-10-17 实验发布说明

本目录存档独立图文指代分割项目截至第六轮融合与容量配对复验的源码与结果，并包含图文数据准备文档。当前完成的实验均为纯图像分割；图文指代能力尚未验证。

## 阅读入口

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
- [图文数据准备](text_stage_preparation/README.md)及[数据接口和评价草案](text_stage_preparation/DATA_PROTOCOL.md)。

第三轮完成 seeds 43/44/45 × Light/Swin 共六次 300 轮训练，三种选模规则共导出 18 组结果。主结果按预先规定的最高验证 mean Dice 选模，并同时报告逐图平均与 pooled 指标。所有结果均来自 1886/808 开发划分，不是独立测试成绩。

第四轮完成 seed42、DINOv3/VMamba/双编码器拼接 × Light/Swin 六组300轮训练，统一主选模为最高mean Dice，另保存最低loss和最后一轮。全部18组的完整指标与逐图记录已存档；开发验证集单seed筛选尚不能替代容量匹配和跨seed确认。`smoke/`及`smoke_analysis/`仅为两轮恢复与导出预检，非正式结果。

第五轮完成seed42三个容量/残差/门控模型各300轮、九组checkpoint评估和门控干预。第六轮复用经过配置/初始化/源文件核验、且当前机器重算概率逐像素相同的第三轮三个DINO+Swin基线；新增容量对照及Concat各三个seed训练，九个模型与seed组合共27组checkpoint完整评估均通过。seed42只作探索背景，不混入seeds43/44/45主汇总。这些seed和开发划分曾用于前期研究，不能据此宣称全新独立验证。

图文准备目录仅是本地数据审计和后续协议草案。检查的23份元数据表尚无已验证的自然语言指代表达字段；旧reference_id是掩码标注记录。没有启动文本训练，也没有将单一固定提示或结构化诊断字段视为真实指代文本。

## 发布内容与本地完整实验的区别

Git 保存自研源码、协议、数据成员清单、配置、日志、状态、校验结果、逐图数值、汇总表与纯数值图。遵循仓库已有的权重发布约定，模型权重不进入 Git 或 Release。

完整实验仍位于 `/home/featurize/referring_image_segmentation/`。以下文件保留在该机器，路径、大小和 SHA256 记录于 [source_inventory.json](publication/source_inventory.json)：

- 训练/预训练权重、优化器断点。
- 逐像素概率图、二值预测、误差数组（NPY/NPZ）；逐图指标 JSON 已提交。
- 含 ISIC2018 原始病例图像的示例拼图；报告中相关历史图片引用仅在本地完整目录可用。
- 上游 VMamba / VM-UNet 源码副本、安装包和 CUDA 扩展；固定来源与 Git SHA 见 [third_party.json](publication/third_party.json)。
- 原始数据目录链接不提交，数据划分和成员 SHA 见 `stage1_visual/data_manifest.json`。

缓存、嵌套 Git 元数据、运行锁和PID文件不属于发布快照。源代码和历史协议/结果保持原字节；项目README同步第五/六轮完成结果并追加发布说明链接，清单同时记录源文件与发布文件的SHA256。第五/六轮图表仅含数值学习曲线，不包含源病例图像。

## 校验与复现

从仓库根目录运行：

```bash
python referring_image_segmentation/publication/verify_archive.py
```

该命令只核对已发布文件的大小和 SHA256，不训练、不评分、不要求本地权重或数据。运行时生成的 `validation.json` 是原实验环境的完成记录，不能理解成克隆仓库后已经重新完成训练校验。

复现训练需要另行恢复合法取得的数据、预训练权重、固定版本上游实现和运行环境。DINOv3 与 VMamba 权重的来源和 SHA 记录在 `stage1_visual/provenance.json`；Python/PyTorch/timm/CUDA 版本也在该文件中。VMamba 使用的 selective-scan 扩展依赖记录在同文件与 `dependencies.log`。

历史代码和 provenance 中保留原绝对路径，尤其第一轮迁移前路径 `/home/featurize/ris_visual_stage1_20261016`。原机器通过兼容链接继续访问。跨机器复现应恢复这些路径映射，或在新的实验目录中显式记录路径调整；不能修改历史证据后声称字节级一致。阶段三至六启动器会校验历史权重及源码，在未恢复依赖时不能直接复跑。第五轮P0/门控分析依赖已保存的概率和权重；第六轮复用审计还需要第三轮DINO权重/概率，均保留本地，克隆仓库不含这些文件。各轮硬件与软件信息见对应 `preflight.json`；第五/六轮使用24GB RTX4090，与早期48GB实例不同。

本轮不改变其他历史实验的结论或文件。完整指标及选模敏感性的解释以各阶段报告为准。
