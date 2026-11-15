MambaVision 范围漏分：文献与代码调研、Stage29 设计记录
日期：2026-11-11
状态：方法设计与实现完成；效果必须以 Stage29 正式验证/测试结果为准。

1. 本轮问题和证据边界
当前系统是 MambaVision-S + 64通道 Swin 多尺度解码器，后两个编码阶段参与微调。
Stage22/24支持微调的整体收益。Stage25否定“大病灶普遍变差”的过强说法，但发现部分重复的大片内部漏分。
Stage26的训练集拟合线性粗范围探针能改善部分开发集坏例，整体精度却远低于完整网络。因此只能说部分案例存在可读取的范围线索，不能证明解码器普遍丢信息。
Stage27只增加f4范围辅助损失没有稳定改善最终输出。Stage28缓解过度自信也没有稳定改善分割。
本轮研究对象：已存在的深层语义是否需要一条更直接的路径进入各级解码，而非再次强化辅助监督。

2. 检索与真实阅读范围
两次2020—2026官方API检索分别获得39、33条去重记录，两个集合未作全局去重，不能称为72篇不同论文。
原始结果及参数在 mambavision/、decoder/ 的 papers.json、search_log.md；这是候选元数据，不是已读全文。
OpenAlex/Crossref可用；部分Semantic Scholar/arXiv API请求遇到429。GitHub API遇403限流，改用官方codeload源码包，保留包哈希及逐文件哈希。
原CLI自动生成的topic_brief包含错误的多智能体连通性固定模板；已标记无效，不作为本报告证据。相关判断由论文原文、摘要和实际代码重建。
已获取7个官方代码仓库的轻量源码快照；重点阅读6个仓库关键模型文件，PraNet-V2另外核查论文方法和后续源码。
MVSeg只取得出版方公开摘要/方法片段，没有据此声称复现其完整方法。EGE-UNet仅作为已核实的相关机制边界，不声称读完其源码。
检索不是穷尽综述，也没有把各论文不同数据协议的分数直接对比当前成绩。

3. 代表方法及对当前问题的意义
HMT-UNet（2024预印本）
https://arxiv.org/html/2408.11289v1
https://github.com/simzhangbest/HMT-Unet
已读 models/hmt_unet.py，核实HMTUNet调用自定义MambaVision_sim，并在load_from中映射编码层到解码层。它说明MambaVision医学U形结构已有前作。仅增加一个Swin或Mamba解码器不是新的研究问题。

MVSeg（CVIU 2026，出版方页面）
https://www.sciencedirect.com/science/article/pii/S1077314226001542
直接使用MambaVision编码器，提出Progressive Prediction Generation和Prediction-Guided Feature Enhancement。它是“基于MambaVision的预测引导融合”非常直接的近邻。公开片段不足以逐层还原，未找到可确认的作者完整仓库，不能编造结构或宣称我们首次做预测引导。

PraNet（MICCAI 2020）
https://arxiv.org/abs/2006.11392
https://github.com/DengPingFan/PraNet
lib/PraNet_Res2Net.py明确用1-sigmoid(coarse_logits)加权深层特征，再将残差logit与粗预测相加，多个尺度接受监督。其启发是让粗预测参与在线细化，而非只做训练辅助。局限：反向概率本身不区分真背景和漏分，不能直接理解成“错误位置”。

UACANet（ACM MM 2021）
https://arxiv.org/abs/2107.02368
https://github.com/plemeri/UACANet
lib/modules/attention_module.py的UACA从预测构造前景、背景和中间概率区域，形成区域上下文，再对像素执行上下文注意力。启发是区域级聚合；但我们已有高置信度漏分证据，不能假定错误都在0.5附近。因此本轮不采用“只修不确定区域”的硬限制。

DCA（2023）
https://arxiv.org/html/2303.17696v1
https://github.com/gorkemcanates/Dual-Cross-Attention
model/utils/dca.py先将多尺度特征池化到统一网格，顺序执行通道、空间跨注意力，再上采样并残差回接各尺度。它说明跨尺度语义桥接已有通用实现。我们不把跨尺度传递本身当创新；也不在现有Swin之上直接叠加完整双注意力。

EMCAD（CVPR 2024）
https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html
https://github.com/SLDGroup/EMCAD
lib/decoders.py的LGAG由上采样解码特征和skip共同生成权重，权重直接乘skip；MSCB使用多核深度卷积。其可继承点是低成本多尺度局部处理和条件融合。
Stage29只借鉴这些常用算子，使用新写的GroupNorm残差适配器，不是EMCAD完整复现。原skip始终保留；门控只作用于新增残差。

MEGANet（WACV 2024）
https://arxiv.org/abs/2309.03329
https://github.com/UARK-AICV/MEGANet
MEGANet-Res2Net/lib/modules.py的EGA融合反向预测、预测Laplacian和输入边缘三路特征，再残差连接和CBAM。该方法针对边界有清晰设计，但我们存在大面积内部漏分，因此当前不优先加入输入边缘支路。

PraNet-V2（2025预印本，读取v2）
https://arxiv.org/html/2504.10986v2
https://github.com/ai4colonoscopy/PraNet-V2
论文DSRA分别学习前景与背景，借助更深层输出调制当前预测。它直接说明“独立前背景头/双向纠错”已有近邻。当前不增加双头和新损失，以免与信息路由同时改变。

EGE-UNet（MICCAI 2023）
https://arxiv.org/abs/2307.08473
https://github.com/JCruan519/EGE-UNet
公开摘要和仓库说明GAB联合不同尺度特征及解码mask；也是皮肤任务中预测引导融合的直接前作。其仓库7:3训练/验证划分不能与我们1886/762/867协议直接比较。

4. 为什么本轮选这个最小模块
现有解码：8x8深层Swin输出 -> 上采样+16x16 skip -> Swin -> 上采样+32x32 skip -> Swin -> 上采样+64x64 skip -> Swin -> mask。
本轮保留此主路径，并为每个融合点增加小残差：
    merged_i = up(d_{i+1}) + skip_i + R_i(context_i, skip_i)
R用128->32通道压缩、3x3/5x5并行深度卷积、GroupNorm/GELU、空间门控、32->64投影。
投影零初始化，门控初值1，确保各组最初与基线输出一致。该门控不是可靠性估计，不赋予其概率解释。
control：原模型。
local_fusion：context_i=当前上采样解码特征；是普通相邻层残差融合。
anchor_fusion：context_i=最初8x8深层Swin输出直接上采样；相同残差模块、相同参数和初始化，只换输入来源。
16x16处两候选的上下文最初来自同一深层结果；差异主要在后续32/64尺度是否再次读取未经逐级融合的深层表示。
不加入额外标签、辅助头、边缘损失、DropPath或新数据增强。代码从官方预训练重新开始训练，并非在验证最优checkpoint上继续挑点。

5. 能证明什么，不能证明什么
anchor超过control但不超过local：支持增加普通融合容量，不支持深层直达机制有额外价值。
anchor同时超过两组且严重漏分减少：初步支持这条信息路径，仍需另一个种子与合理外部基线。
总体上升但漏分或大量正常病例恶化：只能描述指标权衡，不能宣称解决范围漏分。
anchor在同权重推理时替换上下文/关闭残差：只作开发集敏感性分析，不能替代重新训练消融。
失败不会自动触发不同宽度、门控、损失权重扫描；先分析配套结果再决定是否继续。
这个结构继承残差、跨尺度连接、多核卷积和门控，未确认算法原创性。当前贡献目标是验证有针对性的适配是否有效，而不是先承诺可以发论文。

6. 训练、评估与完成标准
正式方案：3组、seed42、300轮、每组每轮30次更新、有效batch64；后两阶段微调，原学习率/损失/增强/余弦日程不变。
新A4000 16GB上共同标定微批量，梯度累积保持有效batch；验证/测试batch64。缓存只含冻结前缀，训练特征每次重新计算。
预检：官方encoder特征一致、共享初始化一致、起始logits一致、模块梯度有效、冻结权重不变、实际停止/恢复、验证输出重放。
正式训练完成后固定3组验证macro Dice最佳权重，核对重放，再对全部867张筛查测试子集评估。测试不参与优化、阈值、轮次选择。
这批测试此前已曝光；不能把新结果称为全新的独立确认，也未证明患者独立。
报告macro/pooled Dice、IoU、precision/recall、Prec@、阈值Jaccard、boundary F1、HD95、逐图指标、严重漏分和面积分组。
实用复核门槛事先固定为：开发集anchor比两对照均至少高0.20个百分点，严重漏分不多于control，pooled Dice不低超过0.10个百分点。这不是显著性检验，也不是用测试选模型。
当前测试成绩只有在results/DONE.json且test_evaluated=true之后才算完成。

7. 产物位置
调研：/home/waas/mambavision_stage22/research/module_design_20261111/
新实验：/home/waas/mambavision_stage22/source/referring_image_segmentation/stage29_context_fusion/
fusion.py：ResidualFusion / FusionDecoder
train.py：三组共享训练与容量、梯度、恢复核查
evaluate.py：冻结权重、重放验证、测试与逐图配对
run_pipeline.py：预检 -> 正式训练 -> 验证+测试，失败立即停止
PROTOCOL.json：训练前固定的假设、对照、指标和判定条件
所有原Stage22—28源文件、权重、结果保持原样。
