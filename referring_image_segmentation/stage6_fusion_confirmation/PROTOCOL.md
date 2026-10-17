# 第六轮：Concat 融合与容量对照的配对复验

冻结日期：2026-10-17。用户在下一步建议后明确要求“继续”。授权范围为三个模型在 seeds 43/44/45 的配对复验，以及后续图文数据与评价准备。

## 问题与预定矩阵

第五轮 seed42 的动态门控未超过静态残差或原 Concat，且未减少新增错误。当前检验的是原 Concat 的收益能否超过训练波动和新增可训练容量。

| 模型 | Seeds | 执行方式 |
|---|---|---|
| DINOv3 + Swin | 43/44/45 | 校验通过后只读复用第三轮三个已完成训练 |
| DINOv3 双 Adapter + Concat + Swin | 43/44/45 | 新训练三个 seed |
| DINOv3 + VMamba + Concat + Swin | 43/44/45 | 新训练三个 seed |

共九个模型与 seed 组合，六个新增训练。seed42仅单列为探索背景，不计入三 seed 主汇总。43/44/45此前用于 Decoder 选择，不能称为完全未接触的确认集；此轮检验训练随机性，不是独立泛化。

本轮没有同 seed 的 VMamba 单分支训练，不使用 seed42 的 VMamba 结果构造跨 seed 的概率平均或互补性对照。不会自动追加门控、Cross-Attention、新 loss、文本训练或额外 seeds。

## 复用门槛

核对第三轮源文件及评价定义 SHA256、冻结 DINO 权重、同 seed 的完整 Probe/Adapter/Decoder 初始 hash、实际数据 manifest、增强生成函数、全部训练超参数、checkpoint 选择历史以及完成校验。当前 GPU 对三个 seed 的三种 checkpoint 重新前向并核对完整区域指标与概率；任何不一致均停止复用并报告原因，不挑选较好的旧结果。

历史文件只读。新结果、审计和独立重算的旧模型指标写入本目录。编码器哈希/训练配置和历史输入哈希在开始和结束时验证。

## 完全固定的训练与评估

ISIC2018旧1886/808开发划分；256×256；ImageNet归一化；原图 bilinear、mask nearest；保留原灰度训练标签；H/V翻转和0～360°旋转各p=.5；epoch几何/顺序seed=run_seed+epoch×100003。

冻结DINOv3-S/16和VMamba-T且永久eval。Adapter每尺度64通道；Swin window8/shift4/4heads/每层两块，dropout/drop-path均为0。双Adapter与Concat直接复用第4/5轮类，不改变结构。容量组638305、双编码器Concat632161个可训练参数，相差0.972%；只近似匹配可训练容量，不匹配总参数或计算量。

每组300epochs/9000updates，batch64并保留尾batch；AdamW lr=.001、weight_decay=.01、betas=.9/.999、eps=1e-8；Cosine T_max300、eta_min1e-5；BCEWithLogits+逐图softDice（smooth1）；BF16前向/FP32 loss、clip1；无warmup、early stopping、梯度累积。不同seed进程独立，同seed两个新模型共享每轮新增强后的冻结特征，参数及优化器独立。

主checkpoint为最高验证mean Dice、精确平局取较早轮；最低样本加权loss和epoch300为次要结果。全部保存权重、概率、二值mask、逐图指标与完整指标。阈值固定.5，不依照结果切换选模规则。cudnn deterministic=True/benchmark=False，不声称全算子逐位确定性。

## 预定分析

主比较：Concat−DINO、Concat−双Adapter容量对照；辅助比较：双Adapter−DINO。逐seed报告差值，三seed报告均值±样本标准差(ddof1)。不将3×808条记录当成2424个独立样本，不把逐图bootstrap区间当作跨seed显著性。

所有checkpoint均报告macro/pooled Dice、IoU、Precision/Recall、Prec@.5/.6/.7/.8/.9、Boundary F1、HD95px/对角线归一化和ISIC Threshold Jaccard。报告相对同seed DINO的纠正/新增/净减少错误，原错误修复率、原正确像素破坏率，以及边界/前景内部/背景、原前景概率区间、病灶面积和DINO表现分组。修复率与破坏率分母不同，不能直接相减。

在同机实测完整推理延迟/吞吐、可训练与总参数及训练显存。所有三seed结果及辅助选模规则一并报告，收益与边界/错误/效率共同解释。不据这次开发结果自动宣称VMamba独特贡献或独立泛化；后续独立数据需先做内容/患者/历史使用审计。

## 运行与完成条件

监督程序依次进行预检、真实数据两轮中断恢复及完整评估预检，然后正式运行三个seed。当前GPU为24GB RTX4090；保持batch64，单seed进程共享冻结特征，按实测峰值评估资源，默认串行seed避免缓存竞争。

每个seed完成后立即生成该seed全指标；全部结束自动汇总配对比较和学习曲线。只有六个新训练、九个模型组合/27组checkpoint全指标、选模/数据/源文件/历史文件校验通过才写complete。断点包含优化器、scheduler、RNG、选模状态与历史，可从监督入口恢复。
