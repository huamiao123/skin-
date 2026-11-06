# MambaVision-S 编码器内部区域扫描：ISIC2018 seed 42

## 研究问题与固定对照

在冻结 ImageNet 预训练 MambaVision-S 原始权重的基础上，把 HiRAM 的区域预测、分组和置信度排序思想引入一个编码器 MambaVisionMixer 的 SSM 扫描。不是直接移植 HiRAM 的双路模块，也不是使用三个独立随机初始化 SSM。仅改 `net.model.levels[2].blocks[0].mixer`（第3阶段首个 Mixer，16×16 网格、384通道）；其余 CNN/Mamba/Attention、原 Swin Decoder 和输出头保留。

- **baseline**：stage17 同机重跑的 MambaVision-S + Swin（冻结 Encoder），只读使用，主参考。
- **ordered_prior**：新增相同区域头和先验，SSM 按窗口内原空间顺序扫描有效位置，填充位置单独扫描。
- **region_scan**：相同区域头、先验、预训练 SSM 权重、区域损失和填充处理；将有效位置按预测背景/边界/内部分类，组内按该类别置信度降序后独立扫描并恢复原位置。

主比较预先固定为 `region_scan - ordered_prior`，研究分组、排序和区域序列状态重置的共同作用；`ordered_prior - baseline` 还包含加入先验/区域监督以及窗口填充扫描处理的差异，不能归结为单一因素。两个新增模型的全部初始参数相同，差异仅在实际扫描序列与训练期间硬路由（分组版使用 Gumbel、验证argmax）。只训练上述两个模型、seed42，不增加其他模型/种子/文本。

## Mixer 算法

保留原 Mixer 的 in_proj、conv1d_x/z、x_proj、dt_proj、A_log、D、out_proj 预训练权重和结构，保持 x/z 卷积在分组前执行。对输入窗口特征训练相同的三类线性区域头，得到 `p_i=softmax(router(h_i))`。每类学习一个初始为零的 SSM 输出先验 `E_r`，有效位置的 SSM 输出参数修改为 `C'_i=C_i+Σ_r p_i(r) E_r`；padding 位置的先验为0。两组 E、区域头和 Decoder 可训练；其余预训练参数及所有 Encoder BN 统计冻结，Encoder 递归保持 eval。

按匹配的有效/填充集合提取独立序列：对有效序列，ordered_prior 保持原序；region_scan 按三个预测类别分组、类别内按置信降序；对填充序列，两组均保持原序并使用同一个原始 SSM。各序列的 SSM 状态独立重置，使用原权重；扫描结果散回窗口原位置，再执行原 concat(z) 和 out_proj。任意窗口、任意样本互不混合。为防止可变长度补齐干扰有效 token，补齐仅在序列尾、输出只保留真实位置。若全部关闭修改、回退原始 Mixer，前向必须与官方模型一致。

这两个变体都会将填充位置与有效位置分开扫描，因此与官方原模型也有这一差别；所有填充位置不参与区域损失，不作病灶背景标签。类别硬选择及排序不反传，但可微区域损失和 `p→C'` 连续先验为区域头提供梯度。验证和推理使用预测标签、argmax，不输入 GT。三个类别共享原 SSM，不是原文三个独立 SSM 的完整复现。

## 数据、标签、训练与评价

复用原始 ISIC2018 1886/808 开发集、图像清单与预处理/增强、256×256、seed42+epoch×100003，冻结原始预训练权重（含BN缓冲区）；沿用 Adapter 与 Swin 初始化。病灶 mask 阈值0.5，经 width2 方形5×5膨胀/腐蚀得到边界；内部为病灶中非边界，背景为其余位置。三个互斥二值区域使用 area pooling 到16×16，得到和为1的软区域标签，避免稀疏边缘被单点插值丢失。区域头仅在16×16有效位置监督，逐类加权交叉熵为各类 `-Σ(t_r log p_r)/(Σt_r+1e-8)` 的均值，系数1；主损失为历史 BCEWithLogits+逐图 SoftDice(smooth1)。aux 的分母对完整有效 batch 计算，分块时只累加分子，避免改变整体loss。

AdamW lr=.001 wd=.01 betas(.9,.999) eps1e-8；CosineAnnealingLR T_max300、eta_min1e-5、无warmup；有效batch64、尾批30、30更新/epoch、300 epochs/9000更新、不早停，BF16 autocast、FP32 loss、训练参数联合clip1。小批累积允许用于硬件约束，要求保持有效 batch、样本顺序及损失聚合；记录真实 microbatch。验证每个 epoch 都要重新计算对应 Encoder，不能缓存会改变的区域特征；训练时不能对整个 Encoder 使用 no_grad，否则新区域头不能学习。训练前实际预检10GB GPU，若无法保持有效批语义则停止并如实报告。

主选模为验证逐图平均 Macro Dice 最大（并列取早），另存 best_loss 和 epoch300；输出全部共同评价指标、808逐图、预测图、像素纠错、边界分析、参数、完整推理延迟。开发验证集非独立测试，患者级独立性未核验；逐图 bootstrap 只描述所选验证图，不代表跨种子或患者独立的不确定性。

## 预检与执行

严格核对官方权重及原始 backbone 参数和缓冲区哈希、阶段/窗口形状、4窗口有效/填充掩码与重组、训练/推理无GT泄露、退回原Mixer时输出一致、两种模式参数与初始化一致、样本之间无串扰、序列分组后能完整恢复、所有可训练参数梯度有限、原预训练参数BN冻结。实测真实GPU显存并完成独立两轮smoke及断点/RNG/优化器恢复；正式训练后自动验证全部9000步、选模、预测、指标与源文件摘要。新内容独立保存，不修改历史实验、不自动推送GitHub。
