# 第五轮：容量来源与选择性纠错

冻结日期：2026-10-17。用户明确授权执行先前讨论的P0分析和三个新增训练。主问题：VMamba能否帮助纠正DINO错误，同时减少对原有正确预测的破坏？

## 依据、范围与顺序

第四轮seed42、同一Swin Decoder下，DINO/Concat/概率平均的mean Dice为89.185%/89.549%/89.449%。Concat纠正679485个DINO错误像素，同时新增614397个错误，其中286241个来自两个单模型原本都正确的位置。全量纠错率26.24%，原正确像素破坏率1.22%。这些结果只提出假设，不证明错误源于VMamba或已具有稳定收益。

先从第四轮不可变概率文件完成P0：按前景概率、单分支正确性、边界/内部/背景、病灶面积分析修复与破坏。再执行以下三组，最后自动完整评估。没有文本、Cross-Attention、额外seed或新loss。

| 新训练 | 结构 | 对照问题 |
|---|---|---|
| capacity_concat | 同一冻结DINO四层特征→两套独立Adapter→Concat+1×1→原Swin | 接近的可训练容量能否解释第四轮Concat收益 |
| static_residual | D + sigmoid(beta_i) * phi(M)，每尺度一个可学习常数 | 保留DINO主路径的残差参数化是否有效 |
| dynamic_gate | D + sigmoid(Conv1×1([D,M])) * phi(M)，每位置一个权重、通道共享 | 输入相关的空间选择是否优于静态残差 |

复用第四轮同seed42的DINO+Swin、VMamba+Swin、Concat+Swin和固定0.5/0.5概率平均作为参照，逐个checkpoint规则匹配，不混用第三轮跨seed均值。编码器、数据manifest和Decoder源码必须校验一致。

## 初始化与模型约束

- 冻结DINOv3-S/16、VMamba-T且永久eval。Adapter沿用Conv1×1/GN/GELU，各尺度64通道，网格64/32/16/8。Swin Decoder完全复用第四轮。
- 各自Adapter与相应第四轮初始张量相同，所有新模型Decoder初值相同。全部Adapter/Decoder从相同规则随机初始化，未加载已有分割checkpoint。
- 容量对照两套Adapter使用同一seed起点但参数存储独立，Concat卷积两个半区不同。预检验证一次优化后两套Adapter实际分化。该组可训练参数638305，对第四轮Concat的632161高0.972%；不匹配冻结Encoder总参数或计算量。
- static/dynamic的两套Adapter、phi和Decoder初值完全相同。phi为带bias的64→64 Conv1×1，普通随机初始化。
- 两组gate logit初值为log(.1/.9)，dynamic的gate卷积权重为零、bias为该值。gate logits/sigmoid均用FP32，再将alpha转为特征dtype相乘，保证BF16环境下静态/动态初始输出一致。门控没有GT或单模型预测作为输入。
- static每尺度1个beta参数；dynamic每尺度128→1卷积，参数量只多512。第一步即检查phi、两套Adapter、Decoder、gate均有有限非零梯度。
- alpha接近零仅接近当前可训练DINO路径，不保证保留第四轮训练后DINO预测。门控数值也不直接代表正确率或分支贡献。

## 完全固定的训练口径

Seed42；ISIC2018旧1886/808开发划分；输入256×256；沿用ImageNet归一化、图像bilinear、mask nearest和原灰度训练标签；HFlip/VFlip/Rotation各p=.5，旋转0～360°，epoch采样seed=42+epoch×100003。

三组每轮共用同一份新增强图像、顺序和冻结特征。每epoch重新编码，容量对照仅使用DINO缓存。独立训练参数和优化器，不共享可训练状态。

每组300epochs/9000updates；batch64、保留末尾batch；AdamW lr=.001、weight_decay=.01、betas=.9/.999、eps=1e-8；Cosine T_max=300、eta_min=1e-5；BCEWithLogits+逐图softDice（smooth1），BF16 autocast/FP32损失、clip1；无warmup、early stopping、梯度累积。cudnn deterministic=True/benchmark=False，但不保证全算子逐位确定性。

主checkpoint最高验证mean Dice，精确相同取较早轮；最低样本加权验证loss和第300轮统一作为次要结果。保存三种完整预测和全指标。当前设备24GB RTX4090；实测记录显存/速度，不以填满显存为由改变训练口径。

## P0和结果分析的预定定义

- 概率桶：[0,.1)、[.1,.4)、[.4,.6)、[.6,.9)、[.9,1]，指DINO前景概率，非经过校准的置信度。概率统计对三种checkpoint均报告，不据结果重新划桶。
- 区域为互斥分区：GT侵蚀边界的欧氏距离≤2px；其余GT前景内部；其余背景。另报告全前景/全背景。每个分区同时报告占比、原DINO错误/正确数、纠正/新增错误、修复率、破坏率和净减少数。
- 单分支正确性分四类：都正确、DINO对VMamba错、DINO错VMamba对、都错误。分层计算融合错误率；尤其审计都正确区域的新增错误。
- 固定面积组<10%、10%～<30%、≥30%；固定DINO逐图Dice组<.8、.8～<.9、≥.9。报告分组mean/pooled指标，承认按baseline表现分组的选择效应。
- 模糊边界、毛发、反光等缺少标签，本轮不据概率或误差图自动命名此类病例。
- 修复率=纠正的DINO错误/全部DINO错误；破坏率=新增错误/全部DINO正确。两者分母不同，不能直接相减。减少新增错误必须连同修复率、mean/pooled Dice、Boundary F1、HD95解释；恒关闭门控不算机制成功。
- 主比较：dynamic−static、capacity−第四轮Concat、dynamic−第四轮Concat及概率平均，同时对DINO报告各组差异。逐图配对bootstrap2000次seed42为开发集探索性区间，不能证明跨seed显著性。
- 全部指标沿用evaluation.py：macro/pooled Dice/IoU/Precision/Recall、Prec@.5/.6/.7/.8/.9、Threshold Jaccard、Boundary F1、HD95像素/对角线。最终概率独立重算并核对训练选模记录。
- 对dynamic主checkpoint追加固定权重诊断：gate置零、每图每尺度空间均值gate、空间循环平移半个网格gate，均不再训练、不选新checkpoint、不用于主模型排名。用于判断是否使用了空间变化，不能单凭gate热图解释因果。记录每尺度gate分布。
- 同机重测DINO/Concat及三新模型完整Encoder+Adapter+Decoder的batch1延迟、batch64吞吐和显存（BF16，3预热10重复，不含磁盘/后处理）；容量对照只前向一次DINO，其余双编码器均全量前向。

## 完成条件

独立后台监督P0、初始化/梯度/效率预检、两轮真实数据中断恢复及完整分析预检，然后正式训练和最终分析。预检不初始化正式训练。全部完成且验证通过才写complete。旧阶段源码、协议、权重、预测和指标只读并保留hash。单seed开发集结论决定后续容量机制的配对seeds43/44/45确认及独立数据验证，不自动扩展研究矩阵。
