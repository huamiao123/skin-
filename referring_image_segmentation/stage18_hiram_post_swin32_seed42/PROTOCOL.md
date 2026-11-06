# HiRAM 在 Swin32 之后：MambaVision-S、ISIC2018、seed 42

## 要验证的问题和对照

上一轮 HiRAM 在 32×32 跳接相加及 Swin32 之前，单种子仅取得很小的平均 Dice 差异。本轮同样在 32×32，改为在同一层 Swin 之后使用 HiRAM，并增加后置区域监督对照。历史完整结果只读复用，不重训：

- 历史共享基线：MambaVision-S + Swin。
- 历史 pre-region：在 Swin32 前，对上采样和跳接特征做区域监督。
- 历史 pre-HiRAM：在 Swin32 前，对上采样和跳接特征做 HiRAM 和区域监督。
- 本轮 post-region：在 Swin32 后，对 Swin 融合结果和原跳接特征做区域监督，不改变输出特征。
- 本轮 post-HiRAM：在 Swin32 后，对 Swin 融合结果和原跳接特征做 HiRAM 和区域监督。

主比较预先固定为 post-HiRAM 减 pre-HiRAM；辅比 post-HiRAM 减 post-region、post-region 减 pre-region、post-HiRAM 减共享基线。改变位置必然改变第一路输入的语义，区域监督的第一路输入也同步变化；这里比较的是这两种具体接法，不能仅凭结果断言是纯粹位置的因果作用。只训练两组新增模型，seed=42，无文本或其他模型扫参。原论文的 SAM/CNN 多阶段双分支并未复现。

## 数学接口

设 `c=upsample(Swin16)`，`s=adapter32(MambaVision)`：

```text
x = Swin32(c + s)
x1, s1 = HiRAM(x, s)
x_out = x1 + (s1 - s)
Swin64(upsample(x_out) + adapter64(MambaVision)) → mask
```

HiRAM 现有两路输出各含残差，这种加法只增加两路修正量，不重复直接加上原跳接特征；当 HiRAM 两路修正为零，输出应与原基线逐位相同。区域监督对照直接对 `(x,s)` 预测区域，输出 `x` 保持不变。HiRAM 与区域头的权重初始化逐项匹配 stage17，使用原 stage16 的官方代码与 batched 计算；BFloat16 下计算 `s1-s` 可能存在消减误差，作为实现局限记录。

## 固定协议

延续 stage17：ISIC2018 同一 1886/808 划分与原数据加载、增强、归一化、256×256、seed42、冻结 ImageNet 预训练 MambaVision-S；300 epochs、30 steps/epoch、有效 batch64、AdamW lr=.001 wd=.01、CosineAnnealingLR T_max=300 eta_min=1e-5、无 warmup 或早停、BCE + Soft Dice、BF16、FP32 loss、裁剪1。训练 mask >=0.5 提取 width2 边界及内部，经 bilinear 缩到32；区域 KL 与上一轮相同，每路目标按完整有效批次归一化、权重1。推理不使用 GT。若物理批量受限，在不改变每轮30个更新及有效批量的前提下进行梯度累积，并记录。

在808张开发验证图按 Macro Dice 最佳轮选模型（精确并列取早），同时留 best_loss 和 last。统计逐图平均与 pooled Dice/IoU、Precision、Recall、Prec@IoU、Threshold Jaccard、Boundary F1、HD95，分别保存逐图、预测概率、二值 mask 和配对像素纠错；GPU batch1 计时整个模型。该开发集被反复用于选模，不是独立测试集，也未验证患者级独立性。结论限于单 seed，bootstrap 只对本验证集图片重采样。

## 运行前审计

检查数据身份、冻结权重 SHA 和参数缓冲区、初始化与上一轮逐项一致、区域对照原始输出精确不变、HiRAM 零修正时精确还原基线、官方单图运算及独立 batching、有效梯度、真实显存、断点恢复及验证复算。所有源码和历史对照结果签名后训练。所有新增内容只存本目录，不自动向 GitHub 推送。
