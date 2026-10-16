# 第三轮：DINOv3 Decoder 的选模与三个新 seeds 确认

冻结日期：2026-10-16。用户在第二轮结果后要求继续实验。

## 依据与问题

第二轮 DINOv3+Swin 的 Boundary F1 改善约 7.04 个百分点，但最小验证 loss 权重的区域指标有取舍；后期 Dice 和 loss 趋势不同。VMamba+Swin 未显示清晰优势。本轮限定为 DINOv3+Light 与 DINOv3+Swin 的确认，不扩展编码器、文本或融合模块。

要回答：统一以 mean Dice 选模后，Swin 的收益是否在三个新 seeds 中方向一致？按 loss 选模与按 Dice 选模会怎样改变区域/边界结果？

## 预先固定的实验矩阵

- Seeds：43、44、45；此前 seed 42 是探索性筛选证据，不混入本轮均值。
- 每个 seed 下两个 Decoder 分别训练 300 epochs / 9000 updates，共 6 次训练。
- 主 checkpoint：验证集 foreground mean Dice 最高；精确相同取较早 epoch。
- 次要 checkpoint：样本加权验证 loss 最低；精确相同取较早 epoch。
- 固定预算 checkpoint：第 300 轮。保存上述三种权重/概率/二值图/逐图指标，所有模型统一，不因某个模型哪个指标好而改变规则。
- 主读数是三个 seeds 的 mean Dice 均值±样本标准差（ddof=1）；同表报告 pooled Dice、mIoU、oIoU、Precision/Recall、Prec@X、Threshold Jaccard、Boundary F1、HD95。同时列出各 seed 的 Swin−Light 配对差，不能把 3×808 图视为独立训练重复。
- 不要求/不预设 Swin 必须获胜。只有三个 seeds，不能据此声称独立测试泛化或稳定的统计显著性。

## 不变的模型和训练配置

完全复用阶段一 FrozenEncoder、LightDecoder 和阶段二 SwinPixelDecoder 的结构，64 通道、Swin window=8/shift=4/4 heads/每层两块。参数量差异继续披露。

固定 ISIC2018 1886/808 旧开发划分、256×256、ImageNet normalization、冻结编码器且 eval、同一 DINOv3 权重、相同 Adapter 结构和二值输出头。训练 loss 为 BCEWithLogits + per-image soft Dice（smooth=1），AdamW lr=.001/weight_decay=.01/betas=.9,.999/eps=1e-8，CosineAnnealingLR T_max=300/eta_min=1e-5，每次更新 batch=64、保留末尾 batch、无梯度累积、无 warmup、无 early stopping、BF16 autocast/FP32 loss/clip=1。

每个 seed 同时控制 Adapter、Decoder 和数据顺序/几何增强。每个 epoch 的数据 Generator seed=seed+epoch×100003；seed=42 时必须复现旧几何序列和模型初始张量。对应 Light/Swin 的 Adapter 与输出头初始张量完全一致；不同 seed 应产生不同的初始张量。

## 配对训练与复现

每个 seed 一个独立进程，持有同一个冻结 Encoder，缓存同一份逐 epoch 新增强后特征供两个独立 Probe 使用。两个 Probe 的参数和优化器互不共享，每个更新读取同一批 feature/target，分别反向并更新。共享的冻结前向只节省重复计算，不对缓存特征施加额外空间变换。

正式运行最多三个 seed 进程并行；预检记录峰值显存。CUDA 数值设置沿用之前的 cudnn.deterministic=True / benchmark=False，未启用全算子确定性；上采样反向可能非确定，固定 seed 不保证逐位相等。

在首次真实数据反向时检查所有训练参数有限梯度、两个优化器参数互斥、编码器无梯度；定期核对编码器 hash，最终核对完整数据 manifest。断点保留两个 Probe、两个优化器、scheduler、三个 checkpoint 选择状态、历史和 RNG。

## 数据解释范围

该开发集已用于前期设计选择，也用于本轮 checkpoint 选择。结果是训练随机性和选模敏感性的开发集确认，仍不是独立测试。患者级身份不完整。所有评估沿用项目 EVALUATION.md 的 256 网格与阈值 .5；不调阈值，不改变数据划分。

## 完成条件

预检通过后运行 launch.py。六次训练与三种 checkpoint 的完整导出、独立概率重算、三 seed 汇总、学习曲线和完成校验全部通过后才写入 run_status.json: complete。历史阶段文件保留原样。
