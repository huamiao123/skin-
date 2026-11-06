# MambaVision-S + Decoder 内 HiRAM：ISIC2018 seed42

## 接入点及依据

将 HiRAM 加到已有 SwinPixelDecoder 的 **32×32 跳接融合处**。32×32 的两路输入分别是 Decoder 经过 8→16 后上采样的粗特征、以及同一 MambaVision Encoder 经 Adapter 输出的 32×32 跳接特征。两路均为 64 通道，经过官方 HiRAM 两分支区域分解、排序、区域 SSM、全局双向扫描、交叉门控和残差后相加，送入原来的 32×32 Swin 层，随后沿用 64×64 层与输出头。

选择 32×32 而非 16×16：它既已聚合更深层的解码特征，也保留更高的空间分辨率；对现有 1886/808 张未增强 Mask，原论文 width=2 形态学边界经 bilinear 缩小到 32×32 后消失数分别为 0/0，而 16×16 分别为 11/1。此统计不保证性能提高。32×32 的扫描开销更大，必须预检显存及吞吐。

这不是完整的论文复现：原论文采用 SAM/CNN 双分支与 SAM RAE 解码器，本次为 MambaVision 单 Encoder、其解码特征与多尺度跳接双输入、原 Swin 解码器。位置在 Decoder 内，不改动预训练 Encoder 的 Mamba 或 Attention。

## 固定对照

全部从头训练独立 Probe，使用相同预训练 MambaVision-S、冻结及 eval、相同 Decoder 初始化、数据顺序和 seed42：

- A `paired_baseline`：原 MambaVision-S + Adapter + Swin Decoder。
- B `region_only`：A + 32×32 处两路区域预测与区域辅助损失；区域预测不改变分割特征。
- C `hiram`：B + 双输入 HiRAM 特征传播和跨路门控后再做原 Decoder 加和。

历史 stage14 的基线 Macro Dice 88.3598% 作为背景；为配对本机 GPU 和初始化，再训练 A。先比较 A/B/C，不混入 DINOv3+VMamba 结构或改变 Encoder。B/C 仍混合 SSM 与新增容量，需后续容量对照才能归因。只跑 seed42，不引入文本。

## 训练、标签与选模

完全沿用 stage14：ISIC2018 1886/808 旧开发划分、256×256、原 ImageNet 归一化与增强顺序、种子 42+epoch×100003；300 epoch、每 epoch 30 更新、有效 batch64（尾批30）、AdamW lr=.001 wd=.01 betas=.9/.999 eps=1e-8、CosineAnnealingLR T_max300 eta_min1e-5、无 warmup/early stopping、BF16 autocast、FP32 loss、clip1。主损失为历史 BCEWithLogits + per-image soft Dice smooth1。

B/C 加同权重1的原论文 region loss：训练 mask 使用 >=.5 二值化，width2 膨胀/腐蚀得到边界和内部，bilinear 到32×32，两路分别用论文的 KL 正则并按整个有效 batch 的标签质量归一化，推理不使用 GT。原始灰度 Mask 仍按历史方式用于主损失。

预检前保留 microbatch8/16/32/64 中选择最高可用批量的空间；若不能原生 batch64，则分块计算并以样本数累积分割损失、以完整有效 batch 的标签质量为分母累积区域损失，每64样本一步，末尾30样本一步，保持9000更新；记录实际分块与浮点差异。验证使用固定的预注册 batch；原实现训练 Gumbel 路由，评估固定 argmax，与 stage16 相同。

根据验证 Macro Dice 最大选 best_dice，精确并列取早；另保存 best_loss 与 epoch300。复用 evaluation.py 和历史 Boundary F1/HD95 规则。开发验证集非独立测试、患者级独立性未核验。

## 预检、追溯与完成

核验官方固定 revision/权重 SHA、无随机初始化、完整冻结参数和 BN、与原 Swin 的 baseline 前向一致、三组初始公共参数和区域头一致、官方 HiRAM 单图等价、不同图像的互不影响、真实 mask/数据 SHA、32×32 形态边界非空数、有限梯度、实际峰值显存、有效 batch 与两轮 smoke 恢复。保存训练状态、各组最佳预测、逐图/逐像素误差、参数和完整延迟；只有300轮、9000步/组、指标复算和完成审计全部通过才标记complete。历史目录只读，不自动推送远程。
