# HGM BMD 接入 MambaVision 的探索实验，seed 42

本协议在正式训练之前固定。只测试一个 HGM 双向 Mamba 解码器（BMD）的局部适配，不是完整 HGM，也没有区域边界判别 BDFM、图网络或论文的原始编码器。

* 冻结 ImageNet 预训练的 MambaVision-S；复用历史四尺度 Adapter、8/16/32 分辨率的 Swin 解码层；替换 64 分辨率的 Swin 层及输出头为 HGM 原版 BMD，经两层 1×1 投影对齐通道。BMD 内部保留 FSEMV2、BiPixel 和 BiWindow 的原始运算，不与已有 HiRAM 或 AXIS 叠加。
* 原 HGM 的 BF16 输入与本地 Mamba 1.2.2 不兼容，BMD 的 Mamba 层显式转为 FP32 运算，其余模型采用 BF16 autocast 和 FP32 loss。官方仓库依赖 Mamba 2.2.4，因此这是依赖适配版；原 BMD 的分块设置在 64/128 网格会回退成全序列扫描，未自行改为窗口扫描。官方 FSEMV2 有两组 forward 未使用的 BN/PReLU 参数，本实验保留但冻结它们。
* 比较基线：stage20 的同机 MambaVision-S + Swin；逐项核验共用层初始化、encoder 权重、数据、评价和训练协议。适配会改变 decoder 拓扑、归一化和参数量，所以无法将差异单独归因于 Mamba 扫描；注意算力与参数量不是严格匹配。
* 数据：历史 ISIC2018 1886 训练、808 开发验证图，seed 42、256×256；历史 ImageData 原样复用，验证集不能称为独立测试。最大验证逐图平均 Dice 选主 checkpoint，并列选更早 epoch；同时保留最低验证 loss 与末轮。
* AdamW、batch 64（因 BMD 的 BatchNorm 不通过梯度累积模拟大批量）、lr 0.001、weight decay 0.01、BCE + Soft Dice、梯度裁剪 1、cosine T_max 300、最低 lr 1e-5、300 epochs，30 updates/epoch，按历史数据增强和验证协议执行。
* 先执行真实数据预检、完整训练批量显存预检、独立两轮 smoke 与断点恢复检查；全部通过后才启动唯一正式训练。显存不够时不得静默缩小训练批量或改变 BN。中断后从包含模型、优化器、调度器、随机状态的 checkpoint 恢复。
* 正式导出全部历史评价指标、808 张逐图数据和区域分析；主要汇报逐图平均 Dice，并披露其余指标及逐像素汇总指标。所有结论仅限单种子开发集实验。

官方 HGM 源码 revision `49ae05afc9785dbbc5a303d39e63588b068e5328`；未找到上游明确许可证，代码快照仅供本机研究，不自动公开推送。
