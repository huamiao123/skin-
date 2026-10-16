# 第二轮：Swin Pixel Decoder 对照

日期：2026-10-16。研究问题：在冻结视觉编码器与相同 Adapter 下，替换轻量 FPN 内的空间处理块是否改善分割和边界表现？

## 实验矩阵

| 编码器 | Light Decoder | Swin Pixel Decoder |
|---|---|---|
| DINOv3-S/16 | 复用 stage1_visual 已完成结果 | 本轮 300 epochs |
| VMamba-T | 复用 stage1_visual 已完成结果 | 本轮 300 epochs |

两组新实验 seed=42。候选方案经本轮筛选后再决定是否进行至少三个 seeds 的确认。未引入文本、双分支可训练融合或新损失。

## 唯一结构变量

保持四层 64 通道 Adapter、64/32/16/8 网格、双线性上采样加法融合和 1×1 输出头。每层原 Conv/GN/GELU 块替换为两个标准 Swin block（W-MSA、SW-MSA）及输出 LayerNorm。

- 采用已安装 timm 1.0.22 的 SwinTransformerBlock。
- 通道 64，4 heads，window=8，交替 shift=0/4，MLP ratio=4，attention/projection dropout 和 drop path 均为 0。
- 最低分辨率 8×8 只有一个窗口，按实现自动禁用该层 shift。
- 新 Decoder 从头训练；两个编码器对应的 Decoder 初始张量完全一致。Adapter 和线性输出头初始权重与相应 Light 实验一致。
- 参数量不同，训练中记录 Adapter/Decoder/Encoder 参数数。因此收益如有出现应解释为这一 Decoder 配置的收益，不能宣称是参数量匹配后的纯注意力机制收益。
- 这是基于标准 Swin block 的受控 FPN Pixel Decoder，不宣称复现完整 Swin-UNet；其功能是生成像素特征 V。

## 固定训练和评估

其余设置沿用 `../stage1_visual/PROTOCOL.md`：ISIC2018 1886/808 开发划分、256 输入、完全冻结且 eval 的编码器、相同预训练权重、相同逐 epoch 数据顺序与增强、batch=64（包括最后部分 batch）、无梯度累积、每 epoch 30 次更新、300 epochs/9000 updates、AdamW lr=1e-3/weight_decay=.01、Cosine T_max=300/eta_min=1e-5、BCE+逐图 soft Dice、BF16、FP32 loss、clip=1。

Encoder/evaluation forward batch 均沿用 64。训练前预检真实 batch=64 的显存和梯度。每 epoch 重新增强并编码训练图像，固定验证特征缓存。训练可以根据 GPU 情况同时或依次执行；每个进程独立维护数据和随机数。

仍按样本加权验证 loss 最低选择最佳权重，不因新增报告指标改变选模。统一报告 `../EVALUATION.md` 的全部指标。分别计算每个编码器 Swin−Light 的逐图配对差及探索性 95% bootstrap 区间，并保留等权概率融合诊断。

## 运行和完成条件

`validate.py` 先检查指标边界案例、历史结果重算一致、Adapter/readout 初值、冻结权重、所有可训练参数的有限梯度。`train.py --smoke` 为独立 2 epoch 预检，正式训练不加载 smoke 权重。正式训练由 `launch.py` 监督并自动分析；`run_status.json` 为 complete 仅表示两个 300 epoch 训练、完整指标导出与汇总检查全部通过。
