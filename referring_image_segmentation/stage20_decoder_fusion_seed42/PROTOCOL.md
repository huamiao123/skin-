# MambaVision-S 解码器融合实验：第一组

## 研究问题

使用同一个冻结的 ImageNet 预训练 NVIDIA MambaVision-S 和已有 Swin Pixel Decoder，检验在 32×32 跳接处筛选编码器特征是否比原来的直接相加更有效。单独对照上下文门控和坐标注意力，避免一次叠加多个不同论文的模块。

| 组别 | 输入特征 | 32×32 跳接 | 其余结构 |
| --- | --- | --- | --- |
| paired_baseline | MambaVision-S 四尺度 | 上采样特征 + 同尺度特征 | 历史 Adapter、Swin Pixel Decoder |
| gate_only | 同左 | HEDS 官方 context gate，关闭 coordinate attention | 同左 |
| axis_bridge | 同左 | HEDS 官方 AxisBridge：context gate + coordinate attention | 同左 |

这不是完整 HEDS-Net 复现：不引入 HVST、深监督、额外数据或额外损失。编码器、其余跳接、分类头和四尺度 Adapter 均未改变。三组公共参数相同初始化；两个桥接组的门控也相同初始化，坐标注意力是二者唯一架构差异。各组参数量见 `preflight.json`。因为桥接组参数稍多，也不能将结果视作严格等参数消融。

HEDS 源码固定在 commit `a9287da787957b6376a9c403a56ad6693cd04191`；本地独立研究用源码为 `references/axis_bridge_source.py`，SHA256 已在 `preflight.json` 中核验。原始模块的布局判断对 NCHW 32×32×64 会错误换轴；本实验把张量显式变为 NHWC，再转换回来。只在尺寸不含糊的 32×32 跳接使用。预检使用手算公式核对输出。

## 数据和训练

- ISIC2018 历史划分：1886 张训练、808 张开发验证；逐文件 SHA256 与历史清单一致。验证集用于选模，不是独立测试集。
- 图像 256×256，ImageNet 标准化；原代码的水平/垂直翻转、旋转及 mask 最近邻插值，数据增强随机序列为 `42 + epoch * 100003`。
- 一个 seed：42；三组分别训练 300 完整 epochs；有效 batch 64，每轮 30 次更新；不足 batch 的尾样本不丢弃。
- AdamW，lr 0.001，weight decay 0.01，betas (0.9, 0.999)，eps 1e-8；CosineAnnealingLR，T_max=300，eta_min=1e-5，按 epoch 更新，不 warmup、不早停。
- 历史 BCEWithLogits + per-image soft Dice，平滑项 1；梯度裁剪 1.0。
- RTX 4090：Encoder 和可训练模块使用 BF16 autocast，loss 在 FP32 计算。模型参数和优化器维持 FP32。编码器和数据依次共享，三组严格配对。
- 最大验证 macro Dice 选模，精确并列取较早 epoch；同时记录 minimum validation loss 和最后一轮。

显存约 48 GiB 不应被用来擅自扩大训练有效 batch：这会改变优化协议。预检尝试 batch 64；其余显存用于图像及验证特征缓存，是否由计算而非容量限制取决于实际吞吐。

## 对照、评估和限制

主比较为 `gate_only - paired_baseline` 和 `axis_bridge - paired_baseline`；再比较 `axis_bridge - gate_only`。与旧 stage17 仅作为历史背景：本次 4090 本地初始哈希匹配 stage14，但不同于 stage17 的记录，且设备不同。不能声称这是一致初始化的 stage17 重跑。

指标复用项目 `evaluation.py` / `stage1_visual.analyze.metrics`：macro/pooled Dice、IoU、Precision、Recall、Boundary F1、HD95、Prec@0.5～0.9、ISIC threshold Jaccard；保存逐图和最佳模型逐像素结果，统计纠正及新增错误和逐图配对区间。单 seed、反复用于选模的开发验证集不能证明统计显著性或外部泛化。

## 运行

依赖当前项目环境与 `stage14` 的冻结官方权重及 CUDA scan：

```
python preflight.py
python capacity_preflight.py
python train.py --smoke --stop-after-epoch 1
python train.py --smoke
python train.py
python evaluate.py
python benchmark.py
python analyze.py
```

`train.py` 每个 epoch 原子保存所有模型、优化器、调度器和 RNG 状态；同样命令自动恢复。单独的 `smoke/` 权重绝不会用于正式训练。`smoke_fp32_2080ti/` 是换 GPU 以前的隔离试跑，不计入本次 BF16 结果。
