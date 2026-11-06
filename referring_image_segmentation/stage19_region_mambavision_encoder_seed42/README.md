# MambaVision 编码器内部区域扫描：seed 42

实验目标、对照、区域标签、硬路由、窗口填充和局限见 [PROTOCOL.md](PROTOCOL.md)。两个新模型均加载并冻结相同的官方 MambaVision-S 原始参数，只在第3阶段首个 Mixer 内训练区域头及先验，训练同一 Swin Decoder。原始 stage17 基线为只读对照。

依赖项目已有的 ISIC2018 数据与清单、stage14 官方预训练权重/runtime、stage17 的 Encoder 加载器和基线预测，以及原通用评估代码。当前设备需支持 CUDA 和 BF16。程序不会上传原始数据、模型或任何结果。

从本目录执行：

```bash
python -u preflight.py
python -u capacity_preflight.py
python -u launch.py
```

启动入口先做两轮独立 smoke 和断点/RNG/优化器恢复及评价复算，随后固定源代码和历史数据摘要，再正式训练 300 轮并自动评价、测速、分析和审计。训练后每轮保存轻量 trainable-only 检查点（原始 Encoder 权重从 stage14 严格加载）；可重新执行 `python -u launch.py` 恢复。日志和状态在 `train.log`、`results/status.json`、`run_status.json`。完成结果见 `analysis/REPORT.md`、`analysis/summary.json` 和 `analysis/per_image.csv`。

逐图平均 Macro Dice 和汇总像素 Pooled Dice 分开报告；808 张旧开发验证集不是独立测试集，结果仅限 seed 42，不能自动推断跨种子稳定性。
