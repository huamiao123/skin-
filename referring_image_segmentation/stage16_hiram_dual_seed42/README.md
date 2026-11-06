# HiRAM 迁移实验

具体结构、对照、数据及选模见 [PROTOCOL.md](PROTOCOL.md)。`vendor_hiram/` 包含官方固定版本源代码及 CC BY-NC 4.0 许可；不自动上传代码、权重或数据。

在现有本地 PyTorch/CUDA、timm、VMamba 与 Mamba runtime 环境中：

```bash
python preflight.py
python capacity_preflight.py
python -u launch.py
```

监督器按预检、隔离 smoke 与恢复、三个 seed42 模型训练、验证与分析顺序执行，进度见 `run_status.json` 和 `results/status.json`，日志见 `train.log`。中断后重新运行 `python -u launch.py` 可从 `results/latest.pth` 恢复。历史阶段只读。正式结果完成后保存于 `analysis/REPORT.md`、`analysis/summary.json` 和 `analysis/per_image.csv`。
