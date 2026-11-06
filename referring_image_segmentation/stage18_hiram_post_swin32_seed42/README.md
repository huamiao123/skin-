# HiRAM 后置 Swin32：seed 42

本目录新增两个实验：后置区域监督与后置 HiRAM。完整方案、对照关系和限制见 [PROTOCOL.md](PROTOCOL.md)；旧基线和前置模型来自只读 stage17，不需要重复训练。实验不包含文本编码器或额外视觉主干。

依赖项目现存 ISIC2018 数据清单、stage14 固定的 NVIDIA 预训练 MambaVision-S 权重和运行时、stage16 的 HiRAM 官方源码及许可证、stage17 结果与项目通用评估代码。运行前确保 CUDA 与 BF16 可用，然后从本目录运行：

```bash
python -u preflight.py
python -u capacity_preflight.py
python -u launch.py
```

启动程序先进行两轮独立 smoke，并在第 1 轮结束后测试恢复；验证复算成功后固定源文件和旧结果摘要，再开始正式 300 轮训练及自动评价、速度测量、逐图分析和审计。训练每轮结束保存状态和检查点，可重新运行 `python -u launch.py` 恢复；状态和日志见 `run_status.json`、`results/status.json` 和 `train.log`。完成后可查看 `analysis/REPORT.md`、`analysis/summary.json` 和 `analysis/per_image.csv`。

不自动推送远程；图片和权重只留本地。各结果只对本开发验证划分和 seed 42 有直接意义。
