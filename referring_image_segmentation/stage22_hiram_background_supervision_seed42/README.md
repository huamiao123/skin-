# HiRAM 背景路由监督实验

本实验复用 stage17 的 MambaVision + HiRAM + Swin，三组在同机、相同初始化和 Gumbel 随机数下对照原损失、1.5 倍原损失与显式背景监督。详细公式、语义及限制见 [PROTOCOL.md](PROTOCOL.md)。

实验依赖 stage14 的官方预训练 backbone 与 Mamba 运行环境、stage16 的 HiRAM 实现和历史 ISIC2018 数据。训练前依次执行：

```bash
python preflight.py
python capacity_preflight.py
python -u run_pipeline.py
```

管线先验证两轮独立 smoke 与完整断点恢复，通过后运行三组各 300 轮、导出各 checkpoint 的指标、审计两路 Router、测速、逐图统计和最终验证。中断后使用相同环境重启管线从 `results/latest.pth` 恢复。`run_status.json`、`results/status.json` 和 `pipeline.log` 分别记录阶段、最近轮次和日志；只有 `validation.json` 通过后才标为完成。历史训练及结果只读，不自动推送。
