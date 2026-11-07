# MambaVision + HGM BMD 局部适配

实验范围、比较局限和训练协议见 [PROTOCOL.md](PROTOCOL.md)。结果仅在 `run_status.json` 显示 `complete`，且 `results/DONE.json` 和分析审计均通过时标为完成。

依赖现有 stage14 的官方预训练 encoder/runtime、stage20 的配对基线结果和 stage3 的数据加载，不对已有实验做修改；不公开推送任何上游无许可证的源码快照或数据。

**GitHub 归档范围：**保留按最高验证逐图平均 Dice 选出的分割探针最佳权重；不含 encoder 预训练权重、优化器恢复点、其余 checkpoint、原图和逐像素预测。所以下方关于 `latest.pth` 恢复的说明仅适用于仍保留该文件的原实验机器，不能直接从 GitHub 快照恢复断点。重新运行需自行准备数据、预训练骨干和运行依赖，并从项目根目录运行 `python publication/restore_stage20_21_references.py` 获取经 SHA256 核验的上游源码。原 `evaluate.py` 同时要求 `best_loss`、`last` 权重；归档中只包含 `best_dice`，重新训练才能生成另外两个 checkpoint。

在当前 Python 环境中，依次运行：

```bash
python preflight.py
python capacity_preflight.py
python -u run_pipeline.py
```

管线会先做独立两轮 smoke 和断点恢复，再启动 300 轮训练、评价、测速和逐图分析。重新运行管线会从 `results/latest.pth` 的模型、优化器、调度器及随机状态恢复；同一显卡环境下适用。若显卡改变，须先重新核验 batch 64 和 checkpoint 兼容，不要绕开检查。日志与指标分别位于 `pipeline.log`、`results/` 和 `analysis/`。

`diagnose_bmd.py` 使用合成输入检查工程兼容性，不属于正式训练或实验结果。
