# MambaVision-S + Swin Decoder 内的 HiRAM（seed 42）

设计、训练设定及比较边界见 [PROTOCOL.md](PROTOCOL.md)。本实验只使用 MambaVision-S 视觉编码器；HiRAM 的两路是该编码器产生的解码特征和跳接特征。三个对照均使用 seed 42；没有新增随机种子或文本输入。

依赖同仓库 `stage14_mambavision_s_isic2018_seed42` 固定的官方预训练权重、CUDA 扩展和 runtime，`stage16_hiram_dual_seed42/vendor_hiram` 原始 HiRAM 源码及其 CC BY-NC 4.0 许可，`stage1_visual`、`stage2_decoder`、`stage3_confirmation` 和共享 ISIC2018 数据与评价代码。数据不在本目录复制。GPU 需支持 CUDA 与 BF16；机器环境详见 `preflight.json`、`capacity_preflight.json`。训练前检验官方权重 SHA、编码器参数和缓冲区、基线前向、HiRAM 等价及数据成员。

从本目录执行：

```bash
python -u preflight.py
python -u capacity_preflight.py
python -u launch.py
```

`launch.py` 首先训练一个独立 smoke 的第 1 轮，再恢复进行第 2 轮并重新评价，随后冻结代码摘要并开始正式训练、评价、推理测速和完成审计。中止时训练在 epoch 尾部保存；重启 `python -u launch.py` 可从 `results/latest.pth` 恢复并核查模型、优化器、调度器和 RNG。日志和状态分别在 `train.log`、`results/status.json` 与 `run_status.json`。

`results` 保存模型检查点和预测，仅供本地研究；`analysis/REPORT.md`、`analysis/summary.json` 和 `analysis/per_image.csv` 由自动分析生成。不会自动向 GitHub 推送任何文件或原始图像。开发验证集用于选模，不是独立测试集；不要将其结果写成官方测试成绩。论文中的 SAM/CNN 系统并未复现，此处只是将其 HiRAM 模块移植到不同的 Decoder 拓扑。
