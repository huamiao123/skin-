# Stage A 协议草案与 GPU 前置检查

目标是区分 head 宽度（A 与 B）和 FCN 辅助监督（B 与 C），不更换 encoder、数据、增强和优化协议。三份 `config/*.yaml` 是候选，尚未冻结共同 microbatch。所有输出只能写入本目录；不改 Stage1–51。使用旧的图像/标签读取、增强与 `evaluation.py` 指标定义。单种子 42，256 输入，AdamW 100 epochs；主/辅使用 FP32 BCE + per-image Soft Dice；总损失 main + 0.4 aux；验证只用主头、阈值0.5、最大验证 macro Dice（平局取最早），不读取历史 test 来选方案。

## 尚未授权执行的 GPU 预检

当前实例 `nvidia-smi` 无法连接驱动，设备型号/空闲情况与显存容量均未确认，过去的 RTX 4090 记录不能视作当前设备事实。恢复驱动、用户单独授权并确认无人占用后，前台执行以下最小合成预检；每次单独进程，只指定一组，以同一 microbatch 比较，先从 2 开始。每组只进行一个 optimizer step；三组共 6 个合成样本，每次包含两次评估前向和一次训练前向/反向。耗时与峰值显存目前未知，脚本打印实际结果；不得自行后台轮询或扩大搜索。

```bash
cd /home/featurize/work/mambavision_seg_rebaseline_20261116
nvidia-smi
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/waas/mambavision_stage22/env/bin/python src/preflight_gpu.py --arm A-64-noaux --micro-batch 2
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/waas/mambavision_stage22/env/bin/python src/preflight_gpu.py --arm B-512-noaux --micro-batch 2
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/waas/mambavision_stage22/env/bin/python src/preflight_gpu.py --arm C-512-aux --micro-batch 2
```

脚本是未执行的候选：检验 direct/cache 一致、有限梯度、冻结参数和 encoder BN buffer、单步显存；遇错误终止，拒绝覆盖报告。它不能替代真实数据、梯度累积、足额 microbatch、吞吐和完整恢复检查，也不能证明实验已可正式训练。训练 BN 的 1×1 PPM 分支不允许把 microbatch=1 当成 batch64。

随后须实现并 CPU 测试独立训练/评估入口；另行获得 GPU 授权后做真实数据预检和每组 2 epoch smoke，完成 RNG、模型、优化器、调度器和非整批加权恢复检查，并汇报实际显存、速度、完整命令。之后取得进一步授权才做正式三组：每组 100×1886=188600 次样本处理，三组共 565800；若按 effective batch 64 且保留最后不足批，约每组 3000 次 optimizer step。不能仅根据单 seed、历史 test 或形状通过声称结构有效。

## 2026-11-16 新实例状态更新

- 新实例为 RTX 3090 24 GiB，PyTorch 2.2.2+cu121、BF16 可用。A/B/C 的 microbatch=2 单步合成预检已在 GPU 前台通过；报告为 `reports/gpu_preflight_*_mb2.json`，显存峰值仅针对合成小批量，不包括完整数据和缓存。
- 云盘开发数据和仓库源码均已核对：5388 个图像/掩膜字节哈希与历史清单一致；19 个源码审计项中13项字节一致、6项换行格式不同但规范化后完全一致。历史测试图像未恢复，也不用于 smoke。
- `src/smoke_train.py` 只支持真实数据、2 epoch 和 validation，不提供正式训练入口。首轮包含 2 个被回滚的实际数据预检更新，检查缓存等价和恢复重放；之后每组60个正式 smoke 更新。恢复可用 `--resume`，但运行应先取得用户针对 GPU 训练的明确许可。
- 对应的前台命令（各组依次运行，microbatch一致）：`PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/waas/mambavision_stage22/env/bin/python src/smoke_train.py --arm A-64-noaux --micro-batch 2`，B/C 只替换 arm。精确耗时和完整缓存显存仍未实测，不根据合成结果外推。
