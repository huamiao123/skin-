# MedSeg research_v1 交接（P0）

本目录承载技术计划中的 P0 协议与可复现实验入口。当前阶段只完成协议、数据审计、实现修复和冒烟训练；没有启动 15 组正式长训练。

## 2026-09-29 状态澄清

2026-09-12 至 09-14 完成的 6 组 seed 42、以及正在执行的 seed 43 稳定性实验，均调用 `train.py` 的旧数据入口：`NPY_datasets` 从 `data/isic2018/train` 读取 1886 张训练图，从 `data/isic2018/val` 读取 808 张验证图。它们没有按 `manifest_v1.csv` 的 `S_seg/R_fit/R_val/V_dev` 划分训练，也没有独立的 `T_external` 测试数据。训练结束时的 `#----------Testing----------#` 仍使用同一个 `val_loader`，因此该结果是开发验证集结果，不是独立测试结果。

这些长训练是在 P0 代码修复后重新运行的旧划分实验，可用于比较修复前后和观察 seed 稳定性；不能称作完成 `research_v1` 数据协议的正式重跑。下方“下一步”仍是 `research_v1` 所需工作，不应被这些运行记录视为已完成。

## 已冻结的输入

- 协议：`configs/protocol_v1.yaml`，`protocol_id=medseg_research_v1`。
- 清单：`data/manifest_v1.csv`，SHA-256 为 `ffdf0e80852f0e5dd60dcfc1986b69978cf3a55d050dd44433545622f4d44ef9`。
- 审计：`data/manifest_audit_v1.json`。共 2694 张；`S_seg=1320`、`R_fit=377`、`R_val=189`、`V_dev=808`。病人/病灶 ID 不可用，`group_id=image_id`；不能宣称为官方 patient-level split。
- 评价：overlap 为逐图 macro + pooled；surface 使用 `surface-distance==0.1`，空掩膜约定和 4-neighbor BF1 容差均写在 `metrics_surface.py`。

## 已完成的 P0 修复

配对改为显式 image-ID join；图像/掩膜插值分别为 bilinear/nearest；旋转角度每次调用重新采样；梯度累积按真实样本数加权并 flush tail；持久化 GradScaler、RNG 和 `micro_step`；EGE gate 统计聚合整个验证集；输出契约、gamma override replay 和路径检查均有测试。

## 验证记录

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q tests
22 passed, 9 warnings
```

P0 冒烟命令（2 epoch，batch=8，accumulation=2）及结果已归档到 `results/p0_smoke_20260912`：epoch 2 的 `V_dev` mIoU=0.6727008498、Dice/F1=0.8043289389。`latest.pth` 含 `scaler_state_dict`、`rng_state`、`micro_step=472`；该结果仅用于链路验证，不是论文结果。

## 下一步（需明确批准后执行）

1. 用相同 manifest 运行 B0/B1/B2/B3 的 reseed 与每图 gate 导出。
2. 在 `R_fit/R_val` 上拟合干预阈值，仅在 `V_dev` 选择一次。
3. 最后才启动正式长训练并锁定 `T_external`；所有报告同时给出宏平均、pooled、surface 与 gate 覆盖率。

重新生成清单示例：

```bash
python research/data_manifest.py --data-root /root/isic2018_data \
  --official-mask-dir /root/isic2018_official/ISIC2018_Task1_Training_GroundTruth \
  --output research/data/manifest_v1.csv \
  --audit-output research/data/manifest_audit_v1.json
```
