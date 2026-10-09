# 统一边界指标重算记录（2026-09-12）

针对旧版 `tests/eval_boundary_metrics.py` 的边界提取、空掩膜跳过和 HD95/ASSD 实现问题，使用 `research/evaluate_boundary_corrected.py` 对相同 808 张验证图和原 checkpoint 重算。

| 模型 | BF1@2px ↑ | HD95(px) ↓ | ASSD(px) ↓ |
|---|---:|---:|---:|
| EGE baseline | 0.3725 | 18.9375 | 6.8986 |
| EGE-Wave v1 | 0.3800 | 18.4153 | 6.6218 |
| Wave LL-only | 0.3688 | 18.6003 | 7.0042 |
| EGE-Dual difflr | 0.3953 | 17.3083 | 6.3835 |
| EGE-Dual unified | 0.4209 | 18.6964 | 6.5705 |

实现：`surface-distance==0.1`、4-neighbor pixel boundary、2 px 双向容差匹配；空掩膜按协议处理。本表仍是现有 legacy train/val 划分上的验证集重评估，不是独立外部测试集。
