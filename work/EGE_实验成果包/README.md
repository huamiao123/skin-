# EGE 实验成果包（CNN+Transformer 双分支与基线对比）

> 打包日期：2026-08-31
> 项目：EGE-UNet 系列改版（ISIC2018 皮肤病灶分割）
> 核心成果：EGE-Dual 双分支基线 DSC 0.8894（5.33M 参数、无预训练）；推理优化后 0.8931；机制证据链完整

## 目录导航

| 文件/目录 | 内容 |
|---|---|
| `01_基线对比与双分支实验交接文档.md` | 总交接文档：时间线、目录结构、recipe、全部结论、复现指南 |
| `02_原项目交接文档.md` | EGE 改版系列（Wave/HRViT/PRH/PSUDR/弱监督）原交接文档 |
| `03_实现过程详述.md` | S1-S8 每步实现过程、排查故事、bug 记录、工程经验 |
| `04_方法详解.md` | 每个模型/方法的公式、shape、超参、代码索引 |
| `05_全部实验数据汇总.md` | 全部 24+5 个实验的指标表 + 机制数据 + checkpoint 映射 |
| `06_任务方案文档/` | 全部技术方案 md + 参考论文（DECTNet PDF） |
| `checkpoints/` | 27 个 best checkpoint（文件名与数据表对应）+ 官方 ImageNet 权重 |
| `code/` | 四个项目的完整代码（EGE-UNet-main / pure_transformer_unet / Swin-Unet-main / Pytorch-UNet-master） |
| `logs/` | 全部训练日志 + gamma/alpha 逐 epoch 曲线 |
| `scripts/` | 启动脚本 + 接力脚本 + 诊断脚本（lr 筛选/过拟合测试/验证） |

## 一分钟速览

1. **最佳基线（推荐定版）**：EGE-Dual 差分 lr —— DSC 0.8894，5.33M 参数，无预训练
2. **最佳推理配置**：EGE-Dual 差分 lr + TTA + th=0.45 + 孔洞填充 → **0.8931**
3. **上界参照**：Swin + ImageNet = 0.8939（41.4M，需预训练）
4. **创新点候选**：尺度条件化融合 SCF（γ 与病灶尺度显著相关，r=-0.636, p≈1e-93）
5. **关键机制**：预训练 +3.26；差分 lr +0.4；从零 transformer 容量饱和；γ 镜像行为

## 复现要点

- 统一 recipe 见 `01` 文档第三章；所有启动命令见各文档复现指南
- 断点续训：PTU 系 `train.py --resume --work_dir <原目录>`
- lr 约定：CNN 1e-3 / 从零 transformer 1e-4 / 双分支差分
