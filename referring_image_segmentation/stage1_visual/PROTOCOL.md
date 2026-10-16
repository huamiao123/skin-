# 第一阶段：DINOv3 与 VMamba 冻结表征分割对照

日期：2026-10-16。用户确认后续研究转向图文指代分割；本轮按指导文档先验证纯视觉分割与错误互补性。

## 冻结的首轮设置

| 项目 | 配置 |
|---|---|
| 数据 | 本地已核对的 ISIC2018 旧划分：1886 train / 808 development validation |
| 结果性质 | 开发验证集筛选，不是独立测试；已按解码像素 SHA 检查两集合无完全重复；无完整病例身份，不能声称患者级完全独立 |
| B0 | 历史 CNN-only Dice 88.59%，仅作历史参照，非共同 Decoder 下的 Encoder 对照 |
| B1 | timm 官方维护的 DINOv3 ViT-S/16 LVD1689M 预训练权重，约21.59M Encoder参数 |
| B2 | VMamba-T s1l8 / vssm1_tiny_0230s，ImageNet-1K预训练，约29.48M Encoder参数 |
| Encoder训练状态 | 两组全部冻结，永久eval，权重严格加载；定期逐张量hash确认未改变 |
| 特征 | DINO第3/6/9/12层，均为16×16；VMamba四层64/32/16/8 |
| Adapter | 各层1×1投影到64通道＋GroupNorm＋GELU，必要时双线性resize到64/32/16/8 |
| Decoder | 两组同一个64通道轻量FPN，四个3×3 Conv/GN/GELU块＋1×1二值输出头；Decoder初始权重完全一致 |
| 原图旁路 | 无，避免用额外原图CNN绕过被评估的冻结表征 |
| 输入/输出 | 256×256；RGB图像bilinear resize，mask nearest；保留原始灰边软标签用于训练 |
| 归一化 | 两种权重均使用ImageNet mean=(.485,.456,.406), std=(.229,.224,.225) |
| 增强 | HFlip p=.5、VFlip p=.5、Rotation p=.5且每次均匀采样0～360°；相同逐epoch采样/几何序列；图像bilinear、mask nearest、零填充 |
| Seed | 42，首轮筛选；候选方案确定后才扩展至少3个seeds |
| 优化器更新batch | 64，最后不足64的批次保留；每epoch30次更新，无梯度累积 |
| 冻结Encoder前向batch | 分别实测64/128/256/512/1024；吞吐最高5%内选更小批次，结果两组均64 |
| 优化器 | AdamW，只更新Adapter/Decoder，lr=1e-3，weight_decay=.01，betas=(.9,.999)，eps=1e-8 |
| Loss | BCEWithLogits＋逐图soft Dice，等权，smooth=1，无深监督 |
| 调度 | 每epoch一次CosineAnnealingLR，T_max=300，eta_min=1e-5；无warmup、无early stopping |
| 预算 | 两组各300epochs，9000次参数更新 |
| 数值 | BF16 autocast；权重/优化器/损失FP32；梯度范数clip1 |
| 选模型 | 按样本数加权的验证loss最低，完全相同则取较早epoch |
| 验证与最终评估batch | 64，阈值.5，GT>=.5 |
| 主要读数 | 同时保存pooled Dice/IoU和macro逐图Dice/IoU/Precision/Recall |
| 边界指标 | 256网格，Boundary F1容差=图像对角线的0.5%；HD95像素及对角线归一化；单方空mask时HD95=对角线，双方空=0 |
| 简单融合 | 两模型概率图等权平均，阈值固定.5，不按验证结果搜索权重 |
| Oracle | 用真值逐图选更好的单模型，仅诊断潜在收益上限，不作为可部署方法 |
| 互补错误 | 两者都对、仅DINO对、仅VMamba对、两者都错；分别统计全部像素、GT前景和GT边界2px带 |
| 不确定性 | 单seed；2000次逐图配对bootstrap是已用于选择模型的开发集探索性区间，不声称独立测试或患者级推断 |

## GPU与存储

实机RTX4090，49140MiB显存。两组独立进程并行；原图/mask、固定验证特征和本epoch重新增强后的训练特征缓存于GPU。冻结特征缓存与当前输入一一对应，不把对旧特征的空间旋转冒充对新图像编码。每epoch仍重新采样增强并执行Encoder。

显存使用以吞吐与训练可比性为准，不因空闲显存改变输入尺寸、优化器batch或模型结构。单任务两轮实测峰值约DINO3.94GiB、VMamba6.63GiB；更大前向batch吞吐没有提高。并行后的实际吞吐/占用会写日志。

只保存小型Probe的best/latest以及优化器状态；冻结预训练Encoder权重仅存一份供复现使用。最终每张开发验证图保存FP32概率图、二值mask、逐图指标及互补误差类别，避免只有均值而无法分析病例。日志、权重下载来源与hash、代码版本、数据manifest均留存。

## 已知比较范围

不同Encoder的预训练来源、模型规模和Adapter参数量不同。本轮测量两套现成冻结特征的分割适用性，不将全部收益归因于架构。归一化、GPU几何实现、BF16、验证loss归约及验证batch与旧CNN实验不完全一致，旧分数只作背景参照。

本轮不以“DINO必然语义强、Mamba必然边界强”为前提。毛发/反光/遮挡等原因需要人工标记或可靠标签，不能仅从错误图自动推断。

## 入口与完成条件

`models.py`：Encoder/Adapter/共同Decoder；`data.py`：数据与几何；`benchmark.py`：吞吐和冻结/梯度检查；`train.py`：正式训练；`analyze.py`：配对与边界分析；`launch.py`：双进程监督、完成后自动分析。

`smoke/`为独立两轮预检，不能初始化正式训练。`results/<model>/DONE.json`仅在300轮与最佳权重最终导出完成后写入。`run_status.json`为complete且`analysis/REPORT.md`存在才表示本轮包括互补分析全部完成。

正式训练由tmux `ris_visual_stage1_20261016` 管理；日志 `dinov3.log`、`vmamba.log`、`analysis.log`。阶段结束依据结果再决定微调、Swin或双分支，不自动扩大后续实验矩阵。后续图文阶段须另行落实带指代表达和目标mask的数据及文本有效性对照。
