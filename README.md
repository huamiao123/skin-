# RSI v2 执行工程

此工程实现 `/home/featurize/task` 的完整 v2 研究方案和首轮测试方案，先执行数据验收、T0，再执行 seed17 的 T1；通过普通消息门槛后执行 lambda=1 的 T2。代码中的结果均由实际运行写出，未运行的实验不能当结果。

## 位置

- 工程、人工审计、测试与阶段结束后 checkpoint 备份：本目录。
- 原始数据：`data` 链接到 `/home/featurize/rsi_data`，避免占满 work 的 30GB 共享盘配额。
- 活跃 checkpoint：`/home/featurize/rsi_runs/P2-IMA-M-v2`。
- IMA 最终可用清单：`data_manifests/final_references.csv`。元数据数目与完成文件/重复隔离后的数目分别记录，不能混用。
- ISIC 2017/2018、PH2 数据状态和来源：`outputs/external_data_status.json` 及 `data_manifests/external_*`。PH2 官方连接失败与镜像验收分别保留。

## 执行

在本目录运行，环境版本见 `requirements.txt`；当前环境使用 CUDA 12.1 的 PyTorch。

```bash
RSI_TEST_DEVICE=cuda python -m pytest -q
python -m rsi.feasibility
python -u -m rsi.run_pilot --seed 17
```

全量训练入口要求完整文件审计成功且绑定当前 manifest SHA256；固定16图的单独审计不能解锁全量训练。断点继续使用相同命令，校验配置、seed、初始化、清单及代码 hash，恢复模型、optimizer、scheduler、scaler、RNG，并清除未提交 epoch 的日志。

AMP 溢出由 GradScaler 跳过更新；日志分别记录实际 optimizer 更新、尝试次数、跳过次数及非有限梯度范数数量。范数均值只汇总有限值，全溢出时保留空均值。第一次 A-CNN 在第31轮诊断汇总中断，完整失败目录已封存；修复后从 ImageNet 起点按原100轮预算重新运行，未把失败 run 的权重混入共同起点。

文档中的分阶段命令已实现为 `rsi.train_anchor`、`rsi.train_message`、`rsi.export_actions`、`rsi.prepare_abs_threshold`、`rsi.train_objective`、`rsi.summarize_pilot`，每个入口可用 `--help` 查看参数。`RUN_B` 应替换为对应 seed 的 B 目录或其 best.pth。

所有 A/B/D 阶段跑满固定预算，按原图 val macro mean-rater hard Dice 选 checkpoint，精确并列取更早 epoch。B/D 只更新消息参数，CNN/Transformer/辅助头全部冻结并保持 eval；逐 epoch 校验冻结 hash 与固定 FP32 anchor 输出。四个 D 目标共享同一个已完成且选定的 B，各自新建 optimizer，使用相同增强与采样序列。q 仅来自 B 的 canonical M train，每图再每参考加权75%分位。

## 结果与限制

`outputs/T0` 保存固定16张真实训练图的曲线、对齐画册、工程验收报告；debug checkpoint 不能初始化全量 A/B/D。T0 的短跑用于检查学习和实现，不证明 RSI 的实际研究效果。

T1 的 gate 和阶段进度写到 `outputs/T1_decision_seed17.json`、`outputs/pilot_progress_seed17.json`。T2 成功执行后，`outputs/seed17` 保存规定主表、逐参考收益、epoch/update 诊断、来源敏感性、工作点和 tradeoff 图及决策记录。全部失败记录保留。补权重只能对 RSI/MeanHinge/AbsHard 同时补0.3和3；seed29 使用冻结选择，正式101/202/303及 test 释放需按文档后续门槛执行。

test 只做文件完整性和重复审计，当前评分入口均锁定。H/T1 是同图来源敏感性，不据其分数选参。已知组隔离与图像重复审计不等于未知病例完全独立；跨年及 P1/IMA 的历史交叠另行记录，不把共享图像当独立外测。

IMA 的原始参考可能存在很大分歧；来源筛选与选择顺序按方案固定，不因为参考外观或模型成绩删图。数据获取来源、许可证和校验见来源清单，工程未改写原始标注。
