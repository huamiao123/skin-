# 皮肤病灶分割：实验结果与源码存档

最新实验：[LocalContour 第二阶段：候选充分性与全局轮廓修补](experiments/LocalContour_ISIC2017_Phase2_20261005/README.md)。在重复使用的 ISIC2017 开发集上，三种子平均 Dice 为：冻结 CNN 0.85305、参数匹配局部模型 0.85564、全局 Transformer 0.85360。当前实现没有证明全局轮廓通信优于强局部对照，决策为 `STOP_GLOBAL_PRIMARY_AFTER_DEVELOPMENT`；官方 test 未评分。[第一阶段候选诊断](experiments/LocalContour_ISIC2017_20261005/README.md)保留作历史对照。

[RSI TailProbe完整报告](experiments/RSI_TailProbe_20261004/experiment_report.md)结论为 `STOP_CURRENT_RSI`。解冻tail后，λ=3 RSI只保留Mean约46.1%的正收益，且H来源回退超过门槛；五组40轮证据、幅度控制和配对统计均保留。以下历史首轮结果与新机制验证分别存档。

2026-10-05按用户要求清理旧RSI/EGE权重、缓存、备份与冗余副本；源码、方案、数值结果及复核记录保留。[清理核验](publication/cleanup_20261005/final_verification.json)说明删除范围。模型权重不上传Git/Release；历史权重清单记录当时SHA，不表示旧文件仍可恢复。官方ISIC2017四张诊断叠图按用户此次发布要求及CC0数据条款公开，IMA/PH2源图画册仍保留本地。

---

# RSI v2：完整首轮实验存档

代码、配置、协议、全部可公开数值结果、日志、审计与失败记录已纳入本仓库。按用户要求，全部模型检查点（含失败/恢复/T0）及分类预训练权重未上传Git或Release；旧训练权重已按2026-10-05要求清理，历史清单继续保留；仓库仅保存路径、大小和SHA清单。原数据及受许可限制的源图画册保留本地，公开来源、哈希与获取说明。

请先读 [存档及复现说明](PUBLICATION.md) 和 [最终可行性报告](FEASIBILITY_REPORT_2026-10-04.md)。[本地权重清单](publication/weights_manifest.json)保存32个独立权重对象及59个原路径映射；实验原README副本保存在publication/originals。

---

# RSI v2 执行工程

此工程实现 `/home/featurize/task` 的完整 v2 研究方案和首轮测试方案。所需数据成员、T0、T1及seed17完整T2对称网格已于2026-10-04完成。工程和相对D0降低退化的作用得到支持，超出简单Fixed-Shrink的实质优势及跨来源稳定性尚未成立；本轮收缩主张，未触发seed29、T3或正式test。实际结果、范围和依据见 [首轮可行性报告](FEASIBILITY_REPORT_2026-10-04.md) 与 [最终根决策](outputs/seed17/root_decision_seed17.json)。

## 位置

- 工程、人工审计、测试与阶段结束后 checkpoint 备份：本目录。
- 原始数据：`data` 链接到 `/home/featurize/rsi_data`，避免占满 work 的 30GB 共享盘配额。
- 历史 checkpoint 路径（旧权重已清理）：`/home/featurize/rsi_runs/P2-IMA-M-v2`。
- IMA 最终可用清单：`data_manifests/final_references.csv`。元数据数目与完成文件/重复隔离后的数目分别记录，不能混用。
- ISIC 2017/2018、PH2 数据状态和来源：`outputs/external_data_status.json` 及 `data_manifests/external_*`。PH2 官方连接失败与镜像验收分别保留。
- `../RSI_datasets/{IMA,ISIC2017,ISIC2018,PH2}` 是原始数据的便捷入口。ISIC2017 三个 split 的图像/掩码为 2000/150/600 对；ISIC2018 为 2594/100/1000 对，均完成解码、尺寸及二值掩码验收。2018 训练集全部官方成员已按 ZIP 成员 CRC/尺寸验证，完整 11GB ZIP 尚未下载完；跨年同一官方成员与已下载成员合并的证据单独保存。
- PH2 完整历史目录归档包括 200 对 BMP、50 份 ROI 及临床文件；200 对与另一独立镜像逐字节一致。创作者控制的 Dropbox 链接无法访问，不能据此声称已认证官方 RAR 的字节一致性，详见 `outputs/ph2_historical_rar_comparison.json`。

## 执行

在本目录运行，环境版本见 `requirements.txt`；当前环境使用 CUDA 12.1 的 PyTorch。

```bash
RSI_TEST_DEVICE=cuda python -m pytest -q
python -m rsi.feasibility
python -u -m rsi.run_pilot --seed 17
```

实际 seed17 的三个共享阶段及完整十项 D40 均已完成，共13项固定预算训练。共享阶段选中 epoch 分别为 36/40/22，M val Dice 分别为 0.855252/0.853821/0.858374。B 对固定 CNN anchor 的平均收益为 0.003122。最初七组学习曲线见 `outputs/pilot_learning_curves_seed17.png`，完整D网格的训练/验证曲线与来源及成本口径见 `outputs/weight_grid_learning_curves_seed17`；这些是开发集结果。

lambda=1 的中间复核保存在 `outputs/lambda1_review_seed17/lambda1_scientific_decision.md`。完整矩阵及原图M/H/T1评价现已发布到 `outputs/seed17`：16个M点、48来源队列、9984逐参考结果、640轮提交日志。按M规则选中的RSI lambda=3，其Dice/H_epsilon/G+为0.858010/0.003259/0.007857；D0为0.858317/0.007422/0.012553。损害下降伴正收益减少，实际Fixed-Shrink仍提供很强的简单对照。完整配对、尾部、边界及来源证据见 `full_scientific_review.md`；最终决策见 `decision_log.md` 与独立 `root_decision_seed17.json`。

已依据 `outputs/weight_grid_decision_seed17.json` 完成三个加权目标的 lambda=0.3/3 六项 D40，共用选定 B、固定 q 和 seed17，最大 D 任务数为10，不增加表外参数。训练日志、历史进程与严格来源验收见 `outputs/weight_grid_console_seed17.log`、`outputs/weight_grid_process_seed17.json`、`outputs/weight_grid_training_seed17.json`；实际使用的命令为：

```bash
python -u tools/train_weight_grid.py --config configs/p2_ima_m_v2.yaml
```

2026-10-04 检查到原补网格进程消失，MeanHinge-0.3 的最后提交断点为第25轮，原因未确定。配置、来源、模型/优化器/随机状态和日志验收通过，使用同一冻结命令恢复；实际第26轮的更新次数恰为4600+184。中断现场与只读复核见 `outputs/recovery_20261004`，没有重置已完成阶段或增加预算。独立监督器绑定恢复进程身份，六项全部DONE后依序执行导出、完整T2汇总、来源阈值及病例对照分析，四命令均已成功。随后完整科学读出和两类图实际发布，85项输入/检查点SHA最终复核一致；学习曲线只修脚注布局，原图及旧工具留档，分数不变。

因原尺寸验证图中位约 6MP、较大图约 30MP，原导出重复计算 anchor 边界耗时较高。当前执行入口是独立运行器，原 `rsi/*.py` 和配置保持冻结，已完成阶段严格复用 DONE：

```bash
python -u tools/cached_pilot_runner.py --config configs/p2_ima_m_v2.yaml --seed 17 --fast-exact-boundaries
```

它仅缓存以完整输入内容 SHA 为键的数值结果，并用精确欧氏最近边界查询替换全图 EDT；原图坐标、指标及训练规则不变。70 项联合 CPU 回归、6MP/30MP 真实掩码逐字段精确一致性证据见 `outputs/cached_runner_preparation.json` 和 `outputs/exact_boundary_native_verification.json`。接管发生于 B 已完成、导出尚未完成且没有 optimizer 活动的阶段，记录见 `outputs/runner_handoff/handoff.json`。

全量训练入口要求完整文件审计成功且绑定当前 manifest SHA256；固定16图的单独审计不能解锁全量训练。断点继续使用相同命令，校验配置、seed、初始化、清单及代码 hash，恢复模型、optimizer、scheduler、scaler、RNG，并清除未提交 epoch 的日志。

AMP 溢出由 GradScaler 跳过更新；日志分别记录实际 optimizer 更新、尝试次数、跳过次数及非有限梯度范数数量。范数均值只汇总有限值，全溢出时保留空均值。第一次 A-CNN 在第31轮诊断汇总中断，完整失败目录已封存；修复后从 ImageNet 起点按原100轮预算重新运行，未把失败 run 的权重混入共同起点。

文档中的分阶段命令已实现为 `rsi.train_anchor`、`rsi.train_message`、`rsi.export_actions`、`rsi.prepare_abs_threshold`、`rsi.train_objective`、`rsi.summarize_pilot`，每个入口可用 `--help` 查看参数。`RUN_B` 应替换为对应 seed 的 B 目录或其 best.pth。

所有 A/B/D 阶段跑满固定预算，按原图 val macro mean-rater hard Dice 选 checkpoint，精确并列取更早 epoch。B/D 只更新消息参数，CNN/Transformer/辅助头全部冻结并保持 eval；逐 epoch 校验冻结 hash 与固定 FP32 anchor 输出。四个 D 目标共享同一个已完成且选定的 B，各自新建 optimizer，使用相同增强与采样序列。q 仅来自 B 的 canonical M train，每图再每参考加权75%分位。

## 结果与限制

`outputs/T0` 保存固定16张真实训练图的曲线、对齐画册、工程验收报告；debug checkpoint 不能初始化全量 A/B/D。T0 的短跑用于检查学习和实现，不证明 RSI 的实际研究效果。

T1 的 gate 和阶段进度写到 `outputs/T1_decision_seed17.json`、`outputs/pilot_progress_seed17.json`。T2 成功执行后，`outputs/seed17` 保存规定主表、逐参考收益、epoch/update 诊断、来源敏感性、工作点和 tradeoff 图及决策记录。全部失败记录保留。补权重只能对 RSI/MeanHinge/AbsHard 同时补0.3和3；seed29 使用冻结选择，正式101/202/303及 test 释放需按文档后续门槛执行。

以下独立校验与汇总均已执行。finalizer按运行ID处理历史B导出的默认method标签，保留原CSV，检查完整预算、共享起点、参考集合和对称矩阵。配对区间属于选中开发集的探索性证据；不能把工作点标记或脚本完成当作方法通过。自动 `decision_inputs.json` 与科学JSON保留发布时快照，最终继续/停止决定以 `decision_log.md` 和 `root_decision_seed17.json` 为准。

```bash
python tools/finalize_pilot.py --scope T2
python tools/source_threshold_sensitivity.py --inputs 'outputs/seed17/corrected_inputs/*.csv'
python tools/paired_control_comparison.py
python tools/full_scientific_readout.py
python tools/plot_weight_grid_learning_curves.py
python tools/plot_weight_grid_tradeoff.py
```

test 只做文件完整性和重复审计，当前评分入口均锁定。H/T1 是同图来源敏感性，不据其分数选参。已知组隔离与图像重复审计不等于未知病例完全独立；跨年及 P1/IMA 的历史交叠另行记录，不把共享图像当独立外测。

IMA 的原始参考可能存在很大分歧；来源筛选与选择顺序按方案固定，不因为参考外观或模型成绩删图。数据获取来源、许可证和校验见来源清单，工程未改写原始标注。
