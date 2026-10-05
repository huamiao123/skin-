2026-10-05 更新：本仓库新增 LocalContour ISIC2017 开发诊断与完成的 RSI TailProbe 结果。旧 RSI/EGE 训练权重、缓存和备份已由用户明确要求删除；以下旧清单与路径是历史训练证据，清理元数据见 `publication/cleanup_20261005`。权重不上传 Git 或 Release。

新 LocalContour 仅发布自研代码、协议、指标、审计与四张官方 ISIC2017 诊断叠图。[官方数据页](https://challenge.isic-archive.com/data/#2017)将2017 Task1标为CC0；病例图的发布来自用户此次明确提交结果的指令。完整原始图像/GT、NPZ、特征缓存、公开CNN权重及本地边界头权重未进入Git。未复制没有明确再分发许可的上游模型实现；上游固定URL/SHA及下载辅助脚本用于另行获取。

# RSI v2 实验存档

本仓库保存2026-10-04完成的RSI v2首轮可行性实验。范围是本轮RSI工程及两份协议，不包括work目录中的其他历史模型仓库。结果是工程与相对D0降低退化风险可行，超出简单Fixed-Shrink的独特优势尚未成立；没有启动seed29/T3/正式测试。

代码、冻结配置、测试、两份方案、执行状态、全部可公开数值结果、日志、分组与来源审计、失败/恢复记录和纯数值图形均进入Git。按用户要求，完整best/latest、失败、恢复、T0 debug与两个分类预训练权重均保留本地，不上传Git或Release。仓库保存32个独立权重对象的SHA、大小及59个原路径映射；原始checkpoint字节保持不变。原始实验README和.gitignore在 `publication/originals`，其余历史实验文件不改写。

## 主要入口

- [首轮结论](FEASIBILITY_REPORT_2026-10-04.md)和[最终根决策](outputs/seed17/root_decision_seed17.json)。
- [完整主表](outputs/seed17/pilot_main_table.csv)、[9984行逐参考结果](outputs/seed17/per_reference_gains.csv)、[640轮epoch诊断](outputs/seed17/epoch_diagnostics.csv)、[完整科学读出](outputs/seed17/full_scientific_review.md)。
- [风险收益图](outputs/seed17/gain_harm_tradeoff_readable.png)、[完整验证曲线](outputs/weight_grid_learning_curves_seed17/weight_grid_validation_curves_seed17.png)。
- `protocols/`：两份原始v2方案和任务状态快照；`runs/seed17/`：13阶段的DONE、环境、诊断与固定q；`artifacts/rsi_runs/`：失败及恢复的小文件。
- `publication/source_inventory.json`：每个原始项目/运行文件的原路径、大小、SHA和Git/本地保留位置。
- `publication/weights_manifest.json`：所有本地checkpoint和分类预训练文件的去重对象及完整路径映射，不包含权重文件或下载链接。

## 存档校验

```bash
python publication/verify_archive.py
```

校验脚本核对已发布实验文件的大小和SHA，以及本地权重清单的完整映射关系。它不读取本地权重、不执行模型、不训练、不评分，也不修改原实验文件。仅克隆本仓库不能取得训练权重。

## 历史路径与冻结版本

实验训练代码的Git版本是 `5b81bb1f812360049a89684dd43ce235b4487753`，源bundle SHA为 `36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb`，该版本保留在本仓库历史中。当前发布commit新增了实验存档，不等于历史训练commit。严格历史验收脚本同时检查Git版本、源代码、清单、检查点与输入SHA，不能在新的发布HEAD上声称原样重跑通过。

冻结YAML、逐参考清单与历史JSON保存 `/home/featurize/work/RSI_Implementation_20261003`、`/home/featurize/rsi_runs/P2-IMA-M-v2` 和 `/home/featurize/rsi_data` 的原绝对路径。没有为了发布改写这些证据。复跑历史验收应在独立环境中恢复对应路径，从历史commit建立工作树，再放入本次保存的tools和证据；获得原数据后按SHA校验。只改YAML的data_root不能同步迁移CSV路径。

需要换路径、改实现或新训练时，使用独立配置与新的证据目录；既有发布结果不能改标成新的运行。复现仍保持test锁定及既有工作点，不自动继续后续研究。

真实环境见各阶段environment.json：Python3.11.8、torch2.2.2+cu121、torchvision0.17.2+cu121、timm1.0.22、RTX4090。冻结requirements.txt以原字节保留；监督器另使用psutil5.9.8，见 `publication/requirements-extra.txt`。未执行的官方PH2 RAR获取分支还依赖rarfile及解RAR能力，属于可选数据获取依赖，不是本次训练环境已验收组件。

## 数据及本地保留项

约50GB逻辑大小的第三方原始数据树、缓存和未完成下载归档不复制到Git。现有下载程序、原始来源、成员CRC/尺寸/SHA、清单及验收记录均保留；所需图像/掩码在原机器完整可用，ISIC2018的整个训练ZIP仍只是部分下载，PH2官方归档字节级真实性未认证。

[PH2官方条款](https://www.fc.up.pt/addi/ph2%20database.html)明确禁止再分发。因此PH2源图片、掩膜、临床逐病例原信息和样本不公开复制。[IMA++官方记录](https://zenodo.org/records/14201693)声明CC BY-NC-ND4.0并要求遵守ISIC条款；源RGB清单中还有CC-0/CC-BY-NC的不同许可标记。包含源RGB/标注的本地对齐与重复审查画册也保留本地，仅公开SHA、审查结论与生成代码。它们列在source_inventory中的 `withheld_third_party_data`，并非遗漏。纯数值曲线、指标和实验诊断正常公开。

不向原始数据、上游分类预训练权重或其他第三方资产附加新的统一许可证。分类权重的上游来源与SHA见既有预训练加载报告和权重清单。仓库不包含SSH目录、私钥、访问令牌、认证缓存、Python缓存或无关旧项目。

自动生成的decision_inputs.json与科学JSON是发布时的输入快照，最终继续/停止决定以独立root_decision_seed17.json和decision_log.md为准。全部95%区间是选择后的开发证据，不代表独立正式验证。
