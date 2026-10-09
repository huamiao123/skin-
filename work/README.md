# work 实验存档（2026-10-09）

本目录新增本机 `/home/featurize/work` 下的实验源码、配置、方案、数值结果、
训练日志和可公开图形。原仓库根目录的 RSI 及 `experiments/` 下 LocalContour
存档继续保留。逐文件来源、大小和 SHA256 见 [archive_manifest.json](archive_manifest.json)。

## 入口

| 目录 | 内容 |
|---|---|
| [EGE-UNet-main](EGE-UNet-main/) | EGE baseline、Wave v1/v2、BGCT、HRViT、PRH、PSUDR、双分支、差分学习率、SCF、弱监督及后续修复与重训 |
| [EGE_实验成果包](EGE_实验成果包/) | 历史汇总、方法说明、旧实现快照及运行日志 |
| [pure_transformer_unet](pure_transformer_unet/) | 自建 PTU 深度/容量扫描及 PTU-Dual |
| [Swin-Unet-main](Swin-Unet-main/) | Swin 从零训练、学习率扫描与 ImageNet 预训练分割基线 |
| [Pytorch-UNet-master](Pytorch-UNet-master/) | U-Net 基线 |
| [WaveFormer-main](WaveFormer-main/) / [hrvit-main](hrvit-main/) | 上游参考实现，含原许可文件；存在目录不表示其全部模型均训练过 |
| [RSI_Implementation_20261003](RSI_Implementation_20261003/) / [RSI_TailProbe_20261004](RSI_TailProbe_20261004/) | RSI 首轮实验与最终报告的本地工作目录快照 |
| [experiments](experiments/) | 启动、诊断脚本和日志；包含当前 Transformer seed42 重训快照 |
| [task](task/) | 历史技术方案，方案存在不等于已经执行 |
| [RSI_datasets](RSI_datasets/) | 数据位置元信息，原始数据不纳入仓库 |

## 结果口径

8 月的历史汇总与 9 月修复后的结果不属于严格同配置实验。后续修复包括每次调用
重新采样旋转角度、掩膜最近邻插值、图像 ID 配对、梯度累积末尾处理和随机状态。
旧边界指标应由 [统一重算记录](EGE-UNet-main/results/corrected_boundary_metrics_v1.md)
替代。旧数据目录中的预处理掩膜本身仍含灰度边缘；最近邻变换保留这些原标签。

9 月 30 日完成的六项标准重训使用 train batch64、val batch8、final batch1、
300 epochs。日志中的 DSC 是 pooled pixel Dice。以下均为旧 1886/808 划分的
开发验证集结果，训练结束时的 Testing 标题不表示独立 test：

| 模型 | seed42 DSC | seed43 DSC |
|---|---:|---:|
| EGE baseline | 0.887022 | 现存镜像只到253轮，未核实最终结果 |
| Wave v1 | 0.886729 | 0.886006 |
| Dual 差分学习率 | 0.891300 | 0.886997 |
| Dual + SCF | 0.888372 | 0.885766（9月29日已全程 val batch8） |

RSI 的结论是工程通过、相对 D0 降低部分退化风险，但超出 Fixed-Shrink 的独特优势
和跨来源稳定性尚未成立；没有完成正式 test，详见
[最终可行性报告](RSI_TailProbe_20261004/FEASIBILITY_REPORT_2026-10-04.md)。

10 月 9 日预训练 Swin 和 PTU `[2,2,6,2]` 的 seed42 重训均已完成300轮。
按最低验证loss选中的模型，在808张开发验证图上最终 pooled DSC 分别为
**0.89719584（Swin，第16轮）**与 **0.83826920（PTU，第174轮）**。
[完整最终报告](experiments/transformer_seed42_20261009/FINAL_RESULTS.md)、
[结果CSV](experiments/transformer_seed42_20261009/final_results.csv)以及
[完整日志和逐epoch记录](experiments/transformer_seed42_20261009/snapshot/)已更新。
训练期间最高 DSC 单独报告，不改变选模型规则。Swin 用激活重算维持物理
batch64/FP32，PTU 从第5轮断点恢复完成预算；失败和恢复记录保留。

## 存档范围与使用

模型权重、原始数据、NumPy/特征缓存及包含源 RGB/标注的画册保留本地。
第三方参考论文 PDF 不复制；上游项目公开的说明图及纯数值曲线按原项目保留。
大于512KiB的日志以 `.log.gz` 无损存档；例如 `gzip -dc FILE.log.gz` 可直接查看。
清单同时记录压缩前后大小和 SHA256。

10 月 5 日清理记录针对当时指定的旧 RSI/EGE 资产，不代表所有目录的权重都已删除。
历史 README、DONE 和绝对路径保持原记录；是否仍有本地权重应以本次清单为准。
部分历史入口依赖 `/root` 路径或本地数据链接，需要按当前环境配置后运行。
此次上传仅归档，不重新训练、不重算指标，不附加覆盖第三方资产的统一许可证。
