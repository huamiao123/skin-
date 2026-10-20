# 第十至十三轮：存档范围与依赖恢复

本仓库提供已执行实验的源码和证据。权重、图像/Mask、第三方文本表、特征/文本缓存、逐像素预测和运行依赖不随 Git 发布。先在仓库根目录运行：

```bash
python referring_image_segmentation/publication/verify_archive.py
```

该检查仅核对发布文件的大小与 SHA256，不会训练或评分。各阶段 validation.json 和 DONE.json 属于原机器的历史记录，不能当作新机器已复现。

## 状态和运行范围

- stage10_mosmed_text：用户暂停。seed42 三组各完成 100 轮；seed43 完成 59 轮及第 60 轮部分更新；seed44 未启动。只读低 Dice 核查未使用测试集，正式测试结果不存在。不要为读取存档而运行其 launch.py，也不要未经用户要求恢复其他 seeds。
- stage11_isic_text_controls：seed42、100 轮，新增无文本 E0；固定/逐图文本两组只读复用 stage9 的已完成训练和权重，完整评估重新核验。
- stage12_predecoder_text：seed42、固定/逐图文本各 100 轮，Adapter 后、Swin 前加入词级 Cross-Attention，之后采用普通 Mask Head。
- stage13_isic_300：seed42、五组各 300 轮 / 15300 次更新，全部重新初始化可训练参数。完整 300 轮 cosine；不从 100 轮分割权重续训。

读取源码应同时保留各阶段目录：stage13 的模型封装复用 stage10 的后置预测头和 stage12 的前置交互；还依赖 stage9 的 ISIC 数据处理与统一 evaluation.py。Python 模块中的 common 名称由当前阶段入口解析，不能随意平铺移动文件。

## ISIC2016 与文本缓存

先按 [第九轮恢复说明](REPRODUCE_STAGE9.md) 恢复合法取得的官方 ISIC2016 图像/Mask、固定版本 MedVLSM 文本、通用视觉预训练权重、CLIP 文本权重及运行环境。无需恢复 RefCOCO 图像或训练出的分割参数。

实际划分为 806/90/379，原始文本中的重复与跨集合 ID 去除规则、图像成员 SHA、归一化 P3 属性提示的来源均由 stage9 清单定义。仅使用颜色/形状，统一类别词 skin lesion，不切换到含 GT 大小/位置的提示。

stage11 至 stage13 只读复用 stage9/assets；stage12/prepare_text.py 生成词级 CLIP 表示、有效词掩码和行索引，保存于 stage12/assets，stage13 继续复用。词级缓存来源与哈希见 token_provenance.json；资产目录中包括第三方标注的派生表示，不随本次发布重新分发。

原启动器会检查早期阶段的完成记录、源文件、资产以及历史最佳权重哈希。复现训练需要恢复这些历史依赖，或在独立的新实验副本中明确记录并调整 lineage/路径检查，重新生成数据、缓存、预检和 provenance。不要直接把存档的完成状态复制成一次新训练成功。新运行中任何调整都需保留新的配置和哈希；原始源码不因此改写。

## MosMedData+ 数据与文本来源

图像/Mask 为 Kaggle `maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset` version 2；文本表为公开镜像 `xiaopan9802/textdiff-mosmeddata` version 1，成员为 `MosMedData/TextDiff_code/datasets/Covid19/{train,val,test}/{Train,Val,Test}_text_MosMedData.xlsx`。固定获取逻辑在 stage10/fetch_data.py，文件 URL、大小和 SHA 在 stage10/sources/download_provenance.json。原始 XLSX 与转换后的完整文本 JSON 只在本地保存，公开来源、哈希和脚本足以说明恢复条件。

镜像文本未与作者 OneDrive 文件逐字节认证；LViT 文本包含感染灶数量/位置并使用 Mask 质检，不能描述为普通临床报告。依据可追溯 CT 体积及来源组修复集合重叠后，2004/216/509 划分与原论文不同；来源组不交叉不代表完整患者独立性已认证。遵循原作者加载逻辑将 Mask > 0 二值化。下载数据或恢复缓存无需自动启动暂停的训练计划。

## 硬件、评价与保留

训练硬件为 48GB RTX4090。ISIC 三轮均冻结 DINOv3 和 CLIP，只训练 Adapter/交互/Decoder/Head，输入 256、batch16、AdamW 1e-4；MosMed 使用自己的 batch64 协议，不与 ISIC 混用。各自 PROTOCOL.md、config.json、preflight.json 与 provenance 保存实际细节。

测试预测还原原始尺寸，报告 macro/pooled Dice、IoU、Precision/Recall、Prec@0.5/0.6/0.7/0.8/0.9、Boundary F1、像素和对角线归一化 HD95；ISIC 增加阈值 Jaccard。逐图 JSON 和独立复核结果已发布，含病例原图的拼图及稠密数组保留本地。

stage11/12/13 仅保留各自新增 1/2/5 个最佳权重，恢复与 smoke 权重已在校验通过后清理。源文件和已完成数值记录原样复制；本次发布只更新总览和发布说明，不重训、不重新选模、不扩展 seed。单 seed 和已用于开发的测试集限制研究结论，图像 bootstrap 不能估计患者或训练随机性。
