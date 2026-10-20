# MosMedData+：逐图文本增益的三组配对实验

2026-10-20，用户授权执行。E0 纯视觉、E1 固定文本 `lung infection`、E2 逐图医学文本；正式 seeds 42/43/44。三组之外不添加模型或文本干预实验。历史阶段保持原样。

## 数据与结论范围

图像/二值感染 Mask：LViT 作者仓库链接的 Kaggle `maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset` version 2，2729 对 PNG。该集合实际合并 Jun、Morozov 和 bjorke 三个来源。PNG 已处理为显示强度，不将其当原始 HU，不重复加窗。

作者 OneDrive/Google Drive 在当前环境不可直接取得；文本使用公开 Kaggle `xiaopan9802/textdiff-mosmeddata` version 1 中三张 LViT 格式 XLSX 的小文件镜像，源 URL、版本、hash 均记录。三张表 2183/273/273，逐文件名与原始图像配对；未能验证镜像与作者原始二进制逐字节相同，明确披露。保留原句，不从本地 Mask 生成或修订描述，不把文本重复当成独立病例。

LViT §IV-A 明确说明文本包含单侧/双侧、病灶数量、位置，标注流程有基于 Mask 的质量检查。因此本实验是“推理时提供额外逐图医学文本标注的感染分割”，不能作为普通临床报告或通用多目标 RIS 的证明。

公开逐切片划分含相同 CT 体积跨集合。使用文件名可追溯的整组重分配：Jun coronacases 按 CT case；radiopaedia 按网站患者 case ID 合并不同 series；Morozov 按 study；无法恢复患者身份的全部 bjorke 图片保守地归为一个来源桶。原图/模型输入精确相同的图片先合并相关组，再去重。每组统一分到该组原始成员涉及的最高优先集合 test > val > train，保留原始测试病例方向，不按模型表现或病灶面积选择划分。记录全部移动及去重，三组和三个 seed 共用唯一冻结清单。

这保证可追溯来源组不重叠；未证明所有跨来源患者身份独立，也未排除全部近重复。原始论文分数仅作背景，不与重新分组后的测试结果直接排名。

## 结构与初始化

DINOv3-S/16 使用通用预训练权重，冻结并 eval；取第 3/6/9/12 层 16x16x384 特征。四套原 Adapter 转为 64 通道、64/32/16/8 网格；现有 Swin Pixel Decoder 每层 2 blocks、4 heads、window 8、shift 0/4、无 dropout/drop-path。

E0 使用普通 64->1 线性卷积头。E1/E2 则使用冻结 CLIP ViT-B/32 文本 EOS 512 维表示，可训练文本 512->64 与像素 64->64 投影，归一化点积乘可训练尺度并加偏置；初始尺度 10，上限100。E1 固定文字，E2 原始逐图文本。没有 VMamba，也没有新交互层。

每个 seed 的 E1/E2 全部可训练参数同初值；三组 Adapter 和 Decoder 公共部分同初值。各组独立参数、优化器；不加载 ISIC/RefCOCO 等历史分割权重。CLIP 最大77 token，必须验证零截断。E2-E1 是主要文本特异增益；E2-E0 为整体增益；E1-E0 含预测头结构差异，不能单独解释为语言语义收益。

## 输入与训练

256x256 等比例缩放并填充；RGB 灰度通道保留源 PNG 内容；ImageNet 通道 mean/std 用于 DINO。部分原始 Mask 含0/255以外的灰度值；按作者 LViT Load_Dataset.py 的定义以原始灰度 > 0 二值化，最近邻缩放，padding 不计损失。验证和测试恢复原尺寸。第一轮关闭所有几何和强度增强，使位置描述一致，并可精确复用冻结的 DINO 特征。三组获得完全相同的输入；这属于冻结特征 probe，不在此轮比较增强或 Encoder 微调。

默认 batch64（最后不足64的 batch保留），以预检实测确认显存余量；冻结特征常驻 GPU，三组按同一批次顺序依次更新。显存预检若不足，三组统一减半并在正式冻结前记录，不依靠填满显存改变训练语义。

AdamW lr1e-4、weight_decay0.01、betas(0.9,0.999)、eps1e-8；warmup5 epochs 后逐更新 cosine 到1e-6。BCEWithLogits + 逐图 SoftDice，等权，smooth1，gradient norm clip1，BF16，损失和匹配计算 FP32。冻结编码器缓存不参与梯度。

## 预实验与正式预算

先进行三组 seed42 各50轮预实验，独立输出到 pilot，不读取测试预测、不将其作为正式结果。预先规定统一预算选择：若三组中任一组最高验证 macro Dice 出现在第41-50轮，或者某组最后10轮最高验证分比前40轮最高分提高至少0.002，则所有正式训练统一150轮；否则统一100轮。规则只决定共同预算，不为某组单独延长，也不选择不同学习率。

预算决定写入 budget_decision.json；正式 seed42/43/44 全部重新初始化，按各自100/150轮完整 cosine 计划训练，batch、样本顺序和增强一致。无 early stopping。每轮以原尺寸验证集 macro Dice 最大选模，精确相同保留最早轮。

全部9个正式训练 checkpoint 选择锁定后，才运行测试指标。测试结果不参与预算、prompt、阈值或结构选择。正式配置和源码/data/text hashes 在训练前冻结；异常中断可恢复。

## 评价与保存

主指标原尺寸 macro foreground Dice；同时保存 macro/pooled Dice、IoU、precision、recall，Prec@0.5/0.6/0.7/0.8/0.9、Boundary F1、HD95 像素/对角线归一化、逐图和逐来源组指标。沿用项目 evaluation.py 和 stage1_visual/analyze.py 的边界定义，BF1 容差为原图对角线0.5%；无 spacing 不报告毫米。双方空 Mask Dice/IoU/BF1=1、HD95=0；单方为空 BF1=0、HD95=对角线；若存在空 GT，额外报告阳性子集和阴性误报。

报告每个 seed 与跨 seed 均值/样本标准差，重点 E2-E1 和 E2-E0 的逐 seed 配对差值。补充按来源组聚类、且在同一组内保留全部切片的配对 bootstrap；它仍不代表未核实身份的独立患者区间。三个 seeds 不作为显著性的充分证明。

参数量实测；记录峰值显存、训练耗时以及含必要 Encoder/Decoder/文本头的完整 GPU batch1 推理延迟。保存最佳权重、完整日志、预测 logits、指标和清单；训练中只另留一个恢复用 latest。所有正式评估验证成功后删除 latest/smoke 权重，保留9个正式最佳，pilot只保留3个最佳和日志。

来源：https://github.com/HUANGLIZI/LViT ，https://arxiv.org/pdf/2206.14718 ，原始及镜像 URL 见 sources/download_provenance.json。
