# ISIC2016 医学文本引导分割：seed 42 探索协议

2026-10-18：用户要求停止 RefCOCO，并试验小规模 ISIC 图文分割。本阶段独立运行，不改变历史模型代码与结果。只运行 seed 42。

## 数据与文本

图像及二值 Mask 来自 ISIC 官方 2016 Task1 ZIP（原始 900 train / 379 test），MedVLSM 作者标注来自固定 commit `dc36a3af83035efa344c8ce26397528ad173569e`，原始记录为 810 train / 90 val / 379 test。逐图使用原文件名配对，并保存源 ZIP 和图像/Mask SHA256；解压校验后删除重复 ZIP。

去重顺序固定为 test > val > train：同 ISIC ID、原尺寸解码像素相同或 256 方形缩放像素相同则只保留前一份。不得根据预测指标决定删除样本。实际保留数量和全部删除记录以训练前冻结的 data_manifest.json 为准。不宣称患者级无重叠。新 Decoder/Adapter 从 seed42 初始化，禁止载入旧 ISIC 或 RefCOCO 训练出的分割权重，避免跨年份图像重叠造成训练污染。

实际核查后为806训练/90验证/379测试：去除1条重复训练记录及3个与测试相同的训练ID。过去的ISIC2018开发图与ISIC2016可能跨年份重叠，因此本轮测试是本次训练的留出评估，不称整个研究流程的全新外部验证。

统一使用 MedVLSM P3（颜色 + 形状 + 类别词）。作者论文说明形状/颜色来自 VQA，数量/大小/位置来自 Mask；本轮排除后者，不使用 JSON bbox，不把 GT 生成的位置/面积提示作为模型输入。原文所有病例使用 `skin melanoma`，该词不是逐例诊断；将其统一替换为 `skin lesion`，所有划分一致。固定文本对照始终为 `skin lesion`。文本是作者生成的属性提示，质量不等同于医生报告，也不宣称模拟了真实诊疗输入。

不按验证或测试效果改选 P1–P9，不生成新的 GT 文本。这轮任务为属性文本辅助病灶分割，不作为多目标自然语言指代推理的证明。

## 模型与控制变量

- DINOv3-S/16 + 原 Adapter + Swin Pixel Decoder + text-to-pixel head。
- VMamba-T + 原 Adapter + 相同设计 Swin + 相同文本头。
- 两视觉 Encoder + 原 Concat 1x1 Conv + 相同 Swin + 相同文本头。
- DINOv3 固定文本训练对照：与 DINOv3 主组全部参数同初值。

视觉 Encoder 和 CLIP ViT-B/32 Text Encoder 全部冻结并 eval。每个模型的参数和优化器独立。DINO/VMamba Adapter 统一输出 64/32/16/8 四尺度、64 通道；Swin 每尺度2 blocks、4 heads、window8。CLIP EOS隐藏表示512维，经可训练投影到64维，与像素向量归一化点积，通过可训练尺度/偏置产生 logits。Decoder、文本/像素投影在三组中同初值。参数量须在 preflight.json 实测。

## 训练预算

固定100 epochs，batch16，保留最后不足16的 batch；每个 epoch 每个训练图恰好一次。三组及固定文本对照完全同样数据顺序、增强、更新次数。AdamW lr1e-4，weight_decay0.01，betas(0.9,0.999)，eps1e-8；逐步 warmup5epochs，再 cosine 到1e-6。梯度范数裁剪1；BF16混合精度，loss和匹配头FP32；无early stopping，不根据测试结果改变预算。损失为有效区域 BCEWithLogits + 逐图 SoftDice，权重1:1，smooth1。

输入256等比缩放并填充；Mask 最近邻。填充区域不参与 BCE/Dice。训练只有0.9–1.1倍亮度/对比度变化，保留颜色类别；不改变色相，不翻转/旋转/裁剪。图像及冻结特征缓存驻留GPU以减少重复IO；encoder batch64，优化器batch16，缓存不改变样本或训练语义。验证无增强。

## 选模与最终评估

每轮在验证集按原图尺寸 macro Dice 选择最佳 checkpoint，相同分数保留最早轮。四模型训练全部结束、选模锁定后，才计算测试集预测指标；准备数据时读取 Mask 仅用于配对、二值性、尺寸等检查。正式测试不用于调参。

预测 64x64 logits 先双线性到256，去padding，再双线性还原原图，阈值0。共享 evaluation.py 的 macro/pooled foreground Dice、IoU、Precision、Recall、Prec@0.5/0.6/0.7/0.8/0.9、threshold Jaccard(0.65)定义。最终每图 BF1 和 HD95 复用 stage1_visual/analyze.py：侵蚀边界，BF1容差原图对角线0.5%，HD95合并双向边界距离95百分位；另报HD95/对角线。记录参数量、原图逐图指标、64x64 logits（可重建原尺寸预测）、原图完整边界指标和完整 GPU 推理延迟。

测试诊断：正确P3、临时固定文本、不同P3错配文本（确定性从另一图选择不同字符串）及输出变化。仅DINO具有独立训练的固定文本容量对照，不将临时干预当成同等对照。单seed，仅报告逐图配对差值及图像bootstrap区间，不推断种子稳定性；不追加新种子。

## 工程与保留规则

正式训练前通过：源数据/图像内容去重，编码器冻结检查，共享初值/独立参数检查，有限梯度，文本影响预测，padding loss不变性，原图几何恢复，短程两轮训练/重启恢复/完整指标评估。冻结本阶段全部源码、协议、数据和文本清单hash。

训练中只保留各组最佳权重和一个组合 latest（故障恢复）。训练和锁定评估都成功后删除 latest及smoke权重，保留四组最佳权重、日志、原始数据、来源和指标。不会保存每轮权重。

来源：[ISIC官方数据](https://challenge.isic-archive.com/data/)，[MedVLSM数据协议](https://github.com/naamiinepal/medvlsm/blob/main/DATASETS.MD)，[MedVLSM论文§2.3](https://arxiv.org/pdf/2308.07706)。本实验使用其文本资源，不声称完整复现其模型或原始划分指标。
