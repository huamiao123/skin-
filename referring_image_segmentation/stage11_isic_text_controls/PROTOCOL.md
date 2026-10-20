# ISIC2016：补齐无文本／固定文本／逐图文本的 seed42 对照

2026-10-20 用户要求切换 ISIC2016。只使用seed42；MosMed暂停不恢复。本阶段补全三组DINOv3+Swin对照：E0无文本；E1固定 `skin lesion`；E2逐图属性文本。原stage9的E1/E2已经独立完成100轮，直接核验并复用它们的最佳权重、原始训练日志及结果，只新训练缺少的E0。明确记录复用来源，不声称重新训练过E1/E2。

## 数据与文本

直接以只读方式复用stage9已校验的806训练／90验证／379测试清单与256缓存。不重新划分、增加或按指标删除样本。全部图像/Mask/文本的历史hash先核验；data_manifest/text_provenance原文快照到本阶段，lineage.json记录来源。

原始ISIC2016官方数据、MedVLSM固定commit及去重规则见历史协议。患者独立性未验证；该测试集在stage9已经查看，因此本次是开发性补充对照，不称全新独立外部验证。

E2严格沿用P3颜色/形状属性（来自VQA），统一类别词 `skin melanoma` -> `skin lesion`；不引入GT生成的数量、面积或位置。文本不等同于医生报告，也不能证明多目标自然语言指代能力。E1/E2输入的历史CLIP缓存保持原样。

## 架构与控制变量

通用预训练DINOv3-S/16冻结eval，取第3/6/9/12层特征；原Adapter四尺度64/32/16/8、64通道；原Swin每层2blocks、4heads、window8、shift0/4。E0普通64->1 Mask Head；E1/E2冻结CLIP ViT-B/32文本EOS512，经512->64投影，与64维像素投影归一化点积，通过可训练尺度/偏置输出logits。

E0从seed42新初始化，Adapter/Swin公共权重与历史E1/E2同初值。E1/E2全部可训练参数同初值，预检逐字节核验初始hash，并重放它们的完整验证分数以证明模型/输入/几何兼容。历史E1/E2权重仅用于比较和评价，不载入E0初始化。

## 新E0的训练预算

完全复刻历史ISIC口径：100epochs、batch16、不丢最后小batch，每轮806样本各一次、每轮51步，总5100步；AdamW lr1e-4、weight_decay0.01、betas(0.9,0.999)、eps1e-8；warmup255步，再逐步cosine到1e-6；BCEWithLogits+逐图SoftDice等权、smooth1，梯度范数裁剪1，BF16，损失FP32。

256等比缩放并填充，Mask最近邻；padding不计损失。沿用历史0.9-1.1倍亮度/对比度增强，不改色相，不翻转/旋转/裁剪。直接继承历史FeatureCache.encode，随机因子种子42+epoch*100003+17，数据顺序种子42+epoch*100003。每轮重新编码增强图像，冻结特征常驻GPU；只计算DINO分支。与MosMed的batch64/无增强不同，这里是匹配既有ISIC实验，不是仅切换数据集的跨数据集消融。

无early stopping、无预算/学习率搜索，不因测试结果延长。每轮原尺寸验证macro Dice最高选模，精确同分保留最早轮。

## 评价与保留

E0选模锁定后，按同一原图评价代码重放三组val/test。E1/E2应复现历史验证和测试宏平均Dice；若不符，停止并检查，不能混合不兼容结果。

主指标macro foreground Dice；同时保存macro/pooled Dice/IoU/precision/recall、Prec@0.5/0.6/0.7/0.8/0.9、ISIC threshold Jaccard0.65、BF1、HD95像素及对角线归一化、逐图记录、参数量、显存和推理耗时。复用evaluation.py及stage1边界定义。阈值logit0，64logits->256->去padding->原图。

重点E2-E1解释逐图文本增益；E2-E0为整体收益；E1-E0还包含输出头结构差异。只跑一个seed；图像配对bootstrap不估计患者或训练随机性。没有错误文本实验或额外视觉分支。

训练前完成冻结编码器、相同公共初值、历史E1/E2初值hash、独立有限梯度、padding不变性、原图几何和两轮断点恢复检查。正式源码/清单hash冻结。训练中保留E0最佳及一个latest，完成评价后删除latest/smoke权重；历史E1/E2仍保留原路径，只在lineage中引用。
