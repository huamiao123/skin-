# ISIC2016：五组统一300轮，seed42

2026-10-20 用户明确要求跑300轮。新增独立阶段，保留所有100轮结果。五组全部重新初始化可训练部分并独立训练300轮，不从已完成100轮的分割checkpoint续训。只有seed42；MosMed与其他seed保持暂停。

## 模型矩阵

- E0_visual：DINOv3 + 原Adapter/Swin + 普通64->1 Mask Head，无文本。
- E1_fixed_text：原Decoder输出端CLIP句子-像素匹配头，固定skin lesion。
- E2_image_text：相同后置匹配结构，逐图P3颜色/形状属性。
- predecoder_fixed_text：原stage12词级Cross-Attention在四层Adapter之后、Swin之前，固定skin lesion；最后普通Mask Head。
- predecoder_image_text：相同前置结构，逐图P3属性；最后普通Mask Head，Decoder后不使用文本。

以上直接复用既有模型定义。所有Adapter/Decoder公共初值与原seed42保持一致；固定/逐图文本配对全部参数同初值、参数与优化器独立。DINOv3和CLIP均冻结、eval、使用相同通用预训练权重，不载入任何历史分割权重。

前置模块仍是四个16x16视觉网格、64通道、4heads单向词级Cross-Attention，x+0.1*Attention，原Swin window8、各尺度2blocks；不新增融合、loss或Encoder。PAD/BOS/EOS屏蔽，既有50种文本缓存只读复用。

## 数据与统一训练

806训练/90验证/379测试、256输入，沿用stage9去重清单和P3颜色/形状；固定类别词skin lesion，不引入GT数量/面积/位置。患者独立性未验证，测试集已经用于开发。

只有预算与cosine总步数相对100轮版本改变：每组300epochs，batch16，保留尾batch，每轮806样本51步，共15300次更新。AdamW lr1e-4、wd0.01、betas(0.9,0.999)、eps1e-8；warmup仍5轮/255步，然后对完整剩余15045步执行cosine，到最终1e-6。无early stopping，无梯度累积，clip_grad_norm1，BF16和FP32损失。

BCEWithLogits+逐图SoftDice等权smooth1；有效区域排除padding。256等比缩放填充，mask nearest，亮度/对比度0.9-1.1，不改色相、不翻转/旋转/裁剪。原增强因子seed42+epoch*100003+17，顺序seed42+epoch*100003，五组每轮共用相同图像与冻结特征。

这统一的是ISIC2016图文对照的300轮预算；仍保留ISIC2016的lr1e-4/batch16/属性一致增强，不把它称为早期ISIC2018 lr1e-3/batch64/几何增强的相同口径。

## 选模、评价与比较

每轮原尺寸验证macro Dice最高选模，精确同分保留最早轮。全部五组完成300轮并锁定checkpoint以后，才运行新测试预测。阈值logit0，64logits->256->去padding->原图。不会因测试表现延长或换提示。

全部新模型同样报告macro/pooled Dice/IoU/precision/recall、Prec@0.5/.6/.7/.8/.9、ISIC threshold Jaccard0.65、BF1、HD95像素/对角线归一化、逐图指标、参数量、完整GPU推理延迟、显存和训练耗时，复用evaluation.py和stage1边界规则。

主要比较为相同300轮预算内：后置逐图-后置固定；前置逐图-前置固定；前置逐图-后置逐图；文本组-纯视觉。前后方案同时包含词级/句子表示与输出头变化，不将全部差异归因位置。另列已完成100轮结果作预算背景，不能将100/300cosine计划不相同的前100轮当成同一路径。

只用seed42，2000次图像配对bootstrap不代表患者级或训练随机性。预检确认五个初始hash与100轮对应模型相同、参数独立、有效梯度、正确句子/词输入路由、冻结Encoder、padding/几何、原图指标及两轮断点恢复。正式源码、缓存和协议hash冻结。

训练中只保留每组验证最优及一个共同latest恢复文件，完成评价和核验后删除latest/smoke权重，保留五个新最佳权重、日志、预测、指标及历史结果引用。
