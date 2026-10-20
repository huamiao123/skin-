# ISIC2016 seed42：Decoder之前的词级图文Cross-Attention

2026-10-20 用户要求更换融合方式，明确不再在Decoder之后融合。只运行seed42，两个新的同结构模型：predecoder_fixed_text固定 `skin lesion`；predecoder_image_text使用原逐图P3属性。两者从相同初值独立训练。MosMed保持暂停，不启动其他seeds、VMamba或错误文本实验。

## 结构与研究问题

Image -> frozen DINOv3-S/16 -> four original Adapters -> word-level Cross-Attention -> resize to four scales -> original Swin Decoder -> ordinary 64->1 Mask Head -> Mask。

Text -> frozen CLIP ViT-B/32 -> contextual word tokens -> trainable 512->64 projection -> predecoder Cross-Attention keys/values。

文本只进入Decoder之前的交互层；Decoder和最后Mask Head都没有额外文本输入。移除原先Decoder输出端的文本投影/像素投影/归一化点积/温度/文本偏置。原图Adapter、Swin、普通输出头保持与纯视觉E0完全相同的seed42初始张量。

DINO第3/6/9/12层各16x16x384，经原Adapter为16x16x64。每层在此原生16x16网格上执行一个单向Cross-Attention：视觉逐位置LayerNorm产生Q；CLIP词级隐藏状态先共享LayerNorm512、Linear512->64，作为K/V；nn.MultiheadAttention dim64、4heads、dropout0。更新为 x + 0.1*Attention(LN(x),words,words)，0.1固定，非新门控超参数搜索；无额外FFN、反向注意力或位置模块。然后按原流程缩放到64/32/16/8，送Swin。四层注意力参数各自独立，词投影共享。

CLIP保持原来通用预训练权重冻结eval，只缓存每种唯一文本的last_hidden_state(77x512)。PAD/BOS/EOS均不作为K/V有效token；关注实际内容词，预检验证padding不影响前向。未使用CLIP图像Encoder，不假定DINO与CLIP天然对齐。

主比较是新逐图文本与新固定文本，两者完全同容量、同初值、同预算，回答提前交互后逐图属性是否有额外价值。与历史晚期匹配头比较的是整体机制替换，同时改变融合时机、词级表示和输出头，因此不把所有差异归因于单独的“融合位置”。本实现借鉴早期图文交互的研究方向，不宣称复现完整LAVT。

## 数据与训练

沿用已核验的ISIC2016 806训练/90验证/379测试及MedVLSM P3颜色/形状属性，类别词统一skin lesion。不引入GT派生数量、面积、位置或bbox，不更换数据划分，不依据结果改选文本。患者独立性未验证；测试集已经用于历史开发，本次为单seed探索，不作为全新外部验证。

DINO与CLIP全部冻结，仅训练Adapter、Cross-Attention、Swin和普通Mask Head。100epochs、batch16、每轮806样本51步，共5100步；AdamW lr1e-4、wd0.01、betas(0.9,0.999)、eps1e-8；warmup255步、cosine到1e-6；BCEWithLogits+逐图SoftDice等权smooth1；clip_grad_norm1；BF16、loss FP32。无early stopping，不为某一组延长。

输入256等比缩放填充、Mask最近邻，padding不计loss。原亮度/对比度0.9-1.1增强；不翻转/旋转/裁剪/改色相。完全复用stage9图像增强实现，因子seed42+epoch*100003+17、数据顺序seed42+epoch*100003。每轮重新编码增强图；两模型共享相同冻结图像特征及数据顺序，参数/优化器独立。

每轮原尺寸验证macro Dice最高选模，精确同分留最早轮。两组训练全部完成、选模锁定后才运行新的测试预测，阈值logit0。正式配置、源代码、数据及word缓存hash事先冻结。

## 验证、评价和保留

预检：公共视觉初值与E0逐字节一致；两新模型全部初值相同但不共享参数；冻结Encoder hash不变；有效梯度；文本可改变Decoder之前的特征与最终输出；被mask的word padding变化不改变结果；decoder仅依赖融合后的视觉金字塔；有效区域loss、原图恢复和两轮断点恢复通过。

报告新旧模型的macro/pooled Dice/IoU/P/R、Prec@0.5/.6/.7/.8/.9、ISIC threshold Jaccard0.65、BF1、HD95像素/对角线归一化、逐图记录、参数量、峰值显存、耗时、完整GPU推理延迟。沿用evaluation.py与stage1边界定义。历史E0/晚期固定/晚期逐图的预测和指标只读引用。

主要配对新逐图-新固定；另报新逐图-旧逐图及新逐图-纯视觉。2000次图像配对bootstrap只描述当前seed和图像，不能代表患者独立区间或多seed稳定性。

训练中每组只保留best_dice和共同latest；成功评价后删除latest和smoke权重，保留两个新最佳、旧权重引用、日志和预测。

参考：LAVT https://arxiv.org/abs/2112.02244 （早期语言交互思想，非本实现完整复现）；PyTorch2.2.2本地MultiheadAttention.forward文档验证key_padding_mask布尔True表示忽略，need_weights=False使用优化注意力。
