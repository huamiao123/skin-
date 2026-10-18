# 第八轮：RefCOCO 图文指代分割基线

2026-10-18，用户授权三种视觉分支的RIS首轮实验，仅使用seed42。新增一组DINO固定文本训练对照检验文本增益。当前为从皮肤单目标分割转向自然图像指代任务的可行性实验，不是CRIS完整复现。

## 数据、来源及语义

RefCOCO UNC标准划分：train16994图/42404目标/120624表达；val1500/3811/10834；testA750/1975/5657；testB750/1810/5095。每个epoch访问全部120624训练表达一次，不按图像数定义epoch，也不随机只抽每目标一句话。三主组和固定文本组读取同样目标、同样表达索引及顺序。

原REFER站与其网页归档在当前环境连接失败，使用固定镜像jxu124/refcoco@9fba8200c5326e996f789191f095bd464ef1d09e，其Parquet保留raw_anns/raw_image_info和原始sentences。图像从COCO的S3端点获取，mask由COCO原始polygon/RLE用pycocotools栅格化，拒绝用bbox替代mask。核对所有图/标注关联、尺寸/解码、完整划分计数、样本唯一性与图像ID隔离，保存来源/哈希。没有取得原ZIP作逐字节一致性验证，明确标记镜像来源。

256×256保持长宽比bilinear缩放，中心填充近似ImageNet均值RGB(124,116,104)，mask nearest。训练有效像素mask排除padding；保留整图不裁目标。关闭翻转、旋转、随机裁剪，不修改表达的left/right等语义。每epoch每图固定同一组brightness/contrast/saturation系数[.9,1.1]；共享三视觉分支的增强图，padding在颜色变换后还原均值。不存在病灶面积/诊断派生语言或统一模板伪装成真实表达。

val与test都恢复原尺寸：64² FP32 logits先用OpenCV INTER_LINEAR到256²，再精确去除padding，再INTER_LINEAR到原宽高，logit>=0为前景（sigmoid阈值.5）。不在native尺寸调整阈值或后处理。官方表达为基本评估样本；不把不同表达当作独立图像做统计推断。

## 四组模型（只有一个seed）

1. dinov3_text：冻结DINOv3-S/16→原四层Adapter→Swin Pixel Feature Decoder→文本匹配头。
2. vmamba_text：冻结VMamba-T→原Adapter→同设计Swin→同匹配头。
3. fusion_text：两Encoder冻结→各自原Adapter→原Concat+1×1Conv→同Swin→同匹配头。
4. dinov3_fixed_text：与1完全相同初始化/参数/数据/预算，所有训练/评价文本替换为“segment the object”；是真正从头训练的固定文本对照，仅用于DINO的文本收益比较。

Adapter保留64通道及64/32/16/8网格，复用stage3/4类的初始张量。Swin保留各级2块、4heads、window8/shift4，删除固定64→1 readout；保持64²×64像素特征。两侧匹配投影Wv=Conv1×1(64→64)、Wt=Linear(512→64)，FP32 L2归一化后余弦匹配。z=exp(log_tau)*cos(Wv V,Wt t)+b；初始tau10/b0，tau上限100。可训练的仅Adapters、融合、Swin、Wt/Wv、tau/b；无额外图文Cross-Attention或视觉新融合。

CLIPTextModel/tokenizer固定openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268；文本Encoder冻结/eval。max77 tokens（含特殊token），attention mask传入，截断数单独记录；取EOS pooler_output的512维隐藏表示，未使用CLIP图像编码器，未假设DINO/VMamba天然与CLIP对齐。相同表达只编码一次，保存FP32句子缓存与源文件hash；Wt保持可训练。三个主组Wt/Wv/Swin初值相同，参数与优化器各自独立；固定文本DINO与真实文本DINO完整初始state相同。

## 训练口径

seed42；30epochs，每epoch7539步（120624/16），每模型226170次更新。batch16，保留全部样本；AdamW lr1e-4、weight_decay.01、betas(.9,.999)、eps1e-8；逐step线性warmup2epochs，再cosine至1e-6。BCEWithLogits+逐样本SoftDice(smooth1)，两项权重1:1；BCE按各样本有效像素归一化，Dice排除padding再计算。BF16 autocast、FP32匹配头/loss、clip_grad_norm1，无early stopping。

顺序seed=42+epoch×100003，颜色seed额外+17，独立于batch或模型。Encoder永久eval/no_grad；每epoch共享新增强图像的冻结特征，val特征固定。当前48GB4090用GPU特征缓存减少重复前向，同时保持batch16。如果预检资源不满足，需在正式训练前记录实现调整，不能在比较中改变有效batch或样本集合。

只按val**逐表达Mean IoU最高**选模，精确相同取较早epoch；最后一轮另存，不混用指标择优。TestA/TestB只在全部模型的checkpoint选择锁定后评分一次。第一轮仅seed42，不将图像样本数当成多seed证据，也不与ISIC成绩直接比较。

## 评价和文本作用

主报告native逐表达mIoU/Overall IoU/Prec@.5/.6/.7/.8/.9；同时macro/pooled Dice、Precision/Recall、Boundary F1及HD95像素/对角线归一化。复用evaluation.py计数和stage1的边界定义：erosion边界、容差.005×原图对角线；单空mask HD95=对角线，双空=0。Threshold Jaccard仅为ISIC特有，本轮不作排名。

每epoch选模只计算区域指标，最终checkpoint计算全部边界指标。额外报告先在同图表达内平均、再对图平均的image-balanced mIoU。配对bootstrap按图像聚类保留全部表达，不能独立重采样142210条表达声称独立图数。三主组比较最佳单分支与双分支时同时列出全部预定比较，避免只挑对照。

保存每表达64² FP32 logits、原尺寸几何和官方GT映射，可重建原图预测；不为每epoch保存巨量全尺寸概率。最后验证保存logits重算counts与训练选模记录完全一致。来源图和原尺寸可视化仅保留本地。

同图多目标诊断固定按image_id/ref_id排序选择不同目标的真实表达对，检查mask改变及目标切换：各表达用各自GT评分，并与另一目标GT比较，不假设任意乱配文本应产生空mask。验证集对真实文本模型作固定文本替换干预，只作响应诊断，不替代从头训练的固定文本对照。目标/关系复杂度没有人工标签时不自动命名“理解关系”或“复杂表达”。

## 运行与完成条件

先落实全部图像/文本/mask，完成预训练/历史源码hash和模型梯度/文本作用/letterbox恢复/有效像素loss校验，再执行真实数据短程中断恢复预检。正式训练与预检目录独立。支持断点保存优化器/scheduler进度、RNG和选模状态；未提交epoch可按相同增强重算。

后台监督负责数据/文本准备、预检、30轮训练、全部原尺寸评估、同图多目标诊断、配对统计及历史hash复核。只有上述全部通过才写complete。状态/日志区分“下载中”“预检”“训练中”“评估中”，不能把数据准备报告成已经训练。

## 参考

- https://github.com/lichengunc/refer （数据定义）
- https://www.tensorflow.org/datasets/catalog/ref_coco （划分计数）
- https://huggingface.co/datasets/jxu124/refcoco （使用的镜像，固定版本见上）
- https://huggingface.co/openai/clip-vit-base-patch32 （冻结文本预训练）
- https://arxiv.org/abs/2111.15174 （CRIS，像素级文本对齐参考；本轮不等同复现）
