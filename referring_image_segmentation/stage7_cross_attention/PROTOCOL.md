# 第七轮：DINOv3 / VMamba Window Cross-Attention

冻结日期：2026-10-18。用户明确要求运行Cross-Attention交互实验；沿用此前讨论的受控实验路线。

## 依据与范围

第六轮Concat在seeds43/44/45的mean Dice均超过DINO和容量对照，但pooled Dice均值略低、HD95更高，新增错误没有稳定减少。第五轮动态门控也未超过静态残差。因此本轮检验：DINO位置相关的Query能否从VMamba窗口内检索更合适的信息，优于静态局部卷积及已有融合。

仅新增seed42两组300epochs：window_cross_attention与local_conv。均为冻结DINOv3-S/16+VMamba-T、相同Adapter及既有Swin Pixel Decoder。旧阶段只读。未追加文本、双向交互、多seed、不同窗口扫参或新loss。

## 新模型与容量对照

四尺度各自Adapter输出64通道，空间网格64/32/16/8。DINO原始多层网格仍是16×16，上采样不意味着原始高频细节增加。

Cross-Attention每尺度一个块：分别对DINO与VMamba做逐位置LayerNorm；DINO生成Q，VMamba生成K/V；8×8非重叠窗口、4heads、每head16维、标准缩放1/sqrt(16)、可学习相对位置偏置225×4，最后线性输出投影。无额外FFN、无新增shift、dropout/drop-path均为0；原Swin Decoder的移位窗口结构保持原样。最粗8×8网格恰为一个窗口。

输出F_i=D_i+0.1*CrossAttention(LN(D_i),LN(M_i))。0.1为两组一致的固定残差系数，不是动态门控，也不加载已训练分割模型。所有Adapter/Decoder从与旧seed42相同的初始化开始。

local_conv采用同样的两路LayerNorm和残差：concat128→1×1Conv到72→GELU→depthwise7×7→GELU→1×1Conv到64。Cross块17796参数/尺度，Conv块17816；四尺度总可训练参数分别670321/670401，相差80（约0.012%），编码器和decoder相同。该对照只近似匹配参数；固定7×7邻域与8×8分块窗口的感受野/边界形式及FLOPs不同，不能称为完全同构消融。两组新交互模块的初始函数不同，公共Adapter、归一化初值及Decoder一致。

## 固定训练口径

ISIC2018旧1886训练/808开发验证；输入256×256；ImageNet归一化；图像bilinear、mask nearest；训练保留旧灰度标签；H/V翻转和0～360°旋转各p=.5；数据顺序/增强seed=42+epoch×100003。

每组300epochs/9000updates；batch64，保留尾batch；AdamW lr=.001、weight_decay=.01、betas=.9/.999、eps1e-8；Cosine T_max300、eta_min1e-5；BCEWithLogits+per-image softDice（smooth1）；BF16 autocast/FP32loss，clip1；无warmup、early stopping、梯度累积。两个Encoder永久eval/no_grad，同epoch共享重新增强后编码的特征；新模型参数、梯度、优化器互相独立。

主checkpoint最高验证mean Dice，精确平局取较早；最低样本加权验证loss及epoch300作为辅助，全部导出。阈值.5。cudnn deterministic=True/benchmark=False，未声明全算子逐位确定性。当前设备实测48GB RTX4090；为公平比较保留batch64，以缓存共享冻结前向减少重复计算，不以填满显存为由改变有效batch或增加实验。

## 参照和评价

统一比较同seed42、同Swin、同选模规则的DINO/VMamba/Concat/概率平均（第四轮）与双Adapter/静态残差/动态门控（第五轮）。预检核对预训练hash、旧源文件、配置及数据，并在当前硬件重新前向旧单模型的三种checkpoint核对区域指标和概率。不能将第六轮三seed均值当成seed42配对参照。

主比较Cross−LocalConv、Cross−Concat；辅助为Cross−静态残差、Cross−动态门控、Cross−概率平均和两新模型对DINO。报告所有macro/pooled Dice、IoU、Precision/Recall、Prec@.5/.6/.7/.8/.9、Boundary F1、HD95像素/对角线归一化、ISIC Threshold Jaccard。保存逐图预测和错误分层，尤其纠正/新增/净减少错误、原错误修复率、原正确像素破坏率、VMamba独有正确位置的恢复、边界/内部/背景、前景概率与面积分组。两种率分母不同，不能直接相减。

配对逐图bootstrap只描述重复使用开发集的探索性差异，不证明跨seed或独立泛化。报告参数、完整双Encoder推理延迟、吞吐和显存；本轮重新测量所有参照，避免混用昨天24GB机器的时间。

## 机制和运行校验

先测试窗口划分/逆变换、手写注意力与SDPA的前向和梯度一致性；测试Q来自DINO、KV来自VMamba、窗口外隔离以及所有参数梯度有限、两分支参与预测、Encoder哈希不变。

正式训练前进行两轮真实数据中断恢复、六组checkpoint导出、完整指标分析和注意力干预预检。仅使用正式Cross主checkpoint做不重训的uniform（窗口内均值V）和same_position（仅同位置V）干预；与learned比较，用于诊断当前模型对空间检索的依赖。分布变化可能导致下降，不能把它们当作分别训练后的公平对照，也不用于重新选模。

后台监督器在两组正式训练和全部评估/干预/历史hash校验通过后才写complete。自动保留断点；本轮完成后再按事实决定是否值得多seed复验。

## 实现参考

- [PyTorch SDPA文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)：本机PyTorch2.2.2接口以本地文档/预检为准，显式dropout_p=0。
- [Swin论文](https://arxiv.org/abs/2103.14030)：窗口与相对位置表示的参考。本轮新增模块为跨Encoder注意力，不能把原论文的自注意力性能结论直接迁移过来。
