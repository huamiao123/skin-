Stage36：顶会作者融合块的受控适配
目标：用户要求再尝试公开顶会模块。检验跳跃连接的条件通道筛选，以及最高分辨率局部的多尺度/多频率融合。两条路线独立，不叠加。
论文和作者代码：
1. UCTransNet, AAAI2022, https://arxiv.org/abs/2109.04335
   https://github.com/McGregorWwww/UCTransNet
   nets/UCTransNet.py: UpBlock_attention -> nearest upsample -> CCA -> concat -> 2 ConvBNReLU。
   train_model.py 标准分支没有对整网额外apply初始化。CTrans.py内部初始化属于未采用的CCT。保留作者默认初始化，只初始化新块，全部预训练和原Swin参数不重置。
   适配：原模型四次上采样、通道递减；本模型三个跳跃层均64通道。只借用CCA完整上采样块，不声称重现UCTransNet整网或CCT。
   对照：完整构造后把CCA替换为ReLU(x)，保留原CCA最后激活、去除学习的通道缩放，保证周围卷积初值逐项相同。
2. MADGNet, CVPR2024, https://arxiv.org/abs/2405.06284
   https://github.com/Inha-CVAI/MADGNet
   IS2D_models/mfmsnet.py: UpsampleBlock -> bilinear upsample -> concat -> ConvBNReLU -> MFMSAttentionBlock -> ConvBNReLU。
   MFMS内部保留作者多尺度分支、16组top频率DCT、通道/空间权重、alpha/beta和残差。
   作者整网构造和训练入口未发现额外全网初始化；采用PyTorch模块默认初始化。原ResNet预训练加载与本次MambaVision权重独立。
   适配：仅最高分辨率64x64融合处使用原块，64通道对应原DCT56x56不改。注意作者第二尺度32x32分支也会adaptive pool到56；此为原块行为，明确保留，不宣称所有分支均降采样。
   对照：完整构造后仅把attention_layer替换为Identity，周围卷积和Swin初值相同。
   未移植原论文E-SDM、边界/距离监督、四层解码、域泛化训练或其他骨干，不把结果当原论文复现。
来源与实现：从已存作者源码提取未经改写的AST节点，避免导入未用ResNet和CCT依赖；记录文件哈希和节点行号。
数据：train1886/val762/test867；测试集已反复暴露，不能再称全新独立测试。当前ID划分非已确认患者互斥。
共同设置：256输入、seed42、物理/有效batch64、100轮、沿用300轮cosine前100轮，encoder stage3/4可训练但全encoder保持eval，作者解码BN训练。
评价：训练只看val，按val逐图macro Dice最高权重（同分最早）选模，阈值固定0.5；五组训练全部结束才做完整test。macro/pooled Dice与IoU、Precision/Recall、Prec@、boundary F1、HD95、threshold Jaccard均保存。
解释：concat优于add不能归功注意力；注意力优于其匹配concat才支持模块本身。单seed只能筛选，不能宣称稳定提升或文章创新。条件通道过滤与误检的关系尚未被证明。
暂不选TransFuse：其BiFusion原本服务两个独立编码分支，当前解码器/跳跃特征并不满足同样语义来源，先选择接口更匹配的上述路线。

正式训练前数值修正（v2）：原生adaptive max/avg pool CUDA backward非严格确定。BF16预检恢复后更新最大差1.34468e-4，超过原1e-4阈值，故未启动正式训练。独立原块梯度复验BF16相对L2约3e-5、FP32约1e-9。两个MADG组的整个作者上采样块均用FP32运算、返回时转回原dtype，其余网络继续BF16。保留作者源码、数学结构和原检查阈值。v1预检记录归档，所有组重新通过检查后才正式训练。
