医学图像分割主会论文与公开源码筛选：面向当前 MambaVision-S＋Swin 工程
核查日期：2026-11-11；本轮只做检索、源码核查与接口验证，未训练新组合。

**结论**

当前可以停止自行创造“注意力小变体”，先直接复用作者的成熟实现。首轮保留三个互相独立的方案：
A. MambaVision-S＋作者完整 EMCAD decoder；
B. MambaVision-S＋现有 Swin decoder＋作者 CGA；
C. MambaVision-S＋现有 Swin decoder＋作者 MFMS。
A检验完整解码机制，B检验decoder与skip的条件交互，C检验多尺度多频率信息。先分别跑，只有形成稳定、互补的收益后才考虑合并。不要第一轮同时叠A+B+C。

这是一项可行的复用/适配实验计划，还不是新的算法贡献，也没有新精度结果。当前证据并不保证“更复杂模块”能解决泛化问题。

**检索范围与证据层级**

覆盖2020—2026年，重点CVPR、ICCV、ECCV、MICCAI主会，并补充AAAI、ACM MM、NeurIPS和ICLR相关论文。按“2D直接融合/解码”“训练与泛化”“3D完整架构”“提示或上下文分割”筛选。不是所有医学分割论文的穷尽清单；没有声称精读全部API候选。

API使用OpenAlex、Crossref、arXiv；Semantic Scholar请求受到429限流。三个查询为 medical image segmentation attention decoder、skin lesion segmentation transformer mamba、medical image segmentation domain generalization。每组最多45条/来源，时间2020—2026。各查询内部去重后111、102、101条；跨查询按DOI、arXiv ID及规范化精确标题合并为279条候选元数据。跨年预印本/会议版本仍可能未完全合并，语义相关性也未自动保证。

在API候选池之外，额外核查会议官网/CVF/OpenReview和作者GitHub。最终人工重点整理29项：25项主会工作，4项期刊/预印本/WACV补充。API池、人工目录与源码审查深度分别保留，不能用279替代“已精读论文数”。

发现检索辅助脚本的自动topic_brief混入与医学无关的“多智能体连通性”模板，已将这些自动简报隔离为INVALID_TEMPLATE_*，原始检索记录和日志保留。本文为人工核查后撰写，不采纳错误模板结论。

**我们的实际接口与问题**

当前256×256输入的MambaVision-S输出96@64²、192@32²、384@16²、768@8²。stage3_confirmation/models.py的adapter把四级都压成64通道。stage2_decoder/models.py的SwinPixelDecoder从8²开始，依次上采样到16²、32²、64²，与同尺度skip相加再进SwinLevel；最终1×1读出并上采样到256²。

因此本轮最直接的复用位置是64通道的多尺度特征、三处skip相加以及完整decoder。编码器Stage3/4可训练；不能继续用“骨干全部冻结”解释当前结果。

Stage29—32显示：复杂通道/尺度注意力没有稳定超过简单局部融合；最新Stage32逐图宏平均测试Dice约89.4715%，Stage29 control约89.5344%，local_fusion约89.8266%。都是既有单seed结果，不是本轮新模型结果。Stage31在验证集做的通道权重打乱/置一检查，也没有证明复杂权重的按图对齐带来稳定收益。

现有验证集诊断中，漏分像素很大一部分离真实边界超过5像素。该统计只能说明错误不全是窄边界偏移，不能直接推断“全局上下文不足”、边界不可见或病灶身份错误。训练后期损失下降而验证损失上升的问题也仍存在。后续模块必须同时检查漏分和误分，不能只展示边缘更漂亮或recall提高。

**三个优先方案的具体接法**

1. 完整EMCAD decoder

作者源码：lib/decoders.py中的EMCAD，由EUCB上采样、LGAG门控、通道/空间注意力与MSCB多尺度卷积串联构成。使用其构造参数channels=[64,64,64,64]，接现有四级adapter输出：最深8²作为x，其余按16²、32²、64²输入skips。读取最高分辨率d1，接现有1×1分割读出。它替换Swin decoder，不能称为“Swin保持不变”的模块消融。

本轮实际测得该64通道decoder为284,043参数，不含encoder、adapter和最终分割头。作者实现未改，FP32/BF16的前向、输入梯度、参数梯度均检查通过。原生[768,384,192,96]配置也能运行，但达到10,289,723参数，并同时取消原64通道压缩；不适合作为第一项“只换decoder”的公平对照。

第一轮可以保持现有单头损失以隔离decoder结构；这应写成“使用作者EMCAD模块的适配”，不能声称复现整篇论文的训练方案。若使用原论文多头监督，应单独记录为另一训练变量，不偷偷合并进结构收益。

直接来源：[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)、[作者源码](https://github.com/SLDGroup/EMCAD/blob/main/lib/decoders.py)。

2. 保留Swin，使用MambaLiteUNet的CGA

直接使用CrossGatedAttentionGate(F_g=64,F_l=64,F_int=32)。在16²、32²、64²三处，令g为上采样后的decoder特征，x为对应encoder skip，先得到CGA(g,x)，再与g相加并送进原SwinLevel。这样保留MambaVision与Swin，用作者交叉门控替代无条件skip输入。

CGA包含两路分组Mamba及卷积交互，最后生成空间门控乘到skip；它与“全图平均后给一个通道系数”不同。单个64通道CGA为68,611参数，三处独立使用为205,833参数。两路需相同空间尺寸且此配置通道相同。所有16/32/64分辨率均通过FP32/BF16检查。

不把MambaLiteUNet整体照搬成第二套backbone。尤其AMF内部的LGFM包含全局MultiheadAttention，不能因为标题含Mamba就把其全部计算当线性。64²特征上的注意力开销需另测。本轮优先拆CGA，完整MambaLiteUNet可以作为后续外部基线。

直接来源：[CVPR论文](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html)、[作者源码](https://github.com/maklachur/MambaLiteUNet/blob/main/models/MambaLiteUNet.py)。

3. 保留Swin，使用MADGNet的MFMS

直接使用MFMSAttentionBlock(in_channels=64)，放在现有adapter之后。首版固定处理三路skip，最深8²不变，再进入原Swin decoder。它组合多尺度、不同感受野和DCT频率信息；这是区别于单一全局均值通道注意力的现成机制。

单个模块86,148参数，三处为258,444。16/32/64分辨率的FP32/BF16检查均通过。源码中的DCT目标尺寸表以通道数作键，只支持32/64/128/256/512；直接把96、192、384、768通道送进去会出问题。现有64通道adapter恰好能作为接口，不能忽略这个约束。

原MADGNet还有分割、距离、边界等输出/监督组织。只复用MFMS应据实命名为MFMS适配实验，不是完整MADGNet复现。它是否改善我们当前错误仍是实验问题；跨域论文中的结论不等于我们已经证明存在同一种域偏移。

直接来源：[CVPR论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf)、[作者源码](https://github.com/Inha-CVAI/MADGNet/blob/main/IS2D_models/mfmsnet.py)。

**备选与排除理由**

UCTransNet的CCA是适合做现成通道注意力对照的模块；完整CCT则需重新匹配四级token网格与KV_size，且我们已经有Swin，额外Transformer收益必须验证。TransFuse的BiFusion也可拆，但原论文融合独立CNN与Transformer，移植到同骨干的skip/decoder属于结构适配。

EGE-UNet的GAB已通过接口检查，但它还接收粗预测mask。当前只有最终分割头，接GAB需要中间预测和监督安排，第一轮变化比CGA大，排第二梯队。

UACANet需要中间预测，其公开UACA实现中x.view(B,H*W,-1)没有先把BCHW变换为逐像素通道顺序。小张量检查显示这与flatten(2).transpose(1,2)不同。这是当前源码布局疑点，需要核实；不把它直接上升为“论文错误”或“方法无效”。本轮遵照不自行设计/重写模块的要求，暂缓而非默默修正后声称原样复用。

LoMix的concat分支在forward内部创建Conv2d，卷积不在该模块state_dict中。同一输入在eval模式连续调用，concat输出最大差2.4030728；add对照为0。这个实现不适合作为首轮直接移植对象。完整日志在compatibility_results.json，不推广为所有LoMix算子都不可用。

MCADS确有作者代码，但为TensorFlow/Keras；当前PyTorch项目直接复用需要跨框架重写。FPN-in-FPN仓库目前仅README，PMRNet目前仅模型文件、缺完整训练工程。这三项均不满足本轮“优先拿现成实现跑通”的目标。

3D模型与交互/文本模型也纳入目录，但分别改变空间维度或可用输入信息，不能因为发表场所强就自动排到当前2D全监督任务首位。

**本轮已经实际验证了什么**

compatibility_checks.py与compatibility_results.json记录作者源文件SHA256、服务器环境、输入形状、模块参数量及梯度检查。RTX A4000、PyTorch 2.2.2+cu121、mamba-ssm 1.2.2，未升级现有环境。

四个模块LGAG、CGA、MFMS、GAB：3种空间尺寸×2种精度，共24项。完整EMCAD原生通道配置及64通道配置：各2项，共4项。合计28项全部通过，未发现参与检查的可训练参数缺失梯度。

这是batch=2随机张量的接口与数值有限性检查，不是完整batch的显存评估、吞吐基准、训练稳定性证明或分割效果证明。没有用测试集比较这些新模块，也没有在本轮启动新训练。

EMCAD此前已被我们用于“设计启发”，但Stage29的自写GN/零初始化残差分支与作者LGAG/完整decoder不是同一实现。不能把旧结果当成“已经试过EMCAD且无效”。

**后续实验矩阵与解释边界**

共同条件：同一MambaVision-S权重、同一解冻范围、同一adapter、输入尺寸、数据增强、优化器、学习率轨迹、批量、损失和选模指标。当前100轮对应原300轮余弦前段的策略也要明确写入配置；若修正学习率日程，应重跑配套control，不能只改新模型训练。

R0：原MambaVision-S＋Swin，作为同协议control。若已有checkpoint与全部条件一致可复用，不机械重训。
R1：仅decoder改为作者EMCAD，channels全64。
R2：原Swin三处skip改为作者CGA。
R3：原Swin三路skip前加作者MFMS。
原local_fusion成绩保留为已有工程参照，不把它与新模块同时堆叠作为第一版。

逐图宏平均Dice为主要指标，同时保留pooled Dice/IoU、precision、recall、边界F1、HD95及逐图结果。先看验证集选最佳checkpoint，再统一用该权重跑完整测试，不用test挑epoch/阈值/模块。发现平均数微涨但大漏分恶化，不能称为对症改进。

候选若在验证集上有稳定收益，再用固定三seed确认，检查改善病例与退化病例；只在两模块解决的错误确有互补时做组合。比较模块置一/禁用与动态权重打乱时，需要说明是训练后干预诊断，不等于重新训练消融。

当前867张测试集已在多轮开发中被查看，不能再描述为从未接触的盲测。继续保留现有测试结果用于一致报告；最终论文结论还应由额外独立数据或外部测试加强。患者级独立性尚未确认，不能声称已有患者级无泄漏证明。

复用论文模块完全可以成为研究起点。论文要诚实说明借用来源与我们实际改变的部分；“三个已有模块拼起来”本身不自动构成足够贡献。后续价值应来自明确问题、适配必要性、受控比较、稳定收益与成本证据，不预先保证发表档次。

**源码公开程度**

本文把“能看到源码”“完整训练测试流程”“已训练权重”“可再分发许可证”分开记录。EMCAD使用UT Austin Research License，仓库文本列有学术研究使用和再分发限制；MambaLiteUNet、MADGNet快照未见LICENSE，不能默认MIT；EGE-UNet为Apache-2.0。当前没有发布整合后的第三方源码。原始许可文本和来源随快照保留，未来公开工程时需逐项处理实际条款。

**追加：三个接线原型已执行**

integration_prototypes.py只负责接口接线，直接调用作者模块与现有SwinPixelDecoder，没有自创注意力公式，也没有改生产训练文件。输入为与当前adapter一致的四级64通道特征；这是decoder集成检查，并未运行完整MambaVision encoder。

三个方案均实际输出[2,1,256,256] logits，BF16前向与反向通过；各自state_dict经过内存保存、严格加载后，eval输出最大绝对差均为0。详细结果见integration_checks.json，执行脚本为check_integration.py。

| 接线原型 | decoder部分参数（含最终读出，不含encoder与adapter） | 状态 |
|---|---:|---|
| EMCAD全64通道 | 284,108 | 通过 |
| 现有Swin＋三处CGA | 613,482 | 通过 |
| 现有Swin＋三处MFMS | 666,093 | 通过 |

原Swin对应407,649参数。因此“EMCAD标题写高效”不应替代实测，本配置确实比现有Swin解码部分小；完整模型速度和显存仍未测。随机特征检查不能替代实际图像训练，尚无新Dice。

复核命令顺序：使用现有env/bin/python依次运行compatibility_checks.py、compatibility_emcad64.py、check_integration.py。第一项重建26项模块检查，第二项补全至28项，第三项检查3条接线路径。以上均不启动训练，也不读取测试集。

**重点目录（全部29项）**

下表每项链接均为会议/论文或作者官方仓库。标记“源码核查”表示检查了实际实现与入口，但只有明确标注的模块做过执行检查；网页核验项未声称完整复现。

**主会工作（25项）**

| 工作 | 场所 | 可复用机制/任务 | 公开代码核查 | 当前决策 |
|---|---|---|---|---|
| [EMCAD](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html) | CVPR 2024 | 多尺度卷积解码、通道/空间注意力、LGAG跳跃连接门控 | [模型、多个任务的训练/测试入口公开；本轮执行模块检查](https://github.com/SLDGroup/EMCAD) | 首选：完整解码器对照；LGAG单独作为保留Swin的消融 |
| [MambaLiteUNet](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | AMF、LGFM、CGA跨分支门控 | [模型、train/test/engine/config和ISIC2017/2018权重文件可见；模块检查通过](https://github.com/maklachur/MambaLiteUNet) | 首选：原始CGA接在现有Swin跳跃连接 |
| [MADGNet](https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf) | CVPR 2024 | 多尺度多频率注意力MFMS，辅助监督 | [模型、任务工程公开；MFMS模块检查通过](https://github.com/Inha-CVAI/MADGNet) | 首选备选：MFMS接现有64通道adapter之后 |
| [EGE-UNet](https://arxiv.org/abs/2307.08473) | MICCAI 2023 | 分组聚合桥GAB、多轴Hadamard注意力 | [模型与训练代码公开；GAB模块检查通过](https://github.com/JCruan519/EGE-UNet) | 第二梯队：需要粗分割预测输入后再接 |
| [UCTransNet](https://ojs.aaai.org/index.php/AAAI/article/view/20144) | AAAI 2022 | CCT多尺度通道Transformer、CCA解码器引导通道注意力 | [作者模型、配置、train/test源码已获取](https://github.com/McGregorWwww/UCTransNet) | 第二梯队：CCA可作现成通道注意力对照；完整CCT成本另测 |
| [TransFuse](https://miccai2021.org/openaccess/paperlinks/2021/09/01/496-Paper0016.html) | MICCAI 2021 | BiFusion双线性交互加空间/通道注意力 | [模型、ISIC训练/测试入口公开](https://github.com/Rayicer/TransFuse) | 备选融合模块或完整外部基线 |
| [UACANet](https://arxiv.org/abs/2107.02368) | ACM MM 2021 | 前景/背景/不确定区域上下文聚合 | [模型及训练测试工程公开；发现布局实现疑点](https://github.com/plemeri/UACANet) | 暂缓原样接入，先核实上下文向量实现 |
| [PraNet](https://arxiv.org/abs/2006.11392) | MICCAI 2020 | 反向注意力与逐级细化 | [模型、训练测试代码公开](https://github.com/DengPingFan/PraNet) | 边界细化参考；当前不排首轮 |
| [LoMix](https://openreview.net/pdf?id=87c2JwNJa0) | NeurIPS 2025 | 不同尺度logits组合形成监督 | [模型与训练代码公开；concat分支重复性测试暴露问题](https://github.com/SLDGroup/LoMix) | 暂缓；不直接接作者当前concat实现 |
| [MCADS](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html) | CVPR 2025 | depth-to-space恢复及注意力解码 | [TensorFlow/Keras作者代码公开](https://github.com/saadwazir/MCADS-Decoder) | 暂不直接复用：需要跨框架重写 |
| [PMRNet](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 物理启发多尺度细化 | [快照仅PMRNet.py模型文件](https://github.com/KangBoce/PMRNet) | 暂缓：缺训练协议和完整复现工程 |
| [FPN-in-FPN](https://papers.miccai.org/miccai-2025/paper/4899_paper.pdf) | MICCAI 2025 | 嵌套特征金字塔 | [本轮仓库只有README.md（约101字节）](https://github.com/Yejin0111/FPN-in-FPN) | 排除首轮：当前没有模型源码 |
| [MixStyleFlow](https://papers.miccai.org/miccai-2025/paper/3460_paper.pdf) | MICCAI 2025 | normalizing flow特征风格增强 | [训练与flow代码、配置公开](https://github.com/Reza-Safdari/MixStyleFlow) | 训练泛化方向储备，不当成即插即用注意力 |
| [Swin-UMamba](https://papers.miccai.org/miccai-2024/749-Paper1627.html) | MICCAI 2024 | ImageNet预训练VMamba和U形解码 | [模型与训练工程源码快照已获取](https://github.com/JiarunLiu/Swin-UMamba) | 外部强相关基线 |
| [UNeXt](https://conferences.miccai.org/2022/papers/535-Paper0077.html) | MICCAI 2022 | 卷积与shifted MLP轻量分割 | [模型、训练测试入口公开](https://github.com/jeya-maria-jose/UNeXt-pytorch) | 效率基线；不优先拆模块 |
| [MedT / Medical Transformer](https://miccai2021.org/openaccess/paperlinks/2021/09/01/304-Paper0116.html) | MICCAI 2021 | 门控轴向注意力、局部全局训练 | [模型与训练代码公开](https://github.com/jeya-maria-jose/Medical-Transformer) | 完整网络对照；模块接入需形状与训练视图适配 |
| [UTNet](https://github.com/yhygao/UTNet) | MICCAI 2021 | 卷积与高效自注意力混合 | [模型与训练工程公开](https://github.com/yhygao/UTNet) | 外部网络基线 |
| [MedNeXt](https://inrepo02.dkfz.de/record/285111/) | MICCAI 2023 | 大核卷积、扩展收缩块、UpKern | [模型和nnUNet训练集成公开](https://github.com/MIC-DKFZ/MedNeXt) | 独立架构/计算预算参考 |
| [SegMamba](https://arxiv.org/abs/2401.13560) | MICCAI 2024 | 三维Mamba上下文建模 | [预处理、训练、预测、评估工程公开](https://github.com/ge-xing/SegMamba) | 3D路线参考，不首轮融入 |
| [EffiDec3D](https://openaccess.thecvf.com/content/CVPR2025/papers/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.pdf) | CVPR 2025 | 高效3D多尺度解码 | [模型和训练工程公开](https://github.com/SLDGroup/EffiDec3D) | 不优先：二维化属于进一步适配 |
| [3D UX-Net](https://openreview.net/forum?id=wsZsjOSytRA) | ICLR 2023 | 大核体积卷积骨干 | [官方仓库有train/finetune/test入口及权重链接](https://github.com/MASILab/3DUX-Net) | 3D架构参考，不首轮模块 |
| [UniverSeg](https://arxiv.org/abs/2304.06131) | ICCV 2023 | 支持集图像-标签驱动上下文分割 | [官方模型、推理示例和权重公开；完整训练流程未确认](https://github.com/JJGO/UniverSeg) | 任务设置不同，不当普通融合模块 |
| [ScribblePrompt](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05664.pdf) | ECCV 2024 | 点、框、涂鸦提示 | [官方模型/权重/交互与示例训练代码公开](https://github.com/halleewong/ScribblePrompt) | 提示依赖任务，不首轮融合 |
| [MultiverSeg](https://github.com/halleewong/MultiverSeg) | ICCV 2025 | 跨样本上下文与交互指导 | [官方推理代码、notebook和v1权重公开；完整训练流程未确认](https://github.com/halleewong/MultiverSeg) | 任务设置不同，知识储备 |
| [MedCLIPSeg](https://arxiv.org/abs/2602.20423) | CVPR 2026 | 概率跨模态注意力、patch级对比学习 | [官方模型、配置、train/test代码可见](https://github.com/IMPACT-L/MedCLIPSeg) | 需要视觉语言骨干与文本，不首轮融合 |

**补充相关工作（4项，单独标注）**

| 工作 | 场所 | 可复用机制/任务 | 公开代码核查 | 当前决策 |
|---|---|---|---|---|
| [Dual Cross-Attention (DCA)](https://doi.org/10.1016/j.engappai.2023.107139) | Engineering Applications of Artificial Intelligence 2023 | 多尺度通道、空间交叉注意力 | [作者模块和实验代码已获取](https://github.com/gorkemcanates/Dual-Cross-Attention) | 跨尺度桥接备选 |
| [MEGANet](https://github.com/UARK-AICV/MEGANet) | WACV 2024 | 多尺度边缘引导、反向注意力 | [作者模型与训练测试代码已获取](https://github.com/UARK-AICV/MEGANet) | 边界方向备用 |
| [HMT-UNet](https://github.com/simzhangbest/HMT-Unet) | arXiv 2024（本轮未核实正式会议） | MambaVision相关U形编码解码 | [模型源码公开；训练工程需参照其他项目](https://github.com/simzhangbest/HMT-Unet) | MambaVision直接相关结构对照 |
| [U-Mamba](https://arxiv.org/abs/2401.04722) | arXiv 2024（本轮未确认主会发表） | CNN与Mamba集成nnUNet | [模型及训练工程公开](https://github.com/bowang-lab/U-Mamba) | 独立基线，不等同MambaVision改进 |


**交付与未完成事项**

API去重候选：279；人工重点目录：29；服务器已保留源码/仓库资料目录：25（包括复用前一轮快照，含README占位仓库，不能全部算完整开源模型）。
实际接口检查：28；通过：28。作者模型文件未改；未修改生产训练模型；未启动新训练。

文件：
- deep_research_report.md / deep_research_report.txt：完整中文报告。
- curated_catalog.json / catalog_zh.md：29项论文、代码状态、入口、排除理由。
- code_audit.md / compatibility_results.json：源码定位与执行结果。
- integration_plan.json：三个现成实现的接入位置与控制变量。
- api_merged.json、api/*/search_log.md、errors.log：检索记录与限制。
- web_supplement.md、source_inventory.json、official_code/*/*MANIFEST.json：来源证据与快照。

尚未完成的事项是生产模型接线、完整训练、实际GPU预算和分割指标验证；不能把本报告写成已经验证精度提高。所有新增文件均在服务器目录/home/waas/mambavision_stage22/research/topconf_open_source_20261111，未向本地桌面写文件。
