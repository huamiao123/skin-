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

**逐项源码记录**

**EMCAD — CVPR 2024**

任务：2D多器官、息肉等。入口：lib/decoders.py: EMCAD, LGAG, MSCB, EUCB。

核查深度：source；UT Austin Research License；不是MIT。64通道配置及原生MambaVision通道配置均检查通过。此前Stage29仅受其启发，不是作者实现复现。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)；[官方仓库](https://github.com/SLDGroup/EMCAD)。

**MambaLiteUNet — CVPR 2026**

任务：2D皮肤病灶。入口：models/MambaLiteUNet.py: CrossGatedAttentionGate, AdaptiveMultiBranchMambaFeatureFusion, LocalGlobalFeatureMixing。

核查深度：source；不是NVIDIA MambaVision。CGA接收同尺寸64通道两路特征；含8个小Mamba。LGFM包含全局MultiheadAttention，不应把整个网络描述为全线性复杂度。快照未见LICENSE；权重未加载验证。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html)；[官方仓库](https://github.com/maklachur/MambaLiteUNet)。

**MADGNet — CVPR 2024**

任务：2D多模态、跨域；含皮肤病灶。入口：IS2D_models/mfmsnet.py: MultiFrequencyChannelAttention, MFMSAttentionBlock。

核查深度：source；原实现DCT尺寸表只支持32/64/128/256/512通道，不能直接接96/192/384/768；64通道可用。只借MFMS不是完整MADGNet。快照未见LICENSE。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2024/papers/Nam_Modality-agnostic_Domain_Generalizable_Medical_Image_Segmentation_by_Multi-Frequency_in_Multi-Scale_CVPR_2024_paper.pdf)；[官方仓库](https://github.com/Inha-CVAI/MADGNet)。

**EGE-UNet — MICCAI 2023**

任务：2D皮肤病灶。入口：models/egeunet.py: group_aggregation_bridge, Grouped_multi_axis_Hadamard_Product_Attention。

核查深度：source；Apache-2.0。GAB同时输入高低层特征和mask；不能按普通无条件注意力直接插入。原始数据划分不等于当前协议。

来源：[论文/发表证据](https://arxiv.org/abs/2307.08473)；[官方仓库](https://github.com/JCruan519/EGE-UNet)。

**UCTransNet — AAAI 2022**

任务：2D医学分割。入口：nets/CTrans.py: ChannelTransformer; nets/UCTransNet.py: CCA。

核查深度：source；整库压缩包超过本轮50MiB上限，已成功用raw方式获取关键文件，不属于代码不可用。CCT需设置KV_size等于输入通道和，并匹配patch尺寸。

来源：[论文/发表证据](https://ojs.aaai.org/index.php/AAAI/article/view/20144)；[官方仓库](https://github.com/McGregorWwww/UCTransNet)。

**TransFuse — MICCAI 2021**

任务：2D息肉、皮肤等。入口：lib/TransFuse.py: BiFusion_block; train_isic.py; test_isic.py。

核查深度：source；原方法融合独立CNN与Transformer分支；移植至同一骨干的encoder/decoder特征需明确属于适配。

来源：[论文/发表证据](https://miccai2021.org/openaccess/paperlinks/2021/09/01/496-Paper0016.html)；[官方仓库](https://github.com/Rayicer/TransFuse)。

**UACANet — ACM MM 2021**

任务：2D息肉。入口：lib/modules/attention_module.py: UACA。

核查深度：source；BCHW直接view成B,HW,C不等于逐像素通道向量；小张量验证确实不等价。另需预测头与监督。该发现不能推出论文整体无效。MIT。

来源：[论文/发表证据](https://arxiv.org/abs/2107.02368)；[官方仓库](https://github.com/plemeri/UACANet)。

**PraNet — MICCAI 2020**

任务：2D息肉。入口：lib/PraNet_Res2Net.py。

核查深度：source；我们存在离边界较远的漏分，单纯强调边界/反向注意力不应视为已对症；若复用仍要整体漏分与误分检查。

来源：[论文/发表证据](https://arxiv.org/abs/2006.11392)；[官方仓库](https://github.com/DengPingFan/PraNet)。

**LoMix — NeurIPS 2025**

任务：医学分割监督/解码训练。入口：trainer.py: CombinatorialMutationsLossModule, WeightedFusion。

核查深度：source；forward内临时创建1x1卷积，未注册在模型中；同输入eval连续两次最大差2.403，add对照为0。本轮未修改作者源码。

来源：[论文/发表证据](https://openreview.net/pdf?id=87c2JwNJa0)；[官方仓库](https://github.com/SLDGroup/LoMix)。

**MCADS — CVPR 2025**

任务：2D生物标志物/细胞分割。入口：mcadsDecoder.py; hist_metrics.py。

核查深度：source；当前工程为PyTorch；不能把下载到代码等同于可直接接入。README后续扩展表格不应混同原CVPR论文实验。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2025/html/Wazir_Rethinking_Decoder_Design_Improving_Biomarker_Segmentation_Using_Depth-to-Space_Restoration_and_CVPR_2025_paper.html)；[官方仓库](https://github.com/saadwazir/MCADS-Decoder)。

**PMRNet — CVPR 2026**

任务：医学分割。入口：PMRNet.py。

核查深度：source；不能将单个模型文件当完整可复现项目；physics-informed名称本身不证明本任务机制。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html)；[官方仓库](https://github.com/KangBoce/PMRNet)。

**FPN-in-FPN — MICCAI 2025**

任务：医学分割。入口：README.md。

核查深度：source；论文给出GitHub链接与实际已发布可用代码是两回事；截至本次快照判断，未来可能更新。

来源：[论文/发表证据](https://papers.miccai.org/miccai-2025/paper/4899_paper.pdf)；[官方仓库](https://github.com/Yejin0111/FPN-in-FPN)。

**MixStyleFlow — MICCAI 2025**

任务：跨域前列腺MRI/眼底。入口：training_normalizing_flows; train_adv_supervised_segmentation_triplet.py。

核查深度：source；需要独立flow训练和分阶段流程；与当前单次模块替换实验条件不同。

来源：[论文/发表证据](https://papers.miccai.org/miccai-2025/paper/3460_paper.pdf)；[官方仓库](https://github.com/Reza-Safdari/MixStyleFlow)。

**Swin-UMamba — MICCAI 2024**

任务：2D医学分割。入口：swin_umamba。

核查深度：source；不是NVIDIA MambaVision。论文旧GitHub账户Jiarun-Liu已失效；会议页当前链接JiarunLiu可用。

来源：[论文/发表证据](https://papers.miccai.org/miccai-2024/749-Paper1627.html)；[官方仓库](https://github.com/JiarunLiu/Swin-UMamba)。

**UNeXt — MICCAI 2022**

任务：2D皮肤/超声等。入口：archs.py; train.py; val.py。

核查深度：source；属于完整架构对照，不能直接把其参数量或论文Dice与我们的协议等同。

来源：[论文/发表证据](https://conferences.miccai.org/2022/papers/535-Paper0077.html)；[官方仓库](https://github.com/jeya-maria-jose/UNeXt-pytorch)。

**MedT / Medical Transformer — MICCAI 2021**

任务：2D医学分割。入口：lib; train.py。

核查深度：source；局部/全局划块流程不等于给当前网络随手增加一个attention。

来源：[论文/发表证据](https://miccai2021.org/openaccess/paperlinks/2021/09/01/304-Paper0116.html)；[官方仓库](https://github.com/jeya-maria-jose/Medical-Transformer)。

**UTNet — MICCAI 2021**

任务：医学分割。入口：model; train_deep.py。

核查深度：source；已有MambaVision与Swin，继续叠加相近全局注意力不是首选证据方向。

来源：[论文/发表证据](https://github.com/yhygao/UTNet)；[官方仓库](https://github.com/yhygao/UTNet)。

**MedNeXt — MICCAI 2023**

任务：主要3D医学分割。入口：nnunet_mednext。

核查深度：source；对当前2D任务的完整移植成本较高；不得说全部代码只支持3D而未逐个核查构造参数。

来源：[论文/发表证据](https://inrepo02.dkfz.de/record/285111/)；[官方仓库](https://github.com/MIC-DKFZ/MedNeXt)。

**SegMamba — MICCAI 2024**

任务：3D脑肿瘤等。入口：model_segmamba; 2_preprocessing_mri.py; 3_train.py; 4_predict.py; 5_compute_metrics.py。

核查深度：source；本轮获取代码目录和入口，未完整复现其3D实验。

来源：[论文/发表证据](https://arxiv.org/abs/2401.13560)；[官方仓库](https://github.com/ge-xing/SegMamba)。

**EffiDec3D — CVPR 2025**

任务：3D医学分割。入口：networks; main_train_MSD_Task01_10.py; main_train_BTCV_TU.py。

核查深度：source；EMCAD提供更直接的2D解码复用路径。

来源：[论文/发表证据](https://openaccess.thecvf.com/content/CVPR2025/papers/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.pdf)；[官方仓库](https://github.com/SLDGroup/EffiDec3D)。

**3D UX-Net — ICLR 2023**

任务：3D多器官/脑结构。入口：main_train.py; main_finetune.py; test_seg.py。

核查深度：official_repo_page；仅网页入口核验；本轮未下载权重或运行。

来源：[论文/发表证据](https://openreview.net/forum?id=wsZsjOSytRA)；[官方仓库](https://github.com/MASILab/3DUX-Net)。

**UniverSeg — ICCV 2023**

任务：跨任务医学分割。入口：universeg/model.py。

核查深度：official_repo_page；需要support image-mask pairs；当前单图全监督条件不能直接比较。

来源：[论文/发表证据](https://arxiv.org/abs/2304.06131)；[官方仓库](https://github.com/JJGO/UniverSeg)。

**ScribblePrompt — ECCV 2024**

任务：交互式医学分割。入口：scribbleprompt; examples。

核查深度：official_repo_page；人工提示增加可用信息，不能作为无提示分割的公平替换。

来源：[论文/发表证据](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05664.pdf)；[官方仓库](https://github.com/halleewong/ScribblePrompt)。

**MultiverSeg — ICCV 2025**

任务：跨图像交互式医学分割。入口：multiverseg; notebooks/inference.ipynb。

核查深度：official_repo_page；代码公开不等于训练全链路公开。

来源：[论文/发表证据](https://github.com/halleewong/MultiverSeg)；[官方仓库](https://github.com/halleewong/MultiverSeg)。

**MedCLIPSeg — CVPR 2026**

任务：文本引导医学分割。入口：train.py; test.py; trainers。

核查深度：official_repo_page；会议身份另由CVF补充材料页面支持；本轮仅网页入口核验。

来源：[论文/发表证据](https://arxiv.org/abs/2602.20423)；[官方仓库](https://github.com/IMPACT-L/MedCLIPSeg)。

**Dual Cross-Attention (DCA) — Engineering Applications of Artificial Intelligence 2023**

任务：2D医学分割。入口：model/utils/dca.py。

核查深度：source；期刊论文，不能记为CVPR/MICCAI；需要统一token网格并检查额外注意力开销。

来源：[论文/发表证据](https://doi.org/10.1016/j.engappai.2023.107139)；[官方仓库](https://github.com/gorkemcanates/Dual-Cross-Attention)。

**MEGANet — WACV 2024**

任务：2D息肉。入口：lib。

核查深度：source；WACV单列；当前错误不限于窄边界带。

来源：[论文/发表证据](https://github.com/UARK-AICV/MEGANet)；[官方仓库](https://github.com/UARK-AICV/MEGANet)。

**HMT-UNet — arXiv 2024（本轮未核实正式会议）**

任务：2D皮肤/息肉。入口：models/hmt_unet.py; models/mamba_vision.py。

核查深度：source；代码可见不等于医学数据已训练权重可直接用。本轮不将其列作顶会论文。

来源：[论文/发表证据](https://github.com/simzhangbest/HMT-Unet)；[官方仓库](https://github.com/simzhangbest/HMT-Unet)。

**U-Mamba — arXiv 2024（本轮未确认主会发表）**

任务：2D/3D医学分割。入口：umamba。

核查深度：source；不能因为常与MICCAI论文一起比较就标成MICCAI 2024；与Swin-UMamba区别。

来源：[论文/发表证据](https://arxiv.org/abs/2401.04722)；[官方仓库](https://github.com/bowang-lab/U-Mamba)。

