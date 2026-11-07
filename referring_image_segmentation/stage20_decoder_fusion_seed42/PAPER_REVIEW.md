# 论文和开源实现审查

## 与本项目相关的事实

此前 MambaVision-S + Swin 的 stage17 开发集最佳 Macro Dice 为 0.884398；这是单 seed、开发集选模，不是官方测试集。阶段 17/19 的 HiRAM 迁移已有实验：放进解码器的完整迁移相对配对基线的提升很小，区域路由的背景/边界分布也需要继续诊断。因此不能根据“增强边界”的名字就推断另一个复杂模块一定有效。

## SegMAN / MMSCoPE

[CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.pdf)，[官方解码器](https://github.com/yunxiangfu2001/SegMAN/blob/9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8/segmentation/mmseg/models/decode_heads/segman_decoder.py)。MMSCoPE 是完整的多尺度上下文解码，而非一个可直接替代 Swin 中相加运算的桥接。实际源码先融合 c2/c3/c4，构造多种分辨率，再通过 pixel-unshuffle 和 SSM 在统一网格上建模并聚合；其该解码路径读取了 c1 但未参与最终计算。这一点与“直接把四个尺度都融合”不完全相同。官方 small 配置的通道为 [64,144,288,512]，channels=144、feat_proj_dim=288，不等于本项目 64 通道 Swin，因此整头替换不能宣称只变了扫描方式或参数量不变。

当前 Python 环境缺 `mmcv`、`mmseg`、`natten` 和 `selective_scan_cuda_oflex`，未发现 `nvcc`。即使这些模块不是所有实际 forward 路径都使用，官方文件导入时仍依赖它们。应尝试隔离必要模块并核验输出和梯度；任何替代 kernel 或框架的移植都必须明确标注为适配版，不得宣称完整官方复现。仓库根目录 Apache-2.0，而解码器文件头另外标注 NVIDIA Source Code License，公开分发前应核验各文件的具体授权。

## HEDS-Net / AXIS-Bridge

[MICCAI 2026 论文](https://papers.miccai.org/miccai-2026/0452-Paper4167.html)，[官方 bridge 代码](https://github.com/Momentisbeused/HEDS-Net/blob/a9287da787957b6376a9c403a56ad6693cd04191/models/axis_bridge.py)。论文完整网络同时包含 HVST、AXIS 和动态深监督；这里仅迁移 AXIS：根据粗尺度 decoder 特征和 encoder skip 计算门控，再在高、宽方向提取坐标注意力。最初提供的 `main` 路径不可用，实际所核验默认分支为 `master`。所核验仓库未见许可证文件；源码仅用于本地研究，公开推送前需确认授权。模块的 NCHW/NHWC 自动判断有维度歧义，因此固定 32×32 并做显式适配和数值校验。

本轮是原始相加、相同初始化的官方 context gate、完整 AXIS 三组；保留其他 decoder 和 loss，先回答 skip 注入能否受益，不宣称复现 HEDS 论文结果。

## HGM / Bi-directional Mamba Decoder

[MICCAI 2025 论文](https://papers.miccai.org/miccai-2025/0423-Paper1195.html)，[官方解码器](https://github.com/YueyueZhu/HGM/blob/49ae05afc9785dbbc5a303d39e63588b068e5328/lib/Decoder.py)。HGM 面向息肉，论文的区域区分主要由 BDFM 处理，完整结构还包含图卷积与 HGMM；BMD 则结合双向像素/窗口 Mamba 与多尺度特征增强。官方 `BMD` 依赖 `FSEMV2`、固定 32 通道和配套 skip 输入，不能把单独 `BiWindowMambaLayer` 当作完整 HGM。

当前缺 `torch_geometric`、相关动态网络依赖与 `Mamba2`，但已有 `Mamba`。可研究把可独立运行的前后向模块或完整 BMD 移植为另一个对照；若只提取一个层，要如实命名，且先验证 kernel 和前后向。该仓库未见 LICENSE，公开分享源码前需确认授权。原论文的数据集和 ISIC 划分不同，数值不能直接比较。

进一步检查 `BMD` 的默认构造有一个重要的尺寸适配问题：两层 `BiPixelMambaLayer` 计算得到分块因子均为 5；对于 256 输入常见解码尺寸 64×64、128×128 均不能整除 5，因此官方代码会回退到完整图像序列而不是执行该分块。两层 `BiWindowMambaLayer` 的构造参数均为 1，因此此时池化也不会降低空间分辨率。它并非不能运行，但直接接入不能被描述为已经按预期进行了局部窗口划分，计算成本也需要真实测量。额外填充或改窗口将改变原算法，应作为单独适配并做对照，而不是悄悄修改。

`analysis/hgm_diagnostic.json` 记录了本机实际试验：从上游源码提取的两个 32×32 单独模块，在项目现有 Mamba 1.2.2 下 FP32 前向/反向均通过，直接接收 BF16 特征均因类型不匹配失败；HGM 仓库要求的 Mamba 版本是 2.2.4。这只是隔离层试验，不能代表完整 HGM 或确认其他版本同样失败。若后续迁移，需要显式转换和/或依赖适配，再评估数值及耗时，不将它误报为已完成的融合训练。

## MCADS、Glasses 和 HiRAM

- [MCADS 官方仓库](https://github.com/saadwazir/MCADS-Decoder)：depth-to-space 和残差注意力针对解码恢复有参考价值，但上游采用 TensorFlow/Keras，并非本项目 PyTorch 直接插入；仓库标识 GPL-3.0。移植需要单独数值验证，不应称为即插即用。
- [Glasses 论文](https://papers.miccai.org/miccai-2026/0429-Paper1781.html)：方法同时涉及 DINOv3 适配器、边缘 CNN、边界监督和 U-Net decoder，不只是一个轻量 decoder，且公开文件引用未找到的适配器实现。若整体迁入会同时改变多个变量；目前不作为第一轮模块对照。
- [HiRAM 官方仓库](https://github.com/deepnoid-ai/HiRAM)：本项目已有 stage16–19 的局部适配和预检，但不能把此前实验等同完整论文复现。新增实验需保留现有路由失衡问题的上下文，不能简单叠加到 AXIS 上声称机制有效。

## 下一步准则

先完成 AXIS 系列并分析区域、边界和逐图纠错；与此同时检查 SegMAN/HGM 的可运行性。若只改变 decoder 的独立实验能通过前后向、显存和同协议检查，再启动相应 seed 42 训练；否则明确记为依赖阻碍或适配版，不虚构结果。任何性能结论都以实际配对基线、固定选模和开发集限制为前提。
