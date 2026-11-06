# MambaVision-S + Swin：冻结编码器单种子基线

正式训练前冻结，2026-10-31。只执行一个模型、seed42；独立 smoke 与正式训练隔离。不修改 stage1–13，不运行 DINOv3/VMamba，不推送 GitHub。

## 唯一改变与来源

将 stage4 的单视觉 Encoder 换为 NVIDIA 官方 ImageNet-1K MambaVision-S。HF revision 为 `e4f9ea93877fba6f1e2d231fd0f0a391368ad895`；官方 GitHub 检查 revision `7860a506b2eb844eaaae676f08461ce8c3c26f43`。实际执行未修改的 HF modeling_mambavision.py，先审查再用 AutoModel 加载本地固定快照。model.safetensors SHA256 `006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180`，与远端 LFS 对象身份核对；全部参数及缓冲区 strict 加载，missing/unexpected/mismatched 必须为零。

保留官方所有 Conv、SSM、Attention 与预训练分类头；分类头不参与分割前向，但完整官方 Encoder 参数计数包含它。输出为四级 NCHW：96×64×64、192×32×32、384×16×16、768×8×8。requires_grad=False，递归 eval，no_grad 提取；记录训练前后完整 state_dict 摘要（包括 BN 缓冲区）。不随机初始化 Encoder、不改扫描、不加入文本。

直接调用 stage3_confirmation.models.Probe(decoder='swin', seed=42, channels=(96,192,384,768))。Adapter、SwinPixelDecoder、输出层和 loss 直接导入历史源码。可训练初值须与 stage4 vmamba_swin 的同通道 Probe 完全一致；不运行该历史 Encoder。无新增注意力、融合或损失。

## 数据、增强和训练

直接继承 stage3_confirmation.data.ImageData / stage1_visual.data，实现逐项核对 stage1 data_manifest：1886 train / 808 development validation。RGB bilinear resize256；Mask NEAREST resize256；ImageNet mean/std。历史 numbered ID 是 split-local 身份，核对原文件/解码内容 SHA，不按编号重新分割。验证集不是官方测试集，也未认证患者独立。

原 geometry 每轮用 42+epoch×100003 同时预生成全部样本顺序及 affine 矩阵，依次采样 H/V flip 各0.5、rotation0.5与均匀0–360°角度；原 affine_grid/grid_sample，align_corners=False，图像 bilinear、Mask nearest、zero padding。矩阵在索引时按原样本身份取用。每轮重新增强与编码；验证固定无随机增强。完全复用 stage3.engine.encode/evaluate，冻结特征 BF16 驻留 GPU；不改变数据或损失语义。

- 300 完整 epochs；Decoder batch64；尾批保留（1886=29×64+30），无梯度累积；30 updates/epoch，共9000。
- AdamW lr0.001、weight_decay0.01、betas(.9,.999)、eps1e-8。
- CosineAnnealingLR T_max300、eta_min1e-5；无 warmup；每轮所有 optimizer 更新后 scheduler.step，再验证，日志同时保存本轮实际 LR 与 next_lr。
- 历史 BCEWithLogits + per-image Soft Dice，smooth1；按样本数加权 epoch loss。
- BF16 autocast，FP32 loss；clip_grad_norm1.0；cudnn deterministic=True / benchmark=False，不宣称所有 CUDA 算子逐位确定。
- Encoder batch64、验证 batch64；若物理显存限制须明确记录，仅允许调整冻结特征分批/缓存方式，绝不改 Decoder batch 或更新数。

本机实际 GPU 为 24GB RTX4090，以实测峰值为准，不沿用此前机器的48GB描述。依赖仅安装到本阶段 runtime，不修改历史运行环境。

## 预检、选模与恢复

真实全量数据清单验收、官方 S strict 加载、CUDA scan、64样本四级前向、256输出、有限梯度、冻结 BN/参数、显存/吞吐通过后，在独立 smoke 做至少2完整 epochs。smoke epoch1 保存后退出，由新进程恢复 epoch2，检查模型/optimizer/scheduler/Python/NumPy/CPU/CUDA RNG 摘要，不重新初始化训练状态。正式训练全新初始化，不读 smoke 模型。

主 best_dice 按最高验证 macro Dice，精确并列取较早；辅助 best_loss 按最低样本加权验证 loss，精确并列取较早；last 固定第300轮。三者均保留。每轮 latest 保存模型、optimizer、scheduler、RNG、selectors、history、完整配置与更新计数，可从下一轮恢复。不得因早期达到峰值提前结束。

## 评价与预定分析

全部在历史256网格，probability>=0.5。完全复用 evaluation.py 及 stage1_visual.analyze.metrics 的边界实现/空 Mask 规则。导出三个 checkpoint 的808张逐图概率、本地二值Mask、ID与指标；主表只用 best_dice，同一 checkpoint 同时报告所有指标。

macro/pooled Dice、IoU、Precision/Recall；Prec@IoU.5/.6/.7/.8/.9；Boundary F1（对角线0.5%容差）；HD95 pixels/diagonal；ISIC threshold Jaccard.65。逐图 TP/FP/FN/TN 独立复算 Dice/IoU，校验808图、历史 ID 顺序、预测/聚合/选模重放一致。

只读比较 stage4 best_dice dinov3_swin / vmamba_swin，不重跑 Encoder。历史概率存在且 SHA 与报告一致时，计算纠正/新增错误像素、逐图改善/退步、边界及病灶面积组（<0.1、0.1–0.3、>=0.3）表现；否则仅用已有逐图数值，明确省略像素比较。图像配对2000次bootstrap seed42仅探索性，不估计患者或训练seed不确定性。

学习曲线含 train loss、val loss、macro/pooled Dice、LR，并标主选中轮；Dice/IoU分布、固定面积组；展示改善、退步和最差病例，含源图拼图仅保留本地。完整 batch1 uncached GPU 延迟包含 Encoder+Adapter+Swin+Head，BF16、预热3/重复10、CUDA前后同步，排除I/O与CPU后处理；分别记录绝对及增量峰值，最终另测无数据缓存进程。若不完整支持 scan，不报告部分 FLOPs 为完整 FLOPs。

只在正式300轮/9000更新、权重身份、冻结状态、三选模规则、808图全指标独立复核、报告与JSON一致、历史及新源码哈希不变全部通过后标 complete。单seed、不同预训练规模及来源不作等容量机制归因，不把开发验证结果称为独立测试。
