# HiRAM 双编码器迁移：ISIC2018 seed 42

## 目标与实验组

复用 stage4 的 DINOv3-S/16、VMamba-Tiny、四尺度 Adapter、Concat+1×1 与 Swin Pixel Decoder。原计划使用历史 fusion_swin best_dice 作为 A；预检发现本机根据相同源码/seed 生成的初始参数摘要与历史摘要不一致，原因尚未确定。因此在正式训练前增加一个本机配对重训的 A，确保与 B/C 初始化、设备和数据完全配对。历史 A 仅作背景。三个模型都只跑 seed 42：

- A `paired_baseline`：原始 DINOv3+VMamba+Swin，无额外区域头和损失。
- B `region_only`：在两个 16×16 Adapter 输出各自附加官方 RPG 的区域预测头；只以区域损失监督，区域头不改变特征。验证区域监督单独的贡献。
- C `hiram`：相同位置将两路特征送入官方 HiRAM 双分支的 RPG、每区域 RA-SSM、全图双向 SSM、跨分支乘法及残差，输出送入原有 concat；在与 B 相同的区域预测头和区域损失上训练。

这不是完整的原论文网络：它没有 SAM、CNN U-Net、RAE token/attention 或两层解码器；仅迁移官方 HiRAM。其他三个尺度完全维持 stage4。两个 Encoder 冻结，A/B/C 的 Adapter、fusion 和 Swin 使用相同源码、seed 和本机完全一致的初始张量；不可声称初始张量与历史运行逐位相同。两组区域头从同一 seed 1042 创建的官方 HiRAM 初始化，其他新增模块独立训练。官方 global SSM 的 conv 因 skip_conv=True 从未执行，冻结这些未用参数并列出。

官方代码 commit d8aac2de3c33f8f2e79e281fcea8e5239db29645，原文件和 CC BY-NC 4.0 许可见 vendor_hiram/。区域扫描采用按每图、每区域置信度排序，末尾补零后 batched selective scan，以保持官方单图语义；预检比较前向、梯度及跨图隔离。训练保留 Gumbel 随机路由，验证和推理改用逐位置 argmax，确保固定评估，这是有明确记录的原实现差异，不自称一比一复现。

## 数据与目标

严格继承 stage4/stage1 数据清单、图像/Mask 插值、ImageNet 归一化、几何增强、seed+epoch×100003 顺序和 1886/808 开发划分；不是独立测试。历史 Mask 包含部分灰度值，原样保留给历史分割损失；单独生成区域监督时，以项目评估阈值 0.5 二值化，再沿用官方 width=2 二值 Mask 形态学操作，内部及边界标签 bilinear 缩到 16×16，对两路预测分别使用官方归一化 KL，权重 1；推理不读 GT。初步对未增强图像的扫描发现 bilinear 下采样后训练集 11/1886、验证集 1/808 边界目标全零，这是低分辨率监督局限，不临时筛除或替换标签。

## 优化与计算

300 完整 epochs，seed42，AdamW lr=.001、wd=.01、betas=(.9,.999)、eps=1e-8；CosineAnnealingLR(T_max=300, eta_min=1e-5)，无 warmup/early stopping；梯度裁剪1，BF16 autocast、FP32 loss。分割 loss 复用 BCEWithLogits + per-image SoftDice(smooth=1)。A 只用分割损失，B/C 总损失=分割损失+区域损失。预检在本机10GB显存中完成了验证缓存驻留下的 HiRAM batch64 前向与反向，峰值 6.835 GiB。因此正式训练和评估均采用实际 batch64（尾批30），不使用梯度累积，与历史 stage4 的 batch 口径一致；每 epoch 30 次 optimizer updates。每轮重新编码增强图像，验证特征缓存。不同 GPU 的浮点运算不保证逐位相同。

验证 Macro Dice 最大、精确并列取更早轮为主选择；最低验证分割损失和 epoch300 为辅助。统一 stage3/evaluation.py 定义和历史边界指标，不能拼接不同轮次。最终报告 Macro/Pooled Dice、IoU、P/R、边界F1、HD95、prec@各阈值和每图指标；比较新增/纠正错误、面积组及新增参数、显存、延迟。先比较 A/C，总变化同时包含区域监督、区域 SSM 和交互；B/C 仍混合新模块与容量变化，后续容量匹配消融才能进一步归因。仅 seed42 不能证明稳定性。

## 执行要求

正式训练前执行数据/冻结/权重 hash、官方单图与批量结果及梯度、不同图片批处理隔离、BF16/显存预检，并用隔离输出执行两轮 smoke 与断点恢复。维护每轮日志、完整模型/optimizer/scheduler/RNG 状态、三个选择规则与完整评估；只有完成 300 轮、每组 9000 步、冻结校验及指标审计后才能标记 complete。不修改历史文件、不推送 GitHub。
