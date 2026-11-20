# MambaVision-S 原生双分支的分割读出

这组实验考察一个问题：UPer512 已经拿到编码器的阶段输出，再让它按位置读取同一 Mixer 内的 SSM 和非 SSM 两路特征，能否改善病灶分割？基线为 MambaVision-S + UPer512 + FCN。输入为 256×256，Stage3、Stage4 的窗口分别为 16、8。

## 从源码看一次前向计算

1. **在哪里取特征。** `attach` 给 Stage3 的 Block 3、Stage4 的 Block 2 注册钩子，两者都是各自阶段最后一个 MambaVision Mixer。Block 编号从 0 开始。钩子读出 `out_proj` 的输入，此时前半通道刚经过 selective scan，后半通道经过一维深度卷积和 SiLU。原来的 `out_proj` 继续执行。[接入位置](branch_model.py#L60) · [NVIDIA Mixer 源码](https://github.com/NVlabs/MambaVision/blob/7860a506b2eb844eaaae676f08461ce8c3c26f43/semantic_segmentation/tools/mamba_vision.py)
2. **怎样对应到图像位置。** 在这套输入和窗口配置下，每张图的每个高层阶段各占一个窗口。代码检查 token 形状，恢复成 Stage3 的 16×16 或 Stage4 的 8×8 特征图，再交给该阶段自己的适配器。[空间重排](branch_model.py#L46)
3. **怎样选择两路。** 两路特征分别用 1×1 卷积投影到 64 通道。阶段末端特征也独立投影到 64 通道。门控读取这三项及两路投影的绝对差，在每个空间位置通过 softmax 给出两个权重。加权和再投影回阶段通道数，乘以该阶段的可学习标量 `tanh(strength)`，加回原阶段特征。权重在分支维度归一化，同一位置的 64 个通道共用这对权重。[适配器前向](branch_model.py#L28)
4. **谁使用修正后的特征。** 适配器只改送入 UPer 主头的 F3、F4 副本。Stage4 的编码器计算仍沿原路径，训练用的 FCN 辅助头仍读取原始 F3。[UPer 主头钩子](branch_model.py#L71) · [主头与辅助头](../src/model.py)

| 取点 | 每路原始通道数 | 两路各自投影后 | 加回的阶段特征 |
|---|---:|---:|---|
| Stage3 / Block 3 | 192 | 64 | 384×16×16 |
| Stage4 / Block 2 | 384 | 64 | 768×8×8 |

## 训练与核验

两阶段合计新增 **222,214** 个可训练参数。`strength` 从零开始，因此新结构在初始化时能逐值复现基线。自检分别比较了推理模式的主头、训练模式的主头和辅助头，并检查了前两步真实参数更新。第一步先检查残差系数，第二步再检查两路投影及门控的梯度。[自检源码](audit.py) · [自检记录](audit.json)

共享部分从官方 ADE20K 分割权重初始化，两个输出层替换为二分类输出。编码器和解码器一起微调，初始学习率分别为 `1e-4` 和 `1e-3`，有效 batch 为 64，辅助损失权重为 0.4。最多训练 100 轮，连续 25 轮验证集逐图前景 Dice 没有提高时早停。最终用最佳验证权重在测试集逐图评测，阈值为 0.5。[基线初始化](../official_full_uper512_fcn_20261120/official.py) · [训练循环](train_branch.py) · [测试与配对比较](compare.py)

5090 上复用同一训练循环，同时把图像和掩膜缓存在显存中。运行依赖实验服务器的数据清单、缓存、预训练权重和与显卡兼容的 selective scan 扩展；脚本中的路径沿用实际训练环境。[5090 训练入口](rtx5090/run5090.py) · [数据读取](rtx5090/resident_images.py) · [迁移核验](rtx5090/check5090.json)
