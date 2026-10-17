# 第四轮：DINOv3 + VMamba 特征拼接融合

冻结日期：2026-10-17。依据：第一阶段概率平均将 macro Dice 从 88.2189% 提高到 88.7011%，但 pooled Dice 未提高；逐图互补与边界差异支持进一步检验特征融合，尚不能证明融合收益或模块分工。

## 预先固定的矩阵和选模

Seed 42，六组重新训练：DINOv3 / VMamba / DINOv3+VMamba × Light / Swin。每组 300 epochs、9000 updates。复用已确定的预训练编码器及 Decoder 定义。主结果统一按最高验证 macro Dice 选模，完全相同取较早轮；最低样本加权验证 loss 和第 300 轮为全组统一的次要结果。旧 seed42 的 loss 选模结果、第三轮的跨 seed 均值仅作背景，不能混入本轮配对差。

本轮不追加 seeds 或门控；完成后先判断融合相对于同 Decoder 的单分支和概率平均的收益，再决定确认实验。Seed42 已用于研究设计，因此本轮是探索性筛选。

## 唯一新增结构

两编码器完全冻结且 eval。各分支独立的 Conv1×1/GN/GELU Adapter 投影到 64 通道、双线性调整到 64/32/16/8 网格。每层 concat 得到 128 通道，再由可训练 Conv1×1（128→64，有 bias）融合。之后使用原 Light 或 Swin Pixel Decoder，没有新增 cross-attention、门控、损失或原图旁路。DINO 的高分辨率网格是插值特征，不视为原生高频细节。

对应单分支与融合分支的 Adapter 初始张量相同；同 Decoder 的六组相应 Decoder 初值一致；两个融合模型的融合层初值一致。所有模型训练参数/优化器独立，不共享可训练参数。共享两个冻结 Encoder 的同图同增强前向缓存以节省重复计算。

## 固定配置

- ISIC2018 旧开发划分 1886/808，256×256，沿用全部图像/mask处理、ImageNet归一化和数据 manifest。
- HFlip/VFlip/Rotation 各 p=.5，旋转均匀0～360°；epoch随机种子为42+epoch×100003。六组读取完全相同的训练图像、增强与顺序。
- AdamW lr=1e-3、weight_decay=.01、betas=.9/.999、eps=1e-8；batch=64，保留尾批，无梯度累积。训练300轮，不early stop；Cosine T_max=300、eta_min=1e-5，无warmup。
- BCEWithLogits + per-image soft Dice（smooth=1）；BF16 autocast，FP32损失，梯度裁剪1；cudnn deterministic=True / benchmark=False，未保证全算子逐位确定性。
- 前向/验证batch=64；各epoch重新增强、重新运行两个编码器。验证缓存固定。数据和特征驻留GPU，六组顺序更新；当前设备实测为24GB RTX4090，预检后确定能否容纳，不改变优化器batch来填显存。
- 主要分析采用统一 evaluation.py，输出macro/pooled Dice/IoU/Precision/Recall、Prec@.5/.6/.7/.8/.9、Threshold Jaccard、Boundary F1和HD95（像素/对角线归一化）、逐图概率/二值mask/指标。

## 预定分析与效率

主比较：同Decoder下 fusion−DINO、fusion−VMamba、fusion−等权概率平均；两Decoder融合收益之差（交互项）。概率平均固定为两单分支概率的0.5/0.5，不调权重或阈值。

对每张图报告改善/退步及差值；DINO原有macro Dice分组 <.8、.8～.9、≥.9，GT面积分组 <.1、.1～.3、≥.3。统计DINO错误像素中融合纠正多少、原本正确像素被破坏多少，尤其统计“DINO错而VMamba对”的像素纠正率，以及GT边界2px区域。分组是开发集描述，不能视为独立测试上的因果证据。逐图配对2000次bootstrap仅作探索性区间，不据单seed宣称统计显著或跨seed稳定。

保存对应纠错/新增错误图，展示改善和退步病例，不单挑成功例。记录各组总参数、可训练参数、完整Encoder+Adapter+Decoder的batch1延迟/batch64吞吐与推理显存；计时不使用训练缓存，不包含读取图片或后处理；统一BF16，预热3次/重复10次，记录设备与软件版本。

本轮未匹配可训练参数或推理计算量。即便纠正了DINO原有错误，也不能据此排除容量增加的解释。若融合显示正收益，后续应加入相近可训练参数的DINO双Adapter/扩大单分支对照，再与融合做seeds43/44/45的配对确认；不把本轮作为机制归因的终点。

## 完成条件

通过模型初始化/分支梯度/冻结权重/推理效率预检和真实数据两轮训练及恢复预检后启动正式训练。所有六组300轮、三种checkpoint的预测导出、完整指标独立重算、误差分析和原有实验文件校验通过后，才标记complete。原有阶段文件只读。本轮目录包含独立日志、断点和监督进程，后台运行。
