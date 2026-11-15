# HiRAM 背景路由监督的配对实验（训练前固定）

## 问题与假设

历史 stage17 的粗/跳接 Router 在开发验证集确定性 argmax 下几乎不选择背景。stage16/17 的辅助损失仅含边界和内部正例，在纯背景目标位置两项都为零；它对前景位置的 log-softmax 梯度又会压低背景 logits。这是坍缩的合理原因候选，不等于已证明根因。训练使用随机 Gumbel，不能根据验证 argmax 推断训练中的实际采样占比。

## 三组单种子配对对照

三组均为**相同完整模型**：冻结 MambaVision-S、stage17 的 32×32 处 HiRAM 接法、原 Adapter 和 Swin Decoder；独立从头训练。只修改辅助损失系数：

* `original`：原边界项 + 内部项；
* `scaled_original`：1.5 ×（原边界项 + 内部项），用于粗略检查加强原辅助损失的影响；
* `background_supervised`：原边界项 + 原内部项 + 背景项。

原两项直接调用 stage17 实现。三类目标在历史同一宽度 2、二值 0.5 阈值、双线性缩到 32×32 的边界/内部上补 `background=1-boundary-interior`（仅对浮点误差做范围裁剪）；验证类别和为 1。各类项均为两路 Router 的分量 KL 求和，分别除以整个有效 batch 的该类软标签质量，辅助权重均为 1；主损失沿用 BCEWithLogits + soft Dice。与原代码一致，分母不额外乘分支数。`scaled_original` 和新增背景组的**名义类别项权重之和**都是 3；实际损失值或梯度范数不会被强制相同，不能视为精确等梯度对照。

三组均从同一个随机初始化副本开始；每步三组使用相同输入、数据顺序和相同的 Gumbel 随机数状态（模型各自的路由决策仍由其输出决定）。数据与 stage17 一致：ISIC2018 原 1886/808 划分、256²、旧增强、seed 42+epoch×100003。AdamW lr 1e-3、wd .01、batch64、300 epochs、30 updates/epoch、cosine T_max300 eta_min1e-5、BF16 autocast / FP32 loss、clip1；不提前停止。若硬件导致同名初始化与 stage17 不同，当前三组仍共享精确初始化，以本轮 `original` 为主要对照；stage17 仅作为历史背景。

主结果每组在 808 张开发验证图中按 Macro Dice 最高选模，精确并列取较早 epoch；保留 lowest validation loss 和 epoch300 辅助权重。开发验证集反复使用，不能称为独立测试；本轮只有 seed42，无法声称跨种子稳定。

## 路由与分割评价

训练时通过不改变采样结果的观测钩子记录两路每轮实际 Gumbel 硬采样占比及平均软概率。验证时分别记录两路确定性 argmax 比例、平均软概率、区域混淆和校准；真值仅用于训练辅助监督和事后统计，推理路由不使用真值。与旧诊断一致，区域对照把原分辨率 5×5 形态目标 area-pool 至 32² 统计，**不同于训练时 bilinear 标签采样**；同时记录两种目标的类别占比。

主要比较 `background_supervised - original` 和 `background_supervised - scaled_original` 的背景使用率、背景召回、边界路由纯度/召回、逐图 Macro Dice、Boundary F1、HD95 及逐图改善/退步；完整报告仍保存项目规定的全部宏平均与 pooled 指标。路由恢复不等于分割改善；如果只有路由恢复，要据实报告。验证阶段类分布不要求人为匹配目标面积，语义纯度和覆盖要同时看。

正式训练前必须检查权重、样本清单、三模型初值、损失与梯度、空类别稳定性、三类目标合法性、有限梯度、冻结 Encoder 及完整 batch64 显存；完成两轮独立 smoke 并验证优化器、scheduler、随机状态断点恢复，全部通过后启动正式训练。保持历史文件只读，不自动上传。
