# 医学图像分割下一阶段：研究计划、技术实现与实验规范

> **版本**：v1.0  
> **编制日期**：2026-09-12（Asia/Singapore）  
> **研究对象**：ISIC 皮肤病灶二值分割；EGE-UNet / EGE-Wave / EGE-Dual / SCF  
> **仓库**：`huamiao123/-`，不是 GCN-Train / TFS 项目  
> **本轮网页核对所见 main 提交**：`223af12b2474961ee9cc6d9a39c2589eed73044b`，提交日期 2026-08-31。[R01]  
> **交付性质**：研究执行规范与实现设计。已核对关键源码，但没有在本轮修改、推送仓库，也没有重新训练模型。文中“拟新增”文件、接口、命令均不是现成实现。附录独立参考函数的验证范围另列。

## 核心决策

**主线不是继续堆 SCF，而是先验证“辅助全局分支的收益是否因样本而异、这种差异是否可预测”，有证据后再实现收益驱动的融合选择。**

执行顺序为：

**P0 修正协议与评估 → P1 可信基线与逐图收益 → P2 SCF 干预与选择空间 → P3 受收益监督的策略选择 → P4 独立泛化与必要对照 → P5 可选的真实动态执行。**

第一轮最重要的三个产物是 `protocol_v1.yaml`、`per_image_metrics.csv` 和 `gate_intervention_report.md`，不是新的网络结构图。

本文件中的超参数、继续／停止阈值均是**拟预注册的工程起点**，不是文献确立的最佳值，也不是预期性能保证。历史成绩、新实验结果、机制解释必须分开记录。

---

## 目录

| 章节 | 内容 |
|---|---|
| [1](#s1) | 当前资产与术语边界 |
| [2](#s2) | 可证伪的研究问题 |
| [3](#s3) | P0：修复清单与回归验收 |
| [4](#s4) | 数据划分、独立测试与标签隔离 |
| [5](#s5) | 统一训练配置 |
| [6](#s6) | 指标、空掩膜与统计口径 |
| [7](#s7) | P1：逐图收益与失败模式分析 |
| [8](#s8) | P2：SCF 干预与理想选择器 |
| [9](#s9) | 模型接口与观测能力改造 |
| [10](#s10) | P3：收益监督的融合策略选择 |
| [11](#s11) | 公平对照、泛化与统计推断 |
| [12](#s12) | P5：真实跳过计算的可选扩展 |
| [13](#s13) | 实验矩阵、阶段关卡与资源预算 |
| [14](#s14) | 目录、配置、执行入口和日志规范 |
| [15](#s15) | 创新边界、论文证据与失败后的转向 |
| [附录 A](#appendix-a) | 可独立验证的参考函数 |
| [附录 B](#appendix-b) | 首轮执行清单与交付模板 |
| [参考资料](#references) | 可追溯源码与一手文献 |

<a id="s1"></a>
## 1. 当前资产与术语边界

### 1.1 保留哪些模型

下表为仓库历史汇总，不是本文件新产生的实验结果。历史使用验证集选 checkpoint 后仍在该验证集报告成绩；主表 DSC 的计算口径是汇总像素混淆矩阵，而非逐图 Dice 的平均值。[R02][R04]

| ID | 模型 | 历史 DSC | 下一阶段用途 |
|---|---|---:|---|
| B0 | EGE-UNet | 0.8817 | CNN 主基线 |
| B1 | EGE-Wave v1 | 0.8863 | 极低额外参数的全局信息补充对照 |
| B2 | EGE-Dual，差分学习率 | 0.8894 | 主要研究基线 |
| B3 | EGE-Dual + SCF | 0.8895 | 检验已有样本条件化门控是否有效 |

PTU-Dual、U-Net、Swin 可以作为后续扩展参照；第一轮不重跑全部历史模型。不同预训练条件应分组报告。新协议训练集、归一化和模型选择规则改变后，**不能要求新数字直接复现或超过这张历史表**。

### 1.2 SCF 不是全局分支

Transformer 编码器是当前所谓的“全局辅助分支”；SCF（Scale-Conditioned Fusion）是控制它如何注入 CNN 的门控机制。[R03]

普通 Dual：

$$
F'_k=F_k+\sigma(\theta_k)P_k(T_k).
$$

SCF：

$$
F'_k=F_k+\gamma_k(x)P_k(T_k),\qquad
\gamma_k(x)=\sigma\!\left(g_k(\operatorname{GAP}(\operatorname{stopgrad}(F_k)))\right).
$$

当前普通门控也是 `sigmoid(gamma_logit)`，不是无限制实数；两种门控初始值均为 0.1。SCF 每个尺度输出一个样本级标量，不是像素级或通道级掩码。[R03]

“局部／全局”在这里是架构分工的简称，不能解释为 CNN 完全没有长程信息、Transformer 只含全局信息。进一步，后几层 SCF 的输入已经受到前面融合的影响，不能把所有层描述量都视为独立的“纯 CNN 特征”。

### 1.3 融合位置与形状

根据当前 256×256 输入和模型实现，四个注入位置为：[R03]

| 名称 | 空间分辨率 | CNN 通道 | Transformer 通道 | 对应阶段 |
|---|---:|---:|---:|---|
| fuse2 | 64×64 | 16 | 48 | 浅层 |
| fuse3 | 32×32 | 24 | 96 | 中浅层 |
| fuse4 | 16×16 | 32 | 192 | 深层 |
| fuse5 | 8×8 | 48 | 384 | 最深注入层 |

后续所有四维策略向量固定采用 `[fuse2, fuse3, fuse4, fuse5]`，禁止不同脚本自行改变顺序。

### 1.4 本阶段不同时做什么

暂不扩展弱监督、新的大型 Transformer、边界修复头、复杂空间门控或新的 Halting。它们不是被证明无效，而是会使“提升来自哪一项”无法判断。Wave 保留作效率参照，不与 Dual 的机制研究强行合并成一个复杂模型。

<a id="s2"></a>
## 2. 可证伪的研究问题

| 假设 | 对应证据 | 不支持时的处理 |
|---|---|---|
| H1：不同样本对辅助分支的收益不同 | 配对逐图收益分布、候选策略上界，多个种子重复 | 差异很小则不继续复杂样本路由 |
| H2：SCF 利用了图像与门控的对应关系 | 原始、固定均值、打乱对应关系的比较 | 打乱无影响则不声称学会条件化需求 |
| H3：不看 GT 也能预测收益 | 隔离数据上的收益预测与路由效果 | 只能拟合训练收益则停止扩大路由器 |
| H4：提升并非仅来自训练预算或候选模型容量 | 固定策略、普通 SCF、简单路由和额外预算基线 | 简单方案同样有效就采用简单方案 |
| H5：收益能迁移到未参与开发的数据 | 锁定协议的独立外部测试 | 只在旧验证集有效，不作泛化贡献 |
| H6（可选）：选择策略能节省真实计算 | 分支调用记录与端到端延迟 | 只乘零却没跳过执行，不作加速贡献 |

**禁止预设“小病灶一定需要更多全局信息”。** 病灶尺度、对比度、预测不确定性都只是候选解释变量；它们与门控相关，不等于它们决定了辅助收益。

独立训练的 CNN 与 Dual 的差距也不自动等于“全局信息的因果作用”。两者还存在参数、优化和表示差异。本阶段使用“模型收益差异”“融合干预效果”，不轻率使用“因果需求”。

<a id="s3"></a>
## 3. P0：先修复协议，再开始模型比较

### 3.1 已核对的问题与风险

本表是本次关键路径静态核对结果，不是对整个仓库完成全量运行审计的证明。所有修复均需落实到代码和回归测试后才能标记完成。

| 编号 | 位置 | 已见问题／风险 | 拟修改 | 验收 |
|---|---|---|---|---|
| A01 | `tests/eval_boundary_metrics.py::hd95/assd` | 采样整个前景区域的距离，不是轮廓距离 | 独立表面距离模块，固定定义 | 相同、平移、分离掩膜测试 |
| A02 | 同文件 `evaluate_boundary` | 空预测／空 GT 直接被跳过 | 逐图保留，输出失败标记和空掩膜约定 | 输出行数等于数据清单行数 |
| A03 | 同文件 `boundary_f1` | 膨胀外环重叠不等于容差匹配 BF1 | 改为边界点到边界点的容差匹配 | 容差 0/1/2 px 的单调性 |
| A04 | `train.py` 尾部 | 最终测试仍接收 `val_loader` | 显式 `split` 和独立测试入口 | test 不参与 checkpoint 选择 |
| A05 | `utils.py::myRandomRotation` | 初始化时只采样一次角度 | 每次调用采样，同步作用于图像和掩膜 | 同种子可复现，连续角度会变化 |
| A06 | `utils.py::myResize` | 掩膜没有显式最近邻插值 | 图像双线性，硬掩膜最近邻 | 硬掩膜值仅为 0/1 |
| A07 | `engine.py::train_one_epoch` | 只有整除累积步数时更新，没有尾窗口收尾 | 按有效窗口实际样本数归一化并更新 | 不整除样本的 toy 梯度测试 |
| A08 | 同函数 | GradScaler 每个 epoch 重建；`step += iter` 非线性累计 | Scaler 放到训练生命周期；分别维护 micro/optimizer step | 恢复训练状态和步数正确 |
| A09 | `train.py` 的 `gamma.csv` | 验证后只读取最后一次 forward 的统计 | 验证循环内按 image_id 收集全部门控 | B=1 与 B>1 统计一致 |
| A10 | `datasets/dataset.py` | 图像和 mask 各自排序后按位置配对 | 根据规范化 ID 显式连接并查重 | 文件缺失／错配立即报错 |
| A11 | `tests/eval_efficiency.py` | eval 模式下测量反向与更新；批次不同；有重复 sigmoid 风险 | 分开推理、训练测试，统一输出适配 | 能区分 forward 与 train step |
| A12 | 多处导入／启动脚本 | `/root` 绝对路径，依赖同名 `models` 包 | 相对工程根目录或明确包命名 | 移动仓库位置仍能导入 |
| A13 | `engine.py` 全局 loss 缓存 | 同进程切换配置时可能复用旧损失对象 | 显式传递 criterion，删除跨实验缓存 | 顺序运行两种配置不串用 |
| A14 | `train.py` checkpoint | 现有恢复未覆盖全部随机和 scaler 状态 | 保存 RNG、generator、scaler、步数、配置摘要 | 支持声明过的恢复粒度 |

A01–A03 见 [R07]；A04/A09/A14 见 [R05]；A05/A06 见 [R06]；A07/A08/A13 见 [R04]；A10 见 [R08]；A11 见 [R11]；A12 见 [R03][R07]。

**A07 的重要限定**：EGE 默认 `gradient_accumulation_steps=1`，不会触发不整除尾窗口问题。改成累积训练后，EGE 现有循环缺少 epoch 开始清零和末尾收尾，残余梯度可能跨 epoch 延续；不能统一描述为“所有模型都丢了最后一个 batch”。不同训练脚本应分别检查。AMP 下跨 epoch 残余梯度叠加 scaler 重建风险更大。[R04][R09]

### 3.2 不误判归一化问题

`myNormalize` 先使用常数 mean/std，再做逐图 min-max 到 `[0,255]`。对于非常量图像、正标准差，在精确算术下前面的仿射变换会被后面的 min-max 抵消。因此，**仅仅看到 train/val 使用不同常数，不能直接断言发生了有效的验证集统计泄漏**。实际应检查零分母、输入尺度一致性和各基线处理流程。[R06]

冻结两个清晰的模式：

| 模式 | 用途 | 输入处理 |
|---|---|---|
| `legacy_eval_v1` | 评估历史 checkpoint，仅用于诊断 | 保留历史有效的逐图 min-max 到 `[0,255]`；加常量图像保护；不重新训练 |
| `research_v1` | 所有新训练和正式对照 | RGB float32 `/255` 到 `[0,1]`；不额外做 split 特定标准化；全部重新训练 |

不能将 `[0,1]` 输入直接送入按旧尺度训练的 checkpoint，再把性能变化归因于模型优劣。所有处理改变记入协议版本。

### 3.3 梯度累积的正确约束

以实际样本为权重，而不只是简单把 loss 除以固定 `accum_steps`。一个更新窗口内若有大小为 32 和 30 的两个 batch，应分别对 batch 平均损失乘 `32/62` 和 `30/62`。只有一个尾 batch 时，其权重为 1。

每个窗口开始 `zero_grad(set_to_none=True)`，窗口内反向累积，窗口末才 unscale、梯度裁剪、optimizer step、scaler update。训练器生命周期只创建一次 scaler，checkpoint 同时保存它。AMP 累积规范参考官方说明。[R13]

即使有效 batch 相同，累积与真实大 batch 也不保证数值完全等价，尤其存在 batch 统计或随机操作时。正式配对实验尽量使用相同 micro-batch 与累积策略。

### 3.4 随机性与中断恢复

seed 必须在模型和 transform 实例化前设置；固定 Python、NumPy、Torch 和 DataLoader generator，并配置 worker 初始化。使用 `PYTHONHASHSEED` 时，应在启动解释器前设置；运行中改环境变量不等于回溯改变当前进程哈希行为。PyTorch 官方也不保证跨版本、平台完全逐位复现。[R12]

第一版只承诺**epoch 边界恢复**。需要记录 loader generator 状态；为减少 worker 状态问题，默认 `persistent_workers=False`。若要求更强恢复能力，建议增强参数由 `(seed, epoch, image_id)` 派生，且记录 sampler 状态；不能仅保存一个 seed 就宣称任意中断严格复现。

### 3.5 P0 回归测试最低清单

数据配对、重复 ID、空文件、尺寸不一致；旋转变化与同步性；mask 最近邻；常量图归一化；模型输出概率契约；边界指标空集；尾梯度窗口；无干预 forward 等价；门控记录无跨 batch 串用；checkpoint 恢复；数据划分互斥。全部通过后，才建立 `research_v1` 基线。

<a id="s4"></a>
## 4. 数据划分：先解决标签角色，再追求高分

### 4.1 数据清单是第一道关卡

历史交接采用训练 1886、验证 808 的内部划分。[R16] 正式执行必须重新数文件并追溯这些图像的来源，不能直接把总数或目录名当成官方划分证明。

每张图像建立以下最小清单：

```text
image_id,image_path,mask_path,source_dataset,source_original_split,
patient_id,lesion_id,group_id,image_sha256,mask_sha256,original_h,original_w,
parent_image_id,assigned_split
```

`group_id` 是划分和统计的统一键：有 patient_id 时先按患者分组，再将同病灶、同 parent 或已确认重复图像建立的关联合并为连通组。不能让同一患者的不同病灶跨 split。`patient_id` 缺失必须写 `unknown`，不可假装已经患者级独立。由同一原图裁剪或增强得到的样本必须跟随 parent 一起分组。先查精确重复，再筛近重复并人工核验；近重复算法命中不是自动确认重复。

### 4.2 推荐的第一版：固定留出路由数据，不急于复杂交叉拟合

将**历史训练池内部**按病灶／患者分组拆成：

| 集合 | 建议比例／来源 | 可以做什么 | 不可以做什么 |
|---|---|---|---|
| `S_seg` | 原训练池约 70% | 训练所有候选分割器和共享路径模型 | 不充当未见样本收益证据 |
| `R_fit` | 原训练池约 20% | 冻结分割器后，训练收益预测器 | 不回流训练第一版分割器 |
| `R_val` | 原训练池约 10% | 选择路由 checkpoint、固定策略及路由 margin | 不充当最终测试 |
| `V_dev` | 现有 808 验证图像，经核验后 | 选择分割 checkpoint、方法开发与诊断 | 不称独立 test |
| `T_external` | 未参与此前开发的外部标注集 | 冻结方案后的最终测试 | 不调阈值、路由、增强或 checkpoint |

比例按组落实，实际样本数可能不是精确 70/20/10。分割 checkpoint 统一由 `V_dev` 选择；路由器由 `R_val` 选择。`V_dev` 上的反复改进只算开发证据。

**严禁使用已经在完整 1886 张上训练过的旧 checkpoint，然后声称拆出的 R_fit/R_val 是该模型未见数据。** 采用此协议时，分割器应重新训练，或使用能证明没有接触这些标签的初始化。

这是一种简单、可审计的 MVP，代价是分割器可用训练样本变少。它的目标是验证收益学习是否成立，不是最大化单数据集分数。

### 4.3 公平性：两种比较都要有

**机制内对照**：B0/B1/B2/B3 和候选路径都用相同 `S_seg`；共享模型的固定策略与路由策略使用完全相同的分割权重，只改变选择方式。

**实用强基线**：后续另训固定融合模型，允许其使用整个原训练池 `S_seg∪R_fit∪R_val`。路由方法虽然没有用全部标签训练分割器，但使用了其中一部分标签训练路由器，必须把这些标注预算算进去。最终不能只与一个人为少给数据的基线比较。

### 4.4 提高数据利用率的升级方案

只有 MVP 证明有效后，再做分组 K 折的 out-of-fold 收益标签：每折候选分割器不能见该折标签，生成折外收益后训练路由器。其额外成本是 K 组候选训练，而且不同折的特征空间与最终全数据重训模型不一定一致。

**OOF 不能被一句“做 K-fold”解决**：必须明确每折初始化、特征来源、模型选择数据，以及最终模型的收益校准。如果重训分割器改变了候选输出，就要重新校准或验证路由，不能原封不动沿用旧收益标签。首轮不做 OOF，也不声称解决了这种分布变化。

### 4.5 外部测试的边界

外部数据必须具有可用的二值标注、合法使用权限，并完成与开发集的病灶／图像重叠核查。另一个 ISIC 年份不自动等于独立来源；复制数据、重编码或缩放图也不能算新测试。

本文件不预设已经获得了某个外部数据集。执行者需先在 `external_data_audit.md` 中写清来源、许可、样本数和去重结果，再注册为 `T_external`。没有这项证据时，结论止于开发集研究。

<a id="s5"></a>
## 5. 统一训练设置：先冻结一个配方

下表为**新协议建议值**；其中模型、优化器和损失的起点参考现有配置，其余变化均需记入新协议，不宣称是历史完全复现。[R06][R09][R10]

| 项目 | research_v1 默认值 | 说明 |
|---|---|---|
| 输入 | RGB，256×256，float32 `[0,1]` | 不混用旧 checkpoint 输入尺度 |
| 分割种子 | 先 42；确认时加 3407、2026 | 同 seed、同 split 配对比较 |
| epoch | 300，完整训练 | 调试短跑不能用来宣布模型优劣 |
| micro-batch | 优先 32 | 最小显存模型组先实测；必要时全组调整 |
| 梯度累积 | 2，名义有效 batch 64 | 尾窗口按实际样本加权 |
| precision | FP32，AMP 关闭 | 先保持输出和损失契约简单 |
| optimizer | AdamW | 两组参数无重复、无遗漏 |
| CNN + projection + gate 学习率 | `1e-3` | 初始值 |
| Transformer 学习率 | `1e-4` | 仅含 `t_encoder` 参数 |
| weight decay | `1e-2` | 第一版沿用统一处理，不额外改 no-decay 规则 |
| betas / eps | `(0.9,0.999)` / `1e-8` | 与基线一致 |
| scheduler | 每 epoch CosineAnnealingLR，T_max=300 | 第一次 step 在首轮训练完成后 |
| eta_min | 两组均 `1e-5` | 这不保持全程 10:1 学习率比例 |
| 梯度裁剪 | global norm 1.0 | 在 unscale 后，仅每次更新时执行 |
| 损失 | final BCE + Soft Dice，加深监督 | 见下式 |
| 模型选择 | `V_dev` 最低逐图平均 final BCE+Dice | 不含辅助头；并列时取较早 epoch |
| 验证频率 | 每 epoch | 计算 final loss 和逐图指标 |
| 二值阈值 | `p >= 0.5` | GT 同样以 `>=0.5` 二值化 |
| TTA / 后处理 | 均关闭 | 只允许在后续独立附加实验中开启 |
| 预训练 | 第一版全部无预训练 | 预训练比较单独成组 |
| early stopping | 分割器关闭 | 所有方法同训练预算；保存 best 与 last |

### 5.1 增强和预处理

先读取 RGB 并转换为 `[0,1]`，resize 到 256×256；训练时水平翻转 p=0.5、垂直翻转 p=0.5、以 p=0.5 采样 `Uniform(0°,360°)` 旋转。图像采用双线性插值，掩膜最近邻，旋转空白填充 0。验证与收益标签生成只执行确定性 resize，不旋转、不随机裁剪。[R14]

第一版不启用额外尺度裁剪、颜色扰动或去毛发。它们会改变研究中的尺度／对比度分布，应作为后续独立因素，而不是只给新方法更强增强。

### 5.2 损失与输出契约

令预测概率为 p，GT 为 y，Soft Dice 使用平滑项 1：

$$
\ell_{seg}(p,y)=\operatorname{BCE}(p,y)+1-
\frac{2\sum p y+1}{\sum p+\sum y+1}.
$$

先逐图计算，再对 batch 求平均。深监督从最深到最浅使用 `[0.1,0.2,0.3,0.4,0.5]`：

$$
L=\ell_{seg}(p_{final},y)+\sum_{j=1}^{5}\lambda_j\ell_{seg}(p_{aux,j},y).
$$

这些权重源于当前实现；本阶段不把调整损失权重当成创新变量。[R06]

当前 EGE 系模型输出已经过 sigmoid。[R03] 第一版保持概率输出，使用概率 BCE，评估不得再次 sigmoid。若以后启用 AMP，建议另建兼容的 logits 输出模式并使用 `BCEWithLogitsLoss`，同时确认桥接模块使用的内部张量不被误改；这应独立测试并让所有对照采用同一契约。

### 5.3 模型初始化与学习率检查

B2/B3 保持相同结构配置：`t_embed=48`、`t_depths=(2,2,2,2)`、`t_head_dim=16`、`t_sr_ratios=(4,2,1,1)`、`t_mlp_ratio=4`、`t_drop_path_rate=0.1`。[R10]

在新协议的配对运行中，能对应的公共参数应来自同一个初始状态，而不是以为设同 seed 就自动使不同结构共享初始化。保存初始化权重的 hash。SCF 最后一层初始化需要在父模型的全局初始化之后检查，确保初始 gate 真正为 0.1。

记录每个参数组的参数量、名字摘要和学习率曲线。统一 `eta_min` 的余弦调度会让两组学习率比例随训练改变，不能描述为“始终 10 倍差分”。

<a id="s6"></a>
## 6. 指标：逐图、空掩膜和表面距离必须写清

### 6.1 主次指标

**主指标为逐图平均 Dice（macro Dice）**，用于研究样本条件化收益。同步保留历史的 pooled foreground Dice / IoU 作为辅助，不把后者命名成“逐图 mIoU”。

$$
D_i=\frac{2TP_i}{2TP_i+FP_i+FN_i},\qquad
D_{macro}=\frac{1}{N}\sum_iD_i.
$$

$$
D_{pooled}=\frac{2\sum_iTP_i}{2\sum_iTP_i+\sum_iFP_i+\sum_iFN_i}.
$$

| 层次 | 必报指标 |
|---|---|
| 分割质量 | macro Dice、macro foreground IoU、pooled foreground Dice/IoU |
| 误检／漏检 | pooled sensitivity、specificity；逐图 FP/FN 像素量 |
| 边界 | HD95、明确版本的 ASSD、BF1@2px |
| 严重失败 | Dice<0.5 的图像比例；非空 GT 对应空预测比例 |
| ISIC 兼容辅助指标 | thresholded Jaccard：IoU<0.65 记 0，否则记原值，再逐图平均 |
| 条件化效果 | 正／负收益率、平均 regret、策略频率与理想选择空间 |

ISIC2018 官方 Task 1 的目标指标是上述 thresholded Jaccard；它不同于仓库历史 DSC，也不同于普通平均 IoU。仅当遵循相应数据和评估协议时，才与官方结果讨论可比性。[R15]

### 6.2 表面距离的固定定义

推荐固定使用一份有版本号的 `surface-distance` 实现，记录其 commit／包版本。2D 图像按 `spacing=(1.0,1.0)` 计算时单位是 **256×256 评估网格上的像素**；参数名即使叫 `spacing_mm`，也不能因此把结果标成毫米。[R17]

HD95 采用两个有向、表面元素加权的 95 分位数的最大值。ASSD 固定采用**两个有向面积／长度加权平均距离的算术平均**，字段命名 `assd_dirmean_px`，避免与将两方向所有元素一起加权的版本混淆。两种定义不等价，全文只能使用一个主版本。[R17]

BF1@2px 使用边界点在对侧边界 2 像素容差内的 precision/recall，再取 F1；明确采用的 4 邻域像素边界提取方式。它与加权 surface Dice 不是同一指标，不混名。边界容差在新方法试验前固定。

原始分辨率评估可作为附加结果：先把预测概率 resize 回原图，再阈值化并与原始 mask 比较；必须另标单位和协议。不得在一张表混入不同分辨率的像素距离。

### 6.3 空掩膜约定

| GT | 预测 | Dice/IoU | BF1 | 原始 HD95/ASSD |
|---|---|---|---|---|
| 非空 | 非空 | 正常计算 | 正常计算 | 正常计算 |
| 非空 | 空 | 0 | 0 | `inf`，并记录失败 |
| 空 | 非空 | 0 | 0 | `inf`，并记录失败 |
| 空 | 空 | 1（本协议约定） | 1（本协议约定） | `NaN`，标记不适用 |

不能用一个 `continue` 把空预测移出所有统计。空 GT 在此任务中应先触发数据质量核查；表格约定是为了代码完整性，不意味着数据本应包含大量空 GT。

边界汇总同时给出：全体样本总数、单侧为空数、双侧为空数、有限值样本的条件统计。若用有限值均值比较两个模型，还必须说明各自有效集合可能不同；配对边界检验应使用预先定义的共同有效集合，同时报告被排除情况。

可选补充 `hd95_capped_px`：单侧为空以图像对角线代价计入，双侧为空计 0。必须明确这是人为有限惩罚的**补充指标**，不是标准原始 HD95。不能只展示补充指标掩盖失败数。

### 6.4 回归案例

相同方形的边界距离为 0；分离轮廓距离为正；非空 GT + 空预测不能被当作完美；BF1 随匹配容差增大不下降；交换 GT/预测后对称指标相同。

已有区域取样缺陷可用“大方形平移 1px”暴露。不同像素边界／表面元素离散化的 ASSD 数值可能不同，不要把某一个像素边界版本的精确数值硬当成所有库的标准答案。

<a id="s7"></a>
## 7. P1：先做逐图收益表，不先改模型

### 7.1 两类收益分别记录

**跨模型收益**：

$$
\Delta_i^{model}=D_i(B2)-D_i(B0).
$$

它描述独立模型预测之间的互补性，受容量和训练差异影响。

**同一模型内干预收益**：

$$
\Delta_i^{intervention}=D_i(f_{native})-D_i(f_{intervened}).
$$

它描述当前已训练系统受到指定干预后的变化。突然关分支可能造成分布偏移，不能直接作为可靠的训练监督目标。正式收益监督使用第 10 节经过训练的候选路径。

### 7.2 每张图保存什么

建议按长表记录每个 `image_id × model × seed × policy`：

```text
run_id,protocol_id,split_hash,source_commit,checkpoint_sha256,
image_id,group_id,split,model,seed,policy_id,threshold,
dice,iou,tp,fp,fn,tn,gt_empty,pred_empty,
hd95_px,assd_dirmean_px,bf1_2px,surface_status,
gt_area_ratio,pred_area_ratio,contrast_band,
g2,g3,g4,g5,r2,r3,r4,r5,probability_file
```

无门控的模型将门控字段写 `null`，不要填看似有意义的 0。导出概率使用 float32，避免降精度后阈值附近像素翻转；最终指标从相同概率文件和统一 evaluator 生成。

`r_k` 是实际注入相对强度：

$$
r_k(x)=\frac{\|\gamma_k(x)P_k(T_k(x))\|_2}{\|F_k(x)\|_2+10^{-6}}.
$$

同时可记录分母、投影范数和两路特征余弦相似度。单看 gamma 大小容易被投影尺度混淆；`r_k` 也只是描述量，不能独自证明重要性。

### 7.3 病灶分组

面积比例用评估网格上的 `GT foreground / (H×W)`。阈值只在 `S_seg` 上计算，保存 Q25/Q75：小病灶 `a≤Q25`，中等 `Q25<a≤Q75`，大病灶 `a>Q75`，其余数据沿用同一阈值。

边界对比度建议定义为灰度图中 GT 边界内外窄带均值差的绝对值，再除以全图灰度标准差加 epsilon。窄带宽度固定 2px；带为空时记缺失。它只用于离线分析；不能将用 GT 计算的带对比度作为部署路由输入。

毛发、遮挡、模糊标签只有在来源可靠时才使用。没有标签时，不要靠主观挑几张图就宣称模型对这类困难有优势。小分组样本过少时只作描述，报告样本数和区间。

### 7.4 必交的分析结果

| 结果 | 判断价值 |
|---|---|
| 各模型逐图 Dice 散点和差值分布 | 普遍小幅改善，还是帮助一部分、损害另一部分 |
| 面积三组配对差值 | 尺度假设是否有支持 |
| FP/FN 与预测面积变化 | 是否只是整体更倾向预测前景 |
| gamma / r 与收益的关系 | 门控是否与真实收益一致，而不只是与面积相关 |
| 最大提升、最大退化、接近不变的案例 | 排查标签、阈值与具体失败模式 |

收益类别先用 `ΔDice>0.01`、`<-0.01`、其余三类，即 ±1 个百分点；同时报告不做阈值分类的原始均值和分位数，不能只报挑选后的正收益样本。

案例按规则选择，例如每类绝对差值排序前 10 个加固定随机样本，而不是只手选好看的。所有图使用相同阈值和显示方法。

### 7.5 早期诊断如何省训练

P0 的 evaluator 可以先应用于旧 checkpoint，使用 `legacy_eval_v1`，探索明显的数据或指标问题。该结果不算修正版训练成绩，也不能修复历史增强问题。正式 B0–B3 必须在新协议重新训练。

第一批新训练先做 B0/B2/B3 的 seed=42。Wave 在关键链路跑通后补；如 B2/B3 结果极不稳定，优先排查实现，而不是立刻扩张实验矩阵。

<a id="s8"></a>
## 8. P2：SCF 干预与理想选择器上界

### 8.1 先做无干预一致性

对每个样本在 native SCF 前向中记录四层 gate，随后用该样本自己的四维 gate 向量重放。由于后续 CNN 特征受先前融合影响，必须从头重跑对应前向，不是只替换最终输出。

同一 checkpoint、eval 模式、相同输入下，native 与 self-replay 预测应在 FP32 约定容差内一致。第一版测试 `atol=1e-6, rtol=1e-5`；若失败，先检查 gate 顺序、形状、随机层和状态污染，不能继续解释干预结果。

### 8.2 干预矩阵

| ID | 操作 | 是否训练新模型 | 主要问题 |
|---|---|---|---|
| G0 | native SCF | 否 | 参考 |
| G0R | 使用自己原始四维 gate 重放 | 否 | 接口是否正确 |
| G1 | 四层分别用 R_fit 上的无标签均值 | 否 | 样本条件化是否必要 |
| G2 | 在 V_dev 图像之间打乱完整四维 gate 向量 | 否，20 个 permutation seed | 匹配当前图像是否重要 |
| G3-k | 第 k 层 mask=0，其余 gate 正常重新计算 | 否 | 含下游自适应响应的总干预效果 |
| G3F-k | 第 k 层 mask=0，其余 gate 固定为该样本原始值 | 否 | 区分后续 gate 反馈与特征变化 |
| G4 | 四层 mask 全 0 | 否 | 当前系统对注入的依赖；不等同独立 CNN |
| G5 | 保持网络结构，训练固定门控版本 B2 | 是，已有基线 | 消除纯推理干预的分布偏移疑虑 |

**打乱实现细节**：不要在 batch 内简单 `randperm(B)`。当前验证常用 B=1，这样根本没有打乱。应先按 image_id 建立整个开发集的 gate bank，再固定一个 dataset-level permutation；完整向量一起置换，以保留层间联合关系。随机结果报告 20 次均值和离散度，不挑最差一次。

G2 仅是开发集机制诊断，不是部署策略；不得把依赖整批测试集建立 gate bank 的过程包装成独立逐图推理。

### 8.3 如何解释干预结果

G1≈G0 且 G2≈G0：说明目前没有足够证据证明门控和图像对应关系重要，不再将 SCF 的样本适应作为既定成果。

G1/G2 明显变差：支持门控对应关系有用，但仍不说明门控读懂了“病灶尺度”或产生临床因果效果。

G3 与 G3F 差异明显：说明多层反馈不能忽略，后续不应把四层 gate 看成四个相互独立的旋钮。

G4 变差：只说明当前参数依赖辅助路径；不证明“没有辅助分支就达不到此性能”。需要与 B0 和训练过的无注入路径分别比较。

### 8.4 理想选择器：先计算空间，再训练预测器

设某个数据划分有 N 张图，A 个候选输出，逐图 Dice 矩阵为 `D ∈ R^(N×A)`。开发集上的候选集合选择空间定义：

$$
H_{dev}=\frac{1}{N}\sum_i\max_aD_{ia}
-\max_a\frac{1}{N}\sum_iD_{ia}.
$$

第一项使用每张图的真实标注选最佳候选，第二项所有图像采用同一个最佳固定候选。这是**相对于所列候选输出的经验选择空间**，不是所有可能分割模型的理论上界。

开发期允许先比较 B0/B2/B3 的输出互补性。正式收益学习前，必须重新计算受训练候选路径的 H，不能把跨独立模型的上界直接当成共享模型也能达到的上界。

部署固定策略 `a_fixed` 只根据 `R_val` 选择并冻结。外部测试上报告的实际增益是：

$$
G_T=\frac1{|T|}\sum_{i\in T}\left(D_{i,\pi(x_i)}-D_{i,a_{fixed}}\right).
$$

测试标签可以在最终离线分析中计算并明确标为 oracle 的成绩，但不能用于选择实际部署策略、参数或 checkpoint。主结果必须来自路由器自己的选择。

### 8.5 预注册的工程关卡

以下只是默认投入决策标准，不是显著性判定或领域标准：

| 开发集上受训练候选路径的 H | 决策 |
|---:|---|
| <0.002（0.2 个百分点） | 通常停止精度导向复杂路由；空间过小 |
| 0.002–0.005 | 灰区：先试简单线性／小树／MLP，不扩张网络 |
| ≥0.005（0.5 个百分点） | 可以投入收益预测；仍需验证泛化 |

这些门槛至少在两个分割训练种子上检查。候选集合临时扩张会抬高 oracle，必须记录候选数和所有变更；不能搜索大量候选后只保留最有利的一组而不披露。

<a id="s9"></a>
## 9. 模型接口改造：可观测、可干预、默认行为不变

### 9.1 最小修改范围

主要修改 `models/ege_dual.py` 的 `FusionInject` 和 `EGEDualUNet.forward`，新增统一输出适配器与研究脚本，不直接重写全部模型。历史 checkpoint 默认加载应尽量保持 `state_dict` key 不变。[R03]

拟新增接口如下，**这些参数目前不是仓库现成 API**：

```python
# 设计签名，不是可以立即调用的现有接口。
def forward(
    self,
    x,
    *,
    fusion_override=None,   # [B,4]，绝对 gamma；None 表示网络自身 gate
    fusion_mask=None,       # [B,4]，0/1 注入开关；None 表示全 1
    return_aux=False,
):
    ...
```

`fusion_override` 的含义固定为替换绝对 gamma，不能又在某些脚本解释为倍率。实际注入为 `mask_k * gamma_k * projected_k`。严格检查范围 `[0,1]`、batch size、层顺序和空间形状；token 数必须等于对应 `H×W`。

### 9.2 输出契约

`return_aux=False` 时，完整保留原来的概率输出和深监督 tuple。`return_aux=True` 时，建议返回明确包装：

```text
{
  legacy_output: 原有输出，
  research_aux: {
    gamma_native: [B,4],
    gamma_effective: [B,4],
    injection_ratio: [B,4],
    descriptor: [B,d]（按需）
  }
}
```

新增 `unpack_segmentation_output`，显式识别研究包装，再取得 final probability / auxiliary probability。当前 `engine.py` 对一般 dict 会走其他损失分支，**不能直接新增 dict 输出而不改调用方**。[R04]

观测数据记录前 `detach`，避免日志保存整张计算图；不同 batch 的引用不能共用可变字典。统计在整个 loader 内聚合，不从“最后一次 forward”推断数据集均值。

### 9.3 让接口可测试

| 测试 | 要求 |
|---|---|
| 默认新 forward vs 原 forward | 同 checkpoint、同输入，概率相同至设定容差 |
| 绝对 gate 重放 | 与 native 输出一致 |
| 全 1 mask | 等价默认注入 |
| 全 0 mask | 数值路径明确；不自动声称等于独立 EGE |
| B=1 和 B=4 | 相同图像的门控和预测与批次组合无关，允许浮点误差 |
| 输出概率 | 范围 `[0,1]`，sigmoid 不重复 |
| shape 错误 | 主动抛错，不做静默错误 reshape |
| 梯度 | 开启的 projection/gate 可获得梯度；日志不改变梯度 |
| 历史权重 | 严格加载；任何新增参数单独记录和初始化 |

第一版只支持当前 `bridge=True, gt_ds=True, 256×256` 主设置；其他配置要么补测试，要么明确拒绝，避免表面上可配置但实际不可用。

<a id="s10"></a>
## 10. P3：收益监督的融合策略选择——第一版技术方案

### 10.1 定位：候选方案，不先定论文名

暂称 **Benefit-guided Fusion Policy（收益指导的融合策略）**，仅是内部工作名。第一版采用离散策略选择，**不是直接把 SCF 的连续 gamma 替换为一个新 MLP 就结束**。

目标是比较：在相同分割权重和候选路径下，使用逐图真实收益训练选择器，是否优于固定策略、一般门控与简单路由。先验证预测质量，不同时承诺计算节省。

### 10.2 候选策略只保留三种

| 策略 | mask `[fuse2,fuse3,fuse4,fuse5]` | 含义 |
|---|---|---|
| A0：none | `[0,0,0,0]` | 不注入辅助特征 |
| A1：deep | `[0,0,1,1]` | 仅在 16×16 和 8×8 位置注入 |
| A2：full | `[1,1,1,1]` | 四个位置都注入 |

启用位置仍使用同一套可学习标量 gate 和投影层，不把启用误写为 gamma=1。候选集不能在测试上改变。

A0 是共享网络中受训练的无注入路径，不是独立 B0。A1 虽然只在深层注入，Transformer 仍需计算其前序编码阶段，**不能按“只开两层 gate”就估算省了一半编码器计算**。

### 10.3 先让三条路径都能工作，再生成收益标签

从相同初始化训练一个共享路径模型，用 `S_seg` 训练，每个输入都评估 A0/A1/A2，损失为：

$$
L_{multi}=\frac13\left(L_{seg}^{A0}+L_{seg}^{A1}+L_{seg}^{A2}\right).
$$

每条路径包含相同的深监督规则。可以在同一 batch 内共享输入和 Transformer 特征计算，减少重复；三条 CNN/decoder 路径仍按各自注入后的特征计算。先构造平均总损失，再一次 backward，避免多次 backward 误释放共享图或 `detach` 掉全局分支。

共享 Transformer 只前向一次时，候选路径使用同一份 DropPath 随机实现，减少比较噪声；生成收益标签时必须 `eval()`。该设计的显存与训练开销需要实测，不默认等于原 Dual。显存不足时可统一减小 micro-batch 并增加梯度累积。

**共享模型 checkpoint 选择**：在 `V_dev` 上，取三条路径 final BCE+Dice 的等权平均，最小者为 best；不按逐图 oracle 选 checkpoint，避免专门优化有标签选择上界。

同时记录每条路径单独的开发集分数。若共同训练导致所有路径相对相应独立训练模型明显退化，先检查多路径训练，而不是直接把低质量候选交给路由器。必要时先用三个独立候选做概念验证，但须如实报告模型总参数和部署存储，不伪装成单一轻量模型。

### 10.4 冻结分割器，生成逐图收益标签

共享模型 best 确定后，将所有分割权重设 `requires_grad=False` 并保持 `eval()`，在 `R_fit` 与 `R_val` 上分别缓存各路径概率和 per-image Dice。第一版不加随机增强，以使收益目标与部署输入分布一致。

**主监督选择 Dice 差值**，直接对齐主指标：

$$
u_{i,a}=D_i(A_a)-D_i(A_0),\qquad u_{i,0}=0.
$$

收益可以为负；正值表示该候选比无注入路径更好。训练代码中统一命名 `utility_delta_dice`，避免与概率或 loss 混淆。

逐图缓存的计算应严格等价于：

```text
utility_delta_dice[:, 0] = 0
utility_delta_dice[:, 1] = dice_deep - dice_none
utility_delta_dice[:, 2] = dice_full - dice_none
```

**预注册消融**：用 final BCE+Soft Dice 的差值替代 Dice 差值：

$$
u^{loss}_{i,a}=\ell_{seg}(p_{i,0},y_i)-\ell_{seg}(p_{i,a},y_i).
$$

该消融的单位不同，不能与 Dice 差值混用阈值。它可能提供更平滑监督，但不保证更好地排序真实 Dice。第一版不叠加边界、多尺度、熵等奖励，避免目标无法解释。

两种收益都不向分割器反向传播。缓存键至少包含 `image_id, checkpoint_sha256, policy_id, preprocessing_hash, threshold`；其中任何一项变化都必须重新计算收益。

### 10.5 收益预测器输入：先用真正能提前得到的描述量

第一版输入使用 **首次注入之前的 `fuse2` CNN 特征**，形状 `[B,16,64,64]`。取每通道空间均值和标准差（`unbiased=False`），拼成 32 维向量。它位于全部辅助注入之前，避免读取已经融合后的特征而宣称“预先预测全局信息需求”。

在 `R_fit` 上拟合描述量的标准化参数，`R_val/V_dev/T_external` 只复用，不重新估计。输入不能包含 GT 面积、真实 Dice、ground-truth 边界、图像 ID 编码或测试集统计。

候选预测器采用：

```text
32维描述量 → Linear(32,32) → GELU → Dropout(0.1) → Linear(32,2)
```

输出对应 A1/A2 相对 A0 的两个有符号收益，A0 收益固定为 0。输出层不加 sigmoid，不强制收益非负。另设同输入的线性回归基线，检验是否真的需要非线性网络。

后续可加仅由无注入预测计算的面积、概率熵、前景附近不确定性，但这需要先跑额外分割路径。它是不同信息预算的方案，推理开销必须计入，不可与廉价特征版本混为一谈。

### 10.6 回归目标与超参数

对 A1/A2 的收益在 `R_fit` 上分别计算均值 `mu_a` 与标准差 `s_a`，设置 `s_a=max(std_a,1e-3)`，训练归一化目标：

$$
\tilde u_{i,a}=(u_{i,a}-\mu_a)/s_a.
$$

采用 Huber 损失，标准化后的 `delta=1.0`。部署时先恢复原尺度，再比较三个候选；**不得在各自标准化的收益空间直接取 argmax**，因为不同输出的均值和尺度不同。

| 项目 | 首版默认 |
|---|---|
| router train / validation | R_fit / R_val |
| 分割器 | 冻结；eval；不更新归一化统计 |
| router optimizer | AdamW，lr=`1e-3`，weight decay=`1e-4` |
| batch size | 64，特征可缓存到 CPU |
| 最大 epoch | 100 |
| 选择规则 | R_val 上部署选择后的 macro Dice 最高；并列取较早 epoch |
| margin | 首版固定 `0.002` Dice；扩展只试 `{0,0.002,0.005}` |
| 训练随机种子 | 首先 42；确认阶段每个分割 seed 固定配对 router seed |
| 优化范围 | 仅收益预测器参数 |
| 数据增强 | 首版不使用 |

额外 router seeds 可便宜地重复，结果应分开标注“分割器种子”与“路由器种子”，不能把同一分割 checkpoint 上的多次路由训练当作多个独立分割模型。

### 10.7 选择规则与安全回退

先根据 `R_val` 确定一个最佳固定策略 `a_fixed`。对新图像预测恢复原尺度的收益向量 `u_hat=[0,u_hat_deep,u_hat_full]`，令 `a_best=argmax(u_hat)`。

如果预测的 `u_hat[a_best]-u_hat[a_fixed]` 至少达到 margin，才采用 `a_best`，否则沿用 `a_fixed`。这样不会因为基准设为 A0 就强制大量选择 none，也避免把几乎并列的路径差异过度放大。

平局使用冻结的 tie-break 顺序，优先 `a_fixed`。第一版回退只是保守工程策略，不是逐图安全保证；没有真实标签时仍可能误选。

### 10.8 必须比较哪些路由学习方式

以下对照共享同一分割 checkpoint、描述量、数据划分和候选集：

| ID | 方法 | 意义 |
|---|---|---|
| R0 | 最佳固定策略 | 最基本且不可跳过的基线 |
| R1 | 按 R_fit 上 oracle 策略频率随机选择 | 判断非均匀使用路径本身是否解释收益 |
| R2 | 线性收益回归 | 排除“只是加了 MLP” |
| R3 | 同规模 MLP，预测 oracle 类别，用交叉熵训练 | 与普通监督分类路由比较 |
| R4 | 同规模 MLP，最小化候选期望 regret | 与直接代价敏感路由比较 |
| R5 | 主方案：Huber 收益回归 + 固定回退 | 待检验候选 |
| R6 | 用无注入预测面积／熵作规则或简单路由 | 检查尺度／不确定性启发式；另计信息成本 |

R1 的频率只在 R_fit 上计算并冻结，重复不同随机种子，不能用测试结果选择频率。

R4 可写为 `mean(sum_a softmax(logits)_a * (D_best-D_a))`，候选 Dice 全部 detach。它是一个重要且可能很强的对照：如果与主方案一样好，就不能声称收益回归本身有独特优势。

普通 SCF 仍作为总体模型对照，但其连续门控空间不同，不能替代上述同候选集比较。

### 10.9 不应过早做的设计

不要假设“预测收益越大就把 gamma 线性调得越大”。提高某一路径的全局贡献可能非单调影响结果；离散候选的收益不能直接当作连续 gamma 的梯度。

不要直接同时训练分割器和收益标签生成器。两者一起移动会使监督目标不断变化，且可能通过分割器改变收益尺度。联合微调只有在冻结版本稳定有效后再尝试，并重新生成标签或明确设计交替优化。

也不要把多模型概率平均作为“融合路由”。概率集成是另一种方案，可能有效，但会要求执行多个候选；需要单独的集成对照和成本报告。

### 10.10 路由评估以收益为核心，不以分类准确率为核心

$$
Regret=\frac1N\sum_i\left(\max_aD_{ia}-D_{i,\pi(x_i)}\right).
$$

必报：实际 macro Dice、相对固定策略提升、regret、每条路径选择比例、误选造成退化的比例与幅度。预测收益的 MAE／相关性仅为辅助；类别准确率高不代表分割收益高，因为错误可能集中在代价最大的图像。

若 oracle 空间足够大，可报告收益利用率：

$$
\eta=\frac{D_{route}-D_{fixed}}{D_{oracle}-D_{fixed}}.
$$

分母接近 0 时不报告该比值；负值保留，不能截断为 0 掩盖退化。未见数据上反复不优于最佳固定策略，即使训练集收益相关性很好，也应判定路线未成功。

<a id="s11"></a>
## 11. 公平对照、外部泛化与统计推断

### 11.1 三层比较，不能只选择最容易的一层

**第一层：机制内公平性。** 固定策略与路由使用同一共享模型，训练预算完全相同，唯一差别是路径选择。这最直接检验“选择是否有用”。

**第二层：总体方法有效性。** 与独立 B2/B3 比较。共享三路径训练增加训练开销，不能因为都叫 300 epoch 就认为计算预算一样。报告实际 GPU 时间、优化步、样本曝光次数和前后向次数。

**第三层：同标注预算和额外训练预算对照。** 固定 Dual 用完整原训练池训练；另给固定 Dual 相近的总训练计算预算。若主要收益能被更多数据或训练解释，论文就应相应收缩主张。

如果要将收益归因于“全局上下文”而非“更大辅助网络”，增加一个预算相近的 CNN 辅助编码器或限制感受野的辅助分支对照。参数接近不等于所有因素匹配，应同时报告计算量和表示差异。

### 11.2 多种子与配对统计

确认阶段使用三个分割训练种子 `[42,3407,2026]`，每个种子都包含完整的候选训练、冻结、收益生成和路由训练链路。方法之间共享 split，并按相同 seed 配对。

主表给出每个 seed 的 macro Dice、三 seed 均值与标准差；不能只报最好 seed。逐图差异用配对 bootstrap，优先按患者／病灶组抽样，10,000 次，报告 95% 区间。没有组 ID 时使用图像级抽样，但明确无法据此保证患者级独立。

bootstrap 时，同一个抽中的图像／组必须同时用于两个方法。一个患者多张图的相关性不能靠把所有像素当独立样本消除。不同种子在同一张图上的结果也不能简单堆成 3N 个独立样本。

可报告每个种子的组 bootstrap 区间，再单独给训练种子波动。使用层次 bootstrap 时说明抽样层级；三个种子不足以精确刻画所有优化随机性，不能把很窄的图像区间当成整体高度确定。

### 11.3 多重比较与预注册

主比较冻结为 **主方案 R5 vs 同共享分割器的最佳固定策略 R0**。B2/B3、简单路由、面积分组和边界指标列为次要或机制结果。先报告效果量和区间；正式进行多个显著性检验时采用明确的校正方式，不从大量子组中筛一个 p<0.05 当主贡献。

默认有实际投入价值的提升门槛设为 macro Dice 0.002（0.2 个百分点），同时检查退化样本和严重失败比例。该门槛是本项目的工程筛选值，不是临床有效性标准；不因跨过门槛就声称具有临床意义。

### 11.4 独立外部测试

冻结 checkpoint、预处理、阈值、路由器、margin、候选集和后处理规则后，统一运行全部方法。外部集不用于重新计算输入标准化、门控均值或路由阈值。

外部表现差时可以分析分布变化，但一旦据此外调参数，该数据就成为新的开发数据，后续需要另一份未参与开发的测试证据。只展示同一数据集随机拆分，不能证明跨来源泛化。

### 11.5 阈值与敏感度／特异度

主表使用统一 0.5 阈值。若研究 SCF 是否只是更偏向预测前景，可在 `V_dev` 上注册阈值曲线，并选择固定特异度目标作辅助比较；外部测试只能使用开发集确定的阈值，不能在测试集上逐模型重新寻找相同特异度点。

TTA、阈值搜索、孔洞填充必须作为独立附加实验，给所有方法同等搜索预算；主模型贡献表仍展示无 TTA、无后处理的结果。

<a id="s12"></a>
## 12. P5：只有收益预测有效后，才研究真实动态执行

### 12.1 当前实现不节省辅助分支计算

当前 `EGEDualUNet.forward` 在 CNN 注入之前就执行 `self.t_encoder(x)`。[R03] 因此，gamma 变小、mask 置零或选择 none，都不自动减少 Transformer 的执行成本。

第一版的贡献目标是**改善融合决策**。不能仅根据门控稀疏率、未使用的 feature 数量或理论 MACs 宣称加速。

### 12.2 最小可实现的改造路线

先把 CNN 前两段拆出无辅助注入的 `encode_probe(x)`，得到描述量和可复用的浅层状态，再让收益预测器决策。选择 A0 时不调用 Transformer，继续无注入路径；选择 A1/A2 时才执行 Transformer，并使用已经算好的浅层状态继续。

模型应有 `forward_fixed_policy` 作为数值参考。对所有输入，路由版选择同一策略时应与固定策略版等价；直接跳过前向与“计算后乘零”要做单元测试对照。

A1 依然需要编码器深层特征所依赖的前序计算。真正按深度早退需要另一组候选定义和相应训练，不属于当前 none/deep/full 掩码自然附带的能力。

### 12.3 先测 batch=1，再考虑批处理

batch=1 最容易核验真实跳过。batch>1 时按选择结果分桶，分别执行 none 与需要 Transformer 的子批次，再恢复原始顺序。必须计入分桶、scatter/gather、kernel launch、探测特征和路由器的开销。

避免每张图在 Python 中单独 dispatch 造成吞吐下降。比较单图延迟与固定批次吞吐，不能用一个指标覆盖两者。

### 12.4 效率测量规范

记录 GPU 型号、驱动、CUDA、PyTorch、线程、输入形状、精度、batch size、是否编译和同步方式。分别测：

| 指标 | 规范 |
|---|---|
| 单图延迟 | B=1；eval + inference_mode；预热 50 次，至少 200 次计时；median/p95 |
| 吞吐 | 固定 B=8 或设备可容纳的统一值；包含路由 dispatch |
| 端到端延迟 | 另报包含预处理和必要数据传输的版本 |
| 显存 | reset peak 后测实际路径；报告 allocated/reserved 的定义 |
| 训练成本 | 真正 train 模式，forward/backward/update；不叫推理时间 |
| 参数／计算 | 静态存储参数、实际执行计算分别报告 |

以上循环次数是起点，若抖动大需要增加重复。TTA、后处理和日志统计是否计入计时必须一致；`return_aux` 引入的 CPU 同步不应只惩罚某一个模型。

至少用 profiler 或调用计数确认：选择 A0 的样本确实没有进入 `t_encoder`。报告真实路由分布，不能用全 none 的最好情况代替测试集平均情况。

<a id="s13"></a>
## 13. 实验矩阵、阶段关卡与资源预算

### 13.1 实验执行顺序

| 阶段 | 实验 | 第一轮规模 | 必交结果 | 进入下一阶段条件 |
|---|---|---|---|---|
| P0 | 数据、训练器、评估与干预接口修正 | 单元测试 + 2 epoch smoke run | fix log、测试报告、锁定配置 | 正确性测试通过 |
| P1a | 旧 checkpoint 统一重评估 | 不重训；兼容预处理 | 历史诊断表 | 发现的问题可解释，不作为新主结果 |
| P1b | B0、B2、B3 新协议训练 | seed=42，共 3 次分割训练 | 逐图表、失败分析 | 基线稳定且可复现 |
| P1c | 补 B1 Wave | seed=42，1 次分割训练 | 参数效率参照 | 不阻塞主线 |
| P2a | G0/G0R/G1/G2/G3/G3F/G4 | 已有 checkpoint；G2 重复 20 次 | 门控干预报告 | 明确 SCF 是否依赖样本匹配 |
| P2b | 三策略共享分割器 | seed=42，1 次训练 | 三策略逐图输出和 H | 受训练候选存在值得利用的空间 |
| P3a | R0–R5 | 缓存特征后训练小路由器 | regret、实际提升、过拟合分析 | R5 优于或解释清楚强简单对照 |
| P3b | R6 面积／熵路由 | 信息预算单独记录 | 简单启发式对照 | 排除收益仅由简单规则解释 |
| P4a | 关键实验增加两个分割 seed | B0–B3 + 共享模型按需重复 | 配对多种子表 | 效果稳定而非最好 seed 偶然 |
| P4b | 全训练池、预算匹配及辅助架构对照 | 按结论需要加入 | 公平性表 | 主要收益不能被预算轻易替代 |
| P4c | 未参与开发的外部测试 | 冻结所有方法后一并运行 | 完整外部结果和负结果 | 支持实际泛化主张 |
| P5 | 前置路由、真实跳过编码器 | 仅在收益路线成立后 | profiler 和端到端性能 | 获得实际收益才写效率贡献 |

所有训练次数指独立分割训练运行，不是 epoch，也不包含自动化小路由器训练。完整 4 个基线加 1 个共享模型、3 个 seed，共 15 次分割训练；全数据强基线和预算对照另计。**不要在第一天就启动这 15 次。**

### 13.2 资源估计使用实测，不猜硬件时间

令各模型一次完整训练实测成本分别为 `C_B0,C_B1,C_B2,C_B3,C_M`。确认主链路的成本为：

$$
C_{confirm}=3(C_{B0}+C_{B1}+C_{B2}+C_{B3}+C_M)+C_{router}+C_{evaluation}.
$$

`C_M` 必须实测，不能简单等同于三倍 Dual，也不能当作普通 Dual 一次。先用 smoke run 记录稳定区间的训练步耗时、显存和样本吞吐，再决定 micro-batch；不要用加载、编译或热身第一步推算全部预算。

第一轮省资源优先级：利用旧权重做评估诊断、减少候选策略数、缓存冻结特征、延后第三方大模型、失败关卡及时停。不能通过只给新方法更多调参次数、删掉失败种子或给基线更短训练来省资源。

### 13.3 明确的继续／停止标准

**Gate 0：工程正确性。** 所有 P0 必测项通过，概率／门控重放一致，数据角色无泄漏。未通过不得用训练数字讨论机制。

**Gate 1：选择空间。** 在受训练候选集合上，多种子 H 达到约定投入门槛；如远低于 0.002，通常停止精度导向路由。

**Gate 2：可预测性。** `R_val` 和 `V_dev` 上，路由实际结果优于固定策略，而不只是收益回归 loss 降低；若只有训练集有效，保持冻结候选，先排查数据量与描述量，不增加主干复杂度。

**Gate 3：方法必要性。** 若线性回归或简单 regret 路由同样好，优先采用更简单方法，并收缩创新主张。若路由不如全训练池固定 Dual，不能称其为更实用的最佳方案。

**Gate 4：泛化与代价。** 独立测试支持收益，且额外开销有合理解释，才进入论文完整叙事。没有通过泛化，不通过继续调同一测试集解决。

**Gate 5（可选）：效率。** profiler 证明跳过实际计算，端到端性能改善，才加入按需计算贡献。

<a id="s14"></a>
## 14. 拟新增目录、配置和运行规范

### 14.1 建议目录

以下树是拟新增文件设计，不表示仓库已具备这些模块：

```text
code/EGE-UNet-main/
  research/
    __init__.py
    train_baseline.py           # 显式配置与划分，复用模型构建
    train_multiroute.py         # none/deep/full 共同训练
    train_router.py             # 冻结候选后的收益学习
    evaluate.py                # 统一概率适配、逐图指标
    export_predictions.py      # 概率、描述量、gate bank
    gate_interventions.py      # dataset-level permutation 与重放
    analyze_utility.py         # 逐图收益、候选上界、regret
    make_splits.py             # ID/组/父图去重后的显式分配
    data_manifest.py
    model_output.py
    metrics_overlap.py
    metrics_surface.py
    statistics.py
    router.py
    provenance.py
    configs/
      protocol_v1.yaml
      baseline_ege.yaml
      baseline_dual.yaml
      baseline_scf.yaml
      baseline_wave.yaml
      multiroute.yaml
      router_utility.yaml
  tests/
    test_data_manifest.py
    test_paired_transforms.py
    test_metric_empty_cases.py
    test_surface_metrics.py
    test_gradient_accumulation.py
    test_fusion_interventions.py
    test_checkpoint_compatibility.py
    test_router_no_leakage.py
    test_policy_execution.py

docs/
  protocol_changes.md
  fix_log.md
  gate_intervention_report.md
  research_decision_log.md
  external_data_audit.md
  related_work_matrix.md
```

将 Transformer 子项目正规注册为可导入包，或使用明确的仓库根目录定位。不要为了临时运行又添加新的 `/root` 路径依赖。

### 14.2 protocol_v1.yaml 参考模板

下面是拟实现的配置 schema，训练入口必须解析、验证必填项并输出展开后的完整配置。`null` 项是必须由数据审计产生的字段，不能偷偷回退到默认路径。

```yaml
protocol:
  id: medseg_research_v1
  source_commit: 223af12b2474961ee9cc6d9a39c2589eed73044b
  stage: baseline
  status: planned
  main_metric: macro_dice

data:
  manifest: null                 # 审计后填写；未填直接报错
  split_hash: null
  train_split: S_seg
  checkpoint_validation_split: V_dev
  router_train_split: R_fit
  router_validation_split: R_val
  final_test_split: T_external
  image_size: [256, 256]
  image_normalization: rgb_float_0_1
  image_interpolation: bilinear
  mask_interpolation: nearest
  group_key: group_id
  missing_group_policy: disclose_and_use_image_id
  horizontal_flip_p: 0.5
  vertical_flip_p: 0.5
  rotation_p: 0.5
  rotation_degrees: [0.0, 360.0]
  scale_crop: false
  color_jitter: false

model:
  name: ege_dual
  c_list: [8, 16, 24, 32, 48, 64]
  bridge: true
  deep_supervision: true
  fusion_type: scalar
  gamma_init: 0.1
  t_embed: 48
  t_depths: [2, 2, 2, 2]
  t_head_dim: 16
  t_sr_ratios: [4, 2, 1, 1]
  t_mlp_ratio: 4.0
  t_drop_path_rate: 0.1
  output_contract: probability
  pretrained: null

train:
  seed: 42
  epochs: 300
  micro_batch_size: 32
  gradient_accumulation_steps: 2
  accumulation_weighting: actual_samples_in_window
  optimizer: AdamW
  cnn_lr: 0.001
  transformer_lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 0.00000001
  scheduler: cosine_per_epoch
  eta_min: 0.00001
  grad_clip_norm: 1.0
  precision: fp32
  amp: false
  loss: probability_bce_plus_soft_dice
  dice_smooth: 1.0
  ds_weights_deep_to_shallow: [0.1, 0.2, 0.3, 0.4, 0.5]
  selection: min_val_final_bce_dice
  validate_every_epoch: true
  persistent_workers: false
  resume_granularity: epoch_boundary

policies:
  order: [none, deep, full]
  masks:
    none: [0, 0, 0, 0]
    deep: [0, 0, 1, 1]
    full: [1, 1, 1, 1]
  multiroute_loss_weights: [1.0, 1.0, 1.0]  # 使用时归一化为等权平均

router:
  enabled: false
  target: delta_dice_vs_none
  descriptor: pre_fuse2_channel_mean_std
  input_dim: 32
  hidden_dim: 32
  output_dim: 2
  dropout: 0.1
  feature_scaler_fit_split: R_fit
  utility_scaler_fit_split: R_fit
  utility_std_floor: 0.001
  loss: huber_normalized_utility
  huber_delta: 1.0
  optimizer: AdamW
  lr: 0.001
  weight_decay: 0.0001
  batch_size: 64
  max_epochs: 100
  frozen_segmentation: true
  margin_dice: 0.002
  fixed_policy: null             # 必须由 R_val 选择并写入模型包
  checkpoint_selection: max_R_val_routed_macro_dice

eval:
  threshold: 0.5
  tta: false
  postprocessing: none
  probability_dtype: float32
  boundary_tolerance_px: 2.0
  surface_spacing: [1.0, 1.0]
  surface_unit: resized_grid_pixel
  assd_definition: mean_of_two_area_weighted_directed_means
  surface_implementation_version: null
  report_empty_cases: true
  report_per_image: true
  bootstrap_unit: group
  bootstrap_repeats: 10000
  bootstrap_seed: 20260912
```

模板中的 `source_commit` 仅表示起始快照；实际运行时自动替换为当前执行代码的真实 Git commit，并保存未提交 diff。

配置校验器应拒绝：router 使用 test 训练、probability 输出配 logits loss、未填 manifest、启用 router 却缺 fixed_policy、选择旧 checkpoint 却换成新归一化、声称动态跳过但执行器仍全算等不一致组合。

### 14.2a 四个基线的配置差异

| 基线 | 模型工厂名称 | 必须显式设置 | 优化组 |
|---|---|---|---|
| B0 | `egeunet` | bridge / deep supervision 与主协议一致 | 全部参数使用 CNN 初始学习率 |
| B1 | `ege_wave_unet` | `wave_mode=full`、`fusion_type=scalar` | 首版小型 Wave 适配器也沿用 CNN 初始学习率 |
| B2 | `ege_dual` | `fusion_type=scalar` | CNN/融合与 t_encoder 分组 |
| B3 | `ege_dual` | `fusion_type=scale_cond` | 与 B2 相同差分学习率 |

SCF 开关的现有值为 `scale_cond`，不是 `scf`。[R22] 训练工厂只向各模型传递其支持的参数，不能把 Dual 的 Transformer 配置直接传给所有模型。Wave 应保留其原有标量残差定义，不因同名 `fusion_type` 而复用错误融合逻辑。

### 14.3 建议的命令行契约

**以下命令仅在 research 包及配置落地后才可执行；本文件没有实现这些入口。** 以仓库中的 `code/EGE-UNet-main` 为工作目录，每个训练命令需独立进程，避免历史全局状态污染。

```bash
# 拟实现入口，先开发并测试，不能直接视为仓库现有命令。
python -m research.train_baseline --config research/configs/baseline_dual.yaml --seed 42
python -m research.export_predictions --run-id <RUN_ID> --split V_dev
python -m research.gate_interventions --run-id <SCF_RUN_ID> --split V_dev --permutations 20
python -m research.train_multiroute --config research/configs/multiroute.yaml --seed 42
python -m research.export_predictions --run-id <MULTIROUTE_RUN_ID> --split R_fit
python -m research.export_predictions --run-id <MULTIROUTE_RUN_ID> --split R_val
python -m research.analyze_utility --run-id <MULTIROUTE_RUN_ID> --split V_dev
python -m research.train_router --config research/configs/router_utility.yaml
python -m research.evaluate --bundle <FROZEN_BUNDLE> --split T_external
```

命令应输出和记录实际 config、split hash、seed 和 checkpoint hash，而不是仅打印模型名。最终 `evaluate` 入口默认禁止写回模型参数或选择新阈值。

### 14.4 每次运行的最小留档

```text
runs/<protocol>/<model>/<seed>/<run_id>/
  config_resolved.yaml
  source_commit.txt
  source_diff.patch
  environment.txt
  hardware.json
  manifest_hash.txt
  init_checkpoint_sha256.txt
  best.pth
  last.pth
  checkpoints_metadata.json
  train_log.jsonl
  validation_log.jsonl
  optimizer_steps.csv
  lr_groups.csv
  per_image_metrics.csv
  probabilities/
  gate_bank.npz
  utility_targets.npz
  router_bundle.pt
  statistical_summary.json
  failure_cases.csv
```

分割器不需要的 router 文件可不生成，但不能生成空内容冒充完成。模型 bundle 必须包含特征标准化、收益反标准化参数、候选顺序、固定策略和 margin，不能只保存 MLP 权重。

日志记录哪些数据标签被哪些组件使用；运行环境只记录必要的依赖和硬件信息，不导出凭据、访问 token 或整个私有环境变量。

### 14.5 修复日志模板

```text
Issue ID:
Source file + function + original line range:
Observed behavior:
Trigger condition and affected experiments:
Root cause:
Patch description:
Regression test:
Before / after result:
Checkpoint compatibility:
Need to retrain or only re-evaluate:
Protocol version before / after:
Commit hash:
Status: planned / implemented / tested / accepted
```

边界指标修复通常需要重评估，增强和训练更新修复通常需要重新训练才能得到新协议成绩。两类修复应明确区分，不笼统写“修完所有成绩自动有效”。

<a id="s15"></a>
## 15. 创新边界与失败后的转向

### 15.1 不能直接作为新颖性的内容

相关一手文献已经覆盖输入相关的动态分割路径、尺度适应、难区域分配更多计算，以及局部／全局特征的自适应融合。[R18][R19][R20] 学习何时把决策交给另一个预测者、用代价敏感目标学习选择器，也有成熟的研究背景。[R21]

因此，本方案中的“小 MLP 预测收益”“三个候选做选择”“按样本分配全局信息”本身都不能直接宣称首次提出。当前文献核对只是最低限度的重叠排查，不是完整查新报告。

| 已有研究线 | 与本方案重叠 | 后续必须区分的内容 |
|---|---|---|
| Learning Dynamic Routing for Semantic Segmentation | 图像尺度相关路径、预算约束 | 是否提出了新的、可证实的收益建模问题，而非换任务重做路由 |
| Deep Layer Cascade | 困难区域自适应计算 | 是否把“难”与“当前辅助网络的实际收益”区分并验证 |
| ScaleFusionNet | 皮肤病灶局部／全局、多尺度自适应融合 | 是否超出常规注意力／融合模块改进 |
| Learning to Defer | 根据预测者能力与代价选择执行者 | 收益学习与已有选择／代价敏感学习相比到底改变了什么 |

这里只有冻住候选后每张图的完整候选反馈，不需要为了显得复杂而加入强化学习。直接监督、回归和代价敏感策略应是强基线；是否需要更复杂训练机制由实验决定。

### 15.2 最有说服力的证据链

**现象**：明确量化辅助信息对不同样本的改善和损害，而不只报平均涨分。

**机制**：通过可重放干预，检验普通 SCF 是否利用了与样本匹配的门控；区分 gamma、实际注入强度、尺度和收益。

**方法**：在没有 GT 泄漏、候选路径受训练的前提下，证明收益指导的选择优于合适的固定和简单动态基线。

**泛化**：不同种子、独立数据来源以及同标注预算对照都支持结论。

**工程（可选）**：决策确实发生在昂贵分支之前，并带来实测收益，而不是全算后乘零。

论文贡献数量应由上述证据决定，不先列三个创新点再倒找实验。工程修复是可信实验前提，不是模型创新。

### 15.3 停止后的明确转向

| 观察结果 | 推荐调整 |
|---|---|
| Oracle 空间很小 | 使用固定 Dual，转向 Wave 的参数效率或训练／蒸馏问题 |
| 空间大但廉价特征不可预测 | 验证是否需要更丰富的无 GT 观测；若成本过高则停止路由 |
| SCF 有用但收益路由无额外改善 | 保留 SCF，聚焦其适用边界和跨来源稳定性 |
| 简单线性或 regret 路由同样好 | 采用简单方法，不把 MLP 或 Huber 包装成独特贡献 |
| 新方法只在少给数据的基线上获益 | 优先改善完整数据强基线，不宣称总体领先 |
| 平均不提升但计算显著下降 | 重新注册精度—代价问题，明确非劣界限并独立检验 |
| 外部测试失败 | 收缩为开发集发现；排查域偏移，不能继续调这份测试后仍称独立 |

**最终判断原则：下一步的成功不一定是产生新模块，也可能是用可靠实验否定一个不值得继续投入的假设。**

<a id="appendix-a"></a>
## 附录 A：独立参考函数

这些函数用于约定关键语义，不是完整训练工程或已经合并的补丁。集成时仍需处理模型输出适配、文件读写、版本与数据划分。

**本轮局部校验**：附录 5 段独立参考代码在 CPU 环境（PyTorch 2.10.0+cpu）通过 29 项合成数值／输入边界检查，包括梯度传播、空掩膜、宏平均与 pooled 区别、收益反标准化、回退和组 bootstrap；另检查了 Python 语法、YAML 解析、目录锚点及 22 个引用定义。没有进行真实模型集成、CUDA 测试或数据集训练，不能据此判定仓库 bug 已修复。

### A1. 受控残差注入：绝对 gamma 与二值 mask 分开

输入 `projected` 是调用方已经对齐空间和通道的 Transformer 投影。此函数不执行 Transformer，因此也不决定能否节省编码器计算。检查包含张量同步，适合调试；正式性能测量应统一关闭额外诊断或统一计入开销。

```python
import torch
from torch import Tensor


def controlled_injection(
    cnn: Tensor,
    projected: Tensor,
    gamma: Tensor | float,
    mask: Tensor | float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """返回融合特征、有效 gamma [B,1,1,1]、离线统计用注入比例 [B]。"""
    if cnn.ndim != 4 or cnn.shape != projected.shape:
        raise ValueError("cnn/projected 必须是相同形状的 [B,C,H,W]")
    if min(cnn.shape) <= 0:
        raise ValueError("特征维度不能为空")
    if cnn.device != projected.device or cnn.dtype != projected.dtype:
        raise ValueError("cnn/projected 的设备和 dtype 必须一致")
    if not cnn.is_floating_point():
        raise TypeError("特征必须是浮点张量")
    batch = cnn.shape[0]

    def expand_weight(value: Tensor | float, name: str) -> Tensor:
        weight = torch.as_tensor(value, device=cnn.device, dtype=cnn.dtype)
        if weight.numel() == 1:
            weight = weight.reshape(1, 1, 1, 1).expand(batch, 1, 1, 1)
        elif tuple(weight.shape) == (batch,):
            weight = weight.reshape(batch, 1, 1, 1)
        elif tuple(weight.shape) != (batch, 1, 1, 1):
            raise ValueError(f"{name} 只接受标量、[B] 或 [B,1,1,1]")
        if not torch.isfinite(weight).all():
            raise ValueError(f"{name} 含非有限值")
        if not ((weight >= 0) & (weight <= 1)).all():
            raise ValueError(f"{name} 必须在 [0,1]")
        return weight

    gate = expand_weight(gamma, "gamma")
    active = expand_weight(1.0 if mask is None else mask, "mask")
    if not ((active == 0) | (active == 1)).all():
        raise ValueError("本协议 mask 必须为二值开关")
    effective = gate * active
    delta = effective * projected
    ratio = (
        delta.detach().flatten(1).norm(dim=1)
        / (cnn.detach().flatten(1).norm(dim=1) + 1e-6)
    )
    return cnn + delta, effective, ratio
```

调用者分别传入原生 gate 或 override。该函数不 detach 参与融合的 gate，因此训练时仍能传播梯度；只有返回的描述性比例被 detach。

### A2. 逐图重叠指标及 pooled 口径

不负责表面距离。GT 必须先按数据协议变为严格二值，不在函数内悄悄接受软标签。

```python
import numpy as np


def overlap_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """输入 [N,H,W] 或 [N,1,H,W]；双空 Dice/IoU 约定为 1。"""
    def as_batch(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array)
        if result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0]
        if result.ndim != 3 or min(result.shape) <= 0:
            raise ValueError("输入必须是非空的 [N,H,W] 或 [N,1,H,W]")
        return result

    p, y = as_batch(probabilities), as_batch(targets)
    if p.shape != y.shape:
        raise ValueError("预测与 GT 形状不一致")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold 必须是 [0,1] 内有限数")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("预测必须是 [0,1] 内有限概率，不是 logits")
    if not np.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("GT 必须严格二值")
    pred, gt = p >= threshold, y.astype(bool)
    axis = (1, 2)
    tp = np.sum(pred & gt, axis=axis, dtype=np.int64)
    fp = np.sum(pred & ~gt, axis=axis, dtype=np.int64)
    fn = np.sum(~pred & gt, axis=axis, dtype=np.int64)
    tn = np.sum(~pred & ~gt, axis=axis, dtype=np.int64)
    dice_den, iou_den = 2 * tp + fp + fn, tp + fp + fn
    dice = np.divide(2 * tp, dice_den, out=np.ones(tp.shape, dtype=float),
                     where=dice_den > 0)
    iou = np.divide(tp, iou_den, out=np.ones(tp.shape, dtype=float),
                    where=iou_den > 0)
    pooled_den = int(dice_den.sum())
    return {
        "dice": dice, "iou": iou,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_empty": np.sum(gt, axis=axis) == 0,
        "pred_empty": np.sum(pred, axis=axis) == 0,
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
        "pooled_foreground_dice": (
            float(2 * tp.sum() / pooled_den) if pooled_den else 1.0
        ),
        "thresholded_jaccard": float(np.where(iou < 0.65, 0.0, iou).mean()),
    }
```

### A3. 候选上界与已冻结策略的成绩

`fixed_policy` 必须由开发数据选定后传入；函数不会为你在当前评估集上选择一个部署用最佳策略。`dice_matrix` 的列顺序必须与模型 bundle 一致。

```python
import numpy as np


def summarize_policy_scores(
    dice_matrix: np.ndarray,
    chosen_policy: np.ndarray,
    fixed_policy: int,
) -> dict[str, float]:
    """dice_matrix: [N,A]；chosen_policy: 路由器选出的 [N] 整数列号。"""
    scores = np.asarray(dice_matrix, dtype=np.float64)
    chosen = np.asarray(chosen_policy)
    if scores.ndim != 2 or min(scores.shape) <= 0:
        raise ValueError("dice_matrix 必须是非空 [N,A]")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Dice 必须是 [0,1] 内有限数")
    n, actions = scores.shape
    if not isinstance(fixed_policy, (int, np.integer)):
        raise TypeError("fixed_policy 必须是整数")
    if not 0 <= fixed_policy < actions:
        raise ValueError("fixed_policy 越界")
    if chosen.shape != (n,) or chosen.dtype.kind not in "iu":
        raise ValueError("chosen_policy 必须是形状 [N] 的整数数组")
    if not ((chosen >= 0) & (chosen < actions)).all():
        raise ValueError("chosen_policy 含越界值")
    fixed = scores[:, fixed_policy]
    routed = scores[np.arange(n), chosen]
    oracle = scores.max(axis=1)
    return {
        "fixed_dice": float(fixed.mean()),
        "routed_dice": float(routed.mean()),
        "oracle_dice_diagnostic": float(oracle.mean()),
        "gain_vs_frozen_fixed": float((routed - fixed).mean()),
        "oracle_gap_vs_frozen_fixed": float((oracle - fixed).mean()),
        "regret": float((oracle - routed).mean()),
    }
```

### A4. 恢复收益原尺度后选择策略

此函数只接受已经训练好的路由器的两个标准化输出。`mean/std/fixed_policy/margin` 都来自冻结 bundle，不在测试数据重新拟合。

```python
import numpy as np


def choose_from_normalized_utility(
    prediction: np.ndarray,
    utility_mean: np.ndarray,
    utility_std: np.ndarray,
    fixed_policy: int,
    margin: float = 0.002,
) -> np.ndarray:
    """返回 [N]，策略列顺序固定为 none/deep/full。"""
    z = np.asarray(prediction, dtype=np.float64)
    mean = np.asarray(utility_mean, dtype=np.float64)
    std = np.asarray(utility_std, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != 2:
        raise ValueError("prediction 必须是非空 [N,2]")
    if mean.shape != (2,) or std.shape != (2,):
        raise ValueError("utility_mean/std 必须是 [2]")
    if not all(np.isfinite(v).all() for v in (z, mean, std)):
        raise ValueError("输入含非有限值")
    if not (std > 0).all():
        raise ValueError("std 必须大于 0")
    if not isinstance(fixed_policy, (int, np.integer)) or not 0 <= fixed_policy < 3:
        raise ValueError("fixed_policy 必须是 0/1/2")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin 必须是非负有限数")
    utility = np.concatenate([np.zeros((len(z), 1)), z * std + mean], axis=1)
    if not np.isfinite(utility).all():
        raise ValueError("收益反标准化产生了非有限值")
    best = utility.argmax(axis=1)
    rows = np.arange(len(z))
    advantage = utility[rows, best] - utility[:, fixed_policy]
    # 恰好并列时优先固定策略；其余并列依照固定列顺序。
    switch = (advantage > 0) & (advantage >= margin)
    return np.where(switch, best, fixed_policy).astype(np.int64)
```

### A5. 单一训练种子的配对、组级 bootstrap

本函数接受**同一批图像、同一 seed**两种方法的逐图差值。点估计为图像平均收益，重采样单位为组；一个被抽中的组的所有图像一起进入样本。不要将多个分割 seed 展平后当成独立图像。

```python
import numpy as np


def paired_group_bootstrap(
    delta: np.ndarray,
    group_ids: np.ndarray,
    repeats: int = 10_000,
    seed: int = 20260912,
) -> dict[str, float | int]:
    """对逐图差值计算组 bootstrap 95% percentile 区间。"""
    difference = np.asarray(delta, dtype=np.float64)
    groups = np.asarray(group_ids)
    if difference.ndim != 1 or difference.size == 0 or groups.shape != difference.shape:
        raise ValueError("delta/group_ids 必须是同长、非空的一维数组")
    if not np.isfinite(difference).all():
        raise ValueError("delta 含非有限值")
    if not isinstance(repeats, (int, np.integer)) or repeats < 100:
        raise ValueError("repeats 至少为 100")
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max()) + 1
    if count < 2:
        raise ValueError("至少需要两个独立组；不能从一个组估计该区间")
    totals = np.bincount(inverse, weights=difference, minlength=count)
    sizes = np.bincount(inverse, minlength=count)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.integers(0, count, size=count)
        draws[index] = totals[sampled].sum() / sizes[sampled].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_delta": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_images": int(difference.size),
        "n_groups": count,
        "repeats": int(repeats),
    }
```

调用前必须检查 group ID 缺失；缺失时根据协议使用独立 image_id 并披露限制，不能把所有 `unknown` 当成同一个真实患者。多 seed 不确定性按第 11 节另外分析。

<a id="appendix-b"></a>
## 附录 B：第一轮执行清单与验收交付

### B1. 建议依次完成的七项任务

| 顺序 | 任务 | 交付 | 当前状态 |
|---|---|---|---|
| 1 | 固定源版本，审计 image/mask/组/重复，建立角色清单 | manifest + split_hash | 待执行 |
| 2 | 修复并测试 P0，冻结新协议 | fix_log + pytest 报告 + protocol_v1 | 待执行 |
| 3 | 用旧权重做兼容评估诊断，再新训 B0/B2/B3 | 历史诊断与新协议结果分表 | 待执行 |
| 4 | 建立逐图指标、概率和 gate bank | per_image_metrics + gate_bank | 待执行 |
| 5 | 完成均值替换、数据集级打乱、分层干预 | gate_intervention_report | 待执行 |
| 6 | 有必要时训练三路径模型，计算候选 H | 三路径表 + go/no-go 决策 | 待执行 |
| 7 | H 足够时训练小路由器并比较强简单策略 | router bundle + regret 表 | 待执行 |

### B2. 每个阶段报告至少回答什么

**P0 报告**：哪些问题确认存在、何种条件触发、修了哪个函数、哪些旧成绩需要重训或重算、哪些问题还没覆盖。

**P1/P2 报告**：收益来自哪些样本；打乱门控是否改变结果；候选选择空间多大；上述结论对不同 seed 是否稳定。

**P3 报告**：收益监督是否优于最佳固定路径和代价敏感路由；是否使用了未见于分割训练的数据；是否存在额外标签和计算预算优势。

**P4/P5 报告**：外部测试是否支持泛化；推理是否真正跳过分支；代价、精度与失败样本是否一起报告。

### B3. 决策记录模板

```text
Decision ID / Date:
Current stage and frozen protocol:
Question being tested:
Evidence (run IDs / dataset roles / seed list):
Effect size and uncertainty:
Known confounders:
Decision: continue / simplify / stop / branch
What will not be changed before the next test:
Next required artifact:
```

**本轮明确交付的是这份计划文件。除下方注明的独立参考函数局部测试外，上述仓库修复、模型训练、外部测试、Git 提交都未在本轮执行。**

<a id="references"></a>
## 参考资料与源码依据

**访问日期：2026-09-12。** 仓库文件链接固定到本轮看到的提交，以避免 main 后续改变导致定位漂移。PyTorch / torchvision 文档用于核对 API 与行为，不代表要求把现有项目升级到该文档版本；实际运行环境需单独锁定并测试。

| 编号 | 来源 | 本文用途 |
|---|---|---|
| [R01] | 仓库提交 `223af12b2474961ee9cc6d9a39c2589eed73044b` | 起始版本与可追溯性 |
| [R02] | `05_全部实验数据汇总.md` | 历史成绩与验证集评估口径 |
| [R03] | `models/ege_dual.py` | 双分支、SCF、注入形状、输出与执行顺序 |
| [R04] | `engine.py` | 训练更新、scaler、指标聚合与损失分发 |
| [R05] | `train.py` | checkpoint、验证／测试、门控日志和恢复 |
| [R06] | `utils.py` | 损失、增强、归一化和 scheduler |
| [R07] | `tests/eval_boundary_metrics.py` | 现有边界指标缺陷 |
| [R08] | `datasets/dataset.py` | 图像与 mask 的配对方式 |
| [R09] | `configs/config_setting.py` | 原始训练默认设置 |
| [R10] | `configs/config_ege_dual_difflr.py` | Dual 差分学习率配置 |
| [R11] | `tests/eval_efficiency.py` | 现有效率测量的行为 |
| [R12] | PyTorch 官方 Reproducibility | 随机性、DataLoader 与平台限制 |
| [R13] | PyTorch 官方 Automatic Mixed Precision examples | scaler 与梯度累积规范 |
| [R14] | torchvision 官方 functional.resize | 插值选择和 mask 处理依据 |
| [R15] | ISIC2018 官方 Task 1 页面 | thresholded Jaccard 定义 |
| [R16] | `04_方法详解.md` | 历史数据划分和方法设置 |
| [R17] | Google DeepMind `surface-distance/metrics.py` | 表面元素距离、加权 HD95 与有向平均距离 |
| [R18] | Li et al., Learning Dynamic Routing for Semantic Segmentation, CVPR 2020 | 动态路径与尺度适应的重叠边界 |
| [R19] | Li et al., Not All Pixels Are Equal: Difficulty-aware Semantic Segmentation via Deep Layer Cascade, CVPR 2017 | 难度相关分割计算的先例 |
| [R20] | Qamar et al., ScaleFusionNet, arXiv v3, 2025 | 皮肤病灶自适应多尺度融合的先例 |
| [R21] | Mozannar & Sontag, Consistent Estimators for Learning to Defer to an Expert, ICML 2020 | 选择器与代价敏感学习的先例 |
| [R22] | `configs/config_ege_dual_difflr_scf.py` | SCF 的配置开关 |

本文对相关论文的引用用于明确最低限度的重叠风险，没有照搬其论文成绩作同协议横向排名，也没有据此完成全面的新颖性确认。

[R01]: https://github.com/huamiao123/-/commit/223af12b2474961ee9cc6d9a39c2589eed73044b
[R02]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/05_%E5%85%A8%E9%83%A8%E5%AE%9E%E9%AA%8C%E6%95%B0%E6%8D%AE%E6%B1%87%E6%80%BB.md
[R03]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/models/ege_dual.py
[R04]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/engine.py
[R05]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/train.py
[R06]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/utils.py
[R07]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/tests/eval_boundary_metrics.py
[R08]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/datasets/dataset.py
[R09]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/configs/config_setting.py
[R10]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/configs/config_ege_dual_difflr.py
[R11]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/tests/eval_efficiency.py
[R12]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[R13]: https://docs.pytorch.org/docs/2.14/notes/amp_examples.html
[R14]: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.resize.html
[R15]: https://challenge.isic-archive.com/landing/2018/45/
[R16]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/04_%E6%96%B9%E6%B3%95%E8%AF%A6%E8%A7%A3.md
[R17]: https://github.com/google-deepmind/surface-distance/blob/master/surface_distance/metrics.py
[R18]: https://arxiv.org/abs/2003.10401
[R19]: https://arxiv.org/abs/1704.01344
[R20]: https://arxiv.org/abs/2503.03327v3
[R21]: https://proceedings.mlr.press/v119/mozannar20b.html
[R22]: https://github.com/huamiao123/-/blob/223af12b2474961ee9cc6d9a39c2589eed73044b/code/EGE-UNet-main/configs/config_ege_dual_difflr_scf.py
