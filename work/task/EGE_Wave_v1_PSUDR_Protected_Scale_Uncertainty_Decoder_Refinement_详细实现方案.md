# EGE-Wave v1 + Protected Scale-Uncertainty Decoder Refinement
## 最终详细实现方案与技术指导

> 目标：在当前 **EGE-Wave v1** 精度主线基础上，不再继续堆叠 Stage3 Transformer / Halting / Wave 模块，而是针对当前更可能存在的瓶颈——**高分辨率 Decoder 阶段的病灶范围恢复、模糊边界恢复和尺度适配不足**——设计一个轻量、受控、可解释的残差细化模块。  
> 本文档直接面向代码实现，要求尽可能复用现有 EGE-Wave v1 主干，不改原 Encoder、Wave Adapter、GAB、Decoder 主逻辑，只在高分辨率 Decoder/GAB 输出之后增加 **Protected Scale-Uncertainty Decoder Refinement（PSUDR）**。

---

# 1. 当前实验背景与为什么要改 Decoder

统一实验配置：

- 数据集：ISIC 2018
- Train / Val：1886 / 808
- 输入尺寸：256 × 256
- Seed：42
- Optimizer：AdamW
- Learning Rate：1e-3
- Weight Decay：1e-2
- Scheduler：CosineAnnealing
- Epoch：300
- Batch Size：64

当前关键结果：

| 模型 | DSC | mIoU | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| EGE-UNet | 0.8817 | 0.7884 | 0.8921 | 0.9577 |
| BGCT-EGEUNet | 0.8825 | 0.7898 | 0.8662 | 0.9689 |
| EGE-HRViT + Halting | 0.8550 | 0.7467 | 0.8969 | 0.9353 |
| EGE-HRViT-NoHalting | 0.8857 | 0.7949 | 0.8716 | 0.9689 |
| EGE-PRH | 0.8817 | 0.7885 | 0.8666 | 0.9681 |
| **EGE-Wave v1** | **0.8863** | **0.7958** | **0.8969** | **0.9591** |

重要现象：

1. BGCT、NoHalting、PRH 都表现出：
   - Specificity 高；
   - Sensitivity 明显下降。
2. 这说明很多“更强语义/更强筛选”的模块容易让模型变得更保守：
   - 背景更干净；
   - 但病灶范围被压缩；
   - FN 增多。
3. Wave v1 是目前唯一一条同时保持较高 Sensitivity 和较高 Specificity 的路线：
   - 说明 Wave v1 的 Stage3 全局结构建模已经比较健康；
   - 当前更值得解决的是 Decoder 高分辨率恢复能力，而不是继续堆深层语义模块。

因此新设计目标是：

\[
\boxed{
\text{保留 Wave v1 的全局结构优势}
+
\text{加强 Decoder 对模糊边界和病灶范围的局部恢复}
}
\]

同时必须遵循：

\[
\boxed{
\text{Original Decoder Feature + Small Controlled Correction}
}
\]

不能再次用复杂分支直接替换原始特征。

---

# 2. 方法名称

建议正式名称：

**Protected Scale-Uncertainty Decoder Refinement**

缩写：

**PSUDR**

中文：

**保护式尺度-不确定性解码细化模块**

完整模型：

**EGE-Wave-PSUDR**

---

# 3. 设计原则

本模块只做三件事：

1. **利用 Decoder 当前 coarse mask 判断哪里不确定；**
2. **从浅层 Encoder feature 中提取多尺度局部细节；**
3. **只在不确定区域，以小幅残差方式修正原 GAB / Decoder feature。**

不做：

- 新 Boundary Head；
- 新 Transformer；
- 新 Mamba；
- Cross Attention；
- 大型 ASPP；
- 新 Decoder；
- 修改 GAB 内部原始逻辑；
- 替换原始 Wave Adapter；
- 修改原始 t3 protected skip。

---

# 4. 插入位置

只在 **最后两个高分辨率 Decoder 阶段** 插入。

如果原 EGE-UNet 有：

```text
GAB5
GAB4
GAB3
GAB2
GAB1
```

推荐只加在：

```text
GAB2 后
GAB1 后
```

对应接近最终高分辨率恢复的两级。

不要在：

```text
GAB5 / GAB4 / GAB3
```

加入。

原因：

- 深层分辨率低，边界位置不精确；
- 模糊边界修复应放在高分辨率阶段；
- 控制计算量；
- 避免过多 refinement 导致模型过拟合。

---

# 5. 整体数据流

原始 EGE-Wave v1 主干保持：

```text
Input
 ↓
Encoder Stage1
 ↓
Stage2
 ↓
Stage3
   ├──────────────→ original t3 → GAB / Decoder
   │
   ↓
Wave Adapter
   ↓
Stage4
 ↓
Stage5
 ↓
Stage6
 ↓
Decoder
```

新模块只放在 Decoder 后段：

```text
Decoder coarse feature
        ↓
      GAB2
        ↓
   F_gab2_original
        │
        ├───────────────────────────────┐
        │                               │
        │                               ↓
        │                     PSUDR-2 correction
        │                               │
        └──────────── + β2 × correction ┘
                        ↓
                    F_gab2_new
                        ↓
                  next decoder stage
                        ↓
                      GAB1
                        ↓
                  F_gab1_original
                        │
                        ├───────────────────────────────┐
                        │                               │
                        │                               ↓
                        │                     PSUDR-1 correction
                        │                               │
                        └──────────── + β1 × correction ┘
                                        ↓
                                   F_gab1_new
                                        ↓
                                Final Segmentation Head
```

---

# 6. 每个 PSUDR 模块的三个输入

对第 \(i\) 个 refinement stage：

### 输入 1：原始 GAB / Decoder 输出

\[
F_i^{gab}
\]

这是永远保留的主特征。

### 输入 2：对应浅层 Encoder feature

\[
E_i
\]

例如：

- PSUDR-2 使用对应 Stage2 / GAB2 的浅层 feature；
- PSUDR-1 使用对应 Stage1 / GAB1 的浅层 feature。

### 输入 3：当前 Decoder coarse mask logits

\[
M_i
\]

通过：

\[
p_i=\sigma(M_i)
\]

得到当前阶段前景概率。

---

# 7. 不确定性图

定义：

\[
\boxed{
U_i = 4p_i(1-p_i)
}
\]

其中：

\[
p_i\in[0,1]
\]

性质：

- \(p_i\approx0\)：背景非常确定，\(U_i\approx0\)
- \(p_i\approx1\)：病灶非常确定，\(U_i\approx0\)
- \(p_i\approx0.5\)：最不确定，\(U_i=1\)

代码：

```python
prob = torch.sigmoid(mask_logits)
uncertainty = 4.0 * prob * (1.0 - prob)
```

输出 shape：

```text
[B,1,H,W]
```

---

# 8. 是否 detach uncertainty

最终推荐：

```python
prob_for_refine = torch.sigmoid(mask_logits.detach())
uncertainty = 4 * prob_for_refine * (1 - prob_for_refine)
```

也就是说：

\[
U_i = 4\sigma(\operatorname{sg}(M_i))
\left(
1-\sigma(\operatorname{sg}(M_i))
\right)
\]

理由：

- coarse mask 只是告诉 refinement “哪里需要修”；
- 不希望 refinement loss 通过 uncertainty path 反向驱动 coarse head 人为制造不确定区域；
- 避免 mask head 和 refinement branch 形成不稳定耦合；
- segmentation 主损失仍然正常训练 mask logits。

这是推荐默认实现。

---

# 9. 多尺度局部细节分支

目标：

从浅层 Encoder feature \(E_i\) 中提取不同感受野的局部信息。

定义三个分支：

\[
R_i^{(1)}
=
DWConv_{3\times3,d=1}(E_i)
\]

\[
R_i^{(2)}
=
DWConv_{3\times3,d=2}(E_i)
\]

\[
R_i^{(3)}
=
DWConv_{3\times3,d=3}(E_i)
\]

实际每个分支建议：

```text
Depthwise 3×3 Conv
→ BatchNorm
→ GELU
→ Pointwise 1×1 Conv
```

参考实现：

```python
class DetailBranch(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)
```

---

# 10. 为什么使用 dilation = 1 / 2 / 3

三个分支对应不同有效感受野：

```text
d=1：
局部细节、细边界

d=2：
中尺度病灶边界

d=3：
更宽的模糊过渡区域
```

不推荐：

```text
d=4 / 6 / 8
```

因为在高分辨率浅层阶段，过大 dilation 容易出现：

- gridding artifact；
- 局部连续性破坏；
- 不必要的计算与特征稀疏采样。

---

# 11. Branch 输出 channel

每个分支最后输出必须统一到与：

\[
F_i^{gab}
\]

相同 channel：

```text
[B,C_i,H_i,W_i]
```

这样才能直接残差相加。

若浅层 Encoder feature：

```text
E_i: [B,C_e,H,W]
```

而 GAB 输出：

```text
F_gab: [B,C_g,H,W]
```

则 branch 的 pointwise conv 输出：

```text
C_g
```

---

# 12. 尺度自适应权重

三个 detail branch 不应固定平均。

需要根据当前 coarse lesion size 动态产生：

\[
w_1,w_2,w_3
\]

---

# 13. Lesion Scale Descriptor

从 coarse probability：

\[
p_i
\]

计算当前预测病灶占比：

\[
\boxed{
a_i = \frac1{HW}\sum_{x,y}p_i(x,y)
}
\]

代码：

```python
area_ratio = prob_for_refine.mean(dim=(2,3))
```

shape：

```text
[B,1]
```

---

# 14. Scale Router

使用极轻量 MLP：

```python
self.scale_mlp = nn.Sequential(
    nn.Linear(1, 16),
    nn.GELU(),
    nn.Linear(16, 3)
)
```

然后：

```python
scale_logits = self.scale_mlp(area_ratio)
scale_weight = torch.softmax(scale_logits, dim=-1)
```

得到：

```text
[B,3]
```

即：

\[
w_1+w_2+w_3=1
\]

---

# 15. 多尺度特征融合

三个分支：

```python
r1 = branch_d1(encoder_feature)
r2 = branch_d2(encoder_feature)
r3 = branch_d3(encoder_feature)
```

reshape weight：

```python
w1 = scale_weight[:,0].view(B,1,1,1)
w2 = scale_weight[:,1].view(B,1,1,1)
w3 = scale_weight[:,2].view(B,1,1,1)
```

融合：

\[
\boxed{
R_i
=
w_1R_i^{(1)}
+
w_2R_i^{(2)}
+
w_3R_i^{(3)}
}
\]

代码：

```python
detail = w1 * r1 + w2 * r2 + w3 * r3
```

---

# 16. Scale Router 的梯度语义

推荐继续使用：

```python
area_ratio = prob_for_refine.mean(...)
```

其中：

```python
prob_for_refine = sigmoid(mask_logits.detach())
```

即：

- scale router 自己可以被 segmentation loss 更新；
- 但 coarse mask head 不会因为 scale selection 被额外反向影响。

---

# 17. Uncertainty Gating

将多尺度 detail 只作用在不确定区域：

\[
\boxed{
C_i
=
U_i \odot R_i
}
\]

由于：

```text
U_i: [B,1,H,W]
R_i: [B,C,H,W]
```

PyTorch 自动 broadcast。

代码：

```python
correction = uncertainty * detail
```

---

# 18. 为什么不直接使用边界图

不要使用：

```text
GT boundary
Pred boundary head
Sobel edge
Canny-like edge
```

作为 refinement gate。

原因：

- 我们已经做过 BGCT / boundary supervision；
- 边界强化容易让模型变得过保守；
- 当前最重要不是“所有边界都增强”，而是“模型拿不准的地方才增强”。

所以 gate 采用：

\[
\boxed{Uncertainty}
\]

而不是：

\[
\boxed{Hard Boundary}
\]

---

# 19. Protected Residual Fusion

最终输出：

\[
\boxed{
F_i^{new}
=
F_i^{gab}
+
\beta_i C_i
}
\]

其中：

\[
\beta_i=\sigma(b_i)
\]

建议每个 stage 独立一个 scalar。

---

# 20. Beta 初始化

推荐：

\[
\boxed{\beta_{init}=0.05}
\]

而不是 0.1。

原因：

- Decoder 高分辨率局部 feature 对 segmentation output 更敏感；
- BGCT 已经说明过强局部/边界修正有风险；
- 从 0.05 开始更稳。

对应 logit：

\[
b_0=\log\frac{0.05}{0.95}\approx-2.9444
\]

代码：

```python
self.beta_logit = nn.Parameter(
    torch.tensor(-2.944439)
)

beta = torch.sigmoid(self.beta_logit)
```

---

# 21. 是否限制 Beta

最终推荐直接：

\[
\beta=\sigma(b)
\]

范围：

\[
0<\beta<1
\]

不额外 hard clamp。

训练日志必须记录：

```text
beta_gab1
beta_gab2
```

如果最后 beta 极端接近 1，需警惕 correction 过强。

---

# 22. 单个 PSUDR 模块完整公式

给定：

\[
F_i^{gab}, E_i, M_i
\]

先：

\[
p_i=\sigma(\operatorname{sg}(M_i))
\]

不确定性：

\[
U_i=4p_i(1-p_i)
\]

面积比例：

\[
a_i=Mean(p_i)
\]

尺度权重：

\[
\mathbf w_i=Softmax(MLP(a_i))
\]

多尺度细节：

\[
R_i=\sum_{k=1}^{3}w_{ik}D_k(E_i)
\]

局部修正：

\[
C_i=U_i\odot R_i
\]

最终：

\[
\boxed{
F_i^{new}=F_i^{gab}+\sigma(b_i)C_i
}
\]

---

# 23. 推荐完整模块实现

```python
class ProtectedScaleUncertaintyRefinement(nn.Module):

    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        scale_hidden=16,
        beta_init=0.05,
    ):
        super().__init__()

        self.branch_d1 = DetailBranch(
            encoder_channels,
            decoder_channels,
            dilation=1
        )

        self.branch_d2 = DetailBranch(
            encoder_channels,
            decoder_channels,
            dilation=2
        )

        self.branch_d3 = DetailBranch(
            encoder_channels,
            decoder_channels,
            dilation=3
        )

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, scale_hidden),
            nn.GELU(),
            nn.Linear(scale_hidden, 3)
        )

        beta_logit = math.log(
            beta_init / (1.0 - beta_init)
        )

        self.beta_logit = nn.Parameter(
            torch.tensor(beta_logit, dtype=torch.float32)
        )

    def forward(
        self,
        gab_feature,
        encoder_feature,
        mask_logits,
    ):

        # 1. Align spatial size
        if encoder_feature.shape[-2:] != gab_feature.shape[-2:]:
            encoder_feature = F.interpolate(
                encoder_feature,
                size=gab_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if mask_logits.shape[-2:] != gab_feature.shape[-2:]:
            mask_logits = F.interpolate(
                mask_logits,
                size=gab_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # 2. Detach coarse mask for routing only
        prob = torch.sigmoid(mask_logits.detach())

        # 3. Uncertainty
        uncertainty = 4.0 * prob * (1.0 - prob)

        # 4. Multi-scale local details
        r1 = self.branch_d1(encoder_feature)
        r2 = self.branch_d2(encoder_feature)
        r3 = self.branch_d3(encoder_feature)

        # 5. Lesion scale descriptor
        area_ratio = prob.mean(dim=(2,3))

        scale_logits = self.scale_mlp(area_ratio)
        scale_weights = torch.softmax(scale_logits, dim=-1)

        B = gab_feature.shape[0]

        w1 = scale_weights[:,0].view(B,1,1,1)
        w2 = scale_weights[:,1].view(B,1,1,1)
        w3 = scale_weights[:,2].view(B,1,1,1)

        detail = w1 * r1 + w2 * r2 + w3 * r3

        # 6. Uncertainty-localized correction
        correction = uncertainty * detail

        # 7. Protected residual fusion
        beta = torch.sigmoid(self.beta_logit)
        out = gab_feature + beta * correction

        return {
            "feature": out,
            "uncertainty": uncertainty,
            "scale_weights": scale_weights,
            "beta": beta,
            "correction": correction,
        }
```

---

# 24. 在 EGE Decoder 中的具体接入原则

假设原 GAB2：

```python
gt_pre2 = self.gt_conv2(out2)

gab2 = self.gab2(
    t3,
    t2,
    gt_pre2
)

out2 = out2 + gab2
```

不要改变原逻辑。

新增：

```python
ref2 = self.psudr2(
    gab_feature=out2,
    encoder_feature=t2,
    mask_logits=gt_pre2
)

out2 = ref2["feature"]
```

同理 GAB1：

```python
gt_pre1 = self.gt_conv1(out1)

gab1 = self.gab1(
    t2,
    t1,
    gt_pre1
)

out1 = out1 + gab1

ref1 = self.psudr1(
    gab_feature=out1,
    encoder_feature=t1,
    mask_logits=gt_pre1
)

out1 = ref1["feature"]
```

---

# 25. gab_feature 应该使用哪个张量

推荐使用：

```python
out_i_after_gab = out_i + gab_i
```

之后的 feature。

即：

\[
F_i^{gab}=out_i+GAB_i(...)
\]

PSUDR 在它之后。

不要只给：

```python
gab_i
```

因为我们希望保护的是完整 decoder state，而不是单独 GAB correction。

---

# 26. 只加两层，不要五层全加

最终默认：

```text
PSUDR-2：ON
PSUDR-1：ON

PSUDR-3：OFF
PSUDR-4：OFF
PSUDR-5：OFF
```

原因：

- 只处理高分辨率恢复；
- 降低参数；
- 降低过拟合；
- 让方法逻辑更清楚。

---

# 27. 参数配置建议

```yaml
psudr:
  enabled: true

  stages:
    gab2: true
    gab1: true

  dilations:
    - 1
    - 2
    - 3

  scale_hidden_dim: 16

  beta_init: 0.05

  detach_mask_for_routing: true

  uncertainty_type: quadratic
  uncertainty_formula: "4*p*(1-p)"

  use_depthwise_conv: true
  use_batchnorm: true
  activation: gelu

  branch_kernel_size: 3
  branch_pointwise_conv: true
```

---

# 28. Loss 设置

PSUDR 默认：

\[
\boxed{
Loss 完全沿用 EGE-Wave v1
}
\]

不要新增：

- uncertainty loss；
- edge loss；
- scale loss；
- consistency loss；
- auxiliary refinement loss。

原因：

这个模块本身是 segmentation-driven correction。

让最终 segmentation loss 自动决定：

- scale router 怎么选；
- detail branch 学什么；
- beta 应该多大。

---

# 29. Training Setting

严格保持 EGE-Wave v1：

```text
Dataset: ISIC 2018
Train: 1886
Val: 808
Input: 256×256
Seed: 42
Optimizer: AdamW
LR: 0.001
Weight Decay: 0.01
Scheduler: CosineAnnealing
Epoch: 300
Batch: 64
```

不允许因为 PSUDR：

```text
换 optimizer
加 warmup
改 lr
改 augmentation
改 segmentation loss
```

否则无法公平比较。

---

# 30. 模块参数学习率

全部使用主 optimizer 相同 lr：

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

不要给：

```text
beta
scale_mlp
detail branch
```

单独学习率。

---

# 31. 初始化

- Depthwise / Pointwise Conv：PyTorch 默认 Kaiming 初始化即可。
- Scale MLP：默认 Linear 初始化即可。
- Beta：0.05。

不要把 beta 初始化为 0。

如果 beta=0，则 refinement branch 初始对最终输出没有有效通路，detail branch 的有效学习会变慢。

---

# 32. 推荐日志

每个 epoch 记录：

```text
train_loss
val_loss
DSC
mIoU
Sensitivity
Specificity

beta_gab1
beta_gab2

scale_w1_gab1
scale_w2_gab1
scale_w3_gab1

scale_w1_gab2
scale_w2_gab2
scale_w3_gab2

uncertainty_mean_gab1
uncertainty_mean_gab2
```

---

# 33. Scale Weight 统计

每个 epoch 可记录平均：

```python
mean_scale_weight = scale_weights.detach().mean(dim=0)
```

最终可分析模型更偏向：

```text
d=1
d=2
d=3
```

哪种感受野。

---

# 34. Uncertainty 可视化

建议训练完成后保存若干样本：

```text
Input
GT
Prediction
Uncertainty map
Correction magnitude
```

Correction magnitude：

```python
correction_mag = (
    correction
    .detach()
    .abs()
    .mean(dim=1, keepdim=True)
)
```

这能验证：

\[
\boxed{
\text{Correction 是否真的集中在模糊病灶边缘 / 未确定区域}
}
\]

---

# 35. 关键安全保护

## 35.1 原 GAB feature 永远保留

必须：

```python
out = gab_feature + beta * correction
```

禁止：

```python
out = correction
```

也不要用大型 `fuse(gab_feature, correction)` 模块直接重写 feature。

## 35.2 uncertainty 只能作为 gate

推荐：

```python
correction = uncertainty * detail
```

不要：

```text
concat(detail, uncertainty)
→ Conv
```

## 35.3 scale router 只控制三个 branch 权重

不要让 scale router 同时预测：

```text
beta
boundary confidence
channel gate
spatial gate
```

职责必须单一。

---

# 36. 参数量与计算量控制

PSUDR 总新增参数建议控制在：

\[
\boxed{< 10\% \text{ of EGE-Wave v1 total params}}
\]

如果超过 10%，优先减少：

```text
scale_hidden_dim 16 → 8
```

不要删掉三尺度 branch。

三个 branch 必须使用：

```text
Depthwise 3×3
+
Pointwise 1×1
```

禁止改成三个普通 full 3×3 Conv。

---

# 37. 推荐模块命名

代码：

```text
ProtectedScaleUncertaintyRefinement
PSUDR
```

实例：

```python
self.psudr2
self.psudr1
```

正式模型：

```text
EGE-Wave-PSUDR
```

不要叫 Wave v3，因为本次不是继续修改 Wave Adapter。

---

# 38. 完整数学定义

对于 Decoder stage \(i\)：

\[
p_i=\sigma(\operatorname{sg}(M_i))
\]

\[
U_i=4p_i(1-p_i)
\]

\[
a_i=Mean(p_i)
\]

\[
\mathbf w_i=Softmax(MLP(a_i))
\]

\[
R_i=\sum_{k=1}^{3}w_{ik}D_k(E_i)
\]

\[
C_i=U_i\odot R_i
\]

\[
\beta_i=\sigma(b_i)
\]

最终：

\[
\boxed{
F_i^{new}
=
F_i^{gab}
+
\beta_i
U_i
\odot
\left(
\sum_{k=1}^{3}w_{ik}D_k(E_i)
\right)
}
\]

其中：

\[
D_k
=
PWConv(
GELU(
BN(
DWConv_{3\times3,d=k}(E_i)
)
)
)
\]

且：

\[
k\in\{1,2,3\}
\]

---

# 39. 为什么这个设计和 Wave v1 互补

Wave v1 负责：

\[
\boxed{
\text{Stage3 低频全局结构建模}
}
\]

主要解决：

- 病灶整体形状；
- 长距离依赖；
- 大尺度上下文。

PSUDR 负责：

\[
\boxed{
\text{Decoder 高分辨率局部恢复}
}
\]

主要解决：

- 模糊边界；
- 不同病灶尺度；
- 高分辨率细节；
- 病灶收缩 / FN 倾向。

两者职责不同，不重复。

---

# 40. 预期指标方向

当前 Wave v1：

```text
DSC  = 0.8863
mIoU = 0.7958
Sens = 0.8969
Spec = 0.9591
```

PSUDR 理想方向应是：

```text
Sensitivity 稳定或小幅提高
Specificity 基本保持
DSC / mIoU 上升
```

不希望出现：

```text
Sens 明显下降
Spec 明显上升
```

如果出现这种情况，说明模型重新走向“保守前景收缩”。优先检查：

- beta 是否过大；
- uncertainty 是否集中错误；
- dilation branch 是否更偏背景纹理；
- shallow feature 是否接错层；
- coarse mask 是否使用了错误阶段的输出。

---

# 41. 本轮禁止同时修改的内容

不要同时改：

- Wave Adapter；
- Wave alpha；
- DWT 层数；
- Transformer 层数；
- Transformer head；
- PRH；
- Loss；
- GAB 内部；
- Decoder channel；
- augmentation；
- threshold strategy。

本轮只验证：

\[
\boxed{
\text{Decoder Scale-Uncertainty Residual Refinement}
}
\]

---

# 42. 实现前必须阅读当前源码

负责实现的 AI 必须先确认：

1. EGE-Wave v1 当前 `t1~t6` 的真实 shape；
2. GAB1 / GAB2 输入输出 shape；
3. `gt_pre1 / gt_pre2` 是 logits 还是 probability；
4. `out1 / out2` 在加 GAB 前后 shape；
5. 最终 segmentation head 输入来源；
6. 是否有 deep supervision；
7. 原 Decoder 每层上采样位置；
8. GAB1 / GAB2 对应的低层 Encoder feature 到底是 t1/t2 还是其它变量；
9. 当前代码中是否已经对 `gt_pre` 做 sigmoid；
10. checkpoint 的选择标准必须和 Wave v1 一致。

不能根据本文档假定具体 tensor shape 后直接硬编码。

本文档规定的是：

\[
\boxed{
\text{结构语义}
}
\]

具体 channel / H / W 必须以源码为准。

---

# 43. 推荐接入伪代码

```python
# =========================
# Decoder stage 2
# =========================

gt_pre2 = self.gt_conv2(out2)

gab2 = self.gab2(
    high_feat,
    low_feat,
    gt_pre2
)

out2 = out2 + gab2

ref2 = self.psudr2(
    gab_feature=out2,
    encoder_feature=t2,
    mask_logits=gt_pre2
)

out2 = ref2["feature"]


# =========================
# Decoder stage 1
# =========================

gt_pre1 = self.gt_conv1(out1)

gab1 = self.gab1(
    high_feat,
    low_feat,
    gt_pre1
)

out1 = out1 + gab1

ref1 = self.psudr1(
    gab_feature=out1,
    encoder_feature=t1,
    mask_logits=gt_pre1
)

out1 = ref1["feature"]


# =========================
# Final segmentation
# =========================

final_output = self.final_head(out1)
```

实际 `high_feat / low_feat / t1 / t2` 变量按源代码对齐，不可机械照抄。

---

# 44. 工程正确性检查

这些不是额外消融，只是必须确认实现正确。

## Shape

```text
gab_feature.shape == correction.shape == output.shape
```

## Uncertainty 范围

```python
assert uncertainty.min() >= 0
assert uncertainty.max() <= 1
```

## Scale Weight

每个 sample：

```python
scale_weights.sum(dim=-1)
```

必须约等于 1。

## Beta

训练开始：

```text
beta ≈ 0.05
```

## Detach

确认 refinement routing path 不会给 `mask_logits` 额外梯度。

---

# 45. 推荐最终结果表

| Model | Params | DSC | mIoU | Sens | Spec |
|---|---:|---:|---:|---:|---:|
| EGE | - | 0.8817 | 0.7884 | 0.8921 | 0.9577 |
| Wave v1 | - | 0.8863 | 0.7958 | 0.8969 | 0.9591 |
| Wave + PSUDR | ? | ? | ? | ? | ? |

额外报告：

```text
β_GAB1
β_GAB2
mean scale weights
Params increment
FLOPs increment
```

---

# 46. 给实现 AI 的最终执行清单

必须遵守：

1. 基于当前最佳 EGE-Wave v1 代码实现；
2. Wave Adapter 完全不改；
3. Stage3 protected skip 完全不改；
4. GAB 原实现不改；
5. 只在 GAB2 / GAB1 后增加 PSUDR；
6. PSUDR 输入必须包含：原 GAB 后 decoder feature、对应 shallow encoder feature、当前 coarse mask logits；
7. coarse mask routing 必须 detach；
8. uncertainty 使用 `4*p*(1-p)`；
9. 三个 detail branch 使用 dilation 1 / 2 / 3；
10. 必须使用 depthwise + pointwise；
11. branch 输出 channel 对齐 decoder feature；
12. lesion scale descriptor 使用 mean probability；
13. scale MLP 默认 hidden=16；
14. scale weight 使用 softmax；
15. correction 只在 uncertainty 区域激活；
16. 最终必须使用 residual：`gab_feature + beta * correction`；
17. beta_init=0.05；
18. GAB1 / GAB2 各自独立 beta；
19. 不增加新的辅助 loss；
20. 完全沿用 Wave v1 训练配置；
21. 不同时改 loss；
22. 不同时加入 PRH；
23. 不加入额外 Transformer / Mamba；
24. 不加 Boundary Head；
25. 记录 beta、scale weight、uncertainty；
26. 保持 checkpoint 选择标准与 Wave v1 完全一致；
27. 最终和 EGE、Wave v1 做公平对比。

---

# 47. 最终一句话定义

> **PSUDR 在 EGE-Wave v1 的高分辨率 Decoder 阶段，以当前 coarse prediction 的不确定性定位需要修正的空间区域，再根据预测病灶尺度自适应融合不同感受野的浅层局部细节，并通过小幅可学习残差注入原 GAB/Decoder feature，从而在保护原始分割表征的同时增强病灶范围与模糊边界恢复能力。**

---

# 48. 最终核心原则

整套方法必须始终满足：

\[
\boxed{\text{Wave v1 不动}}
\]

\[
\boxed{\text{GAB 不替换}}
\]

\[
\boxed{\text{Decoder 原特征不破坏}}
\]

\[
\boxed{\text{只在不确定区域做多尺度浅层残差修正}}
\]

如果实现过程中出现任何设计需要直接覆盖原 Decoder feature，应立即停止并改回 residual correction 方案。
