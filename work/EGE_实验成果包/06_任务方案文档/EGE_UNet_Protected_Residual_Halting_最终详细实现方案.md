# EGE-UNet Protected Residual Halting 最终实现方案

> 目标：基于当前已经验证的 EGE-HRViT / No-Halting 实验结果，重构动态 Token Halting 的信息流，保留动态计算价值，同时避免原 Halting + Reconstruction 对分割特征连续性造成破坏。  
> 本文档直接面向代码实现，不再设计“先做一个简化测试版再决定”的中间路线。实现时以本方案为最终结构，不额外叠加 Mamba、Boundary Head、Spatial Gate、Cross Transformer、新 Decoder 或新 GAB。

---

## 1. 当前实验事实与设计动机

统一实验配置：

- 数据集：ISIC 2018
- Train / Val：1886 / 808
- 输入：256 × 256
- Seed：42
- Optimizer：AdamW
- Learning Rate：0.001
- Weight Decay：0.01
- Scheduler：CosineAnnealing
- Epoch：300
- Batch：64（如显存不足，只允许通过梯度累积保持等效 batch，不允许修改其它训练语义）

当前关键结果：

| 模型 | DSC | mIoU | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| EGE-UNet Baseline | 0.8817 | 0.7884 | 0.8921 | 0.9577 |
| EGE-HRViT + Halting | 0.8550 | 0.7467 | 0.8969 | 0.9353 |
| EGE-HRViT-NoHalting | 0.8857 | 0.7949 | 0.8716 | 0.9689 |
| EGE-Wave v1 | 0.8863 | 0.7958 | 0.8969 | 0.9591 |

最关键的诊断结论：

1. **12 层 Transformer 本身不是主要问题。**
   - No-Halting 恢复到 0.8857，已经明显高于 EGE-UNet baseline。
2. **原动态 Halting 子系统导致主要退化。**
   - Router / Token selection / different-depth representation / reconstruction / router-related losses 这一整套机制加入后，DSC 从 0.8857 降到 0.8550。
3. **原方法最危险的问题不是“少算 Token”本身，而是输出特征的语义深度不一致。**
   - 一部分 token 停留在 Block3；
   - 一部分 token 继续经过 Block4~12；
   - 最终再把它们恢复到同一完整空间网格。
4. EGE-Wave v1 的成功说明：
   - 原始特征不应被复杂分支直接替换；
   - 更稳定的方式是“原始主体 + 小幅可学习修正”。

因此，本方案的核心原则是：

\[
\boxed{
\text{Dense Base Feature 永久保留，Sparse Deep Transformer 只产生 residual correction}
}
\]

即：

\[
\boxed{
\text{Halting 不再决定“最终采用浅层还是深层特征”，而只决定“这个位置是否需要额外深算修正”}
}
\]

---

# 2. 最终方法名称

建议代码内部命名：

```text
EGE_PRH
ProtectedResidualHaltingAdapter
PRHAdapter
```

论文/汇报中建议名称：

**Protected Residual Halting (PRH)**

中文：

**保护式残差动态 Token 深化模块**

如需要强调不再进行传统 Token Reconstruction，可使用：

**Reconstruction-Free Protected Residual Halting**

---

# 3. 最终整体架构

## 3.1 插入位置

必须保持原 EGE-HRViT 的位置不变：

```text
Stage1
  ↓
Stage2
  ↓
Stage3
  ↓
t3
  ├─────────────────────────────→ protected t3 skip → GAB / Decoder
  │
  ↓
Protected Residual Halting Adapter
  ↓
t3_prh
  ↓
Stage4
  ↓
Stage5
  ↓
Stage6
```

严格要求：

- 原始 Stage3 **不能被替换**；
- Adapter 插在 Stage3 输出与 Stage4 输入之间；
- 原始 `t3` 仍然作为 protected skip 进入原有 GAB / Decoder 路径；
- GAB、Decoder、Deep Supervision 全部保持原样；
- 不允许把 `t3_prh` 替换原始 `t3` 送入 GAB；
- 不允许修改其它 EGE 编码器阶段。

---

# 4. PRH Adapter 总体数据流

假设当前 Stage3 输出为：

\[
t_3 \in \mathbb{R}^{B\times24\times32\times32}
\]

则：

```text
t3 [B,24,32,32]
        ↓
Token Projection / Flatten
        ↓
X0 [B,1024,D]
        ↓
Transformer Block 1
        ↓
Transformer Block 2
        ↓
Transformer Block 3
        ↓
S = Dense Base Feature
[B,1024,D]
        │
        ├────────────────────────────────────────────┐
        │                                            │
        │                                            ↓
        │                                      Edge Router
        │                                            ↓
        │                                     edge probability p
        │                                            ↓
        │                                retention schedule / Top-K
        │                                            ↓
        │                                      active index A
        │                                            ↓
        │                                  Gather active tokens
        │                                            ↓
        │                                  Transformer Block4~12
        │                                            ↓
        │                                    Deep Feature D_A
        │                                            ↓
        │                              Δ_A = D_A - S_A
        │                                            ↓
        │                             C_A = p_A × Δ_A
        │                                            ↓
        │                               Scatter correction only
        │                                            ↓
        └────────────────────── + γ × C ─────────────┘
                              ↓
                         F_protected
                              ↓
                      reshape / projection
                              ↓
                           Stage4
```

---

# 5. 最核心数学定义

## 5.1 全 Token 浅层编码

Stage3 feature 经过 tokenization：

\[
X_0 \in \mathbb{R}^{B\times N\times D}
\]

其中：

\[
N = 32\times32=1024
\]

前三个 Transformer Block 对全部 token 计算：

\[
S=T_3(T_2(T_1(X_0)))
\]

得到：

\[
\boxed{
S\in\mathbb{R}^{B\times N\times D}
}
\]

这个 `S` 是整个 PRH 模块的 **Dense Base Feature / Dense Carrier**。

### 关键约束

`S` 后续必须始终完整存在：

- 不删除；
- 不压缩；
- 不重建；
- 不被 deep token 替换；
- 不对 halted token 做任何覆盖。

---

# 6. Router：与主干表征解耦

## 6.1 Router 输入必须 detach

Router 输入：

\[
S_{router} = \operatorname{stopgrad}(S)
\]

代码：

```python
router_input = base.detach()
edge_logits = self.router(router_input)
edge_prob = torch.sigmoid(edge_logits)
```

目的：

- segmentation feature 由 segmentation loss 主导；
- router 只负责“观察”当前 feature；
- edge supervision 不反向塑造 Block1~3 的 representation；
- 避免 Router Loss 再次污染已经被 No-Halting 验证有效的 Transformer 表征。

### 禁止写法

不要：

```python
edge_logits = self.router(base)
```

否则 `L_edge` 会通过 router 回传到 Block1~3。

---

# 7. Router 网络结构

建议维持轻量，不新增大模块。

推荐：

```python
self.router = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, dim // 4),
    nn.GELU(),
    nn.Linear(dim // 4, 1)
)
```

输入：

```text
[B,N,D]
```

输出：

```text
edge_logits: [B,N,1]
```

随后：

```python
edge_logits = edge_logits.squeeze(-1)      # [B,N]
edge_prob = torch.sigmoid(edge_logits)     # [B,N]
```

不使用 softmax。

原因：

- 每个 token 的 edge score 应是独立概率；
- softmax 会导致 token 间强制竞争，不符合 edge/non-edge 二分类语义。

---

# 8. Edge GT 的严格构造

输入 GT lesion mask：

\[
G\in\{0,1\}^{B\times1\times256\times256}
\]

Stage3 token 网格：

\[
32\times32
\]

因此每个 token 对应原图约：

\[
8\times8
\]

区域。

对每个 8×8 patch：

- 全为 0：非 edge token；
- 全为 1：非 edge token；
- 同时含 0 和 1：edge token。

数学上：

\[
r_i=\frac1{64}\sum_{x\in P_i}G(x)
\]

定义：

\[
y_i^{edge}=
\begin{cases}
1,&0<r_i<1\\
0,&r_i=0\text{ or }1
\end{cases}
\]

推荐代码：

```python
def build_edge_token_gt(mask, token_h=32, token_w=32):
    # mask: [B,1,H,W], binary ground-truth mask
    mask = mask.float()

    ratio = F.adaptive_avg_pool2d(
        mask,
        output_size=(token_h, token_w)
    )

    eps = 1e-6

    edge_gt = (
        (ratio > eps) &
        (ratio < 1.0 - eps)
    ).float()

    edge_gt = edge_gt.flatten(1)

    return edge_gt
```

相比写死 `kernel_size=8, stride=8`，优先使用 `adaptive_avg_pool2d`：

- 对未来输入尺寸变化更稳；
- 与实际 token grid 对齐；
- 不依赖 256/32 必须严格整除的硬编码。

---

# 9. Edge Loss

由于边界 token 数远少于普通 token，使用：

```python
BCEWithLogitsLoss(pos_weight=...)
```

而不是普通 BCE。

## 9.1 pos_weight

建议在训练集上离线统计：

\[
N_+=\text{edge token count}
\]

\[
N_-=\text{non-edge token count}
\]

定义：

\[
w_+=\frac{N_-}{N_+}
\]

然后固定：

```python
self.edge_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight)
)
```

### 禁止

不要每个 mini-batch 重新计算 `pos_weight`。

否则：

- loss scale 随 batch 变化；
- Router 梯度抖动；
- 不同 lesion size 会造成不稳定 supervision。

---

# 10. Final Loss

原 HRViT 中与 reconstruction 相关的 auxiliary loss 必须删除。

最终：

\[
\boxed{
L = L_{EGE} + L_{edge}
}
\]

其中：

- `L_EGE`：完全保留当前 EGE / No-Halting 使用的 segmentation + deep supervision loss；
- `L_edge`：仅训练 Router。

因为 Router 输入已经：

```python
base.detach()
```

所以 `L_edge` 不会反向影响 Block1~3。

### 必须删除

- reconstruction auxiliary head；
- reconstruction auxiliary loss；
- router consistency loss（若现有代码存在且专门服务旧 reconstruction/halting 语义）；
- 任何“重建 token 后再监督”的 loss。

---

# 11. Active Token 选择

最终采用：

\[
\boxed{
\text{Top-K retention}
}
\]

而不是绝对 threshold。

定义：

\[
K=\lfloor \rho N \rfloor
\]

对每张图：

\[
A=\operatorname{TopK}(p,K)
\]

其中 `p` 是 edge probability。

代码：

```python
K = max(1, int(round(N * rho)))

active_idx = torch.topk(
    edge_prob,
    k=K,
    dim=1,
    largest=True,
    sorted=False
).indices
```

---

# 12. Retention Schedule

最终固定以下 schedule，不再使用原来最低 15% 的 aggressive schedule。

\[
\rho(e)=
\begin{cases}
1.0,&e<20\\
1-\frac{0.5(e-20)}{20},&20\le e<40\\
0.5,&e\ge40
\end{cases}
\]

即：

```text
Epoch 0–19:
    100% active tokens

Epoch 20–39:
    100% → 50% 线性下降

Epoch 40–299:
    固定 50%
```

代码：

```python
def get_retention_ratio(epoch):
    if epoch < 20:
        return 1.0
    elif epoch < 40:
        return 1.0 - 0.5 * ((epoch - 20) / 20.0)
    else:
        return 0.5
```

推理：

```python
rho = 0.5
```

---

# 13. Gather Active Token

已知：

```text
base:       [B,N,D]
active_idx: [B,K]
```

构造：

```python
idx_expand = active_idx.unsqueeze(-1).expand(-1, -1, D)

active_base = torch.gather(
    base,
    dim=1,
    index=idx_expand
)
```

得到：

```text
active_base: [B,K,D]
```

同时 gather router confidence：

```python
active_prob = torch.gather(
    edge_prob,
    dim=1,
    index=active_idx
).unsqueeze(-1)
```

得到：

```text
active_prob: [B,K,1]
```

---

# 14. Deep Block 4~12 的计算方式

## 14.1 重要原则

Active tokens 继续深算，但不能失去完整上下文。

推荐语义：

\[
Q = Active
\]

\[
K,V = Dense\ Base
\]

即：

```text
active tokens:
    query

complete base tokens:
    memory / context
```

如果当前旧 HRViT Adapter 已经实现 selected-token cross-attention / deformable cross-attention，并且 memory 是 Block3 full tokens，则直接复用该部分。

不要重新设计新的 cross-attention。

---

# 15. Deep Transformer 的输出语义改变

经过 Block4~12：

\[
D_A = T_{4:12}(S_A; S)
\]

注意：

`D_A` **不是最终 feature**。

它只用于计算：

\[
\boxed{
\Delta_A = D_A - S_A
}
\]

代码：

```python
delta_active = deep - active_base
```

这一步是整个方法最关键的设计。

---

# 16. 为什么使用 Deep Residual Difference

Transformer 本身具有 residual form：

\[
X_{l+1}=X_l+F_l(X_l)
\]

因此：

\[
D_A
=
S_A
+
\Delta_4+
\Delta_5+
\cdots+
\Delta_{12}
\]

所以：

\[
D_A-S_A
\]

就是 Block4~12 累积产生的深层修正。

我们不再使用：

```python
final_active = deep
```

而是：

```python
deep_correction = deep - active_base
```

---

# 17. Router Confidence Gate

最终不使用 binary active mask 直接全量注入 deep correction。

定义：

\[
C_A = p_A\odot\Delta_A
\]

代码：

```python
correction_active = active_prob * delta_active
```

意义：

- `p_i` 高：Router 很确信该位置重要，允许更强 deep correction；
- `p_i` 较低：即使因为 Top-K 进入 active set，也降低 correction 强度；
- 避免 Top-K 的硬边界导致 abrupt feature replacement。

---

# 18. Scatter 的只能是 Correction

创建：

```python
correction = torch.zeros_like(base)
```

然后：

```python
correction = correction.scatter(
    dim=1,
    index=idx_expand,
    src=correction_active
)
```

结果：

```text
active token position:
    correction ≠ 0

halted token position:
    correction = 0
```

## 18.1 禁止使用旧 reconstruction

禁止：

```python
out = base.clone()
out.scatter_(..., deep)
```

禁止：

```python
restore = merge(halted_feature, deep_feature)
```

禁止：

```python
reconstructed_tokens = ...
```

主体 feature `base` 从头到尾都完整存在。

---

# 19. Protected Residual Fusion

最终定义：

\[
\boxed{
F
=
S
+
\gamma C
}
\]

其中：

\[
\gamma=\sigma(a)
\]

初始化：

\[
\gamma_0=0.1
\]

因此：

\[
a_0=
\ln\frac{0.1}{0.9}
\approx -2.1972246
\]

代码：

```python
self.gamma_logit = nn.Parameter(
    torch.tensor(-2.1972246)
)
```

forward：

```python
gamma = torch.sigmoid(self.gamma_logit)

out = base + gamma * correction
```

---

# 20. Active / Halted Token 的最终数学行为

## 20.1 Halted Token

因为：

\[
C_i=0
\]

所以：

\[
\boxed{
F_i=S_i
}
\]

即完全保留 Block3 representation。

## 20.2 Active Token

\[
F_i
=
S_i+
\gamma p_i(D_i-S_i)
\]

展开：

\[
F_i
=
(1-\gamma p_i)S_i
+
\gamma p_iD_i
\]

当：

\[
0\le\gamma\le1,\quad0\le p_i\le1
\]

则：

\[
0\le\gamma p_i\le1
\]

所以它本质上是稳定的 shallow/deep interpolation。

例如：

\[
\gamma=0.1,\quad p_i=0.9
\]

则：

\[
F_i=0.91S_i+0.09D_i
\]

这能够显著降低 deep feature 直接覆盖 shallow feature 的风险。

---

# 21. Detokenize / 输出到 Stage4

PRH 输出：

```text
[B,N,D]
```

恢复空间结构：

```python
out = out.transpose(1, 2).reshape(
    B,
    D,
    H,
    W
)
```

若 token dim `D` 与 EGE Stage4 所需输入 channel 不同，则使用原 Adapter 已有的 output projection。

推荐：

```python
out = self.out_proj(out)
```

必须输出与原 Stage3→Stage4 接口完全一致的 shape：

```text
[B,24,32,32]
```

最终：

```python
t3_prh = self.prh_adapter(t3, epoch=epoch)

t4 = self.stage4(t3_prh)
```

同时原始：

```python
t3_skip = t3
```

继续走 GAB / decoder。

---

# 22. 建议的模块定义

```python
class ProtectedResidualHaltingAdapter(nn.Module):

    def __init__(
        self,
        in_channels=24,
        embed_dim=...,
        num_blocks=12,
        shallow_blocks=3,
        final_keep_ratio=0.5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.shallow_blocks = shallow_blocks
        self.final_keep_ratio = final_keep_ratio

        # Use the same tokenization / embedding
        # as current EGE-HRViT-NoHalting implementation.
        self.token_embed = ...

        # Reuse original 12 Transformer blocks.
        self.blocks = nn.ModuleList([
            ...
            for _ in range(num_blocks)
        ])

        self.router = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

        self.gamma_logit = nn.Parameter(
            torch.tensor(-2.1972246)
        )

        self.out_proj = ...
```

注意：

如果当前 Block4~12 与 Block1~3 的实现不同，比如：

- Block1~3：full self-attention；
- Block4~12：active-query + dense-memory cross-attention；

则不要统一成同一种 Block。

应严格沿用现有 HRViT 的深层计算形式，只改变最终 feature 组织方式。

---

# 23. 推荐完整 Forward 参考实现

```python
def forward(self, t3, epoch=None):

    # ============================================================
    # 1. Tokenization
    # ============================================================
    x, H, W = self.tokenize(t3)
    # x: [B,N,D]

    B, N, D = x.shape

    # ============================================================
    # 2. Full-token shallow Transformer: Blocks 1~3
    # ============================================================
    for i in range(3):
        x = self.blocks[i](x)

    # Permanent dense carrier
    base = x

    # ============================================================
    # 3. Decoupled Router
    # ============================================================
    router_input = base.detach()

    edge_logits = self.router(
        router_input
    ).squeeze(-1)
    # [B,N]

    edge_prob = torch.sigmoid(edge_logits)

    # ============================================================
    # 4. Retention Ratio
    # ============================================================
    if self.training:

        if epoch is None:
            raise ValueError(
                "epoch must be provided during PRH training"
            )

        if epoch < 20:
            rho = 1.0

        elif epoch < 40:
            rho = 1.0 - 0.5 * (
                (epoch - 20) / 20.0
            )

        else:
            rho = 0.5

    else:
        rho = 0.5

    K = max(
        1,
        min(N, int(round(N * rho)))
    )

    # ============================================================
    # 5. Top-K Routing
    # ============================================================
    active_idx = torch.topk(
        edge_prob,
        k=K,
        dim=1,
        largest=True,
        sorted=False
    ).indices
    # [B,K]

    idx_expand = (
        active_idx
        .unsqueeze(-1)
        .expand(-1, -1, D)
    )

    active_base = torch.gather(
        base,
        dim=1,
        index=idx_expand
    )
    # [B,K,D]

    active_prob = torch.gather(
        edge_prob,
        dim=1,
        index=active_idx
    ).unsqueeze(-1)
    # [B,K,1]

    # ============================================================
    # 6. Sparse Deep Computation: Blocks 4~12
    # ============================================================
    active = active_base

    for i in range(3, 12):

        # IMPORTANT:
        # Adapt this call to the existing HRViT implementation.
        # Semantic requirement:
        # query  = active tokens
        # memory = full base tokens

        active = self.deep_forward_block(
            block_id=i,
            active=active,
            memory=base
        )

    deep = active

    # ============================================================
    # 7. Residual Correction
    # ============================================================
    delta_active = deep - active_base

    correction_active = (
        active_prob * delta_active
    )

    # ============================================================
    # 8. Scatter correction ONLY
    # ============================================================
    correction = torch.zeros_like(base)

    correction = correction.scatter(
        dim=1,
        index=idx_expand,
        src=correction_active
    )

    # ============================================================
    # 9. Protected Residual Fusion
    # ============================================================
    gamma = torch.sigmoid(
        self.gamma_logit
    )

    out = (
        base
        +
        gamma * correction
    )

    # ============================================================
    # 10. Restore spatial layout
    # ============================================================
    out = self.detokenize(
        out,
        H,
        W
    )

    out = self.out_proj(out)

    # ============================================================
    # 11. Return auxiliary information for training/logging
    # ============================================================
    return {
        "feature": out,
        "edge_logits": edge_logits,
        "edge_prob": edge_prob,
        "active_idx": active_idx,
        "retention_ratio": rho,
        "gamma": gamma,
    }
```

---

# 24. EGE 主模型如何接入

伪代码：

```python
def forward(self, x, gt=None, epoch=None):

    # Encoder
    t1 = self.stage1(x)
    t2 = self.stage2(t1)
    t3 = self.stage3(t2)

    # IMPORTANT: protected original skip
    t3_skip = t3

    prh_out = self.prh_adapter(
        t3,
        epoch=epoch
    )

    t3_prh = prh_out["feature"]

    t4 = self.stage4(t3_prh)
    t5 = self.stage5(t4)
    t6 = self.stage6(t5)

    # Decoder / GAB: keep original EGE logic unchanged.
    # Wherever original GAB requires Stage3 feature,
    # use t3_skip, NOT t3_prh.

    outputs = self.decode(
        t1=t1,
        t2=t2,
        t3=t3_skip,
        t4=t4,
        t5=t5,
        t6=t6
    )

    outputs["edge_logits"] = prh_out["edge_logits"]
    outputs["edge_prob"] = prh_out["edge_prob"]
    outputs["active_idx"] = prh_out["active_idx"]
    outputs["gamma"] = prh_out["gamma"]
    outputs["retention_ratio"] = prh_out["retention_ratio"]

    return outputs
```

---

# 25. Training Loop 修改

训练时需要把 `epoch` 传给模型：

```python
for epoch in range(num_epochs):

    model.train()

    for images, masks in train_loader:

        outputs = model(
            images,
            epoch=epoch
        )

        seg_loss = compute_ege_loss(
            outputs,
            masks
        )

        edge_gt = build_edge_token_gt(
            masks,
            token_h=32,
            token_w=32
        )

        edge_loss = edge_criterion(
            outputs["edge_logits"],
            edge_gt
        )

        loss = seg_loss + edge_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

如果当前训练脚本存在 AMP、gradient accumulation、gradient clipping 等机制，PRH 应与原 No-Halting 的训练方式保持完全一致。

---

# 26. 必须记录的训练日志

每个 epoch 至少记录：

```text
train_total_loss
train_seg_loss
train_edge_loss

val_dsc
val_miou
val_sensitivity
val_specificity

retention_ratio
gamma

router_edge_positive_mean
router_nonedge_mean
```

建议额外记录：

```python
edge_pos_prob = edge_prob[edge_gt > 0.5].mean()
edge_neg_prob = edge_prob[edge_gt < 0.5].mean()
```

目的不是做额外方法实验，而是确认 Router 是否在执行预期语义。

---

# 27. 必须保证的 Autograd 语义

## 27.1 `base.detach()` 只能用于 Router 输入

正确：

```python
router_input = base.detach()
edge_logits = self.router(router_input)
```

但是：

```python
active_base = gather(base)
```

必须从 **原始 base** gather。

不能：

```python
active_base = gather(base.detach())
```

否则 deep branch 的 segmentation gradient 无法回到 Block1~3。

## 27.2 active_prob 的梯度

当前：

```python
active_prob = sigmoid(edge_logits)
correction_active = active_prob * delta_active
```

因此 segmentation loss 会通过 `active_prob` 回传到 Router。

这是有意设计：

- `L_edge` 负责教 Router “边缘在哪里”；
- segmentation loss 允许 Router 学习“哪些边缘/困难位置的 correction 对最终 segmentation 真正有用”。

但是由于 Router 输入是 `base.detach()`：

- segmentation gradient 可以优化 Router 参数；
- 不会通过 Router 再反传进 Block1~3。

---

# 28. Top-K 的不可导问题

`torch.topk()` 的 index selection 对 index 本身不可导，这是正常的。

Router 仍然通过两条路径学习：

1. `L_edge` 直接训练 `edge_logits`；
2. 对已进入 active set 的 token，`active_prob` 参与 correction gating，因此 segmentation loss 可以更新对应 score。

无需加入：

- Gumbel Softmax；
- Straight-Through Estimator；
- soft top-k；
- reinforcement learning。

这些都会无必要地增加复杂度。

---

# 29. 旧 HRViT 代码中必须删除的组件

请在现有 EGE-HRViT 中搜索并删除/禁用以下语义：

```text
token_reconstruction
token_restore
restore_tokens
reconstruct_tokens

halted_token_merge
merge_halted_active

reconstruction_transformer
fusion_after_reconstruction

reconstruction_head
aux_reconstruction_head

aux_reconstruction_loss
router_aux_loss
```

实际函数名可能不同，以“功能语义”判断，而不是只按名称删除。

最终不能再存在：

\[
\text{halted feature}+\text{deep feature}\rightarrow\text{full reconstructed feature}
\]

这条路径。

---

# 30. 旧 HRViT 中应保留的组件

保留：

- Stage3 adapter 插入点；
- 原 token embedding；
- positional encoding；
- Block1~3 full-token Transformer；
- 原 Router 的基本 edge-aware 思路；
- Block4~12；
- active token deep processing；
- full pre-halting tokens 作为 deep context；
- 原 output projection；
- 原 tensor reshape / permutation；
- 原始 EGE 主干；
- 原始 GAB；
- 原始 decoder；
- deep supervision；
- 原始 `t3` protected skip。

---

# 31. 初始化策略

## 31.1 Gamma

```text
gamma_init = 0.1
gamma_logit_init = log(0.1 / 0.9)
                 ≈ -2.1972246
```

## 31.2 Router

使用标准初始化即可，或沿用当前 HRViT Router 初始化。

不要特殊把 Router 初始化成全 0。

否则早期所有 token score 完全相同，Top-K 选择只由 index tie-breaking 决定。

---

# 32. 训练参数必须与 No-Halting 保持公平

除了 PRH 自身结构与 loss 外，不允许修改：

```text
dataset split
seed
input resolution
optimizer
base learning rate
weight decay
scheduler
epochs
augmentation
segmentation loss
deep supervision weights
checkpoint selection strategy
validation metric computation
test threshold strategy
```

这是为了确保最后结论可以直接归因于 PRH。

---

# 33. 参数量与计算量统计

正式输出必须统计：

```text
Params
FLOPs / MACs（如果当前工具可可靠统计动态分支）
Deep active token ratio
Inference latency
Peak GPU memory
```

动态计算 FLOPs 若工具无法正确识别 Top-K，应同时报告：

### Token-linear workload proxy

No-Halting：

\[
12N
\]

PRH-50：

\[
3N+9(0.5N)=7.5N
\]

相对降低：

\[
1-\frac{7.5}{12}=37.5\%
\]

注意：这只是 token-linear workload proxy，不等价于端到端真实加速。

---

# 34. 最终结果表

至少输出：

| Model | Params | DSC | mIoU | Sens | Spec | Active Deep Tokens | Latency | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGE | - | 0.8817 | 0.7884 | 0.8921 | 0.9577 | - | - | - |
| HRViT-Halting | - | 0.8550 | 0.7467 | 0.8969 | 0.9353 | dynamic | - | - |
| NoHalting-12L | - | 0.8857 | 0.7949 | 0.8716 | 0.9689 | 100% | - | - |
| PRH | ? | ? | ? | ? | ? | 50% | ? | ? |

---

# 35. 强制正确性检查

这些不是额外研究实验，而是实现完成后必须确认的工程正确性。

## 35.1 Shape

确保：

```text
input t3:
[B,24,32,32]

token:
[B,1024,D]

base:
[B,1024,D]

active:
[B,K,D]

correction:
[B,1024,D]

adapter output:
[B,24,32,32]
```

## 35.2 Halted position

对于没有进入 `active_idx` 的位置：

```python
correction[b, i].abs().max() == 0
```

应成立。

## 35.3 Gamma

训练开始时：

```text
gamma ≈ 0.1
```

## 35.4 Router gradient isolation

单独对 `edge_loss.backward()` 检查：

- Router 有 gradient；
- Block1~3 不应因为 `edge_loss` 获得 gradient。

但完整 segmentation loss 下，Block1~3 必须正常有 gradient。

## 35.5 No feature replacement

代码中不应出现：

```python
base.scatter(..., deep)
```

最终必须是：

```python
out = base + gamma * correction
```

---

# 36. 常见实现错误

## 错误 1：把 deep token 直接 scatter 回 base

错误：

```python
out = base.scatter(
    1,
    idx_expand,
    deep
)
```

正确：

```python
delta = deep - active_base
correction_active = active_prob * delta
correction = scatter(correction_active)
out = base + gamma * correction
```

## 错误 2：Router 使用 `base` 而不是 `base.detach()`

错误：

```python
edge_logits = router(base)
```

正确：

```python
edge_logits = router(base.detach())
```

## 错误 3：Deep Branch 没有完整上下文

错误：

```python
active = self_attn(active)
```

如果这意味着 active token 只能看 active token，就改变了当前 HRViT 的上下文语义。

应优先保留原实现中的：

```text
query  = active tokens
memory = complete base tokens
```

## 错误 4：GAB 使用 PRH 输出替代原 t3

错误：

```python
gab3(t4, t3_prh, mask)
```

正确：

```python
gab3(t4, t3_skip, mask)
```

## 错误 5：重新加 Reconstruction Transformer

不要在 PRH 输出后再加：

```python
full_tokens = extra_transformer(full_tokens)
```

否则又会把 sparse correction 转成新的 full-token 重写过程。

---

# 37. 推荐代码目录

如果当前工程结构允许，建议：

```text
models/
├── ege_unet.py
├── modules/
│   ├── prh_adapter.py
│   ├── hrvit_blocks.py
│   └── edge_router.py
│
utils/
├── edge_token_target.py
├── metrics.py
└── profiling.py
```

### `prh_adapter.py`

负责：

- tokenization；
- Block1~12 调度；
- Router；
- Top-K；
- gather；
- deep residual correction；
- scatter correction；
- gamma fusion；
- detokenize。

### `edge_token_target.py`

只负责 GT → edge token label。

---

# 38. 建议配置项

```yaml
model:
  name: EGE_PRH

  prh:
    enabled: true

    insertion_stage: 3

    total_transformer_blocks: 12

    shallow_full_blocks: 3

    final_keep_ratio: 0.5

    warmup_full_token_epochs: 20

    transition_end_epoch: 40

    router_detach_input: true

    router_confidence_gate: true

    gamma_init: 0.1

    use_reconstruction: false

    use_reconstruction_aux_loss: false
```

不要把大量未使用的旧 HRViT 配置继续留在最终 config 中，以免后续误开。

---

# 39. 最终实现完成后的方法定义

首先：

\[
S=T_{1:3}(X)
\]

Router：

\[
p=\sigma(R(\operatorname{sg}(S)))
\]

选取：

\[
A=\operatorname{TopK}(p,\rho N)
\]

深层处理：

\[
D_A=T_{4:12}(S_A;S)
\]

深层增量：

\[
\Delta_A=D_A-S_A
\]

置信度加权：

\[
C_A=p_A\odot\Delta_A
\]

恢复到完整空间：

\[
C=\operatorname{Scatter}(C_A,A)
\]

最终：

\[
\boxed{
F=S+\gamma C
}
\]

也就是：

\[
\boxed{
F
=
S+
\gamma
\operatorname{Scatter}
\left[
p_A
\odot
\left(
T_{4:12}(S_A;S)-S_A
\right)
\right]
}
\]

其中：

\[
\gamma=\sigma(a), \qquad \gamma_{init}=0.1
\]

这就是最终模型最核心、最应保持不变的数学定义。

---

# 40. 这套方法解决的具体问题

原 HRViT 动态路径：

```text
Full tokens
→ Halting
→ shallow tokens + deep tokens
→ Reconstruction
→ Full feature
```

PRH：

```text
Full shallow feature
→ 永远保留

只有困难 token：
→ deep compute
→ 产生 correction
→ scatter correction

最后：
Dense Base + Sparse Correction
```

因此不再存在：

```text
某位置 = Block3 feature
邻近位置 = Block12 feature
两者直接拼接成完整主体 feature
```

而变成：

```text
所有位置都有同一个完整 Block3 Dense Base

困难位置额外叠加受控的 deep correction
```

这是本方法最重要的设计逻辑。

---

# 41. 最终禁止继续叠加的模块

实现该版本时，不允许再同时加入：

- Wavelet；
- Mamba；
- Spatial Gate；
- Boundary Head；
- Cross Transformer Bridge；
- Uncertainty GAB；
- 新 Decoder；
- Channel Attention；
- 多级额外 Auxiliary Head。

原因：

当前目标是明确回答：

\[
\boxed{
\text{是否可以通过 Protected Residual Halting 修复原动态 Halting 的性能退化}
}
\]

只有保持结构干净，最后实验结论才具有解释力。

---

# 42. 给实现 AI 的最终执行要求

请严格遵守以下原则：

1. 先阅读当前 EGE-HRViT 与 No-Halting 源码；
2. 复用已经验证正确的 12 层 Transformer 权重结构与 tensor flow；
3. 不重写 EGE 主干；
4. 不修改 Stage3 插入位置；
5. Block1~3 必须 full-token；
6. Block4~12 只处理 active token；
7. Router 输入必须 `base.detach()`；
8. Router 只使用 edge-aware MLP score；
9. 采用 Top-K；
10. 最终 keep ratio 固定 0.5；
11. Epoch 0~20 full-token；
12. Epoch 20~40 线性降到 0.5；
13. 不再允许降到 0.15；
14. Deep active token 必须继续获得 full base context；
15. Deep output 不能直接作为 final token；
16. 必须计算 `deep - active_base`；
17. correction 必须经过 `active_prob` 加权；
18. Scatter 的只能是 correction；
19. Dense Base 必须永久保留；
20. 最终必须为 `base + gamma * correction`；
21. `gamma_init=0.1`；
22. 删除 token reconstruction；
23. 删除 reconstruction transformer；
24. 删除 reconstruction auxiliary head；
25. 删除 reconstruction auxiliary loss；
26. 原始 t3 继续 protected skip 到 GAB / decoder；
27. 训练超参数与 No-Halting 完全一致；
28. 不额外新增其它增强模块；
29. 完整记录 gamma、router score、retention ratio；
30. 最终输出性能、效率、参数量和内存结果。

---

# 43. 最终一句话定义

> **Protected Residual Halting 将完整的浅层 Transformer 特征始终保留为 dense carrier，动态 Router 只选择困难位置进入深层 Transformer，并把深层结果表示为稀疏残差修正而不是最终 token，再通过置信度和可学习全局幅度进行受控注入，从而避免传统 Halting 中浅层/深层 token 重建造成的空间语义不连续。**

