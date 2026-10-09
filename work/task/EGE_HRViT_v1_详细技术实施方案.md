# EGE-HRViT v1：基于 HRViT 动态 Token Halting 融入 EGE-UNet 的详细实施方案

> **目标**：在不破坏 EGE-UNet 原有轻量结构的前提下，引入 HRViT 风格的动态 Token 计算机制，让所有中层 Token 先进行浅层 Transformer 计算，只让边界/困难 Token 继续深层 Transformer，最后恢复完整二维 Token Grid，并重新交给原 EGE-UNet 编码器与解码器。
>
> **第一版原则**：只实现 **Edge-Aware 单阶段 3/12 Halting**，暂不加入 Uncertainty、Multi-stage Halting、Fuzzy Boundary、Boundary Loss、新 Decoder 或额外 GAB 改造。

---

## 1. 第一版要回答的核心问题

这一版不是为了堆更多 Transformer，而是要验证：

> **EGE-UNet Stage3 输出的 1024 个中层 Token，是否真的只有一部分需要继续进行深层 Transformer 计算？**

第一版必须建立如下证据链：

```text
A0 原始 EGE-UNet
        ↓
A1 EGE + Full Transformer
        ↓
A2 EGE + Oracle Edge Halting
        ↓
A3 EGE + Learned Edge Halting
        ↓
A4 EGE + Learned Halting + Context Refresh
```

只有这条链成立，才值得继续做：

```text
Edge-Aware
   ↓
Boundary + Uncertainty
   ↓
Difficulty-Aware
   ↓
Multi-stage 3/6/9/12 Halting
```

---

## 2. 原始 EGE-UNet 与改造位置

原始编码器大致为：

```text
Input
  ↓
Stage1 Conv
  ↓
Stage2 Conv
  ↓
Stage3 Conv
  ↓
Stage4 GHPA
  ↓
Stage5 GHPA
  ↓
Stage6 GHPA
  ↓
EGE Decoder + GAB
  ↓
Mask
```

假设输入为：

\[
256\times256
\]

Stage3 输出：

\[
t_3\in\mathbb{R}^{B\times24\times32\times32}
\]

因此：

\[
N=32\times32=1024
\]

第一版只允许在：

```text
Stage3 输出 t3
        ↓
[ Dynamic Transformer Adapter ]
        ↓
Stage4 GHPA
```

之间插入动态模块。

---

## 3. 改造后的整体网络

```text
Input
  ↓
Stage1 Conv
  ↓
Stage2 Conv
  ↓
Stage3 Conv
  ↓
t3 = [B,24,32,32]
  ↓
┌──────────────────────────────────────┐
│ EGE-HRViT Dynamic Adapter            │
│                                      │
│ 1×1 Projection: 24 → 48              │
│         ↓                            │
│ 32×32 → 1024 Tokens                  │
│         ↓                            │
│ Transformer Block ×3                 │
│ 所有 Token 都参与                    │
│         ↓                            │
│ 保存完整 Block3 Token Memory         │
│         ↓                            │
│ Edge Router                          │
│   ├── Non-edge Token → Halt          │
│   └── Edge Token → Continue          │
│                     ↓                │
│          Transformer Block ×9        │
│          仅 Active Token 参与        │
│                     ↓                │
│          Context Refresh             │
│                     ↓                │
│          Token Reconstruction        │
│                     ↓                │
│ 1×1 Projection: 48 → 24              │
│                     ↓                │
│ Residual: t3 + γ·F_transformer        │
└──────────────────────────────────────┘
  ↓
Stage4 GHPA
  ↓
Stage5 GHPA
  ↓
Stage6 GHPA
  ↓
原 EGE Decoder + GAB
  ↓
Final Mask
```

第一版：

- 不删除 GHPA；
- 不修改 GAB；
- 不修改 Decoder；
- 不重新设计 EGE Loss 主体；
- 不替换原 Skip Connection；
- 不在多个尺度同时加入 Transformer。

---

## 4. 为什么选 Stage3：32×32

Stage3：

\[
[B,24,32,32]
\]

对应：

\[
1024\text{ Tokens}
\]

每个 Token 大约对应原图：

\[
8\times8
\]

区域。

选择该位置的原因：

1. 还有足够的边界和纹理信息；
2. 1024 Token 足够多，动态 Halting 才有计算价值；
3. 比 Stage1 / Stage2 更不容易受到毛发和细碎噪声影响；
4. 比 Stage5 / Stage6 更保留边界；
5. 不需要重新对 RGB 图像 Patch Embedding。

不要第一版放在：

\[
8\times8
\]

因为仅有：

\[
64 Tokens
\]

动态 Token Halting 的实际价值有限。

---

## 5. 一个关键工程约束：前三层不要用全局 1024-Token MHSA

如果：

\[
N=1024
\]

前三个 Block 对全部 Token 做标准全局 MHSA，计算会快速增加。

因此第一版推荐：

> **Block1～Block3 使用 8×8 Window Self-Attention。**

32×32 feature map 分成：

\[
4\times4=16
\]

个窗口。

每个窗口：

\[
8\times8=64
\]

Tokens。

全局注意力关系量：

\[
1024^2=1,048,576
\]

窗口注意力关系量：

\[
16\times64^2=65,536
\]

约减少 16 倍。

Router 之后 Active Tokens 大幅减少，因此 Block4～12 可以采用：

> **仅针对 Active Tokens 的 Global MHSA。**

---

## 6. 第一版 Transformer 固定配置

建议初始参数：

```python
token_dim = 48
num_heads = 4
head_dim = 12

total_blocks = 12
halt_after = 3

window_size = 8
mlp_ratio = 2.0

dropout = 0.0
attn_dropout = 0.0
drop_path = 0.0
```

不要直接使用原始 HRViT 的 768 维 embedding。

原因：EGE-UNet 通道本身很小：

```text
8 / 16 / 24 / 32 / 48 / 64
```

如果直接：

\[
24\rightarrow768
\]

会完全破坏轻量化特征。

第一版固定：

\[
24\rightarrow48
\]

后续若必要再消融：

```text
32
48
64
```

---

## 7. Token 化方式

不要重新对 RGB 输入做 Patch Embedding。

EGE Stage1～3 已经完成局部卷积和 8 倍空间压缩。

输入：

\[
t_3:[B,24,32,32]
\]

先投影：

```python
self.proj_in = nn.Conv2d(
    24,
    48,
    kernel_size=1
)
```

得到：

\[
[B,48,32,32]
\]

然后：

```python
x = self.proj_in(t3)
x = x.flatten(2).transpose(1, 2)
```

得到：

\[
[B,1024,48]
\]

可以理解为：

> EGE 前三层 CNN 已经承担了一个学习型 Patch Embedding 的作用。

---

## 8. 二维位置编码

推荐第一版使用固定 2D sine-cosine position embedding。

```python
x = x + pos_embed
```

其中：

```text
x:
[B,1024,48]

pos_embed:
[1,1024,48]
```

若当前工程已有稳定的位置编码实现，可以复用，但所有消融实验必须统一。

---

## 9. Block1～Block3：所有 Token 浅算

核心流程：

```python
x = feature_to_tokens(t3)

for block in shallow_blocks:
    x = block(x, H=32, W=32)

memory = x.clone()
```

其中：

\[
memory\in\mathbb{R}^{B\times1024\times48}
\]

这是所有 Token 在 Block3 结束后的完整浅层表示。

**必须保留，不允许删除。**

用途：

1. Halt Token 最终使用该特征；
2. Active Token 深层处理中，从完整 memory 中查询上下文；
3. Token Reconstruction 需要该完整网格。

---

## 10. Edge Router

Block3 后：

\[
X_3\in\mathbb{R}^{B\times1024\times48}
\]

Router：

```python
class EdgeRouter(nn.Module):

    def __init__(self, dim=48):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        logits = self.mlp(x).squeeze(-1)
        prob = torch.sigmoid(logits)
        return logits, prob
```

输出：

```text
router_logits: [B,1024]
router_prob:   [B,1024]
```

---

## 11. Edge Token GT 生成

原始 GT：

\[
[B,1,256,256]
\]

每个 Token 对应：

\[
8\times8
\]

GT Patch。

规则：

- 全背景 Patch：`Edge = 0`
- 全病灶内部 Patch：`Edge = 0`
- 同时存在背景和病灶：`Edge = 1`

推荐：

```python
def build_edge_token_label(mask):
    """
    mask:
        [B,1,256,256]
        values must be 0/1

    return:
        [B,1024]
    """

    patches = F.unfold(
        mask,
        kernel_size=8,
        stride=8
    )

    # [B,64,1024]

    patch_min = patches.min(dim=1).values
    patch_max = patches.max(dim=1).values

    edge_label = (
        patch_min != patch_max
    ).float()

    return edge_label
```

第一版先严格使用这个规则。

不要立即加入：

```text
Fuzzy Boundary
Boundary Band
Uncertainty
Difficulty Score
```

---

## 12. Router Loss

Edge Token 一般属于少数类。

因此推荐：

```python
BCEWithLogitsLoss(
    pos_weight=N_negative/N_positive
)
```

即：

\[
L_{router}=WeightedBCE(router\_logits,edge\_label)
\]

训练中必须记录：

```text
Router Precision
Router Recall
Router F1
Active Token Ratio
```

其中最重要：

\[
EdgeRecall
\]

因为真正 Edge Token 被错误 Halt 的代价远高于多保留一些普通 Token。

建议目标：

\[
EdgeRecall\ge95\%
\]

之后再追求更低 Retention。

---

## 13. Halting Rule

默认：

```python
keep_mask = router_prob >= threshold
```

初始：

```python
threshold = 0.5
```

加入最低 Token 保留率：

```python
min_keep_ratio = 0.15
```

如果：

\[
N_{keep}<0.15N
\]

则选择 Router Score 最高的 15% Token。

第一版不要设置 `max_keep_ratio`，因为复杂图像可能确实需要更多深层 Token。

---

## 14. 防止错误早停：Edge 邻域扩张

Router 预测的 keep map 可以 reshape 为：

\[
[B,1,32,32]
\]

然后对其进行轻微空间扩张。

目的：

> 防止真实边界落在 Patch 边缘，或 Router 只激活一部分边界 Token。

第一版建议支持配置：

```python
router_dilation = True
```

可使用一次 `3×3 max-pooling` 或 4-neighbor expansion。

不要连续膨胀多次。

必须进行消融：

```text
without dilation
with dilation
```

---

## 15. Active Token Gather

每个 batch 样本保留的 Token 数可能不同。

推荐：

1. 每个样本得到 `active_idx[b]`；
2. 找 batch 内最大 Active 数 `M_max`；
3. Padding 成：

\[
x_{active}\in[B,M_{max},48]
\]

并生成：

```text
active_padding_mask:
[B,M_max]
```

需要实现：

```python
gather_active_tokens(
    x,
    keep_mask
)
```

返回：

```text
x_active
active_idx
padding_mask
```

---

## 16. Block4～Block12：只更新 Active Tokens

Router 之后：

```python
for block in deep_blocks:
    x_active = block(
        x_active,
        padding_mask
    )
```

如果平均：

\[
Retention=25\%
\]

则：

\[
1024\rightarrow256
\]

Attention Matrix：

\[
1024^2\rightarrow256^2
\]

理论关系量下降约 16 倍。

---

## 17. Active Token 不能失去完整上下文

不能让 Active Token 在 Block4～12 只和其他 Active Token 交流。

否则它们会失去：

- 病灶内部；
- 正常背景；
- 长距离空间结构；
- 全局病灶关系。

必须保留：

```python
memory = block3_all_tokens
```

然后让：

\[
Q=X_{active},\quad K,V=X_{memory}
\]

进行 Cross-Attention / Deformable Cross-Attention。

推荐每 3 个深层 Block 做一次 Context Refresh：

```text
Block4
Block5
Block6
   ↓
Context Refresh 1

Block7
Block8
Block9
   ↓
Context Refresh 2

Block10
Block11
Block12
```

---

## 18. Context Refresh 第一阶段实现策略

如果当前 HRViT 源码已经包含 3D Deformable Cross-Attention：

> 优先阅读其实现，再把 3D coordinate / sampling 改成 2D。

如果二维 Deformable Attention 第一阶段过于难调，开发阶段允许先用：

```python
nn.MultiheadAttention
```

实现：

```python
active = cross_attn(
    query=active,
    key=memory,
    value=memory
)
```

正式论文实验再替换为 2D Deformable Cross-Attention。

必须区分：

```text
Standard Cross Attention
Deformable Cross Attention
```

不要混称。

---

## 19. Token Reconstruction

原始完整 memory：

\[
[B,1024,48]
\]

深层 Active Token：

\[
[B,M,48]
\]

重建逻辑：

```python
recon = memory.clone()
recon[active_idx] = active_final
```

含义：

```text
Halt Token:
使用 Block3 feature

Active Token:
使用 Block12 feature
```

最终得到：

\[
[B,1024,48]
\]

再：

```python
recon = recon.transpose(1, 2)
recon = recon.reshape(
    B,
    48,
    32,
    32
)
```

空间位置必须完全恢复。

---

## 20. Projection 回 EGE 通道

```python
self.proj_out = nn.Sequential(
    nn.Conv2d(
        48,
        24,
        kernel_size=1
    ),
    nn.GroupNorm(...),
    nn.GELU()
)
```

得到：

\[
F_T\in[B,24,32,32]
\]

---

## 21. Residual 融合

不要直接：

```python
t3 = transformer_feature
```

推荐：

\[
t_3'=t_3+\gamma F_T
\]

其中：

```python
self.gamma = nn.Parameter(
    torch.tensor(0.1)
)
```

实现：

```python
t3_dynamic = (
    t3
    + self.gamma
    * transformer_feature
)
```

之后：

```python
t4 = encoder4(
    t3_dynamic
)
```

对应的 Decoder Skip / GAB 也建议使用 `t3_dynamic`。

日志中必须输出 `gamma`。

---

## 22. Dynamic Adapter 总体伪代码

```python
class EGEHRViTAdapter(nn.Module):

    def __init__(
        self,
        in_channels=24,
        dim=48,
        heads=4,
        window_size=8,
        total_depth=12,
        halt_after=3
    ):
        super().__init__()

        self.proj_in = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=1
        )

        self.shallow_blocks = nn.ModuleList([
            WindowTransformerBlock(
                dim=dim,
                heads=heads,
                window_size=window_size
            )
            for _ in range(3)
        ])

        self.router = EdgeRouter(dim)

        self.deep_blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads
            )
            for _ in range(9)
        ])

        self.context1 = ContextRefresh2D(dim)
        self.context2 = ContextRefresh2D(dim)

        self.proj_out = nn.Conv2d(
            dim,
            in_channels,
            kernel_size=1
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(
        self,
        t3,
        threshold=0.5,
        min_keep_ratio=0.15
    ):

        B, _, H, W = t3.shape

        x = self.proj_in(t3)
        x = x.flatten(2).transpose(1, 2)

        x = x + build_2d_pos_embed(
            H,
            W,
            x.shape[-1]
        )

        # Block1~3: all tokens
        for blk in self.shallow_blocks:
            x = blk(x, H, W)

        # complete shallow memory
        memory = x

        router_logits, router_prob = self.router(x)

        keep_mask = build_keep_mask(
            router_prob,
            threshold=threshold,
            min_keep_ratio=min_keep_ratio
        )

        keep_mask = spatial_expand(
            keep_mask,
            H,
            W
        )

        x_active, active_idx, pad_mask = (
            gather_active_tokens(
                x,
                keep_mask
            )
        )

        # Block4~6
        for blk in self.deep_blocks[0:3]:
            x_active = blk(
                x_active,
                pad_mask
            )

        x_active = self.context1(
            query=x_active,
            memory=memory,
            padding_mask=pad_mask
        )

        # Block7~9
        for blk in self.deep_blocks[3:6]:
            x_active = blk(
                x_active,
                pad_mask
            )

        x_active = self.context2(
            query=x_active,
            memory=memory,
            padding_mask=pad_mask
        )

        # Block10~12
        for blk in self.deep_blocks[6:9]:
            x_active = blk(
                x_active,
                pad_mask
            )

        recon = reconstruct_tokens(
            memory,
            x_active,
            active_idx,
            pad_mask
        )

        recon = recon.transpose(1, 2)
        recon = recon.reshape(
            B,
            -1,
            H,
            W
        )

        transformer_feature = self.proj_out(
            recon
        )

        output = (
            t3
            + self.gamma
            * transformer_feature
        )

        stats = {
            "router_logits": router_logits,
            "router_prob": router_prob,
            "keep_mask": keep_mask,
            "retention_ratio": (
                keep_mask.float().mean()
            )
        }

        return output, stats
```

---

## 23. 修改 EGE-UNet Forward

原始：

```python
t3 = encoder3(t2)
t4 = encoder4(t3)
```

修改：

```python
t3 = encoder3(t2)

t3_dynamic, halt_stats = (
    self.dynamic_adapter(t3)
)

t4 = encoder4(
    t3_dynamic
)
```

若 Decoder 中：

```python
GAB3(
    t4,
    t3,
    ...
)
```

则改为：

```python
GAB3(
    t4,
    t3_dynamic,
    ...
)
```

其他模块保持原样。

---

## 24. Block3 Auxiliary Segmentation Head

建议增加一个辅助分割头：

```text
Block3 Memory
[B,1024,48]
        ↓
[B,48,32,32]
        ↓
1×1 Conv
        ↓
[B,1,32,32]
```

得到：

\[
M_{aux}^{32}
\]

GT：

```python
gt32 = F.interpolate(
    gt,
    size=(32, 32),
    mode="nearest"
)
```

辅助 Loss：

\[
L_{aux}=BCE+Dice
\]

作用：让 Block3 时的 Token 已经具备初步病灶语义，从而帮助 Router 判断哪些 Token 是简单区域，哪些需要继续深算。

---

## 25. 总 Loss

第一版只使用：

\[
L=L_{EGE}+\lambda_rL_{router}+\lambda_aL_{aux}
\]

建议：

\[
\lambda_r=0.2
\]

\[
\lambda_a=0.2
\]

其中：

- `L_EGE`：完全复用原 EGE Loss；
- `L_router`：Weighted BCE；
- `L_aux`：BCE + Dice。

第一版禁止加入：

```text
Focal Loss
Boundary Loss
Hausdorff Loss
Uncertainty Loss
Budget Loss
Fuzzy Boundary Loss
```

---

## 26. Halting 训练 Schedule

不要从 Epoch 1 就使用 Hard Halting。

### Epoch 0～20

```python
keep_mask = all_true
```

所有 1024 Token 执行完整深层 Transformer。

但是：

```text
Router 正常训练
Aux Head 正常训练
```

### Epoch 20～40

开始使用 Router，但：

```python
min_keep_ratio = 0.40
```

### Epoch 40 以后

正式动态 Halting：

```python
min_keep_ratio = 0.15
```

建议配置：

```yaml
halting:
  warmup_epochs: 20
  transition_end_epoch: 40
  warmup_keep_ratio: 1.0
  transition_keep_ratio: 0.40
  final_min_keep_ratio: 0.15
  threshold: 0.5
```

---

## 27. 必须先做的单元测试

### Test 1：Token 化与尺寸恢复

输入：

```text
[B,24,32,32]
```

输出必须：

```text
[B,24,32,32]
```

### Test 2：Full Token Path

设置：

```python
keep_mask[:] = True
```

确保 1024 Token 全部经过 12 层，前向、反向均正常。

### Test 3：25% Token

随机保留：

```text
256 / 1024
```

检查：

```text
gather
padding
MHSA
context refresh
reconstruction
```

全部正确。

### Test 4：Token Position Reconstruction

人工构造：

```python
token[i] = i
```

经过：

```text
gather → mock deep processing → reconstruction
```

必须保证所有 Token 回到原二维位置。

该测试失败，禁止正式训练。

---

## 28. 正式实验顺序

| 编号 | 模型 | 目的 |
|---|---|---|
| A0 | 原始 EGE-UNet | 基线 |
| A1 | EGE + Full Transformer Adapter | 验证 Transformer 插入位置 |
| A2 | EGE + Oracle Edge Halting | 验证 Halting 理论可行性 |
| A3 | EGE + Learned Edge Halting | 验证 Router |
| A4 | A3 + Context Refresh | 完整 EGE-HRViT v1 |

---

## 29. A1：Full Transformer Adapter

关闭 Halting：

```python
keep_mask = all_true
```

所有 Token 全部执行 Block1～12。

如果：

```text
EGE + Full Transformer
```

在：

```text
Dice
IoU
Boundary F1
HD95
```

完全没有任何收益，则说明 Transformer 插入 Stage3 这个位置本身可能就没有价值，此时不要继续调 Router。

---

## 30. A2：Oracle Edge Halting

只作为理论验证。

直接使用：

```text
GT Edge Token → Continue
GT Non-edge   → Halt
```

记录：

```text
Oracle Retention Ratio
Dice
IoU
Boundary IoU
Boundary F1
HD95
ASSD
Latency
FPS
```

如果 Oracle 都明显掉点：

> Halting 假设在该位置不成立。

---

## 31. A3：Learned Edge Halting

使用 Router 输出：

```python
keep_mask = router_prob >= threshold
```

评估：

```text
Router Precision
Router Recall
Router F1
Retention Ratio
Segmentation
Boundary Quality
Latency
```

重点看：

\[
EdgeRecall
\]

---

## 32. A4：Context Refresh

在 A3 基础上增加：

```text
Active Token
    ↓
Block4~6
    ↓
Query full Block3 Memory
    ↓
Block7~9
    ↓
Query full Block3 Memory
    ↓
Block10~12
```

验证：

> Halt Token 不参与深层更新，但仍可以作为上下文帮助 Active Token。

---

## 33. ISIC2018 实验设置

第一轮固定：

```text
Input Size = 256×256
Stage3 Grid = 32×32
Token Number = 1024
```

不要第一版直接做：

```text
352×352
512×512
```

数据划分：完全沿用当前 EGE-UNet 实验的数据划分。

数据增强：尽量沿用原 EGE-UNet。

保证 A0～A4 公平。

---

## 34. 训练配置

优先复用当前 EGE-UNet 已验证稳定的训练设置。

若需要统一新的优化器配置：

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999)
)
```

建议：

```text
Epoch = 300
AMP = True
Gradient Clip = 1.0
Scheduler = Cosine
```

如果修改学习率：

> A0 必须使用完全相同的新设置重新训练。

---

## 35. Warm Start 与正式实验

开发阶段可以：

```text
加载已有 EGE best checkpoint
随机初始化 Dynamic Adapter
```

用于 Debug / Sanity Check / 结构测试。

但正式 A0～A4 必须保证初始化策略公平。

---

## 36. Router 可视化

验证阶段固定一批样本，画：

```text
Input
GT
GT Edge Token Map
Router Score Map
Predicted Keep Map
Final Prediction
```

重点观察 Router 是否真正集中在：

- 病灶边界；
- 模糊区域；
- 不规则轮廓；

还是随机保留大量背景/内部 Token。

---

## 37. Retention Ratio 统计

每张图都保存：

```text
sample_id
retention_ratio
active_token_count
gt_edge_token_ratio
router_edge_recall
router_edge_precision
```

统计：

```text
Mean
Median
P25
P75
P95
```

并分析：

\[
LesionArea
\]

与：

\[
RetentionRatio
\]

的相关性。

如果所有图 Retention 都接近固定值，说明 Router 没有真正做到 image-adaptive。

---

## 38. 评价指标

### Segmentation

```text
DSC
IoU
Accuracy
Sensitivity
Specificity
Precision
```

### Boundary

```text
Boundary IoU
Boundary F1
HD95
ASSD
```

### Router

```text
Edge Precision
Edge Recall
Edge F1
Average Active Tokens
Retention Ratio
```

### Dynamic Computation

```text
Average Deep Blocks per Token
Token Survival Ratio
```

### Efficiency

```text
Params
GFLOPs
MACs
Peak GPU Memory
Latency ms/image
FPS
```

不能只报告 GFLOPs。

---

## 39. 效率比较对象

效率最重要的比较不是原 EGE 与动态 EGE-HRViT，而是：

\[
\boxed{
EGE+FullTransformer
\quad vs\quad
EGE+DynamicTransformer
}
\]

必须证明 Dynamic Halting 相比同等 Transformer Capacity 的 Full Transformer 确实降低真实计算成本。

同时再说明：相比原 EGE 增加了多少成本，换来了什么边界/精度收益。

---

## 40. 第一版成功标准

### Accuracy

至少：

\[
\Delta DSC\ge-0.2\%
\]

最好：

\[
\Delta DSC>0
\]

### Boundary

希望至少满足一个：

```text
Boundary F1 +0.5 percentage point
Boundary IoU 提升
HD95 明显下降
ASSD 明显下降
```

### Dynamic Computation

希望：

\[
MeanRetention\le40\%
\]

更理想：

\[
20\%-30\%
\]

### Real Speed

相对于 `EGE + Full Transformer`：

```text
Latency 更低
FPS 更高
```

否则只是理论 FLOPs 优化。

---

## 41. 结果判断树

### 情况 1

```text
Full Transformer 有收益
Oracle Halting 保持收益
Learned Halting 也保持
Retention = 20%~30%
```

结论：第一版成立，非常值得继续。下一步做 Difficulty-Aware Router。

### 情况 2

```text
Full Transformer 有收益
Oracle Halting 有收益
Learned Halting 掉点
```

结论：Router 是主要瓶颈。下一步做 Boundary + Uncertainty。

### 情况 3

```text
Full Transformer 有收益
Oracle Edge Halting 也明显掉点
```

结论：Non-edge Token 并不一定简单。这正好支持从 Edge-Aware 升级到 Difficulty-Aware。

### 情况 4

```text
Full Transformer 本身无收益
```

结论：不要继续做 Halting。问题不是 Router，而是 Transformer 插入位置或 Transformer 本身对当前 EGE 表示没有价值。

---

## 42. 第一版明确禁止加入的模块

```text
× Multi-stage 3/6/9/12
× Uncertainty Router
× Difficulty Score
× Fuzzy Boundary
× Boundary Loss
× Hausdorff Loss
× 352 / 512 输入
× 多个 Dynamic Adapter
× 替换 GHPA
× 修改 GAB
× 新 Decoder
× 直接 Token Deletion
× 多尺度 Transformer
```

第一版只回答：

> **Stage3 的 1024 个 Token 是否真的只有少部分需要深层 Transformer？**

---

## 43. 若第一版成功，第二版路线

第二版：

\[
D_i=\alpha E_i+\beta U_i
\]

即：

```text
Edge-Aware
    ↓
Boundary + Uncertainty
    ↓
Difficulty-Aware Halting
```

Uncertainty 推荐来自 Block3 shallow prediction。

例如：

\[
p_i=P(lesion|x_i^{(3)})
\]

Binary Entropy：

\[
U_i=-p_i\log p_i-(1-p_i)\log(1-p_i)
\]

---

## 44. 第三版路线

只有 Difficulty-Aware 单阶段 Halting 有效后，才考虑：

\[
Depth\in\{3,6,9,12\}
\]

Multi-stage：

```text
Block1~3
 ↓ Router1
Easy Halt

Block4~6
 ↓ Router2
Medium Halt

Block7~9
 ↓ Router3
Hard Halt

Block10~12
Very Hard
```

不要提前实现。

---

## 45. 推荐工程目录

```text
models/
├── egeunet.py
├── dynamic_adapter/
│   ├── ege_hrvit_adapter.py
│   ├── edge_router.py
│   ├── window_transformer.py
│   ├── transformer_block.py
│   ├── context_refresh.py
│   ├── token_ops.py
│   └── position_encoding.py
│
losses/
├── original_ege_loss.py
├── router_loss.py
└── auxiliary_loss.py
│
tools/
├── analyze_router.py
├── visualize_router.py
├── profile_dynamic_model.py
└── oracle_halting_eval.py
│
tests/
├── test_tokenization.py
├── test_gather_reconstruct.py
├── test_full_keep.py
└── test_partial_keep.py
```

---

## 46. 建议配置文件

```yaml
model:
  dynamic_adapter:
    enabled: true
    in_channels: 24
    token_dim: 48
    shallow_depth: 3
    deep_depth: 9
    num_heads: 4
    mlp_ratio: 2.0
    window_size: 8

    context_refresh:
      enabled: true
      positions: [6, 9]

    residual:
      gamma_init: 0.1

router:
  type: edge
  threshold: 0.5
  dilation: true

  min_keep_ratio:
    warmup: 1.0
    transition: 0.40
    final: 0.15

  warmup_epochs: 20
  transition_end_epoch: 40

loss:
  original_ege_weight: 1.0
  router_weight: 0.2
  auxiliary_weight: 0.2

training:
  input_size: 256
  epochs: 300
  optimizer: adamw
  lr: 0.0003
  weight_decay: 0.01
  amp: true
  grad_clip: 1.0
```

---

## 47. 最终交给实现 AI 的硬性要求

> 本阶段只允许在 EGE Encoder Stage3 输出 `[B,24,32,32]` 与原 Stage4 之间插入一个 HRViT-style Dynamic Transformer Adapter。必须保持原 Stage1～6、GHPA、GAB、Decoder 以及原始分割 Loss 主体不变。Dynamic Adapter 使用 24→48 投影，将 `[B,24,32,32]` 转为 1024 Tokens。Block1～3 使用 Window Transformer 并处理全部 Tokens；Block3 后保存完整 Token Memory，并通过 Edge Router 进行单阶段 3/12 Halting。Non-edge Token 保存 Block3 特征并停止深层更新，Edge / Active Token 继续经过 9 个深层 Transformer Blocks。Active Token 必须定期通过 Cross-Attention 或 2D Deformable Cross-Attention 从完整 Block3 Memory 中获取上下文。深层结束后按照原始索引重建完整 1024 Token Grid，再通过 48→24 投影，并以 `t3 + gamma * transformer_feature` 的 Residual 方式融合，之后继续进入原 Stage4 GHPA。第一阶段禁止实现 Uncertainty、Multi-stage Halting、Fuzzy Boundary、新 Decoder、新 GAB 或额外 Boundary Loss。必须依次完成 A0 原 EGE、A1 Full Transformer、A2 Oracle Edge Halting、A3 Learned Edge Halting、A4 Context Refresh 完整版本，并同时报告 Segmentation、Boundary、Router 与真实 Latency/FPS 指标。

---

## 48. 最终研究逻辑

这一版的整体思想可以概括为：

\[
\boxed{\text{EGE负责轻量局部特征}}
\]

\[
+
\]

\[
\boxed{\text{Transformer负责中层困难区域关系建模}}
\]

\[
+
\]

\[
\boxed{\text{HRViT Halting负责控制深层Transformer计算预算}}
\]

最终不是为了证明：

> “EGE 加 Transformer 会更好。”

而是验证：

> **EGE 中层特征中，是否只有一小部分空间位置真正需要深层 Transformer 计算，并且能否在保持 Dense Prediction 空间完整性的同时，把更多计算预算集中在病灶边界等困难区域。**

如果这一点成立，再继续扩展 Difficulty-Aware 与 Multi-stage Halting。
