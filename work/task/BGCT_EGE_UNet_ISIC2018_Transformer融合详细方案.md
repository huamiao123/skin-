# BGCT-EGE-UNet：面向 ISIC2018 的 Transformer 融合与边界增强方案

> **目标**：在尽量保留 EGE-UNet 轻量化优势的前提下，引入 Transformer 的跨尺度语义建模能力，重点改善 ISIC2018 皮肤病灶分割中的边界模糊、轮廓外溢、边界断裂和局部缺口问题。  
> **建议模型名**：Boundary-Guided Cross-scale Transformer EGE-UNet，简称 **BGCT-EGE-UNet**。  
> **核心思想**：Transformer 不直接替代全部卷积，而是负责“高层语义如何指导浅层边界”；原始 GHPA 和 GAB 继续承担轻量局部建模与特征融合。

---

## 1. 研究动机

EGE-UNet 的核心优势包括：

- GHPA：低成本多轴特征建模；
- GAB：融合高层特征、低层特征和上一阶段 Mask；
- 深度监督：不同解码阶段分别生成中间 Mask；
- 参数量和计算量较低。

但其边界表现可能受到以下因素影响：

1. 编码器多次池化造成高频边缘信息损失；
2. 双线性插值只能放大特征图，不能恢复已经丢失的真实边缘；
3. BCE 与 Dice 主要关注像素分类和区域重叠，并不直接优化边界距离；
4. GHPA 属于轻量门控机制，不具备完整的动态 Token 两两交互；
5. GAB 虽然融合高低层特征，但没有显式建立“浅层边缘位置”和“高层病灶语义”的查询关系；
6. EGE-UNet 通道数很少，浅层边界表达能力有限。

因此，本方案把 Transformer 作为：

> **高层语义指导浅层边缘判别的跨尺度关系建模器。**

---

## 2. 总体方案

第一版只做两项核心改动：

1. 在 **Stage 3 对应的 GAB** 中加入窗口交叉注意力；
2. 从浅层特征中增加轻量边界分支，并使用边界预测引导 Transformer 输出。

保留以下部分不变：

- 编码器前 3 层普通卷积；
- 编码器后 3 层 GHPA；
- 原始 GAB 局部融合路径；
- 原始解码器；
- 原始多尺度深度监督；
- 原始 BCE + Dice 分割损失。

这样可以保证：

- 改动集中；
- 参数增长可控；
- 消融关系清晰；
- 即使 Transformer 分支训练不稳定，原始 GAB 残差路径仍能维持基本性能。

---

## 3. 为什么选择 Stage 3 对应的 GAB

输入尺寸设为：

\[
256\times256
\]

EGE-UNet 编码器特征大致为：

| 特征 | 形状 | 主要信息 |
|---|---|---|
| `t1` | `[B, 8, 128, 128]` | 细边缘、颜色、纹理、毛发噪声 |
| `t2` | `[B, 16, 64, 64]` | 局部纹理、初步轮廓 |
| `t3` | `[B, 24, 32, 32]` | 较完整的病灶边界和中层结构 |
| `t4` | `[B, 32, 16, 16]` | 中高层病灶语义 |
| `t5` | `[B, 48, 8, 8]` | 高层语义 |
| `t6` | `[B, 64, 8, 8]` | 整体病灶表示 |

选择 `t3 ↔ t4` 的原因：

- `t3` 的 `32×32` 分辨率仍保留较多边界；
- `t4` 已经具有较强病灶语义；
- 在 `32×32` 上使用窗口注意力，计算量可以接受；
- 比 `64×64` 更不容易受到毛发、反光和纹理噪声干扰；
- 比 `16×16` 或 `8×8` 更适合直接改善边界。

---

## 4. 模型结构

### 4.1 原始 GAB3

```text
高层特征 t4：32 × 16 × 16
低层特征 t3：24 × 32 × 32
Mask3：       1 × 32 × 32
             ↓
            GAB3
             ↓
桥接特征：   24 × 32 × 32
```

### 4.2 改进后的 BGCTB

```text
                              ┌────────────────────────┐
t4 ──1×1 Conv──双线性插值────→│ 高层语义特征：K、V     │
                              │                        │
t3 ─────────────────────────→│ 低层边界特征：Q        │
                              │ Window Cross-Attention │
Mask3 ──────────────────────→│ Mask 语义引导          │
                              └───────────┬────────────┘
                                          │
t1,t2,t3 ──Boundary Head──→ Edge32 ───────┤
                                          ↓
                                  边界门控 Transformer
                                          +
                                    原始 GAB3 输出
                                          ↓
                                  BGCTB 最终桥接特征
```

---

## 5. 数学定义

### 5.1 原始 GAB 路径

\[
F_{\mathrm{local}}
=
\operatorname{GAB}(t_4,t_3,M_3)
\]

### 5.2 高层特征对齐

\[
F_H
=
\operatorname{BI}
\left(
\operatorname{Conv}_{1\times1}(t_4)
\right)
\]

其中：

\[
F_H\in\mathbb{R}^{B\times24\times32\times32}
\]

### 5.3 Mask 引导

\[
M=\sigma(M_3)
\]

\[
\widetilde F_H
=
F_H\odot
\left(
1+\beta M
\right)
\]

其中 \(\beta\) 是可学习缩放参数。

### 5.4 跨尺度交叉注意力

低层特征作为 Query：

\[
Q=W_Qt_3
\]

高层特征作为 Key 和 Value：

\[
K=W_K\widetilde F_H
\]

\[
V=W_V\widetilde F_H
\]

窗口交叉注意力：

\[
F_T
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V
\]

### 5.5 边界门控

边界预测：

\[
E=\sigma(E_{32})
\]

门控系数：

\[
G_E=0.5+E
\]

因此：

\[
G_E\in[0.5,1.5]
\]

### 5.6 最终融合

\[
F_{\mathrm{BGCTB}}
=
F_{\mathrm{local}}
+
\gamma G_E\odot F_T
\]

其中：

- \(\gamma\) 初始化为较小值，如 `0.1`；
- 训练初期以原始 GAB 为主；
- Transformer 分支逐步学习有效残差。

---

## 6. 为什么使用 Cross-Attention

普通 Self-Attention 的 \(Q,K,V\) 来自同一特征。

本方案使用：

\[
Q\leftarrow t_3
\]

\[
K,V\leftarrow t_4
\]

对应的含义是：

> 让浅层每个疑似边缘位置主动查询高层语义：这里检测到的线条究竟是病灶边界，还是毛发、阴影、反光或普通皮肤纹理？

因此 Cross-Attention 直接解决：

\[
\text{浅层位置精确但语义弱}
\]

与：

\[
\text{高层语义强但位置粗糙}
\]

之间的矛盾。

---

## 7. 窗口注意力设置

`32×32` 特征共有：

\[
N=1024
\]

个 Token。全局注意力关系数量为：

\[
1024^2=1,048,576
\]

使用 `8×8` 窗口：

- 窗口数量：16；
- 每个窗口 Token：64；
- 关系数量：

\[
16\times64^2=65,536
\]

相比全局注意力约减少 16 倍。

第一版推荐配置：

```python
dim = 24
num_heads = 4
head_dim = 6
window_size = 8
mlp_ratio = 2.0
attention_dropout = 0.0
projection_dropout = 0.0
```

第一版不实现 Shifted Window。为了补充跨窗口交流，在窗口注意力前后加入 `3×3` 深度卷积。

---

## 8. 推荐工程目录

```text
project/
├── models/
│   ├── egeunet.py
│   ├── ghpa.py
│   ├── gab.py
│   ├── boundary_head.py
│   ├── window_cross_attention.py
│   └── bgct_bridge.py
├── losses/
│   ├── dice_loss.py
│   ├── boundary_loss.py
│   └── total_loss.py
├── metrics/
│   ├── segmentation_metrics.py
│   ├── boundary_metrics.py
│   └── surface_metrics.py
├── datasets/
│   └── isic2018_dataset.py
├── configs/
│   └── bgct_ege_isic2018.yaml
├── train.py
├── validate.py
├── test.py
└── tests/
    ├── test_window_partition.py
    ├── test_attention_shape.py
    ├── test_boundary_gt.py
    └── test_full_forward.py
```

---

## 9. 窗口划分与恢复

```python
import torch


def window_partition(x, window_size):
    """
    Args:
        x: [B, C, H, W]
        window_size: int

    Returns:
        [B * num_windows, window_size * window_size, C]
    """
    B, C, H, W = x.shape

    assert H % window_size == 0
    assert W % window_size == 0

    x = x.permute(0, 2, 3, 1).contiguous()

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    windows = x.view(
        -1,
        window_size * window_size,
        C
    )

    return windows


def window_reverse(
    windows,
    window_size,
    B,
    H,
    W,
    C
):
    """
    Args:
        windows:
            [B * num_windows, window_size * window_size, C]

    Returns:
        x:
            [B, C, H, W]
    """
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)

    return x.permute(0, 3, 1, 2).contiguous()
```

### 9.1 单元测试

```python
x = torch.randn(2, 24, 32, 32)

windows = window_partition(
    x,
    window_size=8
)

x_rebuild = window_reverse(
    windows,
    window_size=8,
    B=2,
    H=32,
    W=32,
    C=24
)

assert torch.allclose(
    x,
    x_rebuild
)
```

如果该测试失败，说明窗口空间顺序被打乱，必须先修复再训练。

---

## 10. Window Cross-Attention

```python
import torch
import torch.nn as nn


class WindowCrossAttention(nn.Module):
    def __init__(
        self,
        dim=24,
        num_heads=4,
        window_size=8,
        mlp_ratio=2.0,
        dropout=0.0
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.window_size = window_size

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        hidden_dim = int(dim * mlp_ratio)

        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.q_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.kv_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

        self.out_pos = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )

    def forward(
        self,
        q_map,
        kv_map
    ):
        B, C, H, W = q_map.shape

        assert kv_map.shape == q_map.shape

        q_map = q_map + self.q_pos(q_map)
        kv_map = kv_map + self.kv_pos(kv_map)

        q_tokens = window_partition(
            q_map,
            self.window_size
        )

        kv_tokens = window_partition(
            kv_map,
            self.window_size
        )

        q_norm = self.norm_q(q_tokens)
        kv_norm = self.norm_kv(kv_tokens)

        attn_out, _ = self.attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=False
        )

        x = q_tokens + attn_out

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        x = window_reverse(
            x,
            self.window_size,
            B,
            H,
            W,
            C
        )

        x = x + self.out_pos(x)

        return x
```

---

## 11. Boundary Head

边界分支使用 `t1、t2、t3`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj1 = nn.Conv2d(8, 8, 1)
        self.proj2 = nn.Conv2d(16, 8, 1)
        self.proj3 = nn.Conv2d(24, 8, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                24,
                24,
                kernel_size=3,
                padding=1,
                groups=24
            ),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv2d(
                24,
                1,
                kernel_size=1
            )
        )

    def forward(
        self,
        t1,
        t2,
        t3
    ):
        target_size = t3.shape[-2:]

        f1 = self.proj1(t1)
        f1 = F.interpolate(
            f1,
            size=target_size,
            mode="bilinear",
            align_corners=True
        )

        f2 = self.proj2(t2)
        f2 = F.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=True
        )

        f3 = self.proj3(t3)

        x = torch.cat(
            [f1, f2, f3],
            dim=1
        )

        edge_logits = self.fuse(x)

        return edge_logits
```

---

## 12. BGCT Bridge

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryGuidedCrossTransformerBridge(nn.Module):
    def __init__(
        self,
        dim_high=32,
        dim_low=24,
        window_size=8,
        num_heads=4,
        gab_cls=None
    ):
        super().__init__()

        if gab_cls is None:
            raise ValueError(
                "gab_cls must be provided."
            )

        self.local_gab = gab_cls(
            dim_high,
            dim_low
        )

        self.high_project = nn.Conv2d(
            dim_high,
            dim_low,
            kernel_size=1
        )

        self.transformer = WindowCrossAttention(
            dim=dim_low,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=2.0,
            dropout=0.0
        )

        self.gamma = nn.Parameter(
            torch.tensor(0.1)
        )

        self.mask_scale = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(
        self,
        x_high,
        x_low,
        mask_logits,
        edge_logits
    ):
        local_feature = self.local_gab(
            x_high,
            x_low,
            mask_logits
        )

        high_feature = self.high_project(
            x_high
        )

        high_feature = F.interpolate(
            high_feature,
            size=x_low.shape[-2:],
            mode="bilinear",
            align_corners=True
        )

        mask_prob = torch.sigmoid(
            mask_logits
        )

        beta = torch.tanh(
            self.mask_scale
        )

        high_feature = high_feature * (
            1.0 + beta * mask_prob
        )

        transformer_feature = self.transformer(
            q_map=x_low,
            kv_map=high_feature
        )

        edge_prob = torch.sigmoid(
            edge_logits
        )

        edge_gate = 0.5 + edge_prob

        output = (
            local_feature
            + self.gamma
            * edge_gate
            * transformer_feature
        )

        return output
```

---

## 13. 修改 EGE-UNet 初始化函数

原始代码：

```python
self.GAB3 = group_aggregation_bridge(
    c_list[3],
    c_list[2]
)
```

替换为：

```python
self.edge_head = BoundaryHead()

self.GAB3 = (
    BoundaryGuidedCrossTransformerBridge(
        dim_high=c_list[3],
        dim_low=c_list[2],
        window_size=8,
        num_heads=4,
        gab_cls=group_aggregation_bridge
    )
)
```

其他模块第一版保持不变：

```python
self.GAB1
self.GAB2
self.GAB4
self.GAB5
```

---

## 14. 修改 Forward

编码器得到 `t1、t2、t3` 后：

```python
edge_logits_32 = self.edge_head(
    t1,
    t2,
    t3
)
```

原 GAB3 调用：

```python
t3 = self.GAB3(
    t4,
    t3,
    gt_pre3
)
```

修改为：

```python
t3_bridge = self.GAB3(
    x_high=t4,
    x_low=t3,
    mask_logits=gt_pre3,
    edge_logits=edge_logits_32
)

out3 = torch.add(
    out3,
    t3_bridge
)
```

建议模型返回字典：

```python
return {
    "deep_supervision": gt_pre,
    "final_output": out,
    "edge_logits": edge_logits_32
}
```

---

## 15. 边界 GT 生成

\[
E_{\mathrm{GT}}
=
\operatorname{Dilate}(Y)
-
\operatorname{Erode}(Y)
\]

```python
import torch.nn.functional as F


def mask_to_boundary(
    mask,
    kernel_size=5
):
    padding = kernel_size // 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    boundary = (
        dilated - eroded
    ).clamp(0, 1)

    return boundary
```

生成 `32×32` 边界监督：

```python
edge_gt_full = mask_to_boundary(
    target,
    kernel_size=5
)

edge_gt_32 = F.adaptive_max_pool2d(
    edge_gt_full,
    output_size=edge_logits.shape[-2:]
)
```

使用最大池化可防止细边缘在缩小时消失。

---

## 16. 损失函数

### 16.1 原始分割损失

\[
L_{\mathrm{seg}}
=
\sum_{i=0}^{5}
\lambda_i
\left(
L_{\mathrm{BCE}}^{i}
+
L_{\mathrm{Dice}}^{i}
\right)
\]

深度监督权重：

\[
\lambda=
\{1,0.5,0.4,0.3,0.2,0.1\}
\]

### 16.2 边界损失

\[
L_{\mathrm{edge}}
=
L_{\mathrm{BCE-edge}}
+
L_{\mathrm{Dice-edge}}
\]

### 16.3 总损失

\[
L_{\mathrm{total}}
=
L_{\mathrm{seg}}
+
\lambda_e L_{\mathrm{edge}}
\]

建议：

\[
\lambda_e=0.2
\]

前 20 个 Epoch 线性升权：

\[
\lambda_e(t)
=
0.2
\min
\left(
1,
\frac{t}{20}
\right)
\]

---

## 17. Dice Loss 实现

```python
import torch


def dice_loss(
    probability,
    target,
    eps=1e-6
):
    dims = (1, 2, 3)

    intersection = torch.sum(
        probability * target,
        dim=dims
    )

    denominator = (
        torch.sum(
            probability,
            dim=dims
        )
        +
        torch.sum(
            target,
            dim=dims
        )
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()
```

---

## 18. 总损失伪代码

```python
import torch
import torch.nn.functional as F


def compute_total_loss(
    model_output,
    target,
    epoch,
    original_seg_loss_fn
):
    deep_outputs = (
        model_output["deep_supervision"]
    )

    final_output = (
        model_output["final_output"]
    )

    edge_logits = (
        model_output["edge_logits"]
    )

    seg_loss = original_seg_loss_fn(
        deep_outputs,
        final_output,
        target
    )

    edge_gt_full = mask_to_boundary(
        target,
        kernel_size=5
    )

    edge_gt_32 = F.adaptive_max_pool2d(
        edge_gt_full,
        output_size=edge_logits.shape[-2:]
    )

    edge_prob = torch.sigmoid(
        edge_logits
    )

    edge_bce = (
        F.binary_cross_entropy_with_logits(
            edge_logits,
            edge_gt_32
        )
    )

    edge_dice = dice_loss(
        edge_prob,
        edge_gt_32
    )

    edge_loss = (
        edge_bce
        +
        edge_dice
    )

    lambda_edge = 0.2 * min(
        1.0,
        epoch / 20.0
    )

    total_loss = (
        seg_loss
        +
        lambda_edge * edge_loss
    )

    return total_loss, {
        "total": total_loss.detach(),
        "seg": seg_loss.detach(),
        "edge": edge_loss.detach(),
        "lambda_edge": lambda_edge
    }
```

---

## 19. ISIC2018 数据划分

所有模型必须使用完全相同的数据划分。

建议：

```text
约 63%：Train
约 7%：Validation
约 30%：Test
```

要求：

- Validation 用于选择最佳 Epoch 和调参；
- Test 只在所有设计确定后使用；
- 原始 EGE-UNet 必须在相同划分上重新训练；
- 不直接拿论文报告值与本地单次实验结果比较；
- 固定样本列表，不允许每个随机种子重新划分。

保存为：

```text
splits/
├── train.txt
├── val.txt
└── test.txt
```

---

## 20. 数据预处理与增强

第一阶段保持接近原始 EGE-UNet：

```python
Resize(256, 256)
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate(p=0.5)
Normalize(...)
```

注意：

```text
图像缩放：双线性插值
Mask 缩放：最近邻插值
```

第一轮不要同时加入：

- CutMix；
- MixUp；
- Elastic Transform；
- CLAHE；
- 毛发去除；
- 复杂颜色增强；
- 多尺度训练。

---

## 21. 推荐训练配置

| 项目 | 设置 |
|---|---|
| 输入尺寸 | `256×256` |
| Epoch | `300` |
| 优化器 | AdamW |
| 初始学习率 | `1e-3` |
| Betas | `(0.9, 0.999)` |
| Weight decay | `1e-2` |
| 学习率策略 | CosineAnnealingLR |
| 最小学习率 | `1e-5` |
| 分割阈值 | `0.5` |
| 梯度裁剪 | `1.0` |
| AMP | 开启 |
| 有效 Batch size | `8` |

8GB 显存建议：

```python
batch_size = 4
gradient_accumulation_steps = 2
effective_batch_size = 8
amp = True
```

---

## 22. 训练循环伪代码

```python
import torch


scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 2

for epoch in range(num_epochs):

    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):

        images = batch["image"].cuda(
            non_blocking=True
        )

        masks = batch["mask"].cuda(
            non_blocking=True
        )

        with torch.cuda.amp.autocast():

            model_output = model(images)

            loss, loss_dict = (
                compute_total_loss(
                    model_output=model_output,
                    target=masks,
                    epoch=epoch,
                    original_seg_loss_fn=(
                        original_seg_loss_fn
                    )
                )
            )

            loss = (
                loss
                /
                accumulation_steps
            )

        scaler.scale(loss).backward()

        should_step = (
            (step + 1)
            % accumulation_steps
            == 0
        )

        if should_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()
            optimizer.zero_grad()

    scheduler.step()
```

---

## 23. 随机种子

至少运行 5 次：

```python
seeds = [
    42,
    3407,
    2023,
    2025,
    2026
]
```

固定函数：

```python
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

最终报告：

\[
\text{mean}\pm\text{std}
\]

---

## 24. 评价指标

### 24.1 区域指标

- Dice / DSC；
- IoU；
- mIoU；
- Precision；
- Recall / Sensitivity；
- Specificity。

### 24.2 边界指标

必须加入：

- Boundary IoU；
- Boundary F1；
- HD95；
- ASSD。

推荐设置：

```text
Boundary IoU 边界宽度：3 pixels
Boundary F1 容忍距离：3 pixels
HD95 单位：pixel
ASSD 单位：pixel
```

### 24.3 轻量化指标

同时报告：

- Params；
- GFLOPs；
- Peak GPU memory；
- Batch=1 latency；
- FPS。

---

## 25. 消融实验

### 25.1 核心组件消融

| 编号 | 模型 | 目的 |
|---|---|---|
| A0 | 原始 EGE-UNet | 基线 |
| A1 | EGE + Edge Head + Edge Loss | 验证单纯边界监督 |
| A2 | EGE + Window Cross-Attention | 验证 Transformer |
| A3 | EGE + Transformer + Mask Gate | 验证 Mask 引导 |
| A4 | 完整 BGCT-EGE-UNet | 验证边界门控 Transformer |

要求：

```text
A1：没有 Transformer
A2：没有 Edge Head、没有 Edge Loss
A3：有 Transformer 和 Mask，没有边界门控
A4：全部启用
```

### 25.2 插入位置消融

| 设置 | 插入位置 | 预期特性 |
|---|---|---|
| P1 | GAB2，64×64 | 边缘细，但计算大、噪声多 |
| P2 | GAB3，32×32 | 推荐，边界与语义平衡 |
| P3 | GAB4，16×16 | 更偏向病灶形状 |
| P4 | Stage 6，8×8 | 主要改善全局完整性 |

### 25.3 窗口大小消融

| 窗口 | 窗口数量 | 每窗口 Token |
|---|---:|---:|
| `4×4` | 64 | 16 |
| `8×8` | 16 | 64 |
| `16×16` | 4 | 256 |

推荐初值：

```text
8×8
```

### 25.4 边界损失权重

测试：

\[
\lambda_e\in
\{0.1,0.2,0.3\}
\]

### 25.5 注意力头数

在 `dim=24` 下测试：

| 头数 | 每头维度 |
|---:|---:|
| 2 | 12 |
| 4 | 6 |
| 6 | 4 |

推荐初值：

```text
num_heads = 4
```

---

## 26. 第一阶段成功标准

建议工程验收标准：

```text
DSC：不能明显下降
mIoU：最好提高至少 0.3 个百分点
Boundary IoU：提高至少 1.0 个百分点
Boundary F1：提高至少 1.0 个百分点
HD95：下降至少 5%
参数量：控制在 0.1M 左右或以内
GFLOPs：控制在 0.12 左右或以内
```

这些是工程目标，不是保证结果。

---

## 27. 常见失败与修复

### 27.1 Dice 提高，但 Boundary IoU 不提高

```text
λ_edge：0.2 → 0.3
window_size：8 → 4
边界门控：0.5 + E → 0.25 + 1.5E
```

### 27.2 Boundary IoU 提高，但 Dice 下降

```text
λ_edge：0.2 → 0.1
gamma：0.1 → 0.05
必须保留原始 GAB 残差路径
```

### 27.3 训练早期 Loss 震荡

```text
开启梯度裁剪 max_norm=1.0
gamma 初始化为 0.05
边界损失前 20 轮线性升权
学习率从 1e-3 降为 5e-4
```

如果修改学习率，基线必须使用相同学习率重新训练。

### 27.4 边界预测全黑

检查：

```python
print(edge_gt_32.mean().item())

print(
    torch.sigmoid(
        edge_logits
    ).mean().item()
)
```

可能原因：

- Mask 是 `0/255`，没有归一化为 `0/1`；
- 边界太细；
- 缩放时错误使用双线性插值；
- 腐蚀实现错误；
- 数据增强后图像与 Mask 未同步。

### 27.5 注意力输出 NaN

```python
assert torch.isfinite(q_map).all()
assert torch.isfinite(kv_map).all()
assert torch.isfinite(
    transformer_feature
).all()
```

并保证：

```python
dim = 24
num_heads = 4
assert dim % num_heads == 0
```

---

## 28. 实施顺序

### C0：复现原始 EGE-UNet

确认：

- 数据读取正确；
- 输出尺寸正确；
- Loss 正常下降；
- Dice 和 IoU 合理；
- 保存固定数据划分；
- 记录 Params、GFLOPs 和速度。

### C1：只加入 Boundary Head

确认：

```text
edge_logits.shape == [B, 1, 32, 32]
edge_gt.shape     == [B, 1, 32, 32]
边界损失能够下降
```

### C2：单独测试 Window Cross-Attention

```python
q = torch.randn(2, 24, 32, 32)
kv = torch.randn(2, 24, 32, 32)

module = WindowCrossAttention(
    dim=24,
    num_heads=4,
    window_size=8
)

out = module(
    q_map=q,
    kv_map=kv
)

assert out.shape == (
    2,
    24,
    32,
    32
)

out.mean().backward()
```

### C3：加入 BGCTB，但暂时不加边界损失

验证：

- 完整前向；
- 反向传播；
- 显存；
- 数值稳定性；
- Transformer 单独贡献。

### C4：加入完整边界监督

加入：

```text
Edge Head
Edge BCE
Edge Dice
λ_edge warm-up
```

### C5：完成 A0—A4 消融

所有模型必须使用：

- 同一数据划分；
- 同一增强；
- 同一训练轮数；
- 同一有效 Batch size；
- 同一优化器和学习率；
- 同一随机种子；
- 同一最佳模型选择规则。

### C6：窗口、位置和权重调优

只在核心方案已经有效后进行。

---

## 29. 最佳模型保存规则

```text
主规则：验证集 Dice 最大
若 Dice 差异小于 0.1%，选择 Boundary F1 更高者
```

禁止根据测试集选择最佳 Epoch。

---

## 30. 日志记录

每个 Epoch 至少记录：

```text
train_total_loss
train_seg_loss
train_edge_loss
val_dice
val_iou
val_boundary_iou
val_boundary_f1
val_hd95
learning_rate
gamma
mask_scale
GPU memory
```

建议额外可视化：

```text
输入图像
GT Mask
最终预测
预测边界
GT 边界
误差图
Mask3
Transformer 门控图
```

---

## 31. 论文表格建议

### 31.1 主结果表

| Model | Params | GFLOPs | Dice | IoU | BIoU | BF1 | HD95 | ASSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGE-UNet |  |  |  |  |  |  |  |  |
| EGE + Edge |  |  |  |  |  |  |  |  |
| EGE + WCA |  |  |  |  |  |  |  |  |
| BGCT-EGE |  |  |  |  |  |  |  |  |

### 31.2 消融表

| Edge Head | Edge Loss | Cross-Attn | Mask Gate | Edge Gate | Dice | BIoU | HD95 |
|---|---|---|---|---|---:|---:|---:|
| × | × | × | × | × |  |  |  |
| ✓ | ✓ | × | × | × |  |  |  |
| × | × | ✓ | × | × |  |  |  |
| × | × | ✓ | ✓ | × |  |  |  |
| ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |

---

## 32. 最终结论

推荐的第一版模型为：

\[
\boxed{
\text{原始 GHPA 编码器}
+
\text{原始 GAB 局部路径}
+
\text{Stage 3 窗口 Cross-Attention}
+
\text{浅层边界分支}
+
\text{边界损失}
}
\]

各部分职责：

- **GHPA**：保持轻量多轴特征提取；
- **GAB**：保留原始高低层局部融合；
- **Cross-Attention**：让高层病灶语义指导浅层边界判别；
- **Boundary Head**：显式预测病灶轮廓；
- **Edge Loss**：迫使模型真正优化边界，而不只是区域重叠；
- **残差融合**：防止 Transformer 分支破坏原始 EGE-UNet。

最终研究假设：

> Transformer 不直接负责发现边缘，而是利用高层语义判断浅层检测到的边缘是否属于病灶；浅层 CNN 和边界监督负责精确定位，二者共同改善 ISIC2018 的病灶轮廓质量。

---

## 33. 实施注意事项

1. 本方案不保证性能一定提升，必须通过严格消融验证；
2. 实际类名和函数接口需按本地 EGE-UNet 仓库适配；
3. 第一版不要同时修改多个 GAB；
4. 第一版不要引入大量额外损失；
5. 第一版不要使用复杂数据增强；
6. 必须先复现原始基线；
7. 必须报告边界指标；
8. 必须运行多个随机种子；
9. 必须保存数据划分文件和完整配置；
10. 所有消融实验必须保证训练条件一致。
