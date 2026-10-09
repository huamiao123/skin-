# Pure Transformer U-Net Baseline 技术方案（ISIC2018）

> 目标：先搭建一个**干净、可复现、可扩展的纯 Transformer 医学图像分割 Pipeline**，暂时不引入 SelfReg、Mamba、Wavelet、CNN 分支等额外创新模块。  
> 后续所有优化都基于该 Baseline 做单变量实验。

---

## 1. 设计目标

当前希望建立的不是“更复杂的模型”，而是一条便于后续研究的基础 Pipeline：

1. **全网络以 Transformer 为主要特征建模单元**；
2. Encoder / Decoder 均采用 Transformer Block；
3. 不采用 CNN Encoder；
4. 不采用 Swin 的固定 Window Attention 作为核心机制；
5. 采用 **Efficient Global Attention**：
   - Stage1 / Stage2 对 K、V 做 Spatial Reduction；
   - Stage3 / Stage4 使用 Full Global Attention；
6. 网络深度先保持中等，避免 ISIC2018 小数据集下模型容量过大；
7. 后续可以独立研究：
   - Stage 深度；
   - Embed Dimension；
   - SR Ratio；
   - Local / Global Attention 分配；
   - Skip Connection；
   - Pretraining；
   - Decoder 结构。

---

# 2. 第一版推荐配置

## 2.1 总体配置

```text
Input Size       = 256 × 256
Patch Size       = 4
Embed Dim        = 64
Encoder Depth    = [2, 2, 4, 2]
Decoder Depth    = [2, 2, 2]
Channels         = [64, 128, 256, 512]
Heads            = [2, 4, 8, 16]
Head Dim         = 32
MLP Ratio        = 4
SR Ratio         = [4, 2, 1, 1]
Dropout          = 0.0
Attention Drop   = 0.0
DropPath         = 0.1（可从 0.0 起步）
Norm             = LayerNorm
Activation       = GELU
Output Channels  = 1
```

核心配置可记成：

```python
embed_dims = [64, 128, 256, 512]
depths     = [2, 2, 4, 2]
num_heads  = [2, 4, 8, 16]
sr_ratios  = [4, 2, 1, 1]

decoder_depths = [2, 2, 2]
```

---

# 3. Encoder 结构

输入：

```text
256 × 256 × 3
```

经过 `4×4 Patch Embedding`：

```text
64 × 64 × 64
```

然后进入四个 Stage。

| Stage | Feature Size | Channel | Heads | Depth | SR |
|---|---:|---:|---:|---:|---:|
| Stage1 | 64×64 | 64 | 2 | 2 | 4 |
| Stage2 | 32×32 | 128 | 4 | 2 | 2 |
| Stage3 | 16×16 | 256 | 8 | 4 | 1 |
| Stage4 | 8×8 | 512 | 16 | 2 | 1 |

整体：

```text
Input
256×256×3
    │
    ▼
Patch Embedding 4×4
    │
    ▼
64×64×64
    │
    ├── Transformer Block ×2
    │      SR = 4
    ▼
Patch Merging
    │
    ▼
32×32×128
    │
    ├── Transformer Block ×2
    │      SR = 2
    ▼
Patch Merging
    │
    ▼
16×16×256
    │
    ├── Transformer Block ×4
    │      Full Attention
    ▼
Patch Merging
    │
    ▼
8×8×512
    │
    ├── Transformer Block ×2
    │      Full Attention
    ▼
Bottleneck Feature
```

---

# 4. 为什么采用 [2,2,4,2]

第一版不建议直接使用：

```text
[2,2,12,2]
```

原因：

1. ISIC2018 训练集只有约 1886 张；
2. 自定义 Pure Transformer 如果没有完全对应的 ImageNet 预训练，过深容易过拟合；
3. Stage1 / Stage2 token 数量大，不适合堆太多 Block；
4. Stage4 虽然计算便宜，但空间分辨率只有 8×8；
5. Stage3 的 16×16 是最适合增加深度的位置。

因此第一版采用：

```text
[2,2,4,2]
```

后续只改变 Stage3：

```text
[2,2,2,2]
[2,2,4,2]
[2,2,6,2]
```

如果 `[2,2,6,2]` 仍有明显收益，再尝试：

```text
[2,2,8,2]
```

暂时不建议直接测试 12 / 18 层。

---

# 5. Attention 设计

## 5.1 基础思想

普通 Full Self-Attention：

```text
Q: N × C
K: N × C
V: N × C
```

计算：

```text
Attention = Softmax(QK^T / sqrt(d)) V
```

复杂度与：

```text
N²
```

相关。

Stage1：

```text
64 × 64 = 4096 tokens
```

直接 Full Attention 会产生：

```text
4096 × 4096
```

的 Attention Matrix，显存和计算成本很高。

因此 Stage1 / Stage2 采用 Spatial Reduction。

---

## 5.2 Spatial Reduction（SR）

SR 只压缩 K / V，Q 保持完整。

### Stage1

输入：

```text
64 × 64
```

SR = 4：

```text
Q:
64×64 = 4096 tokens

K,V:
16×16 = 256 tokens
```

Attention Matrix：

```text
4096 × 256
```

而不是：

```text
4096 × 4096
```

---

### Stage2

输入：

```text
32 × 32
```

SR = 2：

```text
Q = 1024 tokens
K,V = 256 tokens
```

---

### Stage3

```text
16 × 16 = 256 tokens
```

直接 Full Attention：

```text
256 × 256
```

计算已经很小。

---

### Stage4

```text
8 × 8 = 64 tokens
```

直接 Full Attention。

---

# 6. Transformer Block

每一个 Block：

```text
x
│
├── LayerNorm
│
├── Efficient Global Attention
│
├── Residual
│
├── LayerNorm
│
├── MLP
│
└── Residual
```

公式：

```text
x = x + Attention(LN(x))
x = x + MLP(LN(x))
```

MLP：

```text
C
↓
4C
↓ GELU
C
```

---

# 7. Attention 伪代码

```python
class SpatialReductionAttention:
    def __init__(self, dim, num_heads, sr_ratio):
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        self.q = Linear(dim, dim)
        self.kv = Linear(dim, dim * 2)

        if sr_ratio > 1:
            self.sr = SpatialDownsample(
                kernel_size=sr_ratio,
                stride=sr_ratio
            )
            self.norm = LayerNorm(dim)

        self.proj = Linear(dim, dim)

    def forward(self, x, H, W):
        # x: [B, N, C]

        B, N, C = x.shape

        # Q 保持所有 token
        q = self.q(x)
        q = reshape_to_heads(q)
        # [B, heads, N, head_dim]

        if self.sr_ratio > 1:
            # 恢复成空间特征
            x_spatial = x.reshape(B, H, W, C)

            # 对 K/V 的输入做空间降采样
            x_sr = self.sr(x_spatial)

            # flatten
            x_sr = x_sr.reshape(B, -1, C)

            x_sr = self.norm(x_sr)
        else:
            x_sr = x

        # K / V 来自降采样后的 token
        kv = self.kv(x_sr)

        k, v = split_kv(kv)

        k = reshape_to_heads(k)
        v = reshape_to_heads(v)

        attn = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        attn = softmax(attn, dim=-1)

        out = attn @ v

        out = merge_heads(out)
        out = self.proj(out)

        return out
```

---

# 8. Transformer Block 伪代码

```python
class TransformerBlock:
    def __init__(self, dim, heads, sr_ratio, mlp_ratio=4):
        self.norm1 = LayerNorm(dim)

        self.attn = SpatialReductionAttention(
            dim=dim,
            num_heads=heads,
            sr_ratio=sr_ratio
        )

        self.norm2 = LayerNorm(dim)

        self.mlp = MLP(
            in_features=dim,
            hidden_features=dim * mlp_ratio
        )

    def forward(self, x, H, W):

        x = x + self.attn(
            self.norm1(x),
            H,
            W
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x
```

---

# 9. Patch Embedding

Patch Size = 4：

```text
256×256×3
→
64×64×64
```

可以使用线性 Patch Projection。

工程上可使用：

```python
Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=4,
    stride=4
)
```

这里 Conv2d 仅作为 Patch Projection，不承担 CNN 特征提取功能。

伪代码：

```python
class PatchEmbedding:
    def __init__(self):
        self.proj = Conv2d(
            3,
            64,
            kernel_size=4,
            stride=4
        )

        self.norm = LayerNorm(64)

    def forward(self, x):
        x = self.proj(x)

        # [B, C, H, W]
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
```

---

# 10. Patch Merging

目标：

```text
H × W × C
→
H/2 × W/2 × 2C
```

例如：

```text
64×64×64
→
32×32×128
```

基础实现可以仿照 Swin 的 Patch Merging：

```python
class PatchMerging:
    def forward(self, x, H, W):

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = concat([x0, x1, x2, x3], dim=-1)

        # 4C → 2C
        x = Linear(4*C, 2*C)(x)

        return flatten(x), H//2, W//2
```

---

# 11. Encoder 伪代码

```python
class TransformerEncoder:
    def __init__(self):

        self.patch_embed = PatchEmbedding()

        self.stage1 = TransformerStage(
            dim=64,
            depth=2,
            heads=2,
            sr=4
        )

        self.merge1 = PatchMerging(64, 128)

        self.stage2 = TransformerStage(
            dim=128,
            depth=2,
            heads=4,
            sr=2
        )

        self.merge2 = PatchMerging(128, 256)

        self.stage3 = TransformerStage(
            dim=256,
            depth=4,
            heads=8,
            sr=1
        )

        self.merge3 = PatchMerging(256, 512)

        self.stage4 = TransformerStage(
            dim=512,
            depth=2,
            heads=16,
            sr=1
        )

    def forward(self, image):

        x1, H1, W1 = self.patch_embed(image)
        x1 = self.stage1(x1, H1, W1)

        x2, H2, W2 = self.merge1(x1, H1, W1)
        x2 = self.stage2(x2, H2, W2)

        x3, H3, W3 = self.merge2(x2, H2, W2)
        x3 = self.stage3(x3, H3, W3)

        x4, H4, W4 = self.merge3(x3, H3, W3)
        x4 = self.stage4(x4, H4, W4)

        return x1, x2, x3, x4
```

---

# 12. Decoder 设计

第一版 Decoder 不需要做得和 Encoder 一样深。

采用：

```text
Decoder Depth = [2,2,2]
```

流程：

```text
8×8×512
    │
Patch Expand
    ▼
16×16×256
    │
Concat Encoder Stage3
    │
Linear Fusion
    │
Transformer ×2
    ▼
Patch Expand
    ▼
32×32×128
    │
Concat Encoder Stage2
    │
Linear Fusion
    │
Transformer ×2
    ▼
Patch Expand
    ▼
64×64×64
    │
Concat Encoder Stage1
    │
Linear Fusion
    │
Transformer ×2
    ▼
Final Expand ×4
    ▼
256×256
    ▼
Segmentation Head
```

---

# 13. Skip Connection

Baseline 采用最简单的：

```text
Concat + Linear
```

不要一开始加入：

- Cross Attention；
- Gate；
- Feature Distillation；
- Adaptive Fusion；
- CNN Refinement。

例如：

```python
x = concat([decoder_feature, encoder_feature], dim=-1)

x = Linear(
    decoder_dim + encoder_dim,
    target_dim
)(x)
```

后续如果要研究 Skip Connection，再单独替换。

---

# 14. Decoder 伪代码

```python
class TransformerDecoder:
    def forward(self, x1, x2, x3, x4):

        # x4: 8×8×512

        d3 = patch_expand(x4)
        # 16×16×256

        d3 = concat([d3, x3], dim=-1)
        d3 = linear_fuse(d3)
        d3 = transformer_stage3(d3)
        # Transformer ×2

        d2 = patch_expand(d3)
        # 32×32×128

        d2 = concat([d2, x2], dim=-1)
        d2 = linear_fuse(d2)
        d2 = transformer_stage2(d2)

        d1 = patch_expand(d2)
        # 64×64×64

        d1 = concat([d1, x1], dim=-1)
        d1 = linear_fuse(d1)
        d1 = transformer_stage1(d1)

        out = final_patch_expand_4x(d1)

        mask = segmentation_head(out)

        return mask
```

---

# 15. 完整模型

```python
class PureTransformerUNet:
    def __init__(self):
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(self, x):

        x1, x2, x3, x4 = self.encoder(x)

        mask = self.decoder(
            x1,
            x2,
            x3,
            x4
        )

        return mask
```

---

# 16. 第一阶段实验顺序

## Experiment 0：先跑通 Baseline

固定：

```text
Depth       = [2,2,4,2]
Channels    = [64,128,256,512]
Heads       = [2,4,8,16]
SR          = [4,2,1,1]
Decoder     = [2,2,2]
```

只要求：

1. 能完整训练；
2. loss 正常下降；
3. 没有 NaN；
4. 输出尺寸正确；
5. 参数量 / FLOPs / 显存记录完整；
6. validation DSC 正常。

---

## Experiment 1：Stage3 Depth Scaling

只修改：

```text
Stage3 Depth
```

测试：

```text
[2,2,2,2]
[2,2,4,2]
[2,2,6,2]
```

其他参数完全不变。

记录：

```text
Params
FLOPs
Train Time
GPU Memory
Best Epoch
DSC
mIoU
Sensitivity
Specificity
```

如果：

```text
2 → 4 明显提升
4 → 6 无明显提升
```

则选：

```text
[2,2,4,2]
```

作为后续 Baseline。

---

# 17. 第二阶段：Embed Dimension Scaling

Stage Depth 固定后，再测试：

```text
Base-C48
[48,96,192,384]

Base-C64
[64,128,256,512]

Base-C96
[96,192,384,768]
```

目的：

找到 ISIC2018 下参数量和精度的甜点区。

---

# 18. 第三阶段：SR Ratio

模型容量确定后再测试：

```text
SR-A = [8,4,2,1]
SR-B = [4,2,1,1]
SR-C = [2,1,1,1]
SR-D = [1,1,1,1]
```

其中：

```text
[1,1,1,1]
```

代表所有 Stage 使用 Full Global Attention。

注意：

Full Attention 版本可能需要：

- 降低 Batch Size；
- Gradient Accumulation；
- FlashAttention / PyTorch SDPA。

这一组实验的目的不是创新，而是确认：

```text
ISIC2018 对 K/V Spatial Reduction 的敏感程度
```

---

# 19. 训练 Recipe

为了和当前实验对齐，第一阶段建议继续保持：

```text
Dataset     = ISIC2018
Train       = 1886
Val         = 808
Image Size  = 256 × 256
Seed        = 42
Optimizer   = AdamW
LR          = 1e-3（如果不稳定可测试 3e-4）
WeightDecay = 0.01
Scheduler   = CosineAnnealing
Epoch       = 300
Checkpoint  = Best Validation Loss
```

所有 Baseline 对比必须保持完全相同的数据划分和增强。

---

# 20. Pretraining 问题

当前 Swin-Unet 的 0.8939 使用了 ImageNet 预训练。

自定义 Global Transformer 第一版如果没有完全匹配的 ImageNet 权重，需要明确区分：

```text
Scratch Baseline
```

和：

```text
ImageNet Pretrained Swin-Unet
```

两者不能直接用于说明“架构谁更强”。

推荐后续至少有：

```text
Swin-Unet + ImageNet
Pure Transformer + Scratch
Pure Transformer + Pretrain（若后续建立）
```

如果自定义 Pipeline 后续要作为论文主干，建议最终解决预训练问题，否则在小数据 ISIC2018 上会吃亏。

---

# 21. 数据增强建议

基础增强：

```text
Random Horizontal Flip
Random Vertical Flip
Random Rotation
Random Scale / Crop
Mild Brightness / Contrast
```

不建议第一版使用过强颜色扰动，因为病灶颜色本身可能是重要信息。

数据增强只能提高泛化能力，不能等价于增加独立患者数量。

---

# 22. Baseline 阶段暂时不要加入的东西

第一版不要加入：

```text
Mamba
Wavelet
CNN Branch
SelfReg
Cross Attention Skip
SE / CBAM
Dynamic Window
Edge Branch
Deep Supervision
Frequency Module
MoE
Prompt
```

否则很难知道性能变化到底来自哪里。

Baseline 的目标是：

```text
结构简单
变量清晰
容易解释
容易消融
方便之后替换单个组件
```

---

# 23. 推荐代码目录

```text
pure_transformer_unet/
│
├── models/
│   ├── patch_embed.py
│   ├── patch_merging.py
│   ├── patch_expand.py
│   ├── attention.py
│   ├── transformer_block.py
│   ├── encoder.py
│   ├── decoder.py
│   └── pure_transformer_unet.py
│
├── configs/
│   ├── base_2242.yaml
│   ├── depth_2222.yaml
│   └── depth_2262.yaml
│
├── datasets/
│   └── isic2018.py
│
├── train.py
├── evaluate.py
└── utils/
    ├── metrics.py
    └── seed.py
```

---

# 24. 第一版验收标准

## Correctness

必须检查：

```text
Input:
[B,3,256,256]

Stage1:
[B,4096,64]

Stage2:
[B,1024,128]

Stage3:
[B,256,256]

Stage4:
[B,64,512]

Output:
[B,1,256,256]
```

---

## Attention Shape

Stage1：

```text
Q  = [B, heads, 4096, 32]
K  = [B, heads, 256, 32]
V  = [B, heads, 256, 32]
```

Stage2：

```text
Q  = [B, heads, 1024, 32]
K  = [B, heads, 256,32]
```

Stage3：

```text
Q/K/V = 256 tokens
```

Stage4：

```text
Q/K/V = 64 tokens
```

---

## Numerical

检查：

```text
loss != NaN
gradient != NaN
attention != NaN
```

并记录：

```text
max GPU memory
forward latency
training epoch time
total parameters
```

---

# 25. 最终第一版方案

建议目前直接实现：

```text
Pure Transformer U-Net V1

Input:
256×256×3

Patch:
4×4

Encoder:
Depth     = [2,2,4,2]
Channel   = [64,128,256,512]
Heads     = [2,4,8,16]
SR        = [4,2,1,1]

Decoder:
Depth     = [2,2,2]

Attention:
Stage1 = Global Q + SR-KV
Stage2 = Global Q + SR-KV
Stage3 = Full Global Attention
Stage4 = Full Global Attention

Skip:
Concat + Linear

Output:
256×256×1
```

---

# 26. 后续优化顺序

建议严格按以下顺序：

```text
V1：跑通基础 Pipeline
        ↓
Depth Scaling
        ↓
Channel / Embed Dim Scaling
        ↓
SR Ratio
        ↓
确定最终 Baseline
        ↓
再研究 Local / Global Attention 分配
        ↓
再研究 Skip Connection
        ↓
再研究真正的新模块
```

核心原则：

> **先确定一个“容量合适 + Attention 合理 + 能稳定训练”的纯 Transformer 基础模型，再做创新。**

这样后续每个优化都能清楚回答：

```text
为什么改？
改了哪里？
增加多少参数？
增加多少计算？
提升多少 DSC / mIoU？
是否稳定？
```
