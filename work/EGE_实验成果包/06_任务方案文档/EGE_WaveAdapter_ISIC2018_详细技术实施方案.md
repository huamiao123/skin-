# EGE-WaveAdapter：基于频率解耦的轻量 Transformer 融合方案
## ——EGE-UNet × WaveFormer（ISIC 2018）详细技术实施文档

**版本**：v1.0  
**日期**：2026-08-11  
**实验代号**：EGE-WaveAdapter / FD-EGE  
**目标数据集**：ISIC 2018  
**基础网络**：现有已完成训练与评测的 EGE-UNet 工程  
**主要参考论文**：WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation, MICCAI 2025  
**参考源码**：WaveFormer 官方开源仓库  
**本阶段原则**：不重做 EGE-UNet baseline；沿用 EGE-HRViT 对比实验的全部训练协议；只改变网络结构。

---

# 0. 先给执行者看的结论

本方案不是“再往 EGE-UNet 里塞一个 Transformer”。

之前 EGE-HRViT 已经说明：

- Transformer 能一定程度改善整体形状；
- 但直接在 Stage3 后对大量 Token 做 Transformer/动态深算，容易把局部不规则边界变平滑；
- 最终 DSC 基本没有提升，并且训练稳定性变差。

因此本方案改成**明确分工**：

> **EGE/GHPA 保留局部、纹理、边界；Transformer 只处理低频全局信息。**

同时进一步增加一个重要保护机制：

> **Stage3 原始特征继续直接走原 EGE skip connection；Wavelet-Transformer 只修改送往 Stage4 的“深层语义路径”。**

因此核心数据流不是：

```text
Stage3
  ↓
Transformer
  ├────────→ Skip
  └────────→ Stage4
```

而是：

```text
                           ┌──────── 原始 Stage3 ───────→ 原 EGE Skip/GAB
Stage3 原始特征 ───────────┤
                           │
                           └→ DWT
                               ├→ LL → Lightweight Global Transformer
                               ├→ LH ─────────────────────────────┐
                               ├→ HL ── 高频完全保留/旁路 ────────┤
                               └→ HH ─────────────────────────────┤
                                                                ↓
                                                               IDWT
                                                                ↓
                                                        Residual Gating
                                                                ↓
                                                              Stage4
```

这版的第一性问题只有一个：

> **如果 Transformer 不直接碰 Stage3 的高频边缘，也不污染 skip connection，只负责低频全局语义，能否真正补足 EGE-UNet 的全局形状能力？**

---

# 1. 背景与已有实验结论

## 1.1 已有 EGE-UNet 基线

当前已有完整 EGE-UNet 训练结果，不需要重训。

当前参考结果约为：

| 模型 | 最佳 DSC | 最佳 mIoU | 稳定性 |
|---|---:|---:|---|
| EGE-UNet | ~0.882 | ~0.788 | 高 |
| EGE-HRViT | ~0.882 | ~0.789 | 较差，60–90 epoch 达峰后容易下降 |

注意：

- 上表只作为当前结构研发参考；
- 正式汇报时必须使用已有日志中的**完整精度数值**，不要只用三位小数；
- 本阶段不修改 EGE baseline 的训练配置。

## 1.2 从 EGE-HRViT 负结果得到的核心判断

EGE-HRViT 的结果说明不能简单得出“Transformer 对 EGE 无效”。

更可能是：

1. Transformer 对全局形状建模确实有效；
2. 但直接对 Stage3 的全部/困难 Token 做深层 Transformer，会过度修改局部空间结构；
3. 对皮肤病灶而言，“更平滑”并不等于“更准确”；
4. skip connection 中最重要的局部边界特征不应被 Transformer 过度混合；
5. EGE 已经有较强局部建模能力，新增模块应该补充其缺少的全局关系，而不是重复做局部处理。

因此本方案把目标从：

> “Transformer 重新处理 Stage3 特征”

改成：

> “Transformer 只修正 Stage3 的低频语义分量，并且只影响后续深层编码路径。”

---

# 2. 论文依据：WaveFormer 中真正要借鉴的内容

WaveFormer 是 MICCAI 2025 正式论文，官方 MICCAI Open Access 页面提供正式论文和代码仓库。

论文核心思想：

1. 对特征做 DWT（Discrete Wavelet Transform）；
2. 把特征拆成低频 LF 和高频 HF；
3. Self-Attention 主要作用在低频近似系数上；
4. 高频细节被保留，用于后续重建；
5. 通过降低 Attention 实际处理的空间分辨率，减少 Token 数并保留细节；
6. Decoder 使用 IDWT 将低频语义与高频细节重新组合。

WaveFormer 原论文是 **3D 医学图像分割**。

本方案不复制其完整 3D 网络，而只迁移一个核心机制：

```text
DWT
↓
低频 → Transformer
高频 → 保留
↓
IDWT
```

然后针对 2D ISIC + 超轻量 EGE-UNet 重新设计。

## 2.1 只参考这一篇论文源码即可

本阶段需要：

- 用户当前 EGE-UNet 源码；
- WaveFormer 官方源码作为 DWT/IDWT + Wavelet Attention 设计参考。

不要求额外依赖 HRViT、Swin-Unet、TransUNet 等源码。

官方资料：

- MICCAI 2025 页面：  
  https://papers.miccai.org/miccai-2025/1014-Paper4968.html
- WaveFormer 官方代码：  
  https://github.com/mahfuzalhasan/WaveFormer

---

# 3. 本方案的核心假设

## 3.1 低频与高频的任务分工

对 Stage3 特征 \(F_3\) 做 2D Haar DWT：

\[
F_3
\rightarrow
\{F_{LL},F_{LH},F_{HL},F_{HH}\}
\]

其中：

- \(F_{LL}\)：低频近似成分；
- \(F_{LH},F_{HL},F_{HH}\)：三个方向的高频细节成分。

对于皮肤病灶分割，可以近似理解为：

### 低频 LL 更偏向

- 病灶整体范围；
- 大体形状；
- 大尺度色彩分布；
- 长距离区域一致性；
- 全局语义。

### 高频 LH/HL/HH 更偏向

- 病灶边缘；
- 局部突起；
- 纹理；
- 毛发；
- 色彩急剧变化；
- 细粒度不规则结构。

因此提出：

\[
\boxed{
\text{Transformer 只作用于 } F_{LL}
}
\]

而：

\[
\boxed{
F_{LH},F_{HL},F_{HH}
\text{ 第一版完全旁路，不经过 Transformer}
}
\]

---

# 4. 为什么模块放在 Stage3

根据当前 EGE-HRViT 实验，Stage3 输出空间尺寸为：

\[
32\times 32
\]

即：

\[
N=1024
\]

个空间位置。

如果使用原 EGE-UNet 常见通道设置，则 Stage3 通道通常为约 24；最终请执行者以当前源码实际 tensor shape 为准，不允许凭记忆硬编码。

对 Stage3 做一级 DWT：

\[
32\times32
\rightarrow
16\times16
\]

低频 Transformer 只需要处理：

\[
16\times16=256
\]

个 Token。

相比直接对 1024 Token 做全局 Self-Attention，Token 数下降为 1/4。

Self-Attention 的 Token 相关项近似从：

\[
1024^2
\]

下降到：

\[
256^2
\]

即相关矩阵规模约下降到：

\[
1/16
\]

因此 Stage3 同时满足：

- 分辨率还足够保留中层结构；
- DWT 后 Attention 成本很低；
- 比 bottleneck 更早获得全局信息；
- 又不像 Stage1/Stage2 那样高分辨率、计算昂贵。

---

# 5. 最关键的结构改动：保护原始 Skip Connection

这是本方案与“直接 Stage3 + Transformer”最大的区别。

## 5.1 不允许这么写

```python
t3 = encoder3(...)
t3 = wave_adapter(t3)

t4 = encoder4(t3)

# 后面 skip 也继续使用已经被修改后的 t3
skip3 = t3
```

这样会导致：

> Transformer 的低频修改也进入 Stage3 skip，可能再次破坏局部边界信息。

## 5.2 必须拆成两条路径

推荐逻辑：

```python
t3 = encoder3(...)

# 原始 EGE feature，作为 skip，绝对不经过 Transformer
t3_skip = t3

# 只对“送往深层 encoder 的语义路径”做频率增强
t3_deep = self.wave_adapter(t3)

# Stage4 使用增强后的深层特征
t4 = encoder4(t3_deep)

# Decoder / GAB 中仍然使用 t3_skip
```

注意：

- 不要 `detach()`；
- `t3_skip = t3` 仍然需要正常反向传播；
- 不要对 `t3` 做任何 in-place 修改；
- WaveAdapter 返回新的 tensor。

## 5.3 设计含义

这样网络形成：

```text
Stage3
  │
  ├──────────────→ 原始 Skip → Decoder
  │
  └→ WaveAdapter → Stage4 → Stage5 → Stage6
```

于是：

- 浅层/中层边界信息：完全沿用 EGE；
- 深层语义路径：得到 Transformer 提供的全局增强；
- 这正是本方案想要的 Local–Global 分工。

---

# 6. EGE-WaveAdapter v1 完整结构

## 6.1 总体结构

```text
F3 = Stage3 output
shape: [B, C3, H3, W3]
典型：H3=W3=32

              F3
               │
       ┌───────┴───────────┐
       │                   │
       │                   └──────────────→ F3_skip
       │                                      │
       │                                      ↓
       │                                  原 GAB/Decoder
       │
       ↓
   Haar DWT 2D
       │
 ┌─────┼──────────────┬──────────────┐
 ↓     ↓              ↓              ↓
LL     LH             HL             HH
 │      │              │              │
 │      └──── 高频完全旁路 ───────────┘
 │
 ↓
Conv Positional Encoding
 │
 ↓
Flatten → [B, N, C]
 │
 ↓
Lightweight Global Transformer × 1
 │
 ↓
Reshape → [B,C,H/2,W/2]
 │
 ↓
LL'
 │
 └──────────┬──────── LH/HL/HH
            ↓
          IDWT
            ↓
          F_rec
            ↓
ΔF = F_rec - F3
            ↓
F3_deep = F3 + α·ΔF
            ↓
          Stage4
```

---

# 7. DWT / IDWT 的数学实现

## 7.1 一级 2D Haar DWT

对每个 \(2\times2\) 小块：

\[
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
\]

定义：

\[
LL=\frac{a+b+c+d}{2}
\]

\[
LH=\frac{-a-b+c+d}{2}
\]

\[
HL=\frac{-a+b-c+d}{2}
\]

\[
HH=\frac{a-b-c+d}{2}
\]

这里使用正交 Haar 归一化。

对 tensor：

```python
x00 = x[:, :, 0::2, 0::2]  # a
x01 = x[:, :, 0::2, 1::2]  # b
x10 = x[:, :, 1::2, 0::2]  # c
x11 = x[:, :, 1::2, 1::2]  # d

ll = (x00 + x01 + x10 + x11) * 0.5
lh = (-x00 - x01 + x10 + x11) * 0.5
hl = (-x00 + x01 - x10 + x11) * 0.5
hh = (x00 - x01 - x10 + x11) * 0.5
```

输出 shape：

```text
输入：
[B,C,H,W]

输出：
LL [B,C,H/2,W/2]
LH [B,C,H/2,W/2]
HL [B,C,H/2,W/2]
HH [B,C,H/2,W/2]
```

## 7.2 IDWT

逆变换：

\[
a=\frac{LL-LH-HL+HH}{2}
\]

\[
b=\frac{LL-LH+HL-HH}{2}
\]

\[
c=\frac{LL+LH-HL-HH}{2}
\]

\[
d=\frac{LL+LH+HL+HH}{2}
\]

恢复：

```python
a = (ll - lh - hl + hh) * 0.5
b = (ll - lh + hl - hh) * 0.5
c = (ll + lh - hl - hh) * 0.5
d = (ll + lh + hl + hh) * 0.5
```

再交错恢复到原空间。

## 7.3 第一条硬验收

必须满足：

```python
x_rec = idwt(*dwt(x))
err = (x_rec - x).abs().max()
```

FP32 下要求：

```text
max_abs_error < 1e-6
```

若这一条不通过：

> 不允许开始训练。

---

# 8. 奇数尺寸处理

当前 256 输入时 Stage3 为 32×32，不存在问题。

如果后续使用其他分辨率导致 H/W 为奇数：

1. 在 DWT 前只在 bottom/right pad；
2. 推荐 `reflect` padding；
3. 记录原始 H/W；
4. IDWT 后 crop 回原始尺寸。

伪代码：

```python
orig_h, orig_w = x.shape[-2:]

pad_h = orig_h % 2
pad_w = orig_w % 2

if pad_h or pad_w:
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

...
x_rec = idwt(...)

x_rec = x_rec[..., :orig_h, :orig_w]
```

---

# 9. 低频 Transformer 设计

第一版必须保持极轻量。

## 9.1 默认参数

假设 Stage3 channel：

\[
C_3=24
\]

则：

```text
d_model = C3 = 24
num_heads = 4
head_dim = 6
depth = 1
MLP ratio = 2
MLP hidden = 48
attention dropout = 0
MLP dropout = 0
drop path = 0
```

如果当前实际 Stage3 channel 不是 24：

- `d_model = C3`
- `num_heads` 必须整除 `C3`
- 优先从 `{2,4}` 选择
- 不要为了凑 head 数额外大幅扩充通道。

第一版不建议：

```text
24 → 64 → Transformer → 24
```

因为这会明显增加模型容量，使实验难以判断究竟是“频率解耦”有效还是“参数增加”有效。

## 9.2 位置编码

不要使用固定长度 absolute position embedding。

原因：

- 未来可能测试 256 / 352；
- 固定 16×16 的绝对位置表会降低分辨率适应性。

推荐使用 depthwise convolution positional encoding：

```python
self.pos = nn.Conv2d(
    C, C,
    kernel_size=3,
    padding=1,
    groups=C,
    bias=True
)
```

使用：

```python
ll = ll + self.pos(ll)
```

然后再 flatten。

## 9.3 Token 化

```python
B, C, H, W = ll.shape

tokens = ll.flatten(2).transpose(1, 2)
# [B, N, C]
# N = H*W
```

256 输入：

```text
Stage3 = 32×32
LL = 16×16
N = 256
```

## 9.4 Transformer Block

采用 Pre-LN：

```text
x
 │
 ├───────────── residual ───────────────┐
 ↓                                     │
LayerNorm                              │
 ↓                                     │
Global Multi-Head Self-Attention       │
 ↓                                     │
 └─────────────────────────────────────+
 ↓
x1
 │
 ├───────────── residual ───────────────┐
 ↓                                     │
LayerNorm                              │
 ↓                                     │
Linear(C → 2C)
 ↓
GELU
 ↓
Linear(2C → C)
 ↓                                     │
 └─────────────────────────────────────+
 ↓
output
```

PyTorch 可以直接使用：

```python
nn.MultiheadAttention(
    embed_dim=C,
    num_heads=heads,
    dropout=0.0,
    batch_first=True,
    bias=True
)
```

第一版不要：

- Window Attention；
- Token Halting；
- Cross-Attention；
- MoE；
- 多尺度 Transformer；
- 9/12 个 block。

否则又会把问题复杂化。

---

# 10. 高频分支：第一版必须“什么都不做”

这是非常重要的设计约束。

第一版：

```text
LH → bypass
HL → bypass
HH → bypass
```

不要一开始：

- 加 GHPA；
- 加卷积；
- 加注意力；
- 加边界监督；
- 加 learnable gating。

原因：

> 第一轮必须只验证“低频 Transformer”本身是否有效。

因为 Haar DWT + IDWT 本身是可逆的：

如果：

```text
LL 不改变
HF 不改变
```

则：

\[
IDWT(DWT(F_3))=F_3
\]

因此这是一个非常干净的实验设计。

后续若第一版有效但边界仍需加强，才进入 HF-Refine v2。

---

# 11. 重建与 Residual Gating

## 11.1 不直接使用重建特征替换 Stage3

不推荐：

```python
t3_deep = f_rec
```

而是计算：

\[
\Delta F=F_{rec}-F_3
\]

然后：

\[
F_{3}^{deep}=F_3+\alpha\Delta F
\]

等价于：

\[
F_{3}^{deep}
=
(1-\alpha)F_3+\alpha F_{rec}
\]

这样可以明确理解为：

> Transformer 只提供一个“全局语义修正量”。

## 11.2 α 的推荐实现

定义一个可学习 scalar：

```python
self.alpha = nn.Parameter(torch.tensor(0.1))
```

forward：

```python
alpha = torch.clamp(self.alpha, 0.0, 1.0)
out = x + alpha * (x_rec - x)
```

或者使用 sigmoid 参数化：

```python
self.alpha_logit = nn.Parameter(torch.tensor(-2.1972246))
alpha = torch.sigmoid(self.alpha_logit)  # ≈0.1
```

推荐 sigmoid 版本，避免训练中 α 跑出合理范围。

## 11.3 为什么不初始化为 0

如果：

\[
\alpha=0
\]

则第一步反向传播时 Transformer 分支梯度会被 α 乘成 0。

虽然 α 自己可以先学习起来，但会延迟 Transformer 的训练。

所以第一版推荐：

\[
\alpha_0=0.1
\]

这既保持接近原 EGE，又允许 Transformer 从第一步获得梯度。

---

# 12. 推荐 PyTorch 类结构

建议新增：

```text
models/
  wavelet_adapter.py
```

包含：

```python
class HaarDWT2D(nn.Module):
    ...

class HaarIDWT2D(nn.Module):
    ...

class ConvPosEncoding(nn.Module):
    ...

class LFTransformerBlock(nn.Module):
    ...

class WaveletGlobalAdapter(nn.Module):
    ...
```

---

# 13. WaveletGlobalAdapter 参考伪代码

```python
class WaveletGlobalAdapter(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=2.0,
        depth=1,
        alpha_init=0.1,
    ):
        super().__init__()

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.pos = nn.Conv2d(
            dim, dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True
        )

        self.blocks = nn.ModuleList([
            LFTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=0.0
            )
            for _ in range(depth)
        ])

        # sigmoid(alpha_logit) = alpha_init
        init_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]

        # 1. pad if needed
        pad_h = orig_h % 2
        pad_w = orig_w % 2

        if pad_h or pad_w:
            x_pad = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="reflect"
            )
        else:
            x_pad = x

        # 2. DWT
        ll, lh, hl, hh = self.dwt(x_pad)

        # 3. positional encoding only on LF
        ll_t = ll + self.pos(ll)

        # 4. flatten
        B, C, H, W = ll_t.shape
        z = ll_t.flatten(2).transpose(1, 2)

        # 5. lightweight transformer
        for block in self.blocks:
            z = block(z)

        # 6. reshape LF
        ll_t = z.transpose(1, 2).reshape(B, C, H, W)

        # 7. IDWT
        # HF bands are untouched
        x_rec = self.idwt(ll_t, lh, hl, hh)

        # 8. crop
        x_rec = x_rec[..., :orig_h, :orig_w]

        # 9. residual correction
        alpha = torch.sigmoid(self.alpha_logit)

        delta = x_rec - x

        out = x + alpha * delta

        return out
```

注意：

- 不允许对 `x` 做 in-place；
- 不使用 `.detach()`；
- HF bands 完整参与 autograd；
- `alpha` 要写入日志。

---

# 14. LFTransformerBlock 参考伪代码

```python
class LFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q = self.norm1(x)

        attn_out, _ = self.attn(
            q, q, q,
            need_weights=False
        )

        x = x + attn_out

        x = x + self.mlp(self.norm2(x))

        return x
```

---

# 15. 如何接入当前 EGE-UNet

执行者不要重新实现 EGE。

必须直接在现有工程里寻找：

```text
Stage3 output
```

也就是当前 EGE-HRViT 曾插入 Adapter 的位置。

## 15.1 删除/关闭旧 HRViT Adapter

不要同时存在：

```text
EGE-HRViT
+
EGE-Wave
```

本实验必须是独立结构。

建议增加配置：

```yaml
model_variant: ege_wave
```

而不是在旧 HRViT 代码上继续堆模块。

## 15.2 正确接线

假设原逻辑类似：

```python
t1 = self.encoder1(x)
t2 = self.encoder2(t1)
t3 = self.encoder3(t2)
t4 = self.encoder4(t3)
...
```

改成：

```python
t1 = self.encoder1(x)
t2 = self.encoder2(t1)
t3 = self.encoder3(t2)

# IMPORTANT:
# 原始 Stage3 专门留给 skip
t3_skip = t3

# 只有 deep path 经过 WaveAdapter
t3_deep = self.wave_adapter(t3)

t4 = self.encoder4(t3_deep)
...
```

Decoder/GAB 中：

```python
# 原本使用 t3 的地方
# 必须改成 t3_skip

decoder_feature = self.gab3(
    ...,
    t3_skip,
    ...
)
```

如果当前 EGE 的 Stage3 skip 变量命名不同：

> 以真实代码为准，但原则不能变。

---

# 16. 第一版不要改 Decoder

必须保留：

- 原 GAB；
- 原 decoder；
- 原 deep supervision；
- 原输出 head；
- 原 loss；
- 原 mask bridge；
- 原所有 skip connection。

理由：

> 第一轮只允许一个主要结构变量：Stage3 deep path 的频率解耦 Transformer。

---

# 17. 第一版训练参数：全部继承 EGE-HRViT 实验

这是硬性要求。

## 17.1 不重新设计训练超参数

EGE-WaveAdapter 第一轮必须与 EGE-HRViT 对比实验保持一致：

- 相同 ISIC2018 train/val/test split；
- 相同输入分辨率；
- 相同 normalization；
- 相同 augmentation；
- 相同 batch size；
- 相同 effective batch size；
- 相同 epoch；
- 相同 optimizer；
- 相同 learning rate；
- 相同 weight decay；
- 相同 LR scheduler；
- 相同 warmup（如果原实验有）；
- 相同 loss；
- 相同 deep supervision 权重；
- 相同 seed；
- 相同 AMP 设置；
- 相同 checkpoint criterion；
- 相同 test threshold；
- 相同后处理；
- 相同 metric code。

## 17.2 执行者第一步必须做的事情

不要猜参数。

直接读取当前 EGE-HRViT 实验：

```text
config.yaml
config.py
train.py
run script
日志 header
checkpoint metadata
```

将所有实际值复制进一个新的实验清单：

```text
experiments/ege_wave_protocol.yaml
```

然后进行逐项 diff。

目标：

```text
training protocol diff = 0
architecture diff > 0
```

## 17.3 不允许因为 Transformer 而主动调 LR

第一轮禁止：

```text
Transformer 使用更小 LR
backbone 使用更大 LR
新增 warmup
新增 cosine
改变 optimizer
```

全部沿用旧实验。

只有第一轮结构有效后，才允许第二阶段调参。

---

# 18. 必须先完成的单元测试

在任何正式训练之前，必须依次通过。

---

## Test 1：DWT → IDWT 完全重建

```python
x = torch.randn(2, C, 32, 32)

ll, lh, hl, hh = dwt(x)
x2 = idwt(ll, lh, hl, hh)

assert max_abs(x2 - x) < 1e-6
```

同时测试：

```text
32×32
44×44
31×33（测试 padding/crop）
```

---

## Test 2：常量输入

```python
x = torch.ones(1, C, 32, 32)
```

预期：

```text
LH ≈ 0
HL ≈ 0
HH ≈ 0
LL ≠ 0
```

要求：

```text
max(|HF|) < 1e-6
```

这验证高频定义是否正确。

---

## Test 3：棋盘格输入

构造：

```text
+1 -1 +1 -1
-1 +1 -1 +1
...
```

预期：

> 高频能量明显高于常量输入。

无需规定某个 HF band 精确数值，但必须证明 DWT 能识别快速变化。

---

## Test 4：梯度

```python
x = torch.randn(
    2, C, 32, 32,
    requires_grad=True
)

out = adapter(x)
loss = out.mean()
loss.backward()
```

检查：

```text
x.grad != None
Transformer 参数 grad != None
alpha_logit.grad != None
pos conv grad != None
```

且无：

```text
NaN
Inf
```

---

## Test 5：Identity Bypass

手动把：

```text
alpha = 0
```

或者临时将 adapter 设置 bypass。

要求：

```text
adapter(x) == x
```

误差：

```text
< 1e-6
```

---

## Test 6：实际 EGE shape

对真实 batch 打印：

```python
print(t3.shape)
print(t3_skip.shape)
print(t3_deep.shape)
```

要求：

```text
t3 == t3_skip == t3_deep
```

shape 完全一致。

---

# 19. Smoke Test

单元测试通过后，不要马上完整训练。

## 19.1 1 batch forward/backward

验证：

- forward 正常；
- backward 正常；
- loss 正常；
- 显存无异常；
- AMP 无异常。

## 19.2 小样本过拟合测试

取 8–16 张训练图。

训练几十到几百 step。

目的不是测指标，而是确认：

> 模型具有正常拟合能力，loss 能明显下降。

如果小样本都无法下降：

> 不允许开始全量训练。

---

# 20. 正式实验顺序

---

## A0：现有 EGE-UNet

不重训。

直接使用已有：

```text
best checkpoint
test prediction
metrics
```

作为 baseline。

必须重新确认：

```text
当前 evaluation script
```

跑该 checkpoint 仍能复现原指标。

---

## A1：DWT Identity Sanity Check

结构：

```text
Stage3
 ↓
DWT
 ↓
IDWT
 ↓
Stage4
```

不加 Transformer。

理论上：

\[
IDWT(DWT(F))=F
\]

因此性能应该与 EGE 几乎完全一致。

这个实验可以只做：

- forward 数值对比；
- 少量 inference；

不一定需要完整训练。

目标：

> 证明性能变化不是 DWT 实现错误引起。

---

## A2：EGE-WaveAdapter-v1（主实验）

配置：

```text
Stage3 split path
DWT level = 1
Transformer only on LL
depth = 1
heads = 4（若 C 可整除）
MLP ratio = 2
HF = untouched
alpha init = 0.1
original Stage3 skip = protected
```

这是第一版正式结果。

---

# 21. 第一阶段成功/失败判据

由于 EGE-HRViT 只有约：

```text
DSC 0.882 → 0.882
mIoU 0.788 → 0.789
```

这种 0.001 量级不能视为可靠结构收益。

建议：

## 强成功

满足大部分：

```text
ΔDSC >= +0.004
ΔmIoU >= +0.004
Sensitivity 不明显下降
Boundary 指标不下降
训练曲线稳定
参数/FLOPs 增量合理
```

## 有潜力

例如：

```text
ΔDSC = +0.002 ~ +0.004
或
ΔmIoU = +0.002 ~ +0.004
```

同时：

- 没有 Sensitivity 明显下降；
- 边界指标改善；
- 曲线比 EGE-HRViT 稳定。

这种情况值得继续调结构。

## 基本无效

```text
ΔDSC < +0.002
ΔmIoU < +0.002
```

且没有明显边界收益。

不要马上调几十个参数。

优先做结构诊断。

## 失败

出现：

- DSC 下降；
- Sensitivity 明显下降；
- 边界变得更平滑但与 GT 更远；
- 后期明显过拟合；
- alpha 快速饱和；
- 训练不稳定。

则停止该版本。

---

# 22. 必须记录的新诊断量

除了原 EGE 指标，本方案建议记录：

## 22.1 Alpha

每 epoch：

```text
alpha
```

预期：

- 如果 alpha 一直接近 0：模型认为 WaveAdapter 没价值；
- 如果快速接近 1：可能 Adapter 影响过强；
- 如果稳定在中间范围：较合理。

---

## 22.2 Transformer 修正强度

定义：

\[
R_{\Delta}
=
\frac{\|F_{rec}-F_3\|_2}
{\|F_3\|_2+\epsilon}
\]

每 epoch 统计 validation mean。

如果：

```text
R_delta 极小
```

说明 Transformer 几乎没做事。

如果：

```text
R_delta 非常大
```

可能破坏原 EGE 表征。

---

## 22.3 频率能量

定义：

\[
E_{LL}=\|LL\|_2^2
\]

\[
E_{HF}
=
\|LH\|_2^2+
\|HL\|_2^2+
\|HH\|_2^2
\]

记录：

\[
r_{LF}
=
\frac{E_{LL}}{E_{LL}+E_{HF}}
\]

以及：

\[
r_{HF}=1-r_{LF}
\]

目的：

> 了解 Stage3 特征到底有多少信息位于低频和高频。

这对后续论文分析很有价值。

---

# 23. 原有评价指标必须全部保留

至少继续输出：

- DSC；
- IoU / mIoU；
- Accuracy；
- Sensitivity；
- Specificity。

为了验证本方案“不损失边界”，建议额外增加：

- Boundary F1；
- HD95。

注意：

> 新增边界指标只用于评价，不改变训练 loss 和 checkpoint 选择规则。

---

# 24. 必须保留训练稳定性信息

之前 EGE-HRViT 最大问题之一是：

> 60–90 epoch 达峰后下滑。

因此新模型必须保存：

```text
epoch
train loss
val loss
val DSC
val IoU
val Sens
val Spec
alpha
R_delta
```

绘制：

```text
DSC vs epoch
IoU vs epoch
alpha vs epoch
R_delta vs epoch
```

重点比较：

```text
EGE-HRViT
vs
EGE-WaveAdapter
```

是否还存在后期明显下降。

---

# 25. 可视化必须针对之前的问题设计

不要只随机选“看起来好”的图片。

建立固定 case 集：

## Case A：模糊边界

观察：

> Transformer 是否再次把边界过度平滑。

## Case B：不规则病灶

观察：

> 是否保留真实凹凸和突起。

## Case C：整体区域缺失

观察：

> 低频 Transformer 是否能补全 EGE 的大块漏分。

## Case D：毛发干扰

观察：

> 高频旁路是否把毛发噪声直接传播。

## Case E：低对比度病灶

观察：

> 全局语义是否改善弱边界定位。

每个 case 展示：

```text
Input
GT
EGE
EGE-HRViT
EGE-WaveAdapter
```

---

# 26. 第一版之后的消融顺序

只有 A2 显示“有潜力”后才继续。

---

## A3：Transformer Depth

```text
depth = 1
vs
depth = 2
```

不建议第一阶段测试：

```text
4 / 6 / 9 / 12
```

原因：

> 之前 HRViT 已经暴露过深 Transformer 对轻量 EGE 不友好。

---

## A4：DWT 位置

若 Stage3 有效，可测试：

```text
Stage3
vs
Stage4
```

但不要同时插两个位置。

目的：

> 判断全局语义增强应发生在中层还是更深层。

---

## A5：Protected Skip 消融

这是非常重要的一组。

### A5-1：Protected Skip

```text
原 t3 → skip
Wave t3 → Stage4
```

### A5-2：Unprotected Skip

```text
Wave t3
 ├→ skip
 └→ Stage4
```

如果：

```text
Protected > Unprotected
```

则直接支持我们的核心判断：

> Transformer 不应修改 EGE 中负责细粒度恢复的 skip 特征。

这是很有价值的论文消融。

---

## A6：Frequency Separation 消融

### A6-1：只处理 LL

主方案。

### A6-2：直接对完整 Stage3 做同样 Transformer

控制 Transformer：

- depth 相同；
- embedding dim 相同；
- 其他配置尽量相同。

如果：

```text
LL-only > Full-feature Transformer
```

则支持：

> 不是“加 Transformer”有效，而是“频率解耦后再加 Transformer”有效。

---

# 27. v2：只有必要时才增加 HF Refinement

如果 v1 发现：

- 全局形状改善；
- 但高频中存在大量毛发噪声；
- Boundary F1 没改善；

再做 v2。

推荐：

```text
LH
HL  → concat [B, 3C, H/2, W/2]
HH
 ↓
DWConv 3×3
 ↓
GELU
 ↓
Pointwise Conv 1×1
 ↓
Residual
 ↓
split
```

不要直接给 HF 再加 Transformer。

v2 目标：

> 抑制高频噪声，而不是重新建模高频语义。

---

# 28. 不建议做的事情

第一阶段禁止：

### 1. 同时加 HRViT

```text
WaveFormer + HRViT
```

不做。

### 2. 同时改 GAB

不做。

### 3. 同时增加 Boundary Loss

不做。

### 4. 同时增加 Deep Supervision 新权重

不做。

### 5. 使用 WaveFormer 3D 预训练权重

不做。

原因：

- 3D → 2D 不直接匹配；
- 数据域不同；
- 会污染结构验证。

### 6. 直接复制完整 WaveFormer Encoder/Decoder

不做。

本项目不是重新做 WaveFormer，而是在 EGE 上验证频率解耦 Transformer。

---

# 29. 参数量与 FLOPs 要求

EGE-UNet 的重要价值是轻量。

因此新增模块不能无限膨胀。

第一版需要报告：

```text
Params(EGE)
Params(EGE-Wave)

GFLOPs(EGE)
GFLOPs(EGE-Wave)

Latency(EGE)
Latency(EGE-Wave)
```

推荐目标：

> 第一版新增参数控制在非常小的量级，最好远低于百万级。

以：

```text
C = 24
heads = 4
MLP ratio = 2
depth = 1
```

为例：

Transformer 本身仅为数千参数量级，非常适合 EGE 的轻量定位。

---

# 30. 训练协议 Manifest

执行者必须创建：

```text
experiments/ege_wave_protocol.md
```

记录：

```text
dataset split:
input resolution:
batch size:
effective batch:
epochs:
optimizer:
learning rate:
weight decay:
scheduler:
warmup:
loss:
deep supervision weights:
augmentation:
seed:
AMP:
best checkpoint rule:
inference threshold:
post-processing:
metric implementation:
GPU:
PyTorch:
CUDA:
commit:
```

并注明：

```text
来源：EGE-HRViT 对比实验
```

任何不同项必须写：

```text
DIFF:
reason:
```

第一轮理论上除：

```text
model_name
architecture
```

外不应有其他 training diff。

---

# 31. 结果表模板

```markdown
| Model | Params | GFLOPs | DSC | mIoU | Sens | Spec | BF1 | HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGE-UNet | | | | | | | | |
| EGE-HRViT | | | | | | | | |
| EGE-WaveAdapter-v1 | | | | | | | | |
```

稳定性：

```markdown
| Model | Best Epoch | Best DSC | Last-20 Mean DSC | Last-20 Std | Peak-to-Final Drop |
|---|---:|---:|---:|---:|---:|
| EGE | | | | | |
| EGE-HRViT | | | | | |
| EGE-Wave | | | | | |
```

---

# 32. 第一轮完整执行 Checklist

## P0：代码准备

- [ ] Checkout 当前 EGE 工程稳定版本
- [ ] 保存当前 commit hash
- [ ] 保存 EGE-HRViT 配置
- [ ] 新建独立 branch
- [ ] 不修改 baseline 代码路径

建议 branch：

```text
exp/ege-wave-v1
```

---

## P1：实现 2D Wavelet

- [ ] HaarDWT2D
- [ ] HaarIDWT2D
- [ ] even-shape
- [ ] odd-shape pad/crop
- [ ] FP32 reconstruction test
- [ ] gradient test

---

## P2：实现 Low-Frequency Transformer

- [ ] Conv positional encoding
- [ ] Pre-LN MHA
- [ ] FFN ratio 2
- [ ] depth 1
- [ ] no dropout
- [ ] correct shape

---

## P3：实现 WaveletGlobalAdapter

- [ ] DWT
- [ ] LL Transformer
- [ ] HF bypass
- [ ] IDWT
- [ ] residual delta
- [ ] alpha gate
- [ ] logging

---

## P4：接入 EGE

- [ ] 定位真实 Stage3
- [ ] 保存 `t3_skip`
- [ ] `t3_deep = wave_adapter(t3)`
- [ ] Stage4 使用 t3_deep
- [ ] GAB/decoder 使用 t3_skip
- [ ] 不改其他模块

---

## P5：Sanity Test

- [ ] DWT-IDWT identity
- [ ] constant input
- [ ] checkerboard
- [ ] backward
- [ ] mixed precision
- [ ] one real batch
- [ ] small-set overfit

---

## P6：正式训练

- [ ] 完全复制 EGE-HRViT training protocol
- [ ] seed 一致
- [ ] full epochs
- [ ] best checkpoint rule 一致
- [ ] 保存 per-epoch metrics
- [ ] 保存 alpha
- [ ] 保存 R_delta

---

## P7：测试

- [ ] 同一 test set
- [ ] 同一 threshold
- [ ] 同一后处理
- [ ] DSC
- [ ] IoU
- [ ] Sensitivity
- [ ] Specificity
- [ ] BF1
- [ ] HD95
- [ ] Params
- [ ] GFLOPs
- [ ] latency

---

## P8：比较

必须比较：

```text
EGE
vs
EGE-HRViT
vs
EGE-Wave
```

重点回答：

1. DSC/mIoU 是否真正提升；
2. Sensitivity 是否恢复或改善；
3. 边界是否仍被平滑；
4. 后期训练是否仍出现明显下降；
5. Transformer 修正是否集中在合理幅度；
6. 轻量性是否保持。

---

# 33. 如果第一轮失败，如何定位

---

## Case 1：alpha → 0

说明：

> 网络主动关闭 WaveAdapter。

可能原因：

- EGE Stage3 已足够；
- LL Transformer 没提供新增信息；
- 位置不合适。

下一步：

> 优先测试 Stage4，不要先加深 Transformer。

---

## Case 2：alpha → 1，性能下降

说明：

> 新分支影响过强。

下一步：

- 固定 alpha=0.1；
- 或给 alpha 加上限；
- 不要增加 Transformer depth。

---

## Case 3：DSC 不变，但 Boundary F1 提高

属于有价值结果。

说明：

> 频率解耦改变了错误分布。

可进一步研究：

- HF refinement；
- boundary-aware fusion。

---

## Case 4：整体形状变好，但边界再次变平

首先检查：

> 是否错误地让增强后的 t3 进入了 skip。

这是最高优先级 bug 排查项。

如果 protected skip 正确：

- 再检查 alpha 是否过大；
- 再检查 LL Transformer depth；
- 不要先修改 loss。

---

## Case 5：训练 60–90 epoch 后再次下降

检查：

- alpha trajectory；
- R_delta；
- train/val gap；
- Transformer gradient norm。

若明显过拟合：

第二阶段才测试：

```text
dropout = 0.1
或
depth 保持 1
```

不要直接改成复杂 scheduler。

---

# 34. 如果第一轮成功，推荐后续路线

顺序：

```text
v1
LL Transformer + HF bypass + protected skip
         ↓
depth 1 vs 2
         ↓
protected vs unprotected skip
         ↓
LL-only vs full-feature Transformer
         ↓
HF lightweight refinement
         ↓
Stage3 vs Stage4
         ↓
最终模型
```

不要逆序。

---

# 35. 最终论文故事应该怎么形成

如果实验支持，可以形成如下逻辑：

## 问题 1

EGE-UNet：

> 强局部建模，但全局形状一致性有限。

## 问题 2

直接 Transformer / HRViT：

> 可以改善全局形状，但可能破坏局部边缘，并带来训练不稳定。

## 解决方案

对 Stage3 特征进行频率解耦：

\[
F_3
\rightarrow
LL+HF
\]

让：

\[
LL
\rightarrow
Transformer
\]

而：

\[
HF
\rightarrow
Bypass
\]

同时保护：

\[
F_3^{skip}
\]

不被 Transformer 修改。

因此得到：

\[
\boxed{
\text{Global Semantic Enhancement}
+
\text{Local Boundary Preservation}
}
\]

---

# 36. 本方案与 WaveFormer 的区别

不要在论文中声称“我们就是 WaveFormer 2D”。

应该明确：

WaveFormer：

- 完整 3D Transformer segmentation architecture；
- 多阶段 DWT；
- progressive decomposition；
- Wavelet-Attention；
- IDWT decoder；
- 3D volumetric segmentation。

本方案：

- 主体仍然是 EGE-UNet；
- 只在 Stage3 deep path 插入一个 2D Wavelet Global Adapter；
- 只对一级 DWT 的 LL 做轻量 Transformer；
- Stage3 HF 完整保留；
- 原 EGE skip 特征专门保护、不通过 Transformer；
- 原 Decoder/GAB 完全不改；
- 针对 2D dermoscopy lesion segmentation。

因此这是：

> **借鉴 WaveFormer 的频率解耦思想，重新设计适配 EGE-UNet 的轻量 2D 插件。**

---

# 37. 本阶段最终交付物

执行者最终必须给出：

## 代码

```text
wavelet_adapter.py
修改后的 EGE model file
config
train/inference integration
```

## 测试

```text
unit test log
DWT-IDWT error
shape test
gradient test
small-overfit test
```

## 正式实验

```text
完整 training log
best checkpoint
test result
metrics
prediction masks
```

## 分析

```text
EGE vs HRViT vs Wave
alpha curve
R_delta curve
training stability
visual examples
params/FLOPs/latency
```

---

# 38. 最终一句话任务定义

> **在不改变 EGE-UNet 原有训练协议、Decoder、GAB 和 Stage3 skip 特征的前提下，在 Stage3 → Stage4 的深层语义路径加入一级 2D Haar DWT，将 LL 低频分量送入 1 层轻量 Global Transformer，LH/HL/HH 高频分量保持旁路，经 IDWT 恢复后以可学习 residual gate 融入 Stage3 deep feature，从而验证“低频全局 Transformer + 高频/skip 边界保护”能否解决此前 EGE-HRViT 全局增强但边界平滑、训练不稳的问题。**

---

# 参考资料

1. Al Hasan, M. M. et al. **WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation.** MICCAI 2025.  
   Official MICCAI page: https://papers.miccai.org/miccai-2025/1014-Paper4968.html

2. Official WaveFormer code repository:  
   https://github.com/mahfuzalhasan/WaveFormer

