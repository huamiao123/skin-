# EGE-WaveAdapter v2：Spatial Gated Wave Fusion 详细技术实施方案

**基础模型**：EGE-WaveAdapter v1  
**数据集**：ISIC 2018  
**目标**：在保持 DWT、低频 Transformer、Protected Skip、Decoder/GAB 和训练协议不变的前提下，只把 v1 的固定标量融合系数 `alpha` 改为空间自适应 Gate `A(x,y)`。

---

## 1. 当前 v1 与本次修改目标

v1 的融合为：

\[
F_{out}=F+\alpha(F_{wave}-F)
\]

其中 `F` 是原始 EGE Stage3 特征，`F_wave` 是 LL 经 Transformer 后与原始高频带经 IDWT 得到的重建特征。训练后 `alpha` 大约稳定在 0.10～0.12，说明 Wave 分支被模型认可，但只需要小幅修正。

v2 改为：

\[
F_{out}(x,y)=F(x,y)+A(x,y)\odot(F_{wave}(x,y)-F(x,y))
\]

其中：

\[
A(x,y)\in[0,1]
\]

也就是：

- v1：整张图统一使用约 10% 的 Wave 修正；
- v2：每个空间位置自己决定应该使用多少 Wave 修正。

本轮**唯一主要变量**就是：

```text
scalar alpha  →  spatial gate A(x,y)
```

其余全部保持 v1 不变。

---

## 2. 为什么要做 Spatial Gate

v1 已经得到：

| 模型 | DSC | mIoU | Sens | Spec | Params |
|---|---:|---:|---:|---:|---:|
| EGE-UNet | 0.882 | 0.788 | 0.892 | 0.958 | 53K |
| EGE-HRViT | 0.882 | 0.789 | 0.880 | 0.963 | 323K |
| EGE-WaveAdapter v1 | 0.886 | 0.796 | 0.897 | 0.959 | 58K |

说明“低频 Transformer + 高频保留 + Protected Skip”整体方向有效。

但固定 `alpha` 有天然缺陷：某个位置即使 Wave 修正错误，也必须接收相同比例的全局信息。尤其对于 v1 中出现的严重 over-segmentation case，可能就是低频分支给出了错误的全局前景先验，而固定 alpha 无法在局部关闭它。

Spatial Gate 的目标是：

```text
简单背景          → Gate 低
稳定病灶内部      → Gate 中低
局部难分区域      → Gate 较高
EGE 大块漏分区域  → Gate 较高
Wave 明显错误区域 → Gate 尽量压低
```

---

## 3. 总体结构

保持 Stage3 Protected Skip：

```text
                         ┌──────── 原始 t3 ───────→ 原 EGE Skip / GAB / Decoder
Stage3 feature t3 ───────┤
                         │
                         └→ DWT
                              ├→ LL → LF Transformer
                              ├→ LH ───────────────┐
                              ├→ HL ── HF bypass ──┤
                              └→ HH ───────────────┤
                                                   ↓
                                                  IDWT
                                                   ↓
                                                F_wave
                                                   │
                t3 ────────────────┐               │
                                   ├→ Spatial Gate ┤
                F_wave ────────────┘               │
                                                   ↓
                         t3_deep = t3 + A*(F_wave-t3)
                                                   ↓
                                                Stage4
```

注意：

1. 原始 `t3_skip` 必须继续直接进入原 GAB/Decoder；
2. `t3_deep` 只送 Stage4；
3. 不允许 Wave 处理后的特征同时进入 skip。

---

## 4. Gate 输入设计

Gate 推荐同时看：

\[
F,
F_{wave}
\]

输入：

\[
X_g=Concat(F,F_{wave})
\]

shape：

```text
F       : [B,C,H,W]
F_wave  : [B,C,H,W]
Concat  : [B,2C,H,W]
```

为什么不能只看一个？因为 Gate 真正要判断的是：

> Wave 相比 EGE 改了什么，这个改动在当前位置是否应该接受。

第一版不要额外加入 decoder mask、uncertainty、boundary map、channel attention，保持实验变量单一。

---

## 5. Gate 输出只做 1 通道

第一版：

\[
A\in\mathbb{R}^{B\times1\times H\times W}
\]

通过广播乘到所有通道：

```python
out = x + gate * (x_wave - x)
```

不要第一版就做 `[B,C,H,W]` channel-spatial gate。原因：参数更少、稳定、易解释、可直接可视化。

---

## 6. Gate 网络结构

推荐固定为：

```text
Concat(F, F_wave)
      ↓
1×1 Conv: 2C → Cg
      ↓
GELU
      ↓
DWConv 3×3: Cg → Cg
      ↓
GELU
      ↓
1×1 Conv: Cg → 1
      ↓
Sigmoid
      ↓
A(x,y)
```

隐藏通道：

\[
C_g=max(C/2,8)
\]

例如 `C=24`，则 `Cg=12`。

DWConv 的作用是让每个位置不只看自身通道，还能观察局部 3×3 邻域，对模糊边界、毛发、纹理突变判断更合理，同时参数量极小。

---

## 7. 最关键的初始化

v1 已证明约 0.1 的修正强度有效，所以 v2 不能让 Gate 初始约 0.5。

必须初始化为：

\[
A(x,y)\approx0.1
\]

最后一层：

```python
self.out_conv = nn.Conv2d(hidden, 1, 1, bias=True)
```

初始化：

```python
nn.init.zeros_(self.out_conv.weight)
```

bias 使用：

\[
logit(0.1)=\ln\frac{0.1}{0.9}\approx-2.1972246
\]

```python
nn.init.constant_(self.out_conv.bias, -2.1972246)
```

这样训练开始时 v2 基本等价于 v1 的 `alpha=0.1`。

---

## 8. SpatialWaveGate 完整参考代码

```python
import math
import torch
import torch.nn as nn


class SpatialWaveGate(nn.Module):
    def __init__(self, channels, hidden_channels=None, gate_init=0.1):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(channels // 2, 8)

        self.proj = nn.Conv2d(
            channels * 2,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )

        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )

        self.act = nn.GELU()

        self.out_conv = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # 让初始 gate ≈ gate_init
        nn.init.zeros_(self.out_conv.weight)

        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(gate_init / (1.0 - gate_init))
        nn.init.constant_(self.out_conv.bias, init_bias)

    def forward(self, x, x_wave):
        assert x.shape == x_wave.shape, f"{x.shape} vs {x_wave.shape}"

        g = torch.cat([x, x_wave], dim=1)
        g = self.proj(g)
        g = self.act(g)
        g = self.dwconv(g)
        g = self.act(g)
        gate = torch.sigmoid(self.out_conv(g))

        delta = x_wave - x
        out = x + gate * delta

        return out, gate
```

---
## 9. 修改现有 WaveletGlobalAdapter

v1 当前融合逻辑应类似：

```python
alpha = torch.sigmoid(self.alpha_logit)
delta = x_rec - x
out = x + alpha * delta
```

v2 删除 `alpha_logit`，新增：

```python
self.spatial_gate = SpatialWaveGate(
    channels=dim,
    hidden_channels=max(dim // 2, 8),
    gate_init=0.1,
)
```

forward 改为：

```python
out, gate = self.spatial_gate(x, x_rec)
```

完整参考：

```python
class WaveletGlobalAdapterV2(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        mlp_ratio=2.0,
        depth=1,
        gate_init=0.1,
    ):
        super().__init__()

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.pos = nn.Conv2d(
            dim, dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )

        self.blocks = nn.ModuleList([
            LFTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
            )
            for _ in range(depth)
        ])

        self.spatial_gate = SpatialWaveGate(
            channels=dim,
            hidden_channels=max(dim // 2, 8),
            gate_init=gate_init,
        )

    def forward(self, x, return_gate=False):
        orig_h, orig_w = x.shape[-2:]

        pad_h = orig_h % 2
        pad_w = orig_w % 2

        if pad_h or pad_w:
            x_pad = torch.nn.functional.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="reflect",
            )
        else:
            x_pad = x

        # 1. DWT
        ll, lh, hl, hh = self.dwt(x_pad)

        # 2. Low-frequency Transformer only
        ll_t = ll + self.pos(ll)
        B, C, H, W = ll_t.shape
        z = ll_t.flatten(2).transpose(1, 2)

        for block in self.blocks:
            z = block(z)

        ll_t = z.transpose(1, 2).reshape(B, C, H, W)

        # 3. 高频保持原样，直接 IDWT
        x_rec = self.idwt(ll_t, lh, hl, hh)
        x_rec = x_rec[..., :orig_h, :orig_w]

        # 4. Spatial gated fusion
        out, gate = self.spatial_gate(x, x_rec)

        if return_gate:
            return out, gate
        return out
```

---

## 10. EGE 主网络接线保持 v1 不变

```python
t3 = self.encoder3(t2)

# Protected Skip：原始 t3 直接留给 decoder/GAB
t3_skip = t3

# 只有 deep path 做 Wave + Spatial Gate
t3_deep, gate = self.wave_adapter(
    t3,
    return_gate=True,
)

t4 = self.encoder4(t3_deep)

# 后续 GAB / decoder 中继续使用 t3_skip
```

禁止：

```python
t3 = self.wave_adapter(t3)
# 然后 t3 同时用于 Stage4 和 skip
```

也不要 `detach()`，也不要 in-place 修改 `t3`。

---

## 11. 本轮禁止同时改其他结构

只允许：

```text
scalar alpha → Spatial Gate
```

禁止同时：

- 改 DWT/IDWT；
- 改 Transformer depth；
- 改 heads；
- 改 MLP ratio；
- 改位置编码；
- 改 Protected Skip；
- 改 GAB；
- 改 decoder；
- 加 PFESA；
- 加 HRViT；
- 加高频 refine；
- 加 boundary loss；
- 加 channel gate；
- 加 uncertainty branch。

否则无法判断 v2 的收益是否来自 Spatial Gate。

---

## 12. 训练参数：完全继承 Wave-v1

正式 v2 必须和 Wave-v1 使用完全相同的：

- ISIC2018 train/val/test split；
- 输入分辨率；
- batch size；
- effective batch size；
- epochs；
- optimizer；
- learning rate；
- weight decay；
- scheduler；
- warmup；
- loss；
- deep supervision 权重；
- augmentation；
- normalization；
- seed；
- AMP；
- checkpoint selection rule；
- inference threshold；
- post-processing；
- metric code。

不要给 Gate 单独 LR，不要改 optimizer，不要新增 loss。

正式结果推荐从头训练 v2，这样与 v1 最公平。

可以额外做一个快速 warm-start sanity test：加载 v1 checkpoint，`strict=False`，允许：

```text
unexpected key: alpha_logit
missing keys: spatial_gate.*
```

但正式结果仍以从头训练为主。

---

## 13. 训练前必须完成的单元测试

### Test 1：Gate 初始值

```python
x = torch.randn(2, C, 32, 32)
xw = torch.randn(2, C, 32, 32)
out, gate = gate_module(x, xw)
```

要求：

```text
abs(gate.mean() - 0.1) < 1e-4
gate.std() < 1e-4
```

### Test 2：与 v1 的初始等价性

```python
out_v2, gate = gate_module(x, xw)
out_ref = x + 0.1 * (xw - x)
```

要求：

```text
max_abs(out_v2 - out_ref) < 1e-5
```

### Test 3：Gate 范围

```text
0 <= gate <= 1
```

### Test 4：Backward

```python
loss = out.mean()
loss.backward()
```

检查：

```text
proj.weight.grad != None
dwconv.weight.grad != None
out_conv.weight.grad != None
out_conv.bias.grad != None
```

且无 NaN/Inf。

### Test 5：真实 EGE shape

要求：

```text
t3_deep.shape == t3.shape
gate.shape == [B,1,H3,W3]
```

### Test 6：小样本 overfit

8～16 张训练图进行短时间过拟合，必须看到 loss 正常下降，再启动完整训练。

---

## 14. 每个 Epoch 必须新增 Gate 统计

Validation 阶段记录：

```text
gate_mean
gate_std
gate_min
gate_max
gate_p10
gate_p50
gate_p90
```

参考：

```python
vals = gate.detach().flatten()

gate_mean = vals.mean()
gate_std = vals.std()
gate_min = vals.min()
gate_max = vals.max()

q = torch.quantile(
    vals,
    torch.tensor([0.1, 0.5, 0.9], device=vals.device)
)
```

不能只看 mean。即使 `mean≈0.10`，也可能出现：

```text
简单背景 0.02
困难区域 0.18
平均仍为 0.10
```

真正判断 Gate 是否学出空间差异，要看 `std / p10 / p50 / p90`。

---

## 15. Gate Heatmap 必须生成

对固定测试样本保存：

```text
Input | GT | EGE | Wave-v1 | Wave-v2 | Gate Heatmap
```

上采样：

```python
gate_up = F.interpolate(
    gate,
    size=input_image.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
```

热力图范围固定为 `[0,1]`，不要每张图片单独 min-max normalize，否则不能比较不同图片 Gate 强度。

重点检查：

- 简单背景 Gate 是否偏低；
- EGE 大块漏分区域 Gate 是否提高；
- Wave-v1 严重 over-segmentation 的错误区域，v2 Gate 是否主动压低。

---
## 16. 正式比较模型

至少比较：

```text
EGE-UNet
EGE-HRViT
EGE-Wave v1
EGE-Wave v2 Spatial Gate
```

最核心比较是：

```text
Wave-v1 vs Wave-v2
```

因为二者唯一主要差异应该是：

```text
scalar alpha vs spatial A(x,y)
```

---

## 17. 指标

继续输出：

- DSC；
- mIoU；
- Accuracy；
- Sensitivity；
- Specificity；
- Params；
- GFLOPs；
- latency。

强烈建议增加：

- Boundary F1；
- HD95。

注意：Sensitivity 提升不能直接证明边界改善。

---

## 18. 成功标准

### 强成功

例如：

```text
DSC >= 0.889
mIoU >= 0.800
```

或者 DSC 只小幅上涨，但同时满足：

- Boundary F1 明显提升；
- HD95 明显下降；
- catastrophic over-segmentation 减少；
- Gate heatmap 显示清晰空间差异；
- 参数几乎不增加。

### 有效

```text
DSC +0.002 以上
或
mIoU +0.002 以上
```

并且稳定性不下降，Gate 不退化成常数。

### 无效

```text
DSC/mIoU 基本不变
且 gate_std ≈ 0
```

说明 Spatial Gate 退化成近似 scalar gate。

---

## 19. 如果 Gate 退化成常数

如果最后：

```text
gate_mean ≈ 0.10
gate_std ≈ 0.001
```

第二轮才测试把显式 correction 也送进 Gate：

\[
[F,F_w,F_w-F]
\]

即：

```text
Concat(F, F_wave, Delta)
→ 1×1 Conv
→ DWConv
→ 1×1 Conv
→ Sigmoid
```

输入通道从 `2C` 变成 `3C`。

第一轮不要提前使用这个版本。

---

## 20. 如果 Gate 大量饱和到 0 或 1

优先排查：

1. `out_conv.weight` 是否正确初始化为 0；
2. bias 是否为约 -2.1972；
3. 是否误改了学习率；
4. 是否同时保留了旧 `alpha` 又乘了一次 Gate；
5. 是否错误加载 checkpoint；
6. 是否存在 in-place 操作。

第一轮不要马上加入 entropy/sparsity regularization。

---

## 21. 参数量预估

若：

\[
C=24,\quad C_g=12
\]

Gate 参数大致：

- `1×1 Conv`: `48×12 = 576`
- `DWConv 3×3`: `12×9 = 108`
- `Output 1×1`: `12`
- 加 bias 后仍低于约 1K。

因此预计：

```text
Wave-v1 ≈ 58K
Wave-v2 ≈ 59K
```

仍然保持超轻量。

---

## 22. 必须做 Per-case Diagnosis

对 test set 每张图分别计算：

\[
\Delta DSC_i=DSC_{v2,i}-DSC_{v1,i}
\]

排序输出：

```text
Top 20 improved
Top 20 degraded
```

并生成：

```text
Input | GT | Wave-v1 | Wave-v2 | Gate
```

目的不是只看平均指标，而是回答：

> Spatial Gate 到底救了哪些病例，又在哪些病例上失败？

---

## 23. Catastrophic Case 专项分析

对 v1 中“几乎整图预测为前景”的严重失败样本，必须单独输出：

```text
Input | GT | EGE | Wave-v1 | Wave-v2 | Gate
```

并计算：

```text
DSC_v1
DSC_v2
IoU_v1
IoU_v2
FP ratio
FN ratio
gate_mean
gate_std
```

如果 v2 能显著改善这类样本，Spatial Gate 的设计动机就被直接支持。

---

## 24. 最终结果表模板

```markdown
| Model | Fusion | Params | DSC | mIoU | Sens | Spec | BF1 | HD95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EGE | None | 53K | | | | | | |
| EGE-HRViT | Dynamic Halting | 323K | | | | | | |
| Wave-v1 | Scalar alpha | 58K | | | | | | |
| Wave-v2 | Spatial Gate | ~59K | | | | | | |
```

训练稳定性：

```markdown
| Model | Best Epoch | Best DSC | Final DSC | Peak-to-Final Drop |
|---|---:|---:|---:|---:|
| EGE | | | | |
| EGE-HRViT | | | | |
| Wave-v1 | | | | |
| Wave-v2 | | | | |
```

Gate 统计：

```markdown
| Epoch | Mean | Std | P10 | P50 | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | | | | | | | |
| 60 | | | | | | | |
| 90 | | | | | | | |
| Best | | | | | | | |
| Final | | | | | | | |
```

---

## 25. 推荐工程文件结构

建议：

```text
models/
    spatial_wave_gate.py
    wavelet_adapter.py
```

`wavelet_adapter.py` 支持：

```text
fusion_type = scalar       # v1
fusion_type = spatial_gate # v2
```

这样方便同一套工程直接做消融。

---

## 26. 推荐配置项

```yaml
wave_adapter:
  enabled: true
  version: v2
  stage: 3

  dwt:
    type: haar
    level: 1

  transformer:
    depth: 1
    heads: 4
    mlp_ratio: 2.0
    dropout: 0.0

  fusion:
    type: spatial_gate
    hidden_ratio: 0.5
    min_hidden: 8
    gate_init: 0.1
```

若 `dim % 4 != 0`，优先把 heads 改成 2，不要为了凑 head 数扩大通道。

---

## 27. 最终执行顺序

```text
复制 Wave-v1 稳定版本
        ↓
实现 SpatialWaveGate
        ↓
替换 scalar alpha
        ↓
Gate 初始化等价性测试
        ↓
Backward / shape 测试
        ↓
小样本 overfit
        ↓
完整沿用 v1 训练协议
        ↓
从头训练 v2
        ↓
测试 v2
        ↓
计算 Gate statistics
        ↓
生成 Gate heatmap
        ↓
计算 v1 vs v2 per-case Delta DSC
        ↓
专项检查 catastrophic cases
        ↓
输出最终对比
```

---

## 28. 最终交付物

执行 AI 至少要交付：

```text
1. SpatialWaveGate 源码
2. 修改后的 WaveletGlobalAdapterV2
3. EGE 接入位置说明
4. 单元测试结果
5. 完整训练日志
6. best checkpoint
7. 最终测试指标
8. Gate statistics
9. Gate heatmaps
10. v1 vs v2 对比图
11. per-case Delta DSC 排序
12. catastrophic case 专项分析
13. Params / GFLOPs / latency
14. 最终总结报告
```

---

## 29. 本方案真正要验证的科学问题

本实验不是简单验证：

> “再加一个小模块能不能涨一点指标。”

真正要验证的是：

\[
\boxed{Global\ correction\ should\ be\ spatially\ adaptive}
\]

即：

> EGE 的局部信息与 Wave 的全局信息，在不同空间位置的重要性不同。

如果 v2 能在几乎不增加参数的情况下：

- 提升 DSC / mIoU；
- 减少严重 over-segmentation；
- 改善 BF1 / HD95；
- Gate heatmap 出现明确空间差异；

那么最终模型逻辑就可以完整表述为：

```text
EGE
→ Local / Boundary

DWT + LF Transformer
→ Global Structure

Spatial Gate
→ Decide where global correction is useful

Protected Skip
→ Preserve local details
```

最终可概括为：

\[
\boxed{
Local\ Preservation
+
Global\ Frequency\ Modeling
+
Spatially\ Adaptive\ Fusion
}
\]
