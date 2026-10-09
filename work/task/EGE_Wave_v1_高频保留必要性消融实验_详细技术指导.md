# EGE-Wave v1 高频保留必要性消融实验：详细技术指导

> 目标：验证 EGE-Wave v1 中 `LH/HL/HH` 高频子带是否真的有必要保留，并判断“低频负责全局建模、高频负责局部细节”的设计解释是否成立。  
> 当前参考：EGE-UNet baseline DSC=0.8817；EGE-Wave v1 DSC=0.8863。  
> 原则：**只改频率分支，其他网络结构、训练配置、数据划分、随机种子全部保持一致。**

---

## 1. 当前 Wave v1 的数据流

```text
Stage3 feature F
      │
      ├──────────────→ Protected Skip → 原 GAB / Decoder
      │
      ↓
     DWT
      ↓
 ┌────┴──────────────┐
 ↓                   ↓
LL                LH/HL/HH
 ↓                   │
Transformer            │ bypass
 ↓                   │
LL'                  HF
 └────────┬──────────┘
          ↓
         IDWT
          ↓
       F_wave
          ↓
F_deep = F + α(F_wave - F)
          ↓
     Stage4 → Stage6
```

当前结果只能证明“完整 Wave 模块有效”，不能分别证明：

1. `LL → Transformer` 是否有效；
2. `LH/HL/HH` 是否必要；
3. 两者是否互补。

---

# 2. 最核心实验：Full Wave vs LL-only

## A1：Full EGE-Wave v1

保持当前实现：

\[
(LL,LH,HL,HH)=DWT(F)
\]

\[
LL'=T(LL)
\]

\[
F_{wave}=IDWT(LL',LH,HL,HH)
\]

\[
F_{deep}=F+lpha(F_{wave}-F)
\]

这是当前参考模型。

## A2：LL-only

只改一件事：

> DWT 后将 `LH/HL/HH` 全部置零。

\[
LL'=T(LL)
\]

\[
LH'=0,\quad HL'=0,\quad HH'=0
\]

\[
F_{wave}^{LL}=IDWT(LL',0,0,0)
\]

\[
F_{deep}=F+lpha(F_{wave}^{LL}-F)
\]

### 关键要求

不要：

```python
wave = interpolate(ll_transformed)
```

也不要把 LL 直接传到 Stage4。

必须仍然保持：

```text
DWT → LL Transformer → IDWT
```

因为我们只想控制：

```text
“高频是否存在”
```

不能同时改变上采样方式、通道结构、空间尺寸或后续 encoder 输入分布。

---

# 3. 推荐的消融矩阵

| ID | LL | LH/HL/HH | LL Transformer | 目的 |
|---|---|---|---|---|
| A0 | 无 Wave | 无 | 无 | EGE-UNet baseline |
| A1 | 保留 | 保留 | 使用 | 当前 Wave v1 |
| A2 | 保留 | **置零** | 使用 | **直接验证高频是否必要** |
| A3 | 保留 | 保留 | **不使用** | 验证 LL Transformer 是否必要 |
| A4 | 置零 | 保留 | 不使用 | 可选：HF-only |

第一轮最重要的是：

```text
A1 vs A2
```

---

# 4. A3：No-LL-Transformer

定义：

\[
(LL,LH,HL,HH)=DWT(F)
\]

\[
LL'=LL
\]

\[
F_{wave}=IDWT(LL,LH,HL,HH)
\]

标准正交 Haar DWT + IDWT 理论上应满足：

\[
IDWT(DWT(F))pprox F
\]

所以这一组应该接近原 EGE baseline。

它同时可以检查：

> DWT/IDWT 本身是否引入异常数值或结构变化。

如果 A3 与 baseline 差很多，应先排查实现。

---

# 5. 可选 A4：HF-only

如果 A1 vs A2 已证明高频确实重要，再做：

\[
LL'=0
\]

\[
F_{wave}^{HF}=IDWT(0,LH,HL,HH)
\]

用于回答：

> 单独高频残差能否帮助 segmentation？

这不是第一轮必须项。

---

# 6. 代码实现方式

不要复制多份 Adapter，建议在当前 `WaveletGlobalAdapter` 中加入模式参数：

```python
class WaveletGlobalAdapter(nn.Module):
    def __init__(self, dim, mode="full", alpha_init=0.1, ...):
        super().__init__()
        assert mode in {
            "full",
            "ll_only",
            "no_ll_transformer",
            "hf_only",
        }
        self.mode = mode
```

Forward 逻辑：

```python
def forward(self, x):
    ll, lh, hl, hh = self.dwt(x)

    if self.mode == "full":
        ll_out = self.ll_transformer(ll)
        lh_out, hl_out, hh_out = lh, hl, hh

    elif self.mode == "ll_only":
        ll_out = self.ll_transformer(ll)
        lh_out = torch.zeros_like(lh)
        hl_out = torch.zeros_like(hl)
        hh_out = torch.zeros_like(hh)

    elif self.mode == "no_ll_transformer":
        ll_out = ll
        lh_out, hl_out, hh_out = lh, hl, hh

    elif self.mode == "hf_only":
        ll_out = torch.zeros_like(ll)
        lh_out, hl_out, hh_out = lh, hl, hh

    x_wave = self.idwt(ll_out, lh_out, hl_out, hh_out)

    # 沿用你当前 Wave v1 的 alpha 实现，不要改参数化方式
    alpha = torch.sigmoid(self.alpha_logit)
    out = x + alpha * (x_wave - x)

    return out
```

如果你当前 `alpha` 不是 `sigmoid(alpha_logit)`，则完全照现有代码保留。

---

# 7. 为什么 LL-only 要用 `zeros_like`

推荐：

```python
lh_out = torch.zeros_like(lh)
hl_out = torch.zeros_like(hl)
hh_out = torch.zeros_like(hh)
```

不要修改 IDWT 接口。

理由：

1. 保持 IDWT 路径一致；
2. 保持 tensor shape 一致；
3. 减少实现 bug；
4. 真正做到严格控制变量。

---

# 8. 第一轮禁止新增其他模块

LL-only 第一轮禁止同时加入：

```text
新 Upsampling
新 Conv
新 Gate
新 Loss
新 Attention
```

A1 和 A2 必须做到：

```text
完全相同：
Encoder
Decoder
Transformer
Protected Skip
GAB
Deep Supervision
Loss
Optimizer
LR
Scheduler
Batch size
Epoch
Seed
Data split

唯一不同：
LH/HL/HH 是否置零
```

---

# 9. alpha 必须记录

当前融合：

\[
F_{deep}=F+lpha(F_{wave}-F)
\]

因此每个实验都记录：

```text
epoch
val DSC
alpha
```

保存：

```text
best epoch alpha
final epoch alpha
```

例如：

```text
Full alpha = 0.35
LL-only alpha = 0.03
```

这说明模型可能主动削弱 LL-only 分支。

因此 alpha 是重要机制证据。

---

# 10. 推荐附加记录：Wave correction magnitude

定义：

\[
R=F_{wave}-F
\]

记录：

\[
r=
rac{\|R\|_2}{\|F\|_2+\epsilon}
\]

代码：

```python
with torch.no_grad():
    ratio = (
        (x_wave - x).pow(2).mean().sqrt()
        /
        (x.pow(2).mean().sqrt() + 1e-8)
    )
```

比较 Full 和 LL-only 的 correction magnitude。

如果 LL-only 明显更大，说明丢掉高频后重建 feature 与原特征差异明显增大。

---

# 11. 推荐记录各频带能量

在 DWT 后统计：

\[
E_{LL}=mean(LL^2)
\]

\[
E_{LH}=mean(LH^2),\quad
E_{HL}=mean(HL^2),\quad
E_{HH}=mean(HH^2)
\]

高频占比：

\[
R_{HF}=
rac{E_{LH}+E_{HL}+E_{HH}}
{E_{LL}+E_{LH}+E_{HL}+E_{HH}}
\]

注意：

> 高频能量小，不代表高频对 segmentation 不重要。

这只是描述统计。

---

# 12. 训练配置

第一轮完全复用当前正式配置：

```text
Dataset: ISIC2018
Train/Val: 1886 / 808
Input: 256×256
Seed: 42
Optimizer: AdamW
LR: 1e-3
Weight decay: 1e-2
Scheduler: CosineAnnealing
Epoch: 300
Batch size: 64
```

不要给 LL-only 单独调学习率。

---

# 13. Checkpoint 规则

统一：

> 根据 validation DSC 选择 best checkpoint。

不要根据 test 指标选择 epoch。

最终使用：

```text
best-val checkpoint → 正式评测
```

---

# 14. 必须报告的指标

已有：

```text
DSC
mIoU
Sensitivity
Specificity
```

本次建议补：

```text
Boundary F1
HD95
ASSD
```

因为这次机制假设明确涉及：

> 高频是否帮助局部结构和边界恢复。

---

# 15. 结果解释模板

## Case 1：Full 明显优于 LL-only

例如：

```text
EGE              0.8817
LL-only          0.8825
Full Wave        0.8863
```

并且：

```text
Full BF1 ↑
Full HD95 ↓
```

支持：

> LL Transformer 提供全局建模，但单独低频不足；保留高频子带能够维持局部空间变化和精细结构。

---

## Case 2：LL-only ≈ Full

例如：

```text
LL-only       0.8858
Full          0.8863
```

说明：

> 高频 bypass 的增益可能有限。

此时不应强行声称 HF 是核心组件，可以考虑进一步简化 Wave 结构。

---

## Case 3：LL-only > Full

例如：

```text
LL-only       0.8890
Full          0.8863
```

说明一个很有价值的可能：

> Stage3 高频可能同时携带 hair、reflection、skin texture 等噪声，全部无条件保留不一定最优。

这时下一阶段才值得研究：

\[
oxed{	ext{Selective High-Frequency Preservation}}
\]

但结果出来前不要提前设计复杂 HF gate。

---

# 16. 单 seed 的判断阈值

第一轮 seed=42 只用于方向判断。

建议：

## 强信号

\[
|\Delta DSC|\ge 0.003
\]

值得继续。

## 中等信号

\[
0.0015\le|\Delta DSC|<0.003
\]

补多 seed。

## 弱信号

\[
|\Delta DSC|<0.0015
\]

不要急着作机制结论。

正式论文关键模型建议至少 3 seeds，并报告：

\[
mean\pm std
\]

---

# 17. 实验执行顺序

## Phase 0：检查 DWT/IDWT

```python
ll, lh, hl, hh = dwt(x)
x_rec = idwt(ll, lh, hl, hh)
error = (x - x_rec).abs().mean()
```

确认误差非常小，并检查 shape：

```text
x      [B,C,H,W]

LL     [B,C,H/2,W/2]
LH     [B,C,H/2,W/2]
HL     [B,C,H/2,W/2]
HH     [B,C,H/2,W/2]

IDWT   [B,C,H,W]
```

## Phase 1：只跑 LL-only

当前最优先：

```text
A1 Full Wave     已有 0.8863
A2 LL-only       新跑
```

## Phase 2：如果差异明显，再跑 No-Transformer

验证 LL Transformer 的贡献。

## Phase 3：补边界指标与可视化

只有 A1 vs A2 有清晰差异时再做。

## Phase 4：最终多 seed

只对论文关键模型补：

```text
EGE baseline
LL-only
Full Wave
```

---

# 18. 推荐配置方式

YAML 示例：

```yaml
model:
  name: ege_wave
  wave_mode: full
```

LL-only：

```yaml
model:
  name: ege_wave
  wave_mode: ll_only
```

No Transformer：

```yaml
model:
  name: ege_wave
  wave_mode: no_ll_transformer
```

不要靠手动注释代码切换实验。

---

# 19. 推荐输出目录

```text
experiments/
├── ege_baseline/
├── wave_full/
├── wave_ll_only/
├── wave_no_ll_transformer/
└── wave_hf_only/
```

每个目录保存：

```text
config.yaml
best.pth
last.pth
train.log
metrics.json
alpha.csv
```

---

# 20. 结果表模板

| Method | LL Transformer | HF preserved | DSC | mIoU | Sens | Spec | BF1 | HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EGE-UNet | ✗ | ✗ | 0.8817 | 0.7884 | 0.8921 | 0.9577 | - | - |
| Wave No-T | ✗ | ✓ | TBD | TBD | TBD | TBD | TBD | TBD |
| Wave LL-only | ✓ | ✗ | TBD | TBD | TBD | TBD | TBD | TBD |
| **Wave v1** | ✓ | ✓ | **0.8863** | **0.7958** | **0.8969** | **0.9591** | TBD | TBD |

---

# 21. 第二级消融：分别去掉 LH/HL/HH

只有当：

\[
Full \gg LL	ext{-only}
\]

才值得继续。

可做：

```text
Full                  LL + LH + HL + HH
w/o LH                LL + 0  + HL + HH
w/o HL                LL + LH + 0  + HH
w/o HH                LL + LH + HL + 0
```

目的：

> 看哪个方向的高频信息更重要。

第一轮不要先做这三组。

---

# 22. 论文表述边界

即使 Full > LL-only，也不要直接写：

> “高频就是 lesion boundary。”

更准确是：

> High-frequency subbands preserve local spatial variations and fine-grained details that are partially lost in low-frequency-only reconstruction.

只有 BF1 / HD95 同时明显改善时，才进一步说：

> retained high-frequency information is particularly beneficial for boundary reconstruction.

因为 LH/HL/HH 中可能同时包含：

```text
lesion boundary
hair
reflection
texture
noise
```

---

# 23. 这次消融最终要回答的三个问题

### Q1：LL Transformer 是否必要？

通过：

```text
A3 vs A1
```

回答。

### Q2：LH/HL/HH 是否必要？

通过：

```text
A2 vs A1
```

回答。

### Q3：低频全局建模与高频保留是否互补？

如果：

```text
A1 > A2
A1 > A3
```

则可以支持：

\[
oxed{
	ext{Low-frequency global modeling}
+
	ext{High-frequency preservation}
}
\]

共同构成 Wave v1 的有效机制。

---

# 24. 当前最推荐的第一轮实现

现在只实现：

```text
wave_mode = "ll_only"
```

唯一变化：

```python
lh = torch.zeros_like(lh)
hl = torch.zeros_like(hl)
hh = torch.zeros_like(hh)
```

位置：

```text
DWT 之后
IDWT 之前
```

其他全部保持：

```text
LL → 原 Transformer
原 IDWT
原 alpha residual
Protected Skip 不动
训练配置不动
seed=42
300 epochs
```

先得到：

\[
DSC_{LL-only}
\]

再决定下一步。

---

# 25. 决策树

```text
                LL-only 结果
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓

明显低于 Full      ≈ Full       高于 Full
ΔDSC ≥ .003       Δ < .0015     Δ ≥ .003
        │            │            │
        ↓            ↓            ↓
HF 有价值        HF 价值有限     HF 可能含噪声
        │            │            │
        ↓            ↓            ↓
补 BF1/HD95      考虑简化模型     研究 selective HF
```

---

# 26. 最重要的实验原则

这次实验不是为了证明原设计一定正确。

真正目标是：

\[
oxed{
	ext{让实验告诉我们 EGE-Wave v1 到底为什么有效。}
}
\]

三种结果都有价值：

- 高频重要 → 支撑当前设计；
- 高频不重要 → 简化模型；
- 高频有害 → 发现新的研究问题。

这比现在直接继续设计 Wave v3 更重要。
