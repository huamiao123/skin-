# FG-HWS：基于 EGE-Wave 的异构弱监督皮肤病灶分割快速可行性实验

## Box + Semantic Dermoscopic Attribute Points + Granularity–Frequency Alignment

**版本**：v1.0（快速可行性验证版）  
**目标**：只跑 1 个 seed，用最少的新实验快速判断该弱监督方向是否值得继续。  
**Backbone**：已训练成功并验证有效的 **EGE-WaveAdapter v1**。  
**数据集**：ISIC 2018 Task 1 + Task 2。  
**正式训练模型数**：2 个。

- `Weak-EGE-Wave`：普通弱监督 Baseline
- `FG-HWS Full`：完整方法

已有的全监督 EGE-Wave v1 结果只作为上限参考，不重新训练。

---

# 1. 本实验要回答的唯一问题

我们暂时不做完整论文实验，不做多 seed，不做十几组消融。

只回答：

> 在 **完全相同的 EGE-Wave v1 backbone、完全相同的弱标签、完全相同的训练参数和同一个 seed** 下，把不同粒度的弱监督分别施加到适合的全局/局部表征后，是否能显著优于最普通的弱监督训练？

实验只比较：

```text
Full-supervised EGE-Wave v1    = 已有上限（当前约 DSC 0.886）

Weak-EGE-Wave Baseline         = ?
FG-HWS Full Method             = ?
```

如果 Full Method 明显优于 Weak Baseline，再进入论文级消融；如果几乎没差，就及时停止，不继续浪费算力。

---

# 2. 方法核心思想

训练时不让模型使用完整 lesion mask，而只给两种弱监督：

## 2.1 Bounding Box：粗粒度全局监督

从 Task 1 lesion mask 离线提取 tight bounding box：

```text
[xmin, ymin, xmax, ymax]
```

Box 告诉模型：

- 病灶大概在哪里；
- 病灶横向/纵向 extent；
- Box 外区域可作为可靠背景。

但 Box **不能直接作为 lesion mask**，否则模型容易学习矩形偏置。

因此 Full Method 中 Box 主要监督 EGE-Wave 的 **低频 / Global 分支**。

---

## 2.2 Semantic Attribute Points：稀疏但精细的前景监督

ISIC 2018 Task 2 官方提供五类 dermoscopic attribute masks：

- `pigment_network`
- `negative_network`
- `streaks`
- `milia_like_cyst`
- `globules`

这些是临床有意义的病灶内部结构标注。

第一版不直接使用完整 attribute mask，而是对每个非空 attribute mask 只生成 **1 个语义前景点**。

也就是说训练时模型看到的是：

```text
Box：粗范围
Semantic Points：确定 lesion-positive 的稀疏位置
```

而不是完整 lesion GT。

---

# 3. 三类像素定义

设：

- `B`：Bounding Box 内部区域
- `S`：所有 semantic attribute points 的小圆盘支持区域
- `Ω`：整张图

定义：

```text
Positive = S
Negative = Ω - B
Unknown  = B - S
```

最关键原则：

```text
Box 内但未标注的位置 = UNKNOWN
```

绝对不能：

```text
Box 内全部设为 foreground
```

也不能：

```text
Box 内未标注位置设为 background
```

---

# 4. 所需数据和官方下载链接

## 4.1 必需数据 1：ISIC 2018 Task 1 / Task 2 共用训练图像

官方训练集为 2594 张 dermoscopic lesion images。ISIC Archive 当前仍提供 `Challenge 2018: Task 1-2: Training` collection。

### 官方 Collection 页面

https://api.isic-archive.com/collections/63/

### 官方 ZIP 直接下载

https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip

用途：

```text
训练 / 验证 / 测试输入 RGB 图像
```

> 如果当前 EGE-Wave 工程已经有完全相同的 ISIC2018 图像，则不需要重复下载。

---

## 4.2 必需数据 2：ISIC 2018 Task 1 lesion segmentation GT

### 官方 Task 1 页面

https://challenge.isic-archive.com/landing/2018/45/

### Training Ground Truth

https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip

官方命名格式：

```text
ISIC_<image_id>_segmentation.png
```

二值值域：

```text
0   = background
255 = lesion foreground
```

**在本实验中它只能用于：**

1. 离线生成 Bounding Box；
2. validation / test 指标计算；
3. 数据可行性审计。

**正式训练 Dataset 不允许读取完整 Task1 mask。**

---

## 4.3 必需数据 3：ISIC 2018 Task 2 attribute masks

### 官方 Task 2 页面

https://challenge.isic-archive.com/landing/2018/46/

### Training Ground Truth

https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip

官方命名格式：

```text
ISIC_<image_id>_attribute_<attribute_name>.png
```

其中：

```text
pigment_network
negative_network
streaks
milia_like_cyst
globules
```

二值值域：

```text
0   = absent
255 = present
```

官方说明这些 attribute masks 是由皮肤科研究人员在 dermoscopy expert 监督下，通过人工选择 SLIC superpixels 获得的。

---

## 4.4 可选：官方验证/测试文件

如果当前工程已经有固定 train/val/test split，**优先继续使用当前 split**，不要为了本实验更换划分。

如果确实需要官方文件：

```text
Validation Input:
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip

Task1 Validation GT:
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip

Test Input:
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip

Task1 Test GT:
https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip
```

当前快速实验原则仍然是：

> **沿用 EGE-Wave v1 原 split。**

---

# 5. 前提条件

正式写模型前，必须满足：

## 5.1 代码前提

必须已有：

- EGE-UNet 原工程；
- EGE-WaveAdapter v1 已成功训练版本；
- EGE-Wave v1 forward/backward 正常；
- 原来的训练参数、数据划分、metric code 可复现；
- 能取得 Stage3 原始 `t3_skip`；
- 能取得 WaveAdapter 内 **LL Transformer 输出 feature**。

---

## 5.2 数据前提

至少检查：

```text
Task1 image IDs
Task1 segmentation mask IDs
Task2 attribute mask IDs
```

能够通过 `ISIC_<id>` 对齐。

---

# 6. 第一步必须做：数据可行性统计

不要先训练。

先写：

```text
scripts/audit_weak_annotations.py
```

统计：

1. 当前原始 train split 中图像数；
2. Task2 可匹配图像数；
3. 至少有一个非空 attribute 的图像数；
4. non-empty attribute ratio；
5. 五个 attribute 各自出现次数；
6. attribute union 与 Task1 lesion GT 的 precision（仅审计）；
7. attribute union 对 lesion GT 的 coverage（仅审计）；
8. 每幅图最终可生成多少 semantic points。

### 关键定义

```python
seed_precision = intersection(attribute_union, lesion_gt) / area(attribute_union)

seed_coverage = intersection(attribute_union, lesion_gt) / area(lesion_gt)
```

这里 Task1 mask 只用于**离线研究假设是否成立**，不能用它过滤/修正 semantic points。

---

# 7. Go / No-Go 条件

第一版建议：

### GO

如果同时满足：

- Task1/Task2 可匹配率高；
- 有非空 attribute 的训练图像数量足够；
- semantic seed precision 明显较高；
- 每图通常能够得到至少 1 个 semantic point；

则进入训练。

### NO-GO

若发现：

- 大部分训练图 completely no attribute；
- semantic points 与 Task1 lesion 的对应关系很差；
- eligible training subset 过小；

则先停止，不直接跑几百 epoch。

---

# 8. 弱标签离线生成

强烈建议**预生成 JSON**，而不是每个 epoch 从 GT 重新计算。

最终训练时 DataLoader 只读取：

```text
image
weak_annotation.json
```

不读取 Task1 full mask。

推荐目录：

```text
data/
├── ISIC2018_Task1-2_Training_Input/
├── ISIC2018_Task1_Training_GroundTruth/       # 只供 preprocessing / val/test
├── ISIC2018_Task2_Training_GroundTruth/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── weak_annotations/
    └── weak_train.json
```

---

# 9. Bounding Box 生成代码

```python
import numpy as np
from PIL import Image


def mask_to_tight_box(mask_path):
    mask = np.array(Image.open(mask_path)) > 0

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError(f"Empty lesion mask: {mask_path}")

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    return [xmin, ymin, xmax, ymax]
```

必须使用 tight box。

第一版不要人为扩大 box，不加随机 jitter。

这样减少新变量。

---

# 10. Semantic Attribute Point 生成

第一版：

> 每一种非空 attribute 只取 1 个点，最多 5 points / image。

## 10.1 不推荐随机点

随机点会增加噪声，而且单 seed 实验里会多引入一个随机因素。

推荐从 attribute mask 最大 connected component 中选择最“内部”的点：

> **distance transform 最大值位置。**

---

## 10.2 推荐代码

```python
import cv2
import numpy as np
from PIL import Image

ATTRS = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]


def choose_interior_point(mask_path):
    mask = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)

    if mask.sum() == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # label 0 = background
    if num_labels <= 1:
        return None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (labels == largest_label).astype(np.uint8)

    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)

    return {
        "x": int(x),
        "y": int(y),
    }
```

---

# 11. Weak Annotation JSON 格式

每张训练图：

```json
{
  "ISIC_0012345": {
    "box": [45, 33, 301, 279],
    "image_size": [450, 600],
    "points": [
      {"x": 150, "y": 110, "attribute": "pigment_network"},
      {"x": 188, "y": 165, "attribute": "globules"}
    ]
  }
}
```

第一版 attribute 类别仅用于日志，不改变 loss 权重。

所有 semantic points 都作为：

```text
lesion-positive
```

---

# 12. 哪些训练图参与快速实验

为了保证 Baseline 和 Full Method 公平，第一轮建议：

```text
eligible_train_ids = 原 train split
                     ∩ Task2 IDs
                     ∩ 至少一个 non-empty attribute
```

Baseline 和 Full Method 必须使用**完全相同**的 `eligible_train_ids`。

Val/Test：

> 继续使用原 EGE-Wave v1 的完整 val/test split。

推理阶段不需要 attribute / box。

---

# 13. 坐标和 Resize

假设当前 EGE-Wave v1 输入分辨率为：

```text
256 × 256
```

（若实际工程不是 256，则全部继承真实值，不要硬改。）

Box 与 point 必须跟随 Resize：

```python
def resize_point(x, y, old_w, old_h, new_w, new_h):
    nx = x * new_w / old_w
    ny = y * new_h / old_h
    return nx, ny


def resize_box(box, old_w, old_h, new_w, new_h):
    xmin, ymin, xmax, ymax = box

    xmin = xmin * new_w / old_w
    xmax = xmax * new_w / old_w
    ymin = ymin * new_h / old_h
    ymax = ymax * new_h / old_h

    return [xmin, ymin, xmax, ymax]
```

---

# 14. Augmentation 原则

这是非常容易写错的地方。

## 14.1 几何增强

如果使用：

- random flip；
- rotation；
- crop；
- resize；

必须同步变换：

```text
image
box
semantic points
```

建议使用 Albumentations，并开启：

```text
bbox_params
keypoint_params
```

---

## 14.2 Consistency 的两视图

Full Method 要计算 unknown consistency。

为了避免复杂坐标映射，第一版要求：

> **两个 view 使用完全相同的 geometry，只使用不同 photometric augmentation。**

流程：

```text
raw image + box + points
        ↓
一次共享 geometric transform
        ↓
image_geom + transformed box + transformed points
        ├──────── weak photometric → view_w
        └──────── strong photometric → view_s
```

这样：

```text
P_w(x,y)
P_s(x,y)
```

天然逐像素对应。

---

# 15. Point Rasterization

annotation 仍然是一个点，但训练时单个像素梯度太弱。

第一版在 **256×256 输入空间**将每个 point rasterize 成半径：

```text
r = 2 pixels
```

的小圆盘。

这只是数值支持，不视为额外人工标注。

```python
import cv2
import numpy as np


def build_point_mask(points, h, w, radius=2):
    mask = np.zeros((h, w), dtype=np.uint8)

    for p in points:
        x = int(round(p["x"]))
        y = int(round(p["y"]))

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), radius, 1, -1)

    return mask
```

---

# 16. Box / Positive / Negative / Unknown Mask

```python
import torch


def build_weak_masks(box, point_mask, h, w):
    xmin, ymin, xmax, ymax = [int(round(v)) for v in box]

    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))

    box_mask = torch.zeros((h, w), dtype=torch.bool)
    box_mask[ymin:ymax + 1, xmin:xmax + 1] = True

    positive = point_mask.bool()
    negative = ~box_mask
    unknown = box_mask & (~positive)

    return {
        "positive": positive,
        "negative": negative,
        "unknown": unknown,
        "box_mask": box_mask,
    }
```

---

# 17. 模型基础：EGE-Wave v1 完全保留

不要重新设计 backbone。

仍然：

```text
Stage3
 ├──────── original t3_skip → GAB / Decoder
 └──────── WaveAdapter
             ↓
            DWT
       ┌─────┴─────┐
       ↓           ↓
      LL          HF
       ↓           │
 Transformer       │
       └──── IDWT ─┘
             ↓
   scalar residual α
             ↓
           Stage4
```

以下全部不改：

```text
DWT
IDWT
Transformer depth
heads
MLP ratio
HF bypass
Protected Skip
scalar α
GAB
Decoder
deep supervision
```

---

# 18. Full Method 新增两个“训练期辅助头”

只在 training 时用于提供监督。

推理时不使用，所以不增加最终 inference 路径。

---

# 19. LL Global Auxiliary Head

从 **LL Transformer 输出 feature** 取：

```text
ll_feat: [B,C,Hll,Wll]
```

例如：

```text
Stage3 = 32×32
LL = 16×16
```

新增：

```python
self.ll_aux_head = nn.Conv2d(
    C,
    1,
    kernel_size=1
)
```

forward：

```python
ll_logits = self.ll_aux_head(ll_feat)
```

再上采样到网络最终输出大小：

```python
ll_logits_up = F.interpolate(
    ll_logits,
    size=final_logits.shape[-2:],
    mode="bilinear",
    align_corners=False
)
```

该 head 仅计算 Box Projection Loss。

---

# 20. Local Auxiliary Head

从 **Protected Stage3 原始特征 `t3_skip`** 取：

```python
self.local_aux_head = nn.Conv2d(
    C3,
    1,
    kernel_size=1
)
```

forward：

```python
local_logits = self.local_aux_head(t3_skip)

local_logits_up = F.interpolate(
    local_logits,
    size=final_logits.shape[-2:],
    mode="bilinear",
    align_corners=False
)
```

它只接受 Semantic Point positive supervision。

含义：

```text
Box → Global / LL
Semantic Points → Local / Original EGE Stage3
```

这就是“监督粒度–表征粒度对齐”的第一版实现。

---

# 21. 推荐 Forward 接口

```python
def forward(self, x, return_aux=False):
    # ... original EGE stages

    t3 = self.encoder3(...)
    t3_skip = t3

    t3_deep, wave_aux = self.wave_adapter(
        t3,
        return_internal=return_aux
    )

    # ... Stage4~decoder
    final_logits = ...

    if not return_aux:
        return final_logits

    ll_feat = wave_aux["ll_feat"]

    ll_logits = self.ll_aux_head(ll_feat)
    local_logits = self.local_aux_head(t3_skip)

    ll_logits = F.interpolate(
        ll_logits,
        size=final_logits.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    local_logits = F.interpolate(
        local_logits,
        size=final_logits.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    return {
        "final": final_logits,
        "ll": ll_logits,
        "local": local_logits,
    }
```

Inference：

```python
pred = model(x, return_aux=False)
```

与 v1 主推理路径一致。

---

# 22. Loss 1：Balanced Partial Segmentation Loss

Baseline 和 Full Method 都必须使用。

Final prediction：

```text
positive points → 1
box outside     → 0
unknown         → ignore
```

由于负样本远多于正样本，不能把全部 valid pixels 混起来直接求平均。

必须分别归一化：

```python
import torch.nn.functional as F


def balanced_partial_bce(logits, pos_mask, neg_mask):
    # logits: [B,1,H,W]
    # masks:  [B,1,H,W] bool

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    if pos_logits.numel() > 0:
        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits)
        )
    else:
        loss_pos = logits.sum() * 0.0

    if neg_logits.numel() > 0:
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits)
        )
    else:
        loss_neg = logits.sum() * 0.0

    return 0.5 * loss_pos + 0.5 * loss_neg
```

定义：

```text
L_partial
```

---

# 23. Loss 2：Box → Low-Frequency Projection Loss

只在 Full Method 使用。

目的：

> Box 只约束病灶的全局横向/纵向 extent，不把矩形内部全部当 foreground。

---

## 23.1 生成 Box Projection Target

对于 256×256：

```python

def box_projection_targets(boxes, h, w, device):
    B = len(boxes)

    tx = torch.zeros((B, w), device=device)
    ty = torch.zeros((B, h), device=device)

    for b, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(round(v)) for v in box]

        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))

        tx[b, xmin:xmax + 1] = 1.0
        ty[b, ymin:ymax + 1] = 1.0

    return tx, ty
```

---

## 23.2 Prediction Projection

```python

def projection_loss(ll_logits, boxes):
    # [B,1,H,W]
    prob = torch.sigmoid(ll_logits).squeeze(1)

    # along height -> horizontal coverage
    proj_x = prob.amax(dim=1)   # [B,W]

    # along width -> vertical coverage
    proj_y = prob.amax(dim=2)   # [B,H]

    B, H, W = prob.shape

    tx, ty = box_projection_targets(
        boxes,
        H,
        W,
        prob.device
    )

    lx = F.binary_cross_entropy(proj_x, tx)
    ly = F.binary_cross_entropy(proj_y, ty)

    return 0.5 * (lx + ly)
```

定义：

```text
L_box_lf
```

第一版不做其他复杂 BoxSup/CRF/SAM pseudo-mask。

---

# 24. Loss 3：Semantic Point → Local Feature Loss

只在 Full Method 使用。

`local_logits` 只在 semantic positive points 上监督：

```python

def local_positive_loss(local_logits, pos_mask):
    vals = local_logits[pos_mask]

    if vals.numel() == 0:
        return local_logits.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        vals,
        torch.ones_like(vals)
    )
```

定义：

```text
L_point_local
```

注意：

- 不在 local head 上使用 box outside negative；
- 第一版只告诉 local feature“哪些点肯定是 lesion”；
- 避免把全局 Box 信息再次污染 local branch。

---

# 25. Loss 4：Unknown Region Consistency

只在 Full Method 使用。

对于同一几何位置：

```text
view_w = weak photometric augmentation
view_s = strong photometric augmentation
```

模型预测：

```text
P_w
P_s
```

只在：

```text
Unknown = Box inside - semantic points
```

区域做一致性。

第一版使用最简单稳定版本：

```python

def unknown_consistency_loss(
    logits_w,
    logits_s,
    unknown_mask,
):
    pw = torch.sigmoid(logits_w).detach()
    ps = torch.sigmoid(logits_s)

    if unknown_mask.sum() == 0:
        return logits_s.sum() * 0.0

    return F.mse_loss(
        ps[unknown_mask],
        pw[unknown_mask]
    )
```

定义：

```text
L_cons
```

这里 weak view 使用 stop-gradient teacher，避免两个分支互相追逐。

---

# 26. Baseline 定义

## Weak-EGE-Wave Baseline

模型：

```text
EGE-Wave v1
```

监督：

```text
semantic points → final output foreground
box outside     → final output background
box inside rest → ignore
```

Loss：

\[
L_{baseline}=L_{partial}
\]

不要使用：

```text
LL head
Local head
Projection loss
Consistency
```

---

# 27. Full Method 定义

模型：

```text
EGE-Wave v1
+
LL training auxiliary head
+
Local training auxiliary head
```

Loss：

\[
L_{full}
=
\lambda_pL_{partial}
+
\lambda_bL_{box-LF}
+
\lambda_lL_{point-local}
+
\lambda_cL_{cons}
\]

---

# 28. 第一版固定 Loss 权重

为了快速验证，不做权重网格搜索。

第一版固定：

```yaml
lambda_partial: 1.0
lambda_box_lf: 1.0
lambda_local_point: 1.0
lambda_consistency: 0.5
```

原因：

- 三个 supervision loss 各自内部已经做 mean normalization；
- consistency MSE 通常数值更小但早期噪声较大，因此先给 0.5；
- 第一版重点看方向是否有效，不追求最优超参数。

必须把四个 raw loss 每个 epoch 单独记录。

如果某一项长期比其他项高 **10× 以上**，先停下来检查实现/尺度，不要盲目完整跑完。

---

# 29. 训练参数原则

## 29.1 基础训练参数

全部继承当前正式 EGE-Wave v1 对比实验：

```text
train/val/test split
input resolution
batch size
effective batch size
epochs
optimizer
learning rate
weight decay
scheduler
warmup
augmentation 基础策略
normalization
AMP
checkpoint criterion
threshold
post-processing
metric code
```

**不要重新设计 optimizer / LR。**

---

## 29.2 Seed

第一轮只做：

```text
seed = 42
```

如果你的 Wave v1 正式实验实际 seed 不是 42，则使用 Wave v1 的真实 seed。

目标是一个 deterministic feasibility run，不做多 seed。

---

# 30. 推荐 method config

```yaml
experiment: fg_hws_fast_v1
seed: 42

backbone:
  name: ege_wave_v1
  keep_wave_structure: true
  keep_protected_skip: true
  keep_scalar_alpha: true

weak_annotation:
  attributes:
    - pigment_network
    - negative_network
    - streaks
    - milia_like_cyst
    - globules

  point_per_attribute: 1
  point_selection: distance_transform_center
  point_radius_at_256: 2
  box_type: tight

full_method:
  ll_aux_head: true
  local_aux_head: true
  unknown_consistency: true

loss:
  lambda_partial: 1.0
  lambda_box_lf: 1.0
  lambda_local_point: 1.0
  lambda_consistency: 0.5

consistency:
  geometry_shared: true
  weak_view: mild_photometric
  strong_view: strong_photometric
  stop_grad_weak: true
```

---

# 31. Strong Photometric Augmentation 第一版

不要太激进。

可以使用：

```text
ColorJitter
brightness ±20%
contrast ±20%
saturation ±20%
GaussianBlur 小概率
轻度 Gaussian Noise
```

不要第一版加入：

- CutMix；
- MixUp；
- random erase 覆盖 lesion；
- 大范围 geometric deformation。

否则 consistency 很难解释。

---

# 32. Dataset 返回格式

Baseline：

```python
{
    "image": image,
    "box": box,
    "pos_mask": pos_mask,
    "neg_mask": neg_mask,
    "unknown_mask": unknown_mask,
    "image_id": image_id,
}
```

Full：

```python
{
    "image_weak": image_w,
    "image_strong": image_s,
    "box": box,
    "pos_mask": pos_mask,
    "neg_mask": neg_mask,
    "unknown_mask": unknown_mask,
    "image_id": image_id,
}
```

**train Dataset 中不要返回 lesion GT mask。**

---

# 33. Baseline 训练循环伪代码

```python
for batch in train_loader:
    image = batch["image"].cuda()

    pos = batch["pos_mask"].cuda().bool()
    neg = batch["neg_mask"].cuda().bool()

    optimizer.zero_grad(set_to_none=True)

    with autocast():
        logits = model(
            image,
            return_aux=False
        )

        loss_partial = balanced_partial_bce(
            logits,
            pos,
            neg
        )

        loss = loss_partial

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

# 34. Full Method 训练循环伪代码

```python
for batch in train_loader:
    xw = batch["image_weak"].cuda()
    xs = batch["image_strong"].cuda()

    boxes = batch["box"]

    pos = batch["pos_mask"].cuda().bool()
    neg = batch["neg_mask"].cuda().bool()
    unknown = batch["unknown_mask"].cuda().bool()

    optimizer.zero_grad(set_to_none=True)

    with autocast():
        out_w = model(
            xw,
            return_aux=True
        )

        out_s = model(
            xs,
            return_aux=True
        )

        # 1. final partial supervision
        loss_partial = balanced_partial_bce(
            out_w["final"],
            pos,
            neg
        )

        # 2. Box -> LF / Global
        loss_box_lf = projection_loss(
            out_w["ll"],
            boxes
        )

        # 3. Semantic points -> Local
        loss_point_local = local_positive_loss(
            out_w["local"],
            pos
        )

        # 4. Unknown consistency
        loss_cons = unknown_consistency_loss(
            out_w["final"],
            out_s["final"],
            unknown
        )

        loss = (
            1.0 * loss_partial
            + 1.0 * loss_box_lf
            + 1.0 * loss_point_local
            + 0.5 * loss_cons
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

# 35. 一个需要注意的实现细节：共享 BN

同一 batch 做 weak/strong 两次 forward，如果 EGE-Wave 中使用 BatchNorm，BN statistics 会被两次 view 更新。

第一版如果当前模型 batch norm 很少、原模型已如此训练，可以先保留。

但必须 Baseline / Full 都保持同一 normalization 实现，不要临时将 BN 换成 LN/GN。

如果出现明显不稳定，再单独诊断，不要第一版提前改 backbone。

---

# 36. Validation / Test

Validation 与 Test **可以使用 Task1 full lesion mask 做指标计算**。

这是模型选择和最终评价，不进入训练梯度。

使用和 EGE-Wave v1 完全相同：

```text
threshold
resize
post-processing
metric code
```

输出：

```text
DSC
mIoU
Accuracy
Sensitivity
Specificity
```

如果当前工程已有：

```text
Boundary F1
HD95
```

也一起记录。

---

# 37. Checkpoint 选择

第一版继续沿用 EGE-Wave v1 的 checkpoint selection rule。

如果之前是：

```text
best val DSC
```

就继续：

```text
best val DSC
```

不要因为弱监督实验重新换成：

```text
train loss 最低
最后 epoch
测试集最高
```

**绝对禁止用 test set 挑 checkpoint。**

---

# 38. 训练日志必须额外输出

Baseline：

```text
Epoch
LR
L_partial
Total loss
Val DSC
Val mIoU
```

Full：

```text
Epoch
LR
L_partial
L_box_lf
L_point_local
L_cons
Total loss
Val DSC
Val mIoU
```

例如：

```text
Epoch 020
partial=0.6412
box_lf=0.3081
point_local=0.4125
cons=0.0342
total=1.3789
val_dsc=0.8126
```

---

# 39. 训练前必须做的单元测试

## Test A：GT leakage test

检查 train Dataset：

```python
sample = train_dataset[0]
print(sample.keys())
```

必须看不到：

```text
mask
gt
segmentation
lesion_gt
```

---

## Test B：Weak Masks

随机保存 20 张：

```text
Image
Box
Semantic Points
Positive Mask
Negative Mask
Unknown Mask
```

人工检查坐标变换是否正确。

---

## Test C：Box outside correctness

仅用于离线审计：

```text
Task1 lesion pixels outside generated tight box = 0
```

必须严格成立。

---

## Test D：Attribute seed audit

用 Task1 GT 仅作离线验证：

记录 semantic point 是否在 GT 内。

不要利用结果修正训练 point。

---

## Test E：Forward Shape

确认：

```text
final logits = [B,1,H,W]
ll logits    = [B,1,H,W]
local logits = [B,1,H,W]
```

---

## Test F：Loss finite

随机 batch：

```text
L_partial      finite
L_box_lf       finite
L_point_local  finite
L_cons         finite
```

所有参数梯度无 NaN / Inf。

---

## Test G：8–16 image overfit

分别对 Baseline / Full 做小样本 overfit。

要求：

```text
training loss 能明显下降
```

确认 pipeline 可学习后再启动正式训练。

---

# 40. 正式实验只跑两个模型

## Run A：Weak-EGE-Wave Baseline

```text
seed = 42
backbone = EGE-Wave v1
train subset = eligible_train_ids
loss = L_partial
```

---

## Run B：FG-HWS Full

```text
seed = 42
backbone = EGE-Wave v1
train subset = 与 Run A 完全相同
loss = L_partial + L_box_lf + L_point_local + 0.5 L_cons
```

---

# 41. 绝对公平性约束

Run A / Run B 必须相同：

```text
seed
train IDs
val IDs
test IDs
image resolution
batch size
epochs
optimizer
LR
scheduler
base augmentation
normalization
checkpoint criterion
threshold
metric code
```

只有：

```text
weak supervision strategy
```

不同。

---

# 42. 已有全监督模型怎么使用

已有：

```text
Full-supervised EGE-Wave v1
DSC ≈ 0.886
```

不重新训练。

只需要重新用同一 evaluation script 测试一次，确认 checkpoint 结果可复现。

它是：

```text
Upper Bound / Full-supervision Reference
```

不是弱监督 baseline。

---

# 43. 第一轮最终结果表

```markdown
| Model | Supervision | DSC | mIoU | Sens | Spec |
|---|---|---:|---:|---:|---:|
| Full EGE-Wave v1 | Full mask | 0.886 | 0.796 | 0.897 | 0.959 |
| Weak-EGE-Wave | Box + semantic points / Partial | | | | |
| FG-HWS Full | Granularity-Frequency aligned weak supervision | | | | |
```

---

# 44. 建议增加一个 Gap Recovery 指标

定义：

```text
D_fullsup  = Full-supervised Wave DSC
D_base     = Weak baseline DSC
D_method   = FG-HWS DSC
```

监督差距：

\[
G=D_{fullsup}-D_{base}
\]

方法恢复比例：

\[
Recovery=
\frac{D_{method}-D_{base}}
{D_{fullsup}-D_{base}}
\]

比如：

```text
Full sup = 0.886
Baseline = 0.810
FG-HWS   = 0.850
```

则：

```text
恢复了约 52.6% 的 full-supervision gap
```

这比只说 “+0.04 DSC” 更容易理解。

---

# 45. 快速 Go / No-Go 判断

这不是论文最终门槛，只是第一轮决策阈值。

## Strong GO

如果：

```text
FG-HWS - Weak Baseline >= +0.010 DSC
```

且 mIoU 同方向明显改善：

> 直接继续，值得做消融和正式实验。

---

## GO / Promising

如果：

```text
+0.005 <= ΔDSC < +0.010
```

且：

- Sens 不明显恶化；
- 曲线稳定；
- gap recovery 有意义；

继续做第二阶段。

---

## Borderline

```text
+0.003 ~ +0.005
```

先分析：

- 四个 loss 是否尺度失衡；
- LL projection 是否真的学习；
- attribute points 是否过少；
- consistency 是否伤害训练。

只允许做一次针对性修正。

---

## NO-GO

如果：

```text
ΔDSC < +0.003
```

或者 Full Method 明显下降：

> 暂停该方向，不先做多 seed 和大规模消融。

---

# 46. 第一轮必须输出的可视化

固定 8–12 张 test image：

```text
Input
GT
Weak Baseline
FG-HWS
Full-supervised Wave
```

特别找：

- 模糊边界；
- 低对比度 lesion；
- hair interference；
- 小 lesion；
- 大 lesion；
- irregular lesion。

不要只挑 FG-HWS 好看的图。

---

# 47. Full Method 训练期可视化

额外保存：

```text
Input
Box
Semantic Points
LL auxiliary probability
Local auxiliary probability
Final probability
GT（只展示，不参与训练）
```

这能直接判断：

### LL branch

是否学习：

```text
rough lesion extent / global localization
```

### Local branch

是否在 semantic point 附近形成 lesion evidence。

---

# 48. 常见失败模式与修复顺序

## Failure 1：Baseline 接近全背景

原因通常：

- positive point 太少；
- 正负 loss 没有分别归一化。

首先确认 `balanced_partial_bce`。

不要第一时间扩大 points。

---

## Failure 2：LL 预测成矩形

说明 Box supervision 太强。

先检查是否错误地：

```text
把 box_mask 当 pixel GT
```

正确方法只能使用 projection / extent。

---

## Failure 3：Consistency 让模型变差

首先临时：

```text
lambda_consistency = 0
```

跑一个短验证。

如果恢复，说明 early pseudo consistency 噪声太高。

第二阶段才考虑 ramp-up / confidence masking。

---

## Failure 4：Local head 没有效果

先检查：

```text
semantic points transform 是否正确
point mask 是否落在正确位置
local logits 是否真的来自 protected t3_skip
```

不要先增加复杂 attention。

---

# 49. 第一轮禁止做的事情

禁止：

- SAM / MedSAM；
- WeakMed 直接拼接；
- CRF；
- pseudo-mask generation；
- PFESA；
- Spatial Gate v2；
- HRViT；
- 新 Transformer block；
- Channel Attention；
- Boundary Loss；
- 多 seed；
- 多数据集；
- 1/3/5 point 消融；
- loss weight grid search。

第一轮的目标只是：

> **Full-stack idea 是否有明显收益。**

---

# 50. 推荐工程文件改动

```text
project/
├── models/
│   ├── ege_wave_v1.py
│   └── ege_wave_weak.py          # aux outputs
│
├── datasets/
│   └── isic2018_weak.py
│
├── losses/
│   └── weak_losses.py
│
├── scripts/
│   ├── audit_weak_annotations.py
│   ├── build_weak_annotations.py
│   └── visualize_weak_labels.py
│
├── configs/
│   ├── weak_ege_wave_baseline.yaml
│   └── fg_hws_full.yaml
│
└── train_weak.py
```

---

# 51. `weak_losses.py` 建议接口

```python
class WeakLosses:
    def partial(self, logits, pos, neg):
        ...

    def box_projection(self, ll_logits, boxes):
        ...

    def local_point(self, local_logits, pos):
        ...

    def consistency(self, weak_logits, strong_logits, unknown):
        ...
```

便于 Baseline / Full 共用。

---

# 52. 必须保存实验 Manifest

Run 前输出：

```text
seed
commit hash
train IDs hash
val IDs hash
test IDs hash
eligible train count
input size
batch size
epochs
optimizer
LR
scheduler
all lambda values
point count rule
attribute list
```

Baseline 和 Full 的 Manifest 做 diff。

理论上除：

```text
method / enabled losses / aux heads
```

外全部相同。

---

# 53. 第一版最终交付物

另一个 AI 必须交付：

```text
1. 数据可行性统计报告
2. eligible_train_ids.txt
3. weak_train.json
4. 20 张 weak label 可视化
5. Weak-EGE-Wave baseline 代码
6. FG-HWS Full 代码
7. 单元测试日志
8. Baseline 完整训练日志 + best checkpoint
9. Full 完整训练日志 + best checkpoint
10. 三模型最终指标表
11. 分割效果对比图
12. LL / Local auxiliary 可视化
13. Gap Recovery 计算
14. 最终 Go / No-Go 结论
```

---

# 54. 实施顺序

严格按：

```text
Step 0
确认 EGE-Wave v1 可复现
        ↓
Step 1
下载/整理 Task1 + Task2
        ↓
Step 2
做 Task1/Task2 数据审计
        ↓
Step 3
生成 tight boxes + semantic points
        ↓
Step 4
保存 weak_train.json
        ↓
Step 5
可视化 20 张弱标签
        ↓
Step 6
实现 Weak Dataset + balanced partial loss
        ↓
Step 7
8–16 图 overfit Baseline
        ↓
Step 8
正式训练 Weak-EGE-Wave Baseline（seed 42）
        ↓
Step 9
给 EGE-Wave 增加 LL / Local training heads
        ↓
Step 10
实现 projection / local point / consistency loss
        ↓
Step 11
8–16 图 overfit Full
        ↓
Step 12
正式训练 FG-HWS Full（同 seed 42）
        ↓
Step 13
统一 evaluation
        ↓
Step 14
和 Full-supervised Wave v1 比较
        ↓
Step 15
Go / No-Go
```

---

# 55. 最终方法逻辑

整个第一版研究问题可以概括为：

```text
Bounding Box
= 粗粒度、全局、extent 信息
        ↓
Low-Frequency Global Representation

Semantic Attribute Points
= 稀疏、精细、确定的 lesion evidence
        ↓
Original EGE Local Representation

Box 内未标注区域
= Unknown
        ↓
Self-consistency

最终：
EGE-Wave Decoder
        ↓
Whole Lesion Mask
```

也就是：

\[
\boxed{
\text{Coarse Supervision}
\rightarrow
\text{Global / Low-Frequency Representation}
}
\]

\[
\boxed{
\text{Sparse Precise Supervision}
\rightarrow
\text{Local Representation}
}
\]

这就是第一版 **Granularity–Frequency Aligned Heterogeneous Weak Supervision (FG-HWS)**。

---

# 56. 最重要的工程红线

**Task1 full lesion mask 在训练阶段绝对不能被网络或 loss 读取。**

正确：

```text
Task1 GT
  ├→ preprocessing: generate tight box
  ├→ validation metrics
  └→ test metrics
```

错误：

```text
Task1 GT
  → crop foreground
  → filter semantic points
  → pseudo mask
  → boundary loss
  → training loss
```

只要出现后者，弱监督实验就发生 GT leakage，结果无效。

---

# 57. 官方数据依据

ISIC 2018 官方 Challenge 将任务分成：

- Task 1：Lesion Boundary Segmentation
- Task 2：Lesion Attribute Detection

Task 1 官方说明 lesion segmentation response 是与原图同尺寸的二值 PNG mask；Task 2 官方说明 attribute response 是五种临床有意义 dermoscopic visual patterns 的二值 PNG masks，并给出了标准文件命名规则。

官方入口：

- ISIC 2018 Challenge：https://challenge.isic-archive.com/landing/2018/
- Task 1：https://challenge.isic-archive.com/landing/2018/45/
- Task 2：https://challenge.isic-archive.com/landing/2018/46/
- 2018 Task1-2 Training Collection：https://api.isic-archive.com/collections/63/
- Challenge Dataset Downloads：https://challenge.isic-archive.com/data/

数据使用前请同时检查 ISIC 官方数据许可与引用要求。

