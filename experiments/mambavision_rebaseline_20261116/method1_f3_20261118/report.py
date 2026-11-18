import csv
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root/"comparison.json").read_text())
lines = ["# F3 语义锚定跨尺度修正：单种子验证", "",
         "设备 RTX 4090（24 GB），训练 1886 张，验证 762 张，seed 42。三组各 100 轮，使用验证逐图 Dice 选择权重。开发过程只使用训练集和内部验证。", "",
         "| 模型 | 最佳轮 | 验证逐图 Dice | 初稿 Dice | 解码头参数 |",
         "|---|---:|---:|---:|---:|"]
for arm, row in summary["best_by_validation"].items():
    config=json.loads((root/arm/"config.json").read_text())
    coarse=row.get("coarse_val_macro_dice")
    coarse_text=f"{100*coarse:.4f}%" if coarse is not None else "—"
    lines.append(f"| {arm} | {row['epoch']} | {100*row['val_macro_dice']:.4f}% | {coarse_text} | {config['head_parameters']:,} |")
lines += ["","## 配对验证",""]
for arm, data in summary["paired_validation"].items():
    low, high=data["image_bootstrap_95pct_pp"]
    lines.append(f"- F3 相对 {arm} 的逐图平均 Dice 差为 {data['f3_minus_control_pp']:+.4f} 个百分点，图像配对 bootstrap 95% 区间 [{low:+.4f}, {high:+.4f}]。")
lines += ["", "两组锚定模型的参数和初始预测头构造相同，只改变指导修正的语义层。UPer64 是完整系统对照；其解码结构和监督均与候选有差异。重采样仅描述这批验证图，不反映训练种子波动。", "",
          "## 最佳权重的修正结果", "",
          "| 模型 | 初稿漏分救回像素 | 原正确前景被去除 | 原误报被去除 | 原正确背景新增误报 | 低前景分数内部漏分的救回数/总数 | 有此类漏分的图/762 | 有救回的图 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for arm in ("f3","f4"):
    with (root/arm/"best_validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    totals=summary["best_by_validation"][arm]["correction_counts"]
    hit=total=0
    for row in rows:
        n={k:int(row[k]) for k in ("coarse_fp","coarse_fn","fp","fn",*totals)}
        assert n["fn"] == n["coarse_fn"] - n["saved_fn"] + n["new_fn"]
        assert n["fp"] == n["coarse_fp"] - n["saved_fp"] + n["new_fp"]
        assert 0<=n["conf_interior_fixed"]<=n["conf_interior_fn"]<=n["conf_fn"]
        assert n["conf_interior_fixed"]<=n["conf_fn_fixed"]<=n["saved_fn"]
        total += n["conf_interior_fn"] > 0
        hit += n["conf_interior_fixed"] > 0
    for key, value in totals.items():
        assert sum(int(row[key]) for row in rows)==value
    c=totals
    lines.append(f"| {arm} | {c['saved_fn']:,} | {c['new_fn']:,} | {c['saved_fp']:,} | {c['new_fp']:,} | {c['conf_interior_fixed']:,}/{c['conf_interior_fn']:,} | {total} | {hit} |")
lines += ["", "这里的内部漏分要求初稿前景分数小于 0.1，像素属于标注前景且距离背景超过 8 像素。像素计数是补充诊断，病例权重与逐图 Dice 不同。端到端训练令两个初稿各自改变，跨模型的计数不能直接作共同初稿上的因果比较。", "",
          "验证明细保存在各组 best_validation.csv、best.pth、history.json；配对统计在 comparison.json。"]
dest=root/"RESULTS_zh.md"
temp=dest.with_suffix(".md.tmp")
temp.write_text("\n".join(lines)+"\n")
temp.replace(dest)
