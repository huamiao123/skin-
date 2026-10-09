import os
import numpy as np
from PIL import Image
from collections import Counter

ISIC_OFFICIAL_DIR = '/root/isic2018_official'
TASK1_GT_DIR = os.path.join(ISIC_OFFICIAL_DIR, 'ISIC2018_Task1_Training_GroundTruth')
TASK2_DIR = os.path.join(ISIC_OFFICIAL_DIR, 'ISIC2018_Task2_Training_GroundTruth_v3')
EXISTING_DATA = '/root/EGE-UNet-main/data/isic2018'

ATTRS = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cyst",
    "globules",
]

def get_sorted_isic_ids():
    ids = []
    for f in os.listdir(TASK1_GT_DIR):
        if f.endswith('_segmentation.png'):
            ids.append(f.split('_')[1])
    return sorted(ids)

def main():
    all_ids = get_sorted_isic_ids()
    print(f"官方 Task1 GT 总数: {len(all_ids)}")

    train_imgs = sorted(os.listdir(os.path.join(EXISTING_DATA, 'train/images')))
    print(f"现有数据: train={len(train_imgs)}")

    train_ids = [all_ids[i] for i in range(min(len(train_imgs), len(all_ids)))]
    if len(train_imgs) > len(all_ids):
        train_ids += [f'extra_{i}' for i in range(len(train_imgs) - len(all_ids))]

    task2_available = set()
    for f in os.listdir(TASK2_DIR):
        if f.endswith('.png'):
            task2_available.add(f.split('_')[1])
    print(f"官方 Task2 属性图像数: {len(task2_available)}")

    train_with_task2 = [iid for iid in train_ids if iid in task2_available]
    print(f"Train 中有 Task2 属性的图像: {len(train_with_task2)}/{len(train_ids)}")

    attr_counts = Counter()
    files_in_dir = os.listdir(TASK2_DIR)
    for f in files_in_dir:
        if not f.endswith('.png'):
            continue
        for attr in ATTRS:
            if f'attribute_{attr}.png' in f:
                attr_counts[attr] += 1

    print(f"\n各 attribute 存在次数 (文件数):")
    for attr in ATTRS:
        print(f"  {attr}: {attr_counts[attr]}")

    has_any_nonzero = 0
    attr_nonzero = Counter()
    for isic_id in train_ids[:500]:
        if isic_id not in task2_available:
            continue
        has = False
        for attr in ATTRS:
            apath = os.path.join(TASK2_DIR, f'ISIC_{isic_id}_attribute_{attr}.png')
            if os.path.exists(apath):
                mask = np.array(Image.open(apath).convert('L'))
                if mask.max() >= 128:
                    has = True
                    attr_nonzero[attr] += 1
        if has:
            has_any_nonzero += 1

    sample_total = min(500, len(train_ids))
    print(f"\n前500张中至少有一个非空attribute: {has_any_nonzero}/{sample_total}")
    print(f"各 attribute 非空次数 (前500张):")
    for attr in ATTRS:
        print(f"  {attr}: {attr_nonzero[attr]}")

    print(f"\n=== Go/No-Go 判断 ===")
    match_rate = len(train_with_task2) / len(train_ids) if train_ids else 0
    nonzero_rate = has_any_nonzero / sample_total if sample_total else 0
    print(f"Task2 匹配率: {match_rate*100:.1f}%")
    print(f"非空 attribute 占比: {nonzero_rate*100:.1f}%")

    if match_rate > 0.8 and nonzero_rate > 0.5:
        print("GO: 数据满足条件，可进入训练")
    else:
        print("WARNING: 需检查数据覆盖")

if __name__ == '__main__':
    main()
