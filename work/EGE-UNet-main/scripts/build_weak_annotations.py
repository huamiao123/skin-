import os
import json
import numpy as np
from PIL import Image

ISIC_OFFICIAL_DIR = '/root/isic2018_official'
TASK1_GT_DIR = os.path.join(ISIC_OFFICIAL_DIR, 'ISIC2018_Task1_Training_GroundTruth')
TASK2_DIR = os.path.join(ISIC_OFFICIAL_DIR, 'ISIC2018_Task2_Training_GroundTruth_v3')
EXISTING_DATA = '/root/EGE-UNet-main/data/isic2018'
OUTPUT_DIR = os.path.join(EXISTING_DATA, 'weak_annotations')

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


def mask_to_tight_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return [xmin, ymin, xmax, ymax]


def choose_center_point(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))
    if mask[cy, cx] == 0:
        valid = np.where(mask > 0)
        idx = valid[0].shape[0] // 2
        cy, cx = int(valid[0][idx]), int(valid[1][idx])
    return {"x": cx, "y": cy}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_ids = get_sorted_isic_ids()
    train_imgs = sorted(os.listdir(os.path.join(EXISTING_DATA, 'train/images')))
    train_ids = [all_ids[i] for i in range(min(len(train_imgs), len(all_ids)))]

    task2_available = set()
    for f in os.listdir(TASK2_DIR):
        if f.endswith('.png'):
            task2_available.add(f.split('_')[1])

    weak_train = {}
    skipped_no_box = 0
    skipped_no_points = 0

    for idx, isic_id in enumerate(train_ids):
        if idx % 200 == 0:
            print(f"Processing {idx}/{len(train_ids)}...")

        gt_path = os.path.join(TASK1_GT_DIR, f'ISIC_{isic_id}_segmentation.png')
        if not os.path.exists(gt_path):
            skipped_no_box += 1
            continue

        les_mask_full = np.array(Image.open(gt_path).convert('L'))
        les_mask_256 = np.array(Image.fromarray(les_mask_full).resize((256, 256), Image.NEAREST))
        les_mask_256 = (les_mask_256 >= 128).astype(np.uint8)

        box = mask_to_tight_box(les_mask_256)
        if box is None:
            skipped_no_box += 1
            continue

        points = []
        if isic_id in task2_available:
            for attr in ATTRS:
                apath = os.path.join(TASK2_DIR, f'ISIC_{isic_id}_attribute_{attr}.png')
                if not os.path.exists(apath):
                    continue
                amask_full = np.array(Image.open(apath).convert('L'))
                amask_256 = np.array(Image.fromarray(amask_full).resize((256, 256), Image.NEAREST))
                amask_256 = (amask_256 >= 128).astype(np.uint8)

                if amask_256.sum() == 0:
                    continue

                point = choose_center_point(amask_256)
                if point is not None:
                    point["attribute"] = attr
                    points.append(point)

        if len(points) == 0:
            skipped_no_points += 1

        weak_train[isic_id] = {
            "box": box,
            "image_size": [256, 256],
            "points": points,
        }

    output_path = os.path.join(OUTPUT_DIR, 'weak_train.json')
    with open(output_path, 'w') as f:
        json.dump(weak_train, f, indent=2)

    print(f"\n生成完成: {output_path}")
    print(f"总样本: {len(weak_train)}")
    print(f"无 box: {skipped_no_box}")
    print(f"无 semantic points: {skipped_no_points}")
    print(f"有至少1个 point: {len(weak_train) - skipped_no_points}")

    n_points_dist = {}
    for k, v in weak_train.items():
        n = len(v["points"])
        n_points_dist[n] = n_points_dist.get(n, 0) + 1
    for k in sorted(n_points_dist.keys()):
        print(f"  {k} points: {n_points_dist[k]}")


if __name__ == '__main__':
    main()
