import json
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DiagDataset(Dataset):
    def __init__(self, data_path, weak_json_path, config, is_train=True):
        self.data_path = data_path
        self.config = config
        self.is_train = is_train

        with open(weak_json_path, 'r') as f:
            self.weak_annotations = json.load(f)

        self.image_ids = []
        img_dir = os.path.join(data_path, 'train' if is_train else 'val', 'images')
        all_ids = self._get_all_ids(data_path)
        for fname in sorted(os.listdir(img_dir)):
            img_idx = int(os.path.splitext(fname)[0])
            if img_idx < len(all_ids):
                isic_id = all_ids[img_idx]
                if isic_id in self.weak_annotations and len(self.weak_annotations[isic_id]["points"]) > 0:
                    self.image_ids.append((fname, isic_id))

        self.point_radius = 2

    @staticmethod
    def _get_all_ids(data_path):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.build_weak_annotations import get_sorted_isic_ids
        return get_sorted_isic_ids()

    def __len__(self):
        return len(self.image_ids)

    def _build_point_mask(self, points, h, w, radius=2):
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in points:
            x = int(round(p["x"]))
            y = int(round(p["y"]))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
            x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
            mask[y_min:y_max, x_min:x_max] = 1
        return mask

    def _build_weak_masks(self, box, point_mask, h, w):
        xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w - 1))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h - 1))

        outside_box = np.ones((h, w), dtype=np.bool_)
        outside_box[ymin:ymax + 1, xmin:xmax + 1] = False

        positive = point_mask.astype(np.bool_)
        box_mask = np.zeros((h, w), dtype=np.bool_)
        box_mask[ymin:ymax + 1, xmin:xmax + 1] = True
        unknown = box_mask & (~positive)

        return positive, outside_box, unknown, box

    def __getitem__(self, idx):
        fname, isic_id = self.image_ids[idx]
        ann = self.weak_annotations[isic_id]
        box = ann["box"]
        points = ann["points"]

        split = 'train' if self.is_train else 'val'
        img_path = os.path.join(self.data_path, split, 'images', fname)
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float64)

        h, w = img.shape[:2]
        point_mask = self._build_point_mask(points, h, w, radius=self.point_radius)
        pos_mask, outside_box, unknown_mask, box_coords = self._build_weak_masks(box, point_mask, h, w)

        if self.is_train:
            if random.random() < 0.5:
                img = np.fliplr(img)
                pos_mask = np.fliplr(pos_mask)
                outside_box = np.fliplr(outside_box)
                unknown_mask = np.fliplr(unknown_mask)
                point_mask = np.fliplr(point_mask)

            if random.random() < 0.5:
                img = np.flipud(img)
                pos_mask = np.flipud(pos_mask)
                outside_box = np.flipud(outside_box)
                unknown_mask = np.flipud(unknown_mask)
                point_mask = np.flipud(point_mask)

            if random.random() < 0.5:
                k = random.choice([0, 1, 2, 3])
                img = np.rot90(img, k)
                pos_mask = np.rot90(pos_mask, k)
                outside_box = np.rot90(outside_box, k)
                unknown_mask = np.rot90(unknown_mask, k)
                point_mask = np.rot90(point_mask, k)

        mean, std = 157.561, 26.706
        img = (img - mean) / std
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.
        img = img.astype(np.float32)

        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        pos_t = torch.from_numpy(pos_mask.copy()).unsqueeze(0)
        neg_t = torch.from_numpy(outside_box.copy()).unsqueeze(0)
        unk_t = torch.from_numpy(unknown_mask.copy()).unsqueeze(0)
        box_t = torch.tensor(box_coords, dtype=torch.float32)

        result = {
            "image": img_t,
            "point_mask": pos_t,
            "outside_box_mask": neg_t,
            "unknown_mask": unk_t,
            "box_coords": box_t,
            "image_id": isic_id,
        }

        if self.config.enable_consistency:
            noise = torch.randn_like(img_t) * 5.0
            img_s = img_t + noise
            img_s = torch.clamp(img_s, 0, 255)
            result["image_weak"] = img_t.clone()
            result["image_strong"] = img_s
            del result["image"]

        return result
