"""GPU resident images, paired geometry, and deterministic epoch augmentation."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

DATA = Path(__file__).resolve().parent.parent / 'data' / 'isic2018'


class ImageData:
    def __init__(self, root=DATA):
        self.images, self.masks, self.ids, self.audit = {}, {}, {}, []
        for split, expected in [('train', 1886), ('val', 808)]:
            image_paths = sorted((root / split / 'images').glob('*'))
            masks = {p.stem.removesuffix('_segmentation').removesuffix('_mask'): p
                     for p in (root / split / 'masks').glob('*') if p.is_file()}
            image_paths = [p for p in image_paths if p.is_file()]
            assert len(image_paths) == expected and len(masks) == expected
            assert {p.stem for p in image_paths} == set(masks)

            def read(p):
                q = masks[p.stem]
                image = np.asarray(Image.open(p).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)).copy()
                mask = np.asarray(Image.open(q).convert('L').resize((256, 256), Image.Resampling.NEAREST)).copy()
                return image, mask, dict(split=split, id=p.stem, image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                         decoded_image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                         mask_sha256=hashlib.sha256(q.read_bytes()).hexdigest())
            with ThreadPoolExecutor(8) as pool:
                rows = list(pool.map(read, image_paths))
            self.images[split] = torch.from_numpy(np.stack([r[0] for r in rows])).permute(0, 3, 1, 2).contiguous().cuda()
            self.masks[split] = torch.from_numpy(np.stack([r[1] for r in rows])).unsqueeze(1).cuda()
            self.ids[split] = [p.stem for p in image_paths]
            self.audit.extend(r[2] for r in rows)
        # Legacy files are numbered separately inside each split (e.g. 0.png).
        # Check decoded image content, not these split-local filenames.
        hashes = {split:{r['decoded_image_sha256'] for r in self.audit if r['split']==split}
                  for split in ['train','val']}
        assert not (hashes['train'] & hashes['val']), 'Exact image leakage across splits'
        self.mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)

    def geometry(self, epoch):
        # Draw all geometry before batching so encoder batch size cannot alter augmentation.
        g = torch.Generator().manual_seed(42 + epoch * 100003)
        n = len(self.ids['train'])
        order = torch.randperm(n, generator=g)
        h = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
        v = torch.where(torch.rand(n, generator=g) < .5, -1., 1.)
        rotate = torch.rand(n, generator=g) < .5
        angle = torch.rand(n, generator=g) * (2 * torch.pi) * rotate
        matrix = torch.zeros(n, 2, 3)
        matrix[:, 0, 0] = angle.cos() * h
        matrix[:, 0, 1] = -angle.sin() * h
        matrix[:, 1, 0] = angle.sin() * v
        matrix[:, 1, 1] = angle.cos() * v
        return order.cuda(), matrix.cuda()

    def batch(self, split, indices, matrices=None):
        images = self.images[split][indices].float() / 255.
        masks = self.masks[split][indices].float() / 255.
        if matrices is not None:
            grid = F.affine_grid(matrices[indices], images.shape, align_corners=False)
            images = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            masks = F.grid_sample(masks, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        return (images - self.mean) / self.std, masks

    def save_audit(self, path):
        text = json.dumps(self.audit, sort_keys=True)
        path.write_text(json.dumps(dict(root=str(DATA), members=self.audit,
                          manifest_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=2) + '\n')
