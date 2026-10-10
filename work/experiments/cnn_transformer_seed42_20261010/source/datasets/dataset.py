from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        split = 'train' if train else 'val'
        image_dir = Path(path_Data) / split / 'images'
        mask_dir = Path(path_Data) / split / 'masks'
        self.data = self._join_by_id(image_dir, mask_dir)
        self.transformer = config.train_transformer if train else config.test_transformer

    @staticmethod
    def _sample_id(path):
        stem = Path(path).stem
        for suffix in ('_segmentation', '_mask'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        if not stem:
            raise ValueError(f'empty sample id for {path}')
        return stem

    @classmethod
    def _index_files(cls, directory, kind):
        if not directory.is_dir():
            raise FileNotFoundError(f'{kind} directory does not exist: {directory}')
        result = {}
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            if path.stat().st_size == 0:
                raise ValueError(f'empty {kind} file: {path}')
            sample_id = cls._sample_id(path)
            if sample_id in result:
                raise ValueError(f'duplicate {kind} id {sample_id}: {result[sample_id]} and {path}')
            result[sample_id] = path
        if not result:
            raise ValueError(f'no {kind} files found in {directory}')
        return result

    @classmethod
    def _join_by_id(cls, image_dir, mask_dir):
        images = cls._index_files(image_dir, 'image')
        masks = cls._index_files(mask_dir, 'mask')
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f'image/mask id mismatch: missing_masks={missing_masks[:10]}, '
                f'missing_images={missing_images[:10]}'
            )
        return [(str(images[key]), str(masks[key])) for key in sorted(images)]
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
        
    
