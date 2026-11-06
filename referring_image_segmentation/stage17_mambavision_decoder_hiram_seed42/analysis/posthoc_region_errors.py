"""Post-hoc error locations for the already selected stage-17 checkpoints.

The boundary is the width-two, 5x5 square morphological band on the 256x256
ground-truth grid, matching the experiment's region-target construction. This
is descriptive validation analysis; ground truth is not used at inference.
"""
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT.parent
NAMES = ('paired_baseline', 'hiram')


def main():
    paths = {n: ROOT / 'results' / n / 'best_dice' for n in NAMES}
    ids = json.loads((paths[NAMES[0]] / 'ids.json').read_text())
    assert ids == json.loads((paths[NAMES[1]] / 'ids.json').read_text())
    masks = {f.stem.removesuffix('_segmentation').removesuffix('_mask'): f
             for f in (PROJECT / 'data/isic2018/val/masks').iterdir() if f.is_file()}
    probs = {n: np.load(paths[n] / 'probabilities.npy', mmap_mode='r') for n in NAMES}
    assert len(ids) == 808 and all(p.shape == (808, 256, 256) for p in probs.values())
    zones = {k: dict(pixels=0, baseline_errors=0, hiram_errors=0,
                     corrected=0, introduced=0)
             for k in ('boundary_5x5', 'foreground_interior', 'background_outside')}
    for i, sid in enumerate(ids):
        gt = np.asarray(Image.open(masks[sid]).convert('L').resize(
            (256, 256), Image.Resampling.NEAREST)) >= 128
        count = convolve(gt.astype(np.uint8), np.ones((5, 5), dtype=np.uint8),
                         mode='constant', cval=0)
        boundary = (count > 0) & (count < 25)
        partitions = dict(boundary_5x5=boundary, foreground_interior=gt & ~boundary,
                          background_outside=~gt & ~boundary)
        base_error = (probs[NAMES[0]][i] >= .5) != gt
        hiram_error = (probs[NAMES[1]][i] >= .5) != gt
        for name, region in partitions.items():
            row = zones[name]
            row['pixels'] += int(region.sum())
            row['baseline_errors'] += int((region & base_error).sum())
            row['hiram_errors'] += int((region & hiram_error).sum())
            row['corrected'] += int((region & base_error & ~hiram_error).sum())
            row['introduced'] += int((region & ~base_error & hiram_error).sum())
    for row in zones.values():
        row['net_new_errors'] = row['introduced'] - row['corrected']
        assert row['net_new_errors'] == row['hiram_errors'] - row['baseline_errors']
    summary = json.loads((HERE / 'summary.json').read_text())['comparisons']['hiram']
    assert sum(r['corrected'] for r in zones.values()) == summary['corrected_pixels']
    assert sum(r['introduced'] for r in zones.values()) == summary['introduced_errors']
    result = dict(scope='post-hoc descriptive comparison, 808 development-validation images',
                  split='ISIC2018 legacy development validation', resolution=[256, 256],
                  thresholds='prediction probability >= 0.5; GT gray >= 128',
                  boundary='square 5x5 binary dilation minus erosion, zero padding',
                  checkpoints='each model selected by highest validation macro Dice',
                  inference_uses_ground_truth=False, zones=zones)
    destination = HERE / 'posthoc_region_errors.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
