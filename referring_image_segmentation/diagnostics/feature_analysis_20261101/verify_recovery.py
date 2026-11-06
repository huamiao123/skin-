"""Independently check recovered masks and the intact baseline against GT."""
import json
import numpy as np
from PIL import Image
from diagnose import OUT, PROJECT, S17, save_json


def main():
    ids=json.loads((S17/'results/paired_baseline/best_dice/ids.json').read_text())
    maskdir=PROJECT/'data/isic2018/val/masks'
    gtfiles={p.stem.removesuffix('_segmentation').removesuffix('_mask'):p for p in maskdir.iterdir() if p.is_file()}
    baseline=np.load(S17/'results/paired_baseline/best_dice/probabilities.npy',mmap_mode='r')
    packed={n:np.load(OUT/'recovered_masks'/f'stage17_{n}_bitpacked_masks.npy',mmap_mode='r')
            for n in ('region_only','hiram')}
    stored={n:np.load(S17/'results'/n/'best_dice/counts.npy')
            for n in ('paired_baseline','region_only','hiram')}
    actual={n:[] for n in stored}
    for i,identifier in enumerate(ids):
        gt=np.asarray(Image.open(gtfiles[identifier]).convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
        predictions={'paired_baseline':baseline[i]>=.5,
                     **{n:np.unpackbits(a[i],axis=-1).astype(bool) for n,a in packed.items()}}
        for n,p in predictions.items():
            actual[n].append([(p & gt).sum(), (p & ~gt).sum(), (~p & gt).sum(), (~p & ~gt).sum()])
    result={n:dict(all_808_per_image_counts_match=bool(np.array_equal(np.asarray(actual[n]),stored[n]))) for n in stored}
    assert all(x['all_808_per_image_counts_match'] for x in result.values())
    save_json('recovery_validation.json',dict(seed=42,n=len(ids),models=result,
      note='Counts matching does not prove bitwise equality to unavailable historical masks; these are checkpoint-regenerated masks.'))
    print(result)


if __name__=='__main__':main()
