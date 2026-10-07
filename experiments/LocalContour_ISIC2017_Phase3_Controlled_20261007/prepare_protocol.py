"""Record the official ISIC 2017 split without opening locked test images or GT."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from project_paths import ISIC2017,TASKBOOK

ROOT = Path(__file__).resolve().parent
DATA = ISIC2017
SRC = TASKBOOK

def main():
    (ROOT/'protocols').mkdir(exist_ok=True)
    (ROOT/'splits').mkdir(exist_ok=True)
    destination=ROOT/'protocols'/SRC.name
    if SRC.resolve()!=destination.resolve():shutil.copyfile(SRC,destination)
    ids = {}
    for split, expected in [('train',2000),('val',150),('test',600)]:
        images=sorted((DATA/split/'images').glob('ISIC_*.jpg'))
        masks=sorted((DATA/split/'masks').glob('ISIC_*_segmentation.png'))
        imgids={p.stem for p in images}
        maskids={p.stem.replace('_segmentation','') for p in masks}
        if len(imgids)!=expected or imgids!=maskids or len(images)!=expected or len(masks)!=expected:
            raise RuntimeError(f'Incomplete {split} split')
        ids[split]=imgids
        (ROOT/'splits'/f'{split}.txt').write_text(''.join(x+'\n' for x in sorted(imgids)))
    if any(ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise RuntimeError('Split overlap')
    files={f'{split}.txt':hashlib.sha256((ROOT/'splits'/f'{split}.txt').read_bytes()).hexdigest() for split in ids}
    statement={
      'split':'official ISIC2017 Task1, fixed before phase-2 training',
      'counts':{k:len(v) for k,v in ids.items()},'sha256':files,
      'test_image_bytes_opened':False,'test_gt_bytes_opened':False,
      'cnn':'public MSGU-Net checkpoint 3a7fb064383cc68faf33b76561379b69a84141cf51e94679dc31e3a3e66a515d',
      'cnn_source_training':'author script combined official train+val and shuffled 70/30; therefore val was potentially used for checkpoint selection; official test is absent from source training script',
      'cnn_model_frozen':True,
      'development_2017_val_exposure':'all 150 previously used in prior contour probe; phase-2 candidate diagnostics are development only',
      'phase2_repairer_training':'official train only; val for tuning; official test stays locked until method frozen',
      'phase2_primary_independent_test_claim':'only official test, and only if no earlier test access is found',
      'historical_test_scoring_known':False,
      'weights_publication':'excluded from GitHub',
    }
    (ROOT/'protocols'/'split_provenance.json').write_text(json.dumps(statement,indent=2,ensure_ascii=False))
    print(json.dumps(statement,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
