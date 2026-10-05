"""Publish training metadata and curves without any model tensors."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FAMILIES=('local','local_large','local8','medium32','global','local8_gate','global_gate')
SEEDS=(17,23,42)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    output=ROOT/'results/training_records';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in FAMILIES:
        for seed in SEEDS:
            folder=ROOT/'models'/f'{family}_seed{seed}'
            done=json.loads((folder/'DONE.json').read_text())
            if done['status']!='PASS' or done['seed']!=seed:raise RuntimeError(f'Incomplete model {folder}')
            weight=folder/'best.pth';curve=folder/'training_log.csv'
            if not weight.is_file() or not curve.is_file():raise RuntimeError(f'Missing local provenance {folder}')
            row={'family':family,'seed':seed,'parameters':done['parameters'],
                 'best_epoch':done['best_epoch'],'epochs_run':done['epochs_run'],
                 'source_git_commit':done['source_git_commit'],
                 'checkpoint_sha256_local_only':sha(weight),
                 'checkpoint_bytes_local_only':weight.stat().st_size,
                 'checkpoint_publication':'excluded',
                 'training_log_sha256':sha(curve),
                 'official_test_images_opened':done.get('official_test_images_opened',done.get('test_images_opened')),
                 'best_val_objective':done.get('best_val_loss',done.get('best_val_candidate_loss')),
                 'optimizer':done.get('optimizer'),'batch_size':done.get('batch_size')}
            rows.append(row)
            shutil.copyfile(curve,output/(f'{family}_seed{seed}_training_log.csv'))
    with (output/'training_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summary={'model_runs':len(rows),'families':FAMILIES,'seeds':SEEDS,
             'total_training_epochs':sum(int(r['epochs_run']) for r in rows),
             'weights_in_training_records':False,
             'source_checkpoints_local_only':True,
             'official_test_images_opened_any':any(r['official_test_images_opened'] not in (None,0,False) for r in rows)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
