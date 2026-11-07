"""Offline integrity and per-image Dice/IoU audit for the published archive."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT/'publication/stage20_21_archive_manifest.json'
SPECS = [
    ('stage20_decoder_fusion_seed42', 'paired_baseline', 127, 'paired'),
    ('stage20_decoder_fusion_seed42', 'gate_only', 32, 'paired'),
    ('stage20_decoder_fusion_seed42', 'axis_bridge', 159, 'paired'),
    ('stage20_decoder_fusion_seed42/segman_adaptation', 'segman', 105, 'single'),
    ('stage21_hgm_bmd_seed42', 'hgm_bmd', 116, 'single'),
]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest=read_json(MANIFEST)
    for item in manifest['files']:
        path=ROOT/item['path']
        assert path.is_file() and path.stat().st_size==item['size'], item['path']
        assert digest(path)==item['sha256'], item['path']
    print('Published file hashes verified:',len(manifest['files']))
    for directory, model, expected_epoch, kind in SPECS:
        path=ROOT/directory
        done=read_json(path/'results/DONE.json')
        assert done['seed']==42 and done['epochs']==300 and not done['smoke_only']
        assert done['encoder_unchanged']
        updates=done['updates'][model] if kind=='paired' else done['updates']
        assert updates==9000
        selected=done['selectors'][model]['best_dice'] if kind=='paired' else done['selectors']['best_dice']
        assert selected['epoch']==expected_epoch
        summary=read_json(path/'analysis/summary.json')['models'][model]
        assert summary['selected_epoch']==expected_epoch
        assert abs(summary['macro']['dice']-selected['score'])<1e-12
        history=read_json(path/'results/history.json')
        assert len(history)==300 and [x['epoch'] for x in history]==list(range(1,301))
        scores=[(x['models'][model]['val']['macro']['dice'] if kind=='paired'
                 else x['val']['macro']['dice']) for x in history]
        assert max(scores)==selected['score'] and scores.index(max(scores))+1==expected_epoch
        rows=[]
        with (path/'analysis/per_image.csv').open(newline='') as f:
            rows=[row for row in csv.DictReader(f) if row['model']==model]
        assert len(rows)==808 and len({row['id'] for row in rows})==808
        for row in rows:
            tp,fp,fn,tn=(int(row[k]) for k in ('tp','fp','fn','tn'))
            assert min(tp,fp,fn,tn)>=0 and tp+fp+fn+tn==256*256
            dd=2*tp+fp+fn
            iden=tp+fp+fn
            assert abs(float(row['dice'])-(2*tp/dd if dd else 1.))<1e-12
            assert abs(float(row['iou'])-(tp/iden if iden else 1.))<1e-12
        mean_dice=math.fsum(float(row['dice']) for row in rows)/len(rows)
        mean_iou=math.fsum(float(row['iou']) for row in rows)/len(rows)
        assert abs(mean_dice-summary['macro']['dice'])<1e-12
        assert abs(mean_iou-summary['macro']['iou'])<1e-12
        print(f'{model:16} epoch={expected_epoch:3} n={len(rows)} macro_dice={mean_dice:.9f}')
    print('Archive integrity and per-image metrics verified.')


if __name__=='__main__':
    main()
