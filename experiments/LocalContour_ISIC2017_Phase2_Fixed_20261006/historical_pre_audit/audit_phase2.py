"""Final read-only numerical and provenance audit before deleting temporary cache."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from project_paths import PHASE1

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    errors=[]
    split=json.loads((ROOT/'protocols/split_provenance.json').read_text())
    ids={s:set((ROOT/'splits'/f'{s}.txt').read_text().splitlines()) for s in ('train','val','test')}
    for s,n in [('train',2000),('val',150),('test',600)]:
        if len(ids[s])!=n or sha(ROOT/'splits'/f'{s}.txt')!=split['sha256'][f'{s}.txt']:
            errors.append('split integrity '+s)
    if ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']:
        errors.append('split overlap')
    feature=json.loads((CACHE/'COMPLETE.json').read_text())
    old_head=json.loads((PHASE1/'boundary_head/DONE.json').read_text())
    if feature['cnn_state_hash']!=old_head['cnn_state_hash_before']:errors.append('CNN state hash drift')
    train_ids=[r['image_id'] for r in feature['records'] if r['split']=='train']
    val_records=[r for r in feature['records'] if r['split']=='val']
    if set(train_ids)!=ids['train'] or {r['image_id'] for r in val_records}!=ids['val']:
        errors.append('cache/split mismatch')
    pred=np.load(CACHE/'prediction.npy',mmap_mode='r')
    gt=np.load(CACHE/'gt.npy',mmap_mode='r')
    candidate=json.loads((CACHE/'candidates_R32/COMPLETE.json').read_text())
    points=np.load(CACHE/'candidates_R32/points.npy',mmap_mode='r')
    source=np.load(CACHE/'candidates_R32/source_points.npy',mmap_mode='r')
    valid=np.load(CACHE/'candidates_R32/valid.npy',mmap_mode='r')
    distance=np.load(CACHE/'candidates_R32/gt_distance.npy',mmap_mode='r')
    if candidate['status']!='PASS' or candidate.get('gt_distance_label_source') not in ('complete 2D official GT mask','repaired complete 2D official GT mask'):
        errors.append('candidate label repair incomplete')
    if not np.array_equal(points[:,:,32,:],source):errors.append('mandatory zero candidate differs from source')
    if not np.all(valid[:,:,32]):
        # A single empty-prediction case is expected to have no contour/candidates.
        has=np.load(CACHE/'candidates_R32/has_contour.npy',mmap_mode='r').astype(bool)
        if not np.all(valid[has,:,32]) or np.any(valid[~has]):errors.append('zero candidate validity mismatch')
    if int(np.min(np.asarray(gt).sum(axis=(1,2))))<=0:errors.append('zero-area GT')
    if not np.isfinite(np.asarray(distance[:10])).all():errors.append('nonfinite GT-distance target')
    prediction_changed=[];GT_changed=[]
    for rec in val_records:
        i=rec['index']
        with np.load(PHASE1/'results/cases'/(rec['image_id']+'.npz')) as old:
            prediction_changed.append(int(np.count_nonzero(pred[i].astype(bool)!=old['masks_G0'])))
            GT_changed.append(int(np.count_nonzero(gt[i].astype(bool)!=old['gt_mask'])))
    if any(GT_changed):errors.append('GT differs from Phase1')
    old3=[r for r in read(ROOT/'results/candidates/per_image.csv') if r['config']=='old_3']
    maxdiff=max(abs(float(r['oracle_dice'])-float(r['historical_g4_dice'])) for r in old3)
    if maxdiff>0:errors.append('historical G4 reproduction differs')
    training=json.loads((ROOT/'results/training_records/summary.json').read_text())
    if training['model_runs']!=21 or training['weights_in_training_records'] or training['official_test_images_opened_any']:
        errors.append('training record incomplete or test exposure')
    final=json.loads((ROOT/'results/phase2_summary/summary.json').read_text())
    if final['decision']['status']!='STOP_GLOBAL_PRIMARY_AFTER_DEVELOPMENT' or final['n_images']!=100:
        errors.append('final stage-C decision mismatch')
    gate=[]
    for seed in (17,23,42):
        result=json.loads((ROOT/'results/transformer'/f'gated_seed{seed}'/'summary.json').read_text())
        gate.append({'seed':seed,'local_threshold':result['models']['local8_gate']['threshold'],
                     'global_threshold':result['models']['global_gate']['threshold'],
                     'local_moved':result['models']['local8_gate']['moved_fraction'],
                     'global_moved':result['models']['global_gate']['moved_fraction']})
        if len(read(ROOT/'results/transformer'/f'chosen_keep_seed{seed}'/'per_image.csv'))!=900:
            errors.append('chosen keep rows '+str(seed))
        if len(read(ROOT/'results/transformer'/f'gated_seed{seed}'/'per_image.csv'))!=450:
            errors.append('gated rows '+str(seed))
    cases=json.loads((ROOT/'results/phase2_summary/case_panels/selection.json').read_text())
    if cases['counts']!={'A':14,'B':0,'C':3,'D':30,'E':2}:errors.append('case count mismatch')
    audit={'status':'PASS' if not errors else 'FAIL','errors':errors,
           'split_counts':{s:len(v) for s,v in ids.items()},
           'training_rows':training['model_runs'],'phase2_val100_images':100,
           'phase1_val150_GT_changed_pixels':sum(GT_changed),
           'phase1_val150_prediction_changed_cases':sum(v>0 for v in prediction_changed),
           'phase1_val150_prediction_changed_pixels':sum(prediction_changed),
           'phase1_val150_prediction_max_changed_pixels_per_image':max(prediction_changed),
           'difference_explanation':'same frozen CNN state; Phase1 combined cached and separate inference, Phase2 recomputed every image. Exact numerical cause of 405 changed threshold pixels was not isolated; every Phase2 method shares the recomputed CNN baseline',
           'historical_old3_G4_max_absolute_dice_difference':maxdiff,
           'gated_thresholds_and_movement':gate,
           'official_test_image_or_GT_bytes_opened_by_phase2':False,
           'independent_test_performance_claim':False,
           'temporary_cache_ready_to_delete':not errors}
    out=ROOT/'results/phase2_summary/audit.json';out.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if errors:raise RuntimeError('Phase2 audit failed')

if __name__=='__main__':main()
