"""Audit repaired train/val-only inputs, training records and new development readouts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from candidate_layout import CANDIDATE_COUNT,ZERO_INDEX

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'


def read_csv(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    errors=[]
    feature=json.loads((CACHE/'COMPLETE.json').read_text())
    prior=json.loads((ROOT/'historical_pre_audit/cache_provenance/feature_cache_COMPLETE.json').read_text())
    ids={part:set((ROOT/'splits'/f'{part}.txt').read_text().splitlines()) for part in ('train','val','test')}
    if [len(ids[part]) for part in ('train','val','test')]!=[2000,150,600]:errors.append('split counts')
    if any(ids[a]&ids[b] for a,b in (('train','val'),('train','test'),('val','test'))):errors.append('split overlap')
    if feature.get('gt_shape_verified_for_all')!=2150 or feature['records']!=prior['records']:
        errors.append('fresh GT manifest / record order')
    if any(feature[key]!=prior[key] for key in ('cnn_sha256','cnn_state_hash')):
        errors.append('frozen CNN hash changed')
    if feature.get('test_image_or_gt_opened') is not False:errors.append('feature test exposure')
    candidate=json.loads((CACHE/'candidates_R32/COMPLETE.json').read_text())
    if candidate.get('gt_distance_label_source')!='complete 2D official GT mask':
        errors.append('fresh candidate GT provenance')
    points=np.load(CACHE/'candidates_R32/points.npy',mmap_mode='r')
    source=np.load(CACHE/'candidates_R32/source_points.npy',mmap_mode='r')
    valid=np.load(CACHE/'candidates_R32/valid.npy',mmap_mode='r')
    has=np.load(CACHE/'candidates_R32/has_contour.npy',mmap_mode='r').astype(bool)
    if points.shape!=(2150,256,CANDIDATE_COUNT,2) or not np.array_equal(points[:,:,ZERO_INDEX],source):
        errors.append('zero-offset geometry')
    if not valid[has,:,ZERO_INDEX].all() or np.any(valid[~has]):errors.append('zero-offset validity')
    if candidate.get('test_image_or_gt_opened') is not False:errors.append('candidate test exposure')
    training=read_csv(ROOT/'results/training_records/training_manifest.csv')
    if len(training)!=21:errors.append('training run count')
    for row in training:
        name=f"{row['family']}_seed{row['seed']}"
        folder=ROOT/'models'/name
        done=json.loads((folder/'DONE.json').read_text())
        log=folder/'training_log.csv';weight=folder/'best.pth';scores=folder/'candidate_scores.npy'
        if sha(log)!=row['training_log_sha256'] or sha(weight)!=row['checkpoint_sha256_local_only']:
            errors.append('training hash '+name)
        logs=read_csv(log)
        if len(logs)!=8 or int(done['best_epoch'])!=int(min(logs,key=lambda r:float(r['val_loss']))['epoch']):
            errors.append('best epoch '+name)
        if np.load(scores,mmap_mode='r').shape!=(2150,256,CANDIDATE_COUNT):
            errors.append('candidate score shape '+name)
        if row['family'].endswith('_gate') and np.load(folder/'gate_logits.npy',mmap_mode='r').shape!=(2150,256):
            errors.append('gate score shape '+name)
        commit=row['source_git_commit']
        source_text=subprocess.check_output(['git','show',f'{commit}:train_local.py'],cwd=ROOT,text=True)
        gate_text=subprocess.check_output(['git','show',f'{commit}:train_gated_refiner.py'],cwd=ROOT,text=True)
        if 'torch.full_like(fixable,ZERO_INDEX' not in source_text or 'torch.full_like(repair,ZERO_INDEX' not in gate_text:
            errors.append('model trained from uncorrected source '+name)
        if done.get('official_test_images_opened',done.get('test_images_opened',0)) not in (0,False,None):
            errors.append('model test exposure '+name)
    expected=ids['val']
    for seed in (17,23,42):
        chosen=read_csv(ROOT/'results/transformer'/f'chosen_keep_seed{seed}/per_image.csv')
        if len(chosen)!=1050 or {r['image_id'] for r in chosen}!=expected:
            errors.append('chosen readout '+str(seed))
        if any(r['model'] not in ('CNN','G1_zero_reconstruction','local','local_large','local8','medium32','global') for r in chosen):
            errors.append('chosen model names '+str(seed))
        gate=read_csv(ROOT/'results/transformer'/f'gated_seed{seed}/per_image.csv')
        if len(gate)!=450 or {r['image_id'] for r in gate}!=expected:errors.append('gate readout '+str(seed))
    main_summary=json.loads((ROOT/'results/phase2_summary/summary.json').read_text())
    if main_summary.get('official_test_images_or_GT_opened')!=0 or main_summary.get('n_images')!=100:
        errors.append('summary scope')
    for model,record in main_summary['model_means'].items():
        values=[r['dice'] for r in record['per_seed']]
        if abs(float(np.mean(values))-record['dice_mean'])>1e-12:errors.append('main Dice arithmetic '+model)
    for seed in (17,23,42):
        for family,label,path in (
          ('local','StrongLocal',ROOT/f'results/local/seed{seed}/summary.json'),
          ('local_large','LocalLarge',ROOT/f'results/transformer/local_large/seed{seed}/summary.json'),
          ('global','GlobalTransformer',ROOT/f'results/transformer/global/seed{seed}/summary.json')):
            dp=json.loads(path.read_text())
            grid=read_csv(path.with_name('dp_grid.csv'))
            best=max(grid,key=lambda row:float(row['validation_dice']))
            if len(grid)!=21 or abs(float(best['validation_dice'])-dp['methods'][label+'_DP']['dice'])>1e-12:
                errors.append('DP readout '+family+str(seed))
            if dp.get('test_images_opened')!=0:errors.append('DP test exposure '+family+str(seed))
    cases=json.loads((ROOT/'results/phase2_summary/case_panels/selection.json').read_text())
    for group,chosen in cases['selected_ids'].items():
        if len(chosen)!=len(list((ROOT/'results/phase2_summary/case_panels'/group).glob('*.png'))):
            errors.append('case panel count '+group)
    result={'status':'PASS_REPAIRED_DEVELOPMENT_AUDIT' if not errors else 'FAIL',
            'errors':errors,'train_images':2000,'val_images':150,'official_test_images_or_GT_opened':0,
            'training_runs':len(training),'DP_readouts':9,'all_weight_files_local_only':True,
            'main_readout_images':100,'independent_confirmation':False,
            'audit_limits':'Checks source commits, cache geometry, logs, checkpoint hashes and archived readout arithmetic; does not prove a new independent clinical generalization result.'}
    out=ROOT/'results/repair_validation/audit.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if errors:raise RuntimeError('Repaired audit failed')


if __name__=='__main__':main()
