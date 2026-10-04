"""Read-only source/weight/data audit for the isolated RSI tail probe.

This command writes only the four new audit JSON files in the supplied new
output directory. It never scores test, trains, rewrites a manifest, or invokes
the historical mirror/threshold interfaces.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rsi.datasets import read_manifest
from rsi.models import ControlledModel
from rsi.runtime import atomic_json, environment, set_seed

REVIEW_COMMIT = 'abfb17bc9a175cdce2da82dfcebe827d1b5ba489'
TRAINING_COMMIT = '5b81bb1f812360049a89684dd43ce235b4487753'
BUNDLE = '36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb'
MANIFEST_SHA = '3d99874f14ae5f316eec36e21141043d75bd4fd17ccfb31e3174198dee269596'
EXPECTED_WEIGHTS = {
    'B': ('1703ca22a1c39afeca05ffd0c12fe01606f41d5af81442d20102f130cc2ca491', 22),
    'D0': ('6e0811712543811a10b6cde1ec29db65b7bb4f25142c6991f2738d3d5c073f0a', 7),
    'RSI-3': ('40c40abc702c65bb2f4b3381335796730c89eb4e152188b699dd6f3d0bb4b784', 1),
}
EXPECTED_COUNTS = {'M': {'train': (1471, 3102), 'val': (223, 471), 'test': (451, 948)},
                   'H': {'val': (47, 104)}, 'T1': {'val': (20, 49)}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def csv_rows(path):
    import csv
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def asset_check(old_run_root, manifest_sha):
    publication_path = PROJECT / 'publication/weights_manifest.json'
    publication = json.loads(publication_path.read_text())
    listed = {x['source_path']: x for x in publication['paths']}
    audits = []
    for run_name, (expected_hash, expected_epoch) in EXPECTED_WEIGHTS.items():
        root = old_run_root / 'seed17' / run_name
        done_path = root / 'DONE.json'
        done = json.loads(done_path.read_text())
        epoch_path = root / 'epoch_diagnostics.csv'
        epochs = csv_rows(epoch_path)
        assert [int(x['epoch']) for x in epochs] == list(range(1, 41)), run_name
        signature = done['signature']
        cp_path = root / 'best.pth'
        size = cp_path.stat().st_size
        actual_sha = sha(cp_path)
        assert size == 125077601 and actual_sha == expected_hash, run_name
        assert listed[str(cp_path)]['sha256'] == actual_sha
        assert listed[str(cp_path)]['bytes'] == size
        assert done['best_sha256'] == actual_sha and done['epochs'] == 40
        assert done['best_epoch'] == expected_epoch
        assert signature['manifest_hash'] == manifest_sha
        assert signature['source_bundle_hash'] == BUNDLE
        assert signature['seed'] == done['seed'] == 17
        config_path = root / 'config_resolved.json'
        config = json.loads(config_path.read_text())
        assert config['config'] == signature['config']
        state = torch.load(cp_path, map_location='cpu')
        assert state['seed'] == 17 and state['epoch'] == expected_epoch
        assert state['best_epoch'] == expected_epoch and state['manifest_hash'] == manifest_sha
        assert state['code']['commit'] == TRAINING_COMMIT
        assert state['code']['source_bundle_hash'] == BUNDLE
        source_files = state['code']['source_sha256']
        bundle = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
        assert bundle == BUNDLE
        source_checks = {f: sha(PROJECT / f) == expected for f, expected in source_files.items()}
        assert all(source_checks.values()), source_checks
        model = ControlledModel(pretrained=False, input_size=256)
        model.load_state_dict(state['model'], strict=True)
        finite = all(bool(torch.isfinite(v).all()) for v in state['model'].values() if v.is_floating_point())
        assert finite
        latest_path = root / 'latest.pth'
        latest_sha = sha(latest_path)
        assert latest_sha == done['latest_sha256'] == listed[str(latest_path)]['sha256']
        latest = torch.load(latest_path, map_location='cpu')
        assert latest['epoch'] == 40 and latest['best_epoch'] == expected_epoch
        assert latest['optimizer_step_attempts'] == latest['optimizer_steps'] == 7360
        assert int(epochs[-1]['optimizer_step_attempts']) == int(epochs[-1]['optimizer_steps']) == 7360
        assert all(int(x['amp_skipped_steps']) == 0 for x in epochs)
        assert all(float(x['anchor_max_absolute_error']) == 0 for x in epochs)
        best_row = max(epochs, key=lambda x: float(x['val_dice']))
        assert int(best_row['epoch']) == expected_epoch
        assert float(best_row['val_dice']) == done['best_val_dice'] == state['best_dice']
        for key in ['seed', 'stage', 'method', 'weight', 'q']:
            assert signature[key] == state[key] == config[key]
        assert state['init_checkpoint_hash'] == signature['init_checkpoint_hash']
        if run_name != 'B':
            assert state['init_checkpoint_hash'] == EXPECTED_WEIGHTS['B'][0]
        audits.append(dict(run=run_name, status='PASS', path=str(cp_path), sha256=actual_sha,
                           bytes=size, seed=17, stage=state['stage'], method=state['method'],
                           weight=state['weight'], best_epoch=expected_epoch,
                           budget_epochs=40, budget_completed=True,
                           final_epoch=latest['epoch'], optimizer_steps=latest['optimizer_steps'],
                           optimizer_attempts=latest['optimizer_step_attempts'], amp_skips=0,
                           checkpoint_read_success=True, strict_model_load_success=True,
                           finite_model_tensors=finite, model_state_keys=len(state['model']),
                           init_sha256=state['init_checkpoint_hash'], manifest_sha256=manifest_sha,
                           source_training_commit=TRAINING_COMMIT, source_bundle_hash=BUNDLE,
                           historical_source_file_match=source_checks,
                           done_path=str(done_path), done_sha256=sha(done_path),
                           config_path=str(config_path), config_sha256=sha(config_path),
                           epoch_log_path=str(epoch_path), epoch_log_sha256=sha(epoch_path),
                           latest_path=str(latest_path), latest_sha256=latest_sha,
                           environment=json.loads((root / 'environment.json').read_text()),
                           frozen_hash=state['frozen_hash'],
                           checkpoint_selection=state['selection_rule']))
        print(f'Asset PASS {run_name}: best epoch {expected_epoch}; complete budget 40', flush=True)
        del state, latest, model
    return dict(status='PASS', assets=audits, publication_manifest_path=str(publication_path),
                publication_manifest_sha256=sha(publication_path), weights_local_only=True,
                frozen_F_reuse_requires_new_forward_loss_gradient_export_acceptance=True)


def verify_image(item):
    path, row = item
    actual_sha = sha(path)
    assert actual_sha == row['image_sha256'], path
    with Image.open(path) as f:
        mode = f.mode
        a = np.asarray(f.convert('RGB'))
    assert a.shape[:2] == (int(row['original_h']), int(row['original_w'])), path
    pixel_sha = hashlib.sha256(a.tobytes()).hexdigest()
    assert pixel_sha == row['image_pixel_sha256'], path
    return dict(kind='image', path=path, sha256=actual_sha, pixel_sha256=pixel_sha,
                original_h=a.shape[0], original_w=a.shape[1], mode=mode, decode_success=True)


def verify_mask(item):
    path, row = item
    data = Path(path).read_bytes()
    actual_sha = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    assert actual_sha == row['mask_sha256'], path
    assert md5 == row['mask_md5'] == row['actual_mask_md5'], path
    with Image.open(path) as f:
        mode = f.mode
        a = np.asarray(f.convert('L'))
    assert a.shape == (int(row['original_h']), int(row['original_w'])), path
    values = np.unique(a)
    assert np.isin(values, [0, 1, 255]).all(), path
    return dict(kind='mask', path=path, sha256=actual_sha, md5=md5,
                h=a.shape[0], w=a.shape[1], mode=mode, binary_values=[int(v) for v in values],
                decode_success=True)


def data_check(manifest, file_audit, workers):
    manifest_sha = sha(manifest)
    assert manifest_sha == MANIFEST_SHA
    historical = json.loads(file_audit.read_text())
    assert historical['final_references_sha256'] == manifest_sha
    assert historical['file_audit_complete'] and not historical['file_errors']
    assert not historical['near_duplicate_unresolved_candidates']
    rows = read_manifest(manifest)
    counts = {}
    for subset in sorted({r['subset'] for r in rows}):
        counts[subset] = {}
        for split in ['train', 'val', 'test']:
            members = [r for r in rows if r['subset'] == subset and r['split'] == split]
            counts[subset][split] = dict(images=len({r['image_id'] for r in members}), references=len(members))
            if split in EXPECTED_COUNTS.get(subset, {}):
                images, refs = EXPECTED_COUNTS[subset][split]
                assert counts[subset][split] == dict(images=images, references=refs)
    images, masks = {}, {}
    for row in rows:
        assert row['audit_complete'] == 'true' and row['spatial_size_aligned'] == 'true' and not row['audit_error']
        images.setdefault(row['image_path'], row)
        masks.setdefault(row['mask_path'], row)
        assert images[row['image_path']]['image_sha256'] == row['image_sha256']
        assert masks[row['mask_path']]['mask_sha256'] == row['mask_sha256']
    checks, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(verify_image, x): x[0] for x in images.items()}
        jobs.update({pool.submit(verify_mask, x): x[0] for x in masks.items()})
        for n, fut in enumerate(as_completed(jobs), 1):
            try:
                checks.append(fut.result())
            except Exception as exc:
                errors.append(dict(path=jobs[fut], error=repr(exc)))
            if n % 500 == 0:
                print(f'Data identity/decode audit {n}/{len(jobs)}; errors={len(errors)}', flush=True)
    checks.sort(key=lambda x: (x['kind'], x['path']))
    m_val = [r for r in rows if r['subset'] == 'M' and r['split'] == 'val']
    groups = {r['group_id'] if r['group_id'] and r['group_id'].lower() != 'unknown' else r['image_id'] for r in m_val}
    return dict(status='PASS' if not errors else 'FAIL', file_audit_complete=not errors,
                scope='all_selected_M_H_T1', test_scoring_locked=True, test_model_scoring_performed=False,
                test_files_only_identity_decoded=True, manifest_path=str(manifest),
                final_references_sha256=manifest_sha, manifest_rows=len(rows),
                input_manifest_sha256=historical['input_manifest_sha256'],
                historical_audit_path=str(file_audit), historical_audit_sha256=sha(file_audit),
                historical_final_manifest_binding_verified=True, counts=counts,
                image_counts_by_subset_split={s:{sp:v['images'] for sp,v in d.items()} for s,d in counts.items()},
                reference_counts_by_subset_split={s:{sp:v['references'] for sp,v in d.items()} for s,d in counts.items()},
                unique_image_files=len(images), unique_mask_files=len(masks),
                decoded_image_count=sum(x['kind']=='image' for x in checks),
                decoded_mask_count=sum(x['kind']=='mask' for x in checks),
                group_bootstrap_units_M_val=len(groups), image_bootstrap_units_M_val=223,
                known_group_cross_split=historical['known_group_cross_split'],
                source_subset_note='H/T1 use M predictions with reference subsets; they are not independent cohorts.',
                selected_rows_unchanged=True, path_mapping_performed=False,
                errors=errors, checks=checks,
                rights=dict(raw_data_public_upload=False, source_case_panels_public_upload=False,
                            clinical_content_public_upload=False))


def run(args):
    output = args.output.resolve()
    old_root = args.old_run_root.resolve()
    assert output != old_root and old_root not in output.parents
    assert output.name == 'RSI-TailProbe-20261004'
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest = PROJECT / 'data_manifests/final_references.csv'
    audit = PROJECT / 'data_manifests/file_audit.json'
    paths = dict(schema_version=1, timestamp_utc=now, project=str(PROJECT), run_root=str(output),
                 review_commit=REVIEW_COMMIT, source_training_commit=TRAINING_COMMIT,
                 taskbook=str(args.taskbook.resolve()), taskbook_sha256=sha(args.taskbook),
                 manifest=str(manifest), original_manifest='/home/featurize/work/RSI_Implementation_20261003/data_manifests/final_references.csv',
                 manifest_sha256=sha(manifest), historical_audit=str(audit),
                 new_audit=str(output/'data_audit.json'), data_root='/home/featurize/rsi_data',
                 canonical_cache=str(output/'cache/canonical256'),
                 old_canonical_cache='/home/featurize/rsi_data/ima/cache256',
                 new_logits_cache=str(output/'cache/logits'),
                 teacher_B=str(old_root/'seed17/B/best.pth'),
                 historical_F_Mean=str(old_root/'seed17/D0/best.pth'),
                 historical_F_RSI=str(old_root/'seed17/RSI-3/best.pth'),
                 old_run_root_read_only=str(old_root),
                 new_group_run_template=str(output/'runs/seed17/{group}'),
                 path_mapping_required=False,
                 old_write_interfaces_used=False)
    atomic_json(output/'resolved_paths.json', paths)
    set_seed(17)
    torch.set_num_threads(4)
    env = environment()
    env.update(schema_version=1, timestamp_utc=now, platform=platform.platform(),
               gpu_driver=command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),
               gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
               gpu_compute_capability=list(torch.cuda.get_device_capability(0)),
               cuda_available=torch.cuda.is_available(),
               cudnn_version=torch.backends.cudnn.version(),
               cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
               cudnn_tf32=torch.backends.cudnn.allow_tf32,
               cudnn_benchmark=torch.backends.cudnn.benchmark,
               cudnn_deterministic=torch.backends.cudnn.deterministic,
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    x=torch.tensor([[1.,2.],[3.,4.]],device='cuda'); y=x@x
    assert y.cpu().tolist() == [[7.,10.],[15.,22.]]
    env['actual_cuda_tensor_matmul_success']=True
    atomic_json(output/'environment.json',env)
    assets=asset_check(old_root,sha(manifest))
    missing_names=['code_decision_audit.md','decision_numeric_audit.md','nearest_work_audit.md',
                   '当前交互方向_继续或停止的研究判断_2026-10-04.md']
    found=[]
    for base in [Path('/home/featurize/task'),Path('/home/featurize/work')]:
        for parent, dirs, names in os.walk(base):
            dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','node_modules','.venv','rsi_data','cache256'}]
            for name in missing_names:
                if name in names:
                    path=Path(parent)/name;found.append(dict(name=name,path=str(path),sha256=sha(path)))
    assets['windows_review_materials']=dict(found=found,
        missing=[dict(name=n,status='missing_not_blocking') for n in missing_names if n not in {x['name'] for x in found}],
        search_roots=['/home/featurize/task','/home/featurize/work'],
        unavailable_windows_root='D:\\Apaper\\paper\\27_RSI_Go_NoGo_20261004',
        taskbook_is_complete_authorized_execution_specification=True)
    assets['timestamp_utc']=now
    assets['historical_training_semantics']=dict(
        seed=17,input_size=256,epochs=40,effective_batch=8,microbatch=4,accumulation=2,
        workers=4,eval_batch=4,amp=True,old_evaluation_forward_autocast=True,
        objective_loss_outside_autocast_FP32=True,tf32=False,
        train_image_shuffle='torch DataLoader generator seed17 + epoch * 100003',
        augmentation_rng='sha256(seed:epoch:image_id); same geometry and colour augmentation across groups',
        optimizer='AdamW',weight_decay=1e-4,D_lr=1e-4,gradient_clip=1.,
        warmup_epochs=2,lr_schedule='historical LambdaLR warmup then cosine',
        last_accumulation_window_actual_image_weighting=True,
        exact_epoch_optimizer_attempts=184,total_optimizer_attempts=7360,
        FP32_eval_alignment_requires_separate_AMP_historical_reproduction=True)
    atomic_json(output/'asset_audit.json',assets)
    data=data_check(manifest,audit,args.workers)
    data['timestamp_utc']=now
    atomic_json(output/'data_audit.json',data)
    if data['errors']:
        raise RuntimeError(f"Data audit errors: {len(data['errors'])}")
    print(json.dumps(dict(P0='PASS',output=str(output),unique_images=data['unique_image_files'],
                          unique_masks=data['unique_mask_files'],counts=data['counts']),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/home/featurize/rsi_runs/RSI-TailProbe-20261004'))
    p.add_argument('--old-run-root',type=Path,default=Path('/home/featurize/rsi_runs/P2-IMA-M-v2'))
    p.add_argument('--taskbook',type=Path,default=Path('/home/featurize/task/皮肤病灶分割_RSI方向去留验证_AI执行任务书_2026-10-04.txt'))
    p.add_argument('--workers',type=int,default=8)
    run(p.parse_args())


if __name__=='__main__':
    main()
