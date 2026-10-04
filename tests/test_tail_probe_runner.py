"""CPU-only trainer control checks; fixtures are not research data/results.

The actual 1471-image budget and 184-step epoch control are exercised with a
tiny CPU scalar model. Each run is intentionally interrupted after a few
epochs. No CUDA execution, actual model training or test scoring occurs.
"""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from tools import tail_probe_training as training


class TinyImageDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.epoch = 0
        self.count = 1471 if split == 'train' else 223

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        refs = [{'reference_id': f'{index}:a'}, {'reference_id': f'{index}:b'}]
        mask = torch.tensor([index % 2, (index + 1) % 3 == 0], dtype=torch.float32).reshape(2, 1, 1, 1)
        return dict(image=torch.full((3, 1, 1), (index % 7 + 1) / 7), masks=mask,
                    rater_present=torch.ones(2, dtype=torch.bool), pixel_valid=torch.ones(1, 1, 1),
                    image_id=f'image-{index:04}', group_id=f'group-{index:04}', references=refs,
                    geometry=dict(original_h=1, original_w=1, resized_h=1, resized_w=1,
                                  top=0, left=0, size=1), subset='M')


class TinyTailModel(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.register_buffer('teacher', torch.tensor(0.125))

    def trainable_parameters(self):
        return (self.weight,)

    def trainable_parameter_names(self):
        return ['weight']

    def forward_reference(self, image):
        return image[:, :1] * self.teacher

    def forward_details(self, image, valid):
        reference = self.forward_reference(image).detach()
        return dict(z0=reference, z1=reference + image[:, :1] * self.weight,
                    message=image[:, :1] * self.weight)

    def teacher_state_hash(self):
        return hashlib.sha256(self.teacher.numpy().tobytes()).hexdigest()

    frozen_state_hash = teacher_state_hash

    def architecture_audit(self):
        return dict(group=self.group, trainable_parameters=1,
                    trainable_parameter_names=['weight'], student_reference_storage_independent=True)


def cpu_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=None)


def checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def make_config(root, microbatch=4):
    root.mkdir()
    task = root / 'task.txt'
    task.write_text('CPU-only control-flow fixture, not an experimental protocol.')
    task_hash = training.sha256_file(task)
    cfg = dict(output_root=str(root), run_root=str(root/'runs'), cache_dir=str(root/'cache'), seed=17, epochs={'D': 40},
               input_size=[1, 1], effective_batch=8, microbatch_start=microbatch, workers=0,
               eval_batch=4, amp=False, learning_rate=1e-4, weight_decay=1e-4,
               gradient_clip=1., gradient_diagnostic_every_epochs=5,
               taskbook=str(task), taskbook_sha256=task_hash)
    (root/'preregistration.json').write_text(json.dumps(dict(taskbook_sha256=task_hash)))
    (root/'code_provenance.json').write_text(json.dumps(dict(source_bundle_hash='cpu-only-fixture')))
    for name in ['asset_audit.json', 'data_audit.json', 'environment.json', 'resolved_paths.json']:
        (root/name).write_text(json.dumps(dict(fixture='CPU-only trainer control-flow')))
    write_fixture_acceptance(cfg)
    return cfg


def write_fixture_acceptance(cfg):
    root=Path(cfg['output_root'])
    cpu_path=root/'engineering_cpu_tests.json'
    cpu_path.write_text(json.dumps(dict(status='PASS',passed_tests=47,
                         fixture='mock upstream CPU gate; not a real research acceptance report')))
    binding=dict(status='PASS',
        configuration_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
        manifest_sha256='manifest-fixture',teacher_B_sha256='B-fixture',
        taskbook_sha256=cfg['taskbook_sha256'],code_provenance=dict(source_bundle_hash='cpu-only-fixture'),
        cpu_tests_sha256=training.sha256_file(cpu_path))
    (root/'engineering_acceptance.json').write_text(json.dumps(binding))
    return binding


def install_cpu_fixture(monkeypatch):
    # P0/preflight themselves use real assets elsewhere; this test isolates the
    # durable trainer path after those gates have passed.
    monkeypatch.setattr(training, 'check_inputs', lambda cfg: (Path(cfg['output_root']), 'manifest-fixture', 'B-fixture'))
    monkeypatch.setattr(training, 'provenance', lambda cfg: dict(source_bundle_hash='cpu-only-fixture'))
    monkeypatch.setattr(training, 'build_model', lambda cfg, group: TinyTailModel(group))
    monkeypatch.setattr(training, 'EXPECTED_TRAINABLE', {'F-Mean': 1})
    monkeypatch.setattr(training, 'set_seed', cpu_seed)
    monkeypatch.setattr(training, 'rng_state', cpu_rng)
    monkeypatch.setattr(training, 'environment', lambda: dict(device='CPU control fixture'))
    monkeypatch.setattr(training, 'datasets', lambda cfg: (TinyImageDataset(), TinyImageDataset(), TinyImageDataset('val')))
    def cpu_loader(dataset, batch_size, workers=0, *, seed=17, epoch=0, shuffle=False):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed+epoch*100003), num_workers=0,
            pin_memory=False, collate_fn=training.collate_multi_reference, drop_last=False)
    monkeypatch.setattr(training, 'loader', cpu_loader)
    real_device_batch = training.device_batch
    monkeypatch.setattr(training, 'device_batch', lambda batch, device: real_device_batch(batch, torch.device('cpu')))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)

    def evaluate(model, dataset, cfg, **kwargs):
        # Consume every RNG family to detect failed restoration on resume.
        random.random(); np.random.rand(); torch.rand(1)
        if kwargs.get('per_image_path'):
            training.append_csv(kwargs['per_image_path'], dict(epoch=kwargs['epoch'], split=dataset.split,
                                image_id='fixture-case', group_id='fixture-group', reference_count=2))
        result = {name: 0. for name in training.FIELDS}
        result['images'] = len(dataset)
        if kwargs.get('original', True):
            result.update(dice=.5, iou=.3, G_plus=.01, H_minus=.01, H_epsilon=.005, mean_gain=0.)
        return result
    monkeypatch.setattr(training, 'evaluate', evaluate)


def interrupt_after_latest(monkeypatch, epoch):
    durable_save = training.save_checkpoint
    def stop(path, payload):
        durable_save(path, payload)
        if Path(path).name == 'latest.pth' and payload['epoch'] == epoch:
            raise RuntimeError('intentional CPU fixture interrupt at durable epoch')
    monkeypatch.setattr(training, 'save_checkpoint', stop)
    return durable_save


def run_interrupted(cfg):
    with pytest.raises(RuntimeError, match='intentional CPU fixture interrupt'):
        training.train_group(cfg, 'F-Mean')
    return Path(cfg['run_root'])/'seed17/F-Mean'


def test_micro4_actual_last_seven_image_gradient_matches_micro8(monkeypatch, tmp_path):
    install_cpu_fixture(monkeypatch)
    interrupt_after_latest(monkeypatch, 1)
    real_step = training.finish_optimizer_step
    gradient_runs = []
    current_gradients = []
    def observe(optimizer, scaler, params, max_norm):
        current_gradients.append(params[0].grad.detach().clone())
        return real_step(optimizer, scaler, params, max_norm)
    monkeypatch.setattr(training, 'finish_optimizer_step', observe)
    for micro in (4, 8):
        cfg = make_config(tmp_path/f'micro-{micro}', micro)
        run = run_interrupted(cfg)
        state = checkpoint(run/'latest.pth')
        assert state['optimizer_step_attempts'] == state['optimizer_steps'] == 184
        assert state['epoch'] == 1 and state['config']['epochs']['D'] == 40
        assert torch.equal(state['model']['teacher'], torch.tensor(.125))
        gradient_runs.append(current_gradients[:])
        current_gradients.clear()
    assert len(gradient_runs[0]) == len(gradient_runs[1]) == 184
    # The shuffle order and first 183 full windows agree. The final actual
    # seven-image window must retain its image-average gradient, rather than
    # silently scaling it by 7/8, dropping its final microbatch, or overweighting.
    for a, b in zip(gradient_runs[0], gradient_runs[1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-6)


def test_resume_restores_rng_optimizer_scheduler_and_removes_pending_rows(monkeypatch, tmp_path):
    install_cpu_fixture(monkeypatch)
    cfg = make_config(tmp_path/'resumed')
    durable_save = interrupt_after_latest(monkeypatch, 2)
    run = run_interrupted(cfg)
    second = checkpoint(run/'latest.pth')
    assert second['epoch'] == 2 and second['optimizer_steps'] == 368
    # Mimic evaluation/log writes from an epoch whose latest checkpoint never
    # committed. The next invocation must truncate and replay this epoch.
    with (run/'epoch_diagnostics.csv').open() as f:
        prior = list(csv.DictReader(f))[-1]
    prior['epoch'] = 3
    training.append_csv(run/'epoch_diagnostics.csv', prior)
    training.append_csv(run/'per_image_canonical_diagnostics.csv',
                        dict(epoch=3, split='val', image_id='stale-case', group_id='fixture-group', reference_count=2))
    monkeypatch.setattr(training, 'save_checkpoint', durable_save)
    interrupt_after_latest(monkeypatch, 3)
    run_interrupted(cfg)
    resumed = checkpoint(run/'latest.pth')
    continuous_cfg = make_config(tmp_path/'continuous')
    continuous_run = run_interrupted(continuous_cfg)
    continuous = checkpoint(continuous_run/'latest.pth')
    assert resumed['epoch'] == continuous['epoch'] == 3
    assert resumed['optimizer_steps'] == continuous['optimizer_steps'] == 552
    assert resumed['scheduler'] == continuous['scheduler']
    for key, value in continuous['model'].items():
        assert torch.equal(value, resumed['model'][key])
    for key, value in continuous['optimizer']['state'][0].items():
        assert torch.equal(value, resumed['optimizer']['state'][0][key])
    assert resumed['rng']['python'] == continuous['rng']['python']
    np.testing.assert_equal(resumed['rng']['numpy'], continuous['rng']['numpy'])
    assert torch.equal(resumed['rng']['torch'], continuous['rng']['torch'])
    with (run/'epoch_diagnostics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['epoch']) for r in rows] == [1, 2, 3]
    assert [float(r['lr_group0']) for r in rows] == pytest.approx([5e-5, 1e-4, 1e-4])
    assert [int(r['optimizer_steps']) for r in rows] == [184, 368, 552]
    assert all(r['anchor_max_absolute_error'] == '0.0' for r in rows)
    with (run/'per_image_canonical_diagnostics.csv').open() as f:
        image_rows = list(csv.DictReader(f))
    keys = [(int(r['epoch']), r['split'], r['image_id']) for r in image_rows]
    assert len(keys) == len(set(keys)) == 6
    assert all('stale' not in r['image_id'] for r in image_rows)
    # Equal validation metrics select the first epoch, while every later epoch
    # remains committed in latest. This is not risk-based checkpoint selection.
    assert checkpoint(run/'best.pth')['epoch'] == resumed['best_epoch'] == 1
    assert not (run/'DONE.json').exists()


def test_resume_recovers_best_after_latest_saved_before_best(monkeypatch, tmp_path):
    install_cpu_fixture(monkeypatch)
    cfg = make_config(tmp_path/'durable-latest')
    original_save = interrupt_after_latest(monkeypatch, 1)
    run = run_interrupted(cfg)
    assert not (run/'best.pth').exists()
    monkeypatch.setattr(training, 'save_checkpoint', original_save)
    interrupt_after_latest(monkeypatch, 2)
    run_interrupted(cfg)
    assert checkpoint(run/'best.pth')['epoch'] == 1
    assert checkpoint(run/'latest.pth')['epoch'] == 2


def test_resume_rejects_changed_learning_rate_before_any_update(monkeypatch, tmp_path):
    install_cpu_fixture(monkeypatch)
    cfg = make_config(tmp_path/'reject-change')
    interrupt_after_latest(monkeypatch, 1)
    run_interrupted(cfg)
    changed = dict(cfg, learning_rate=2e-4)
    # A fresh matching mechanical gate must still not authorize resuming a
    # checkpoint whose frozen configuration used a different learning rate.
    write_fixture_acceptance(changed)
    with pytest.raises(ValueError, match='Resume'):
        training.train_group(changed, 'F-Mean')


def test_stale_acceptance_or_cpu_evidence_rejected_before_model_build(monkeypatch, tmp_path):
    install_cpu_fixture(monkeypatch)
    cfg=make_config(tmp_path/'reject-stale-acceptance')
    root=Path(cfg['output_root'])
    binding=write_fixture_acceptance(cfg)
    called=[]
    def forbidden_build(*args,**kwargs):
        called.append(True)
        raise AssertionError('a stale gate must reject before model construction')
    monkeypatch.setattr(training,'build_model',forbidden_build)
    for field in ['configuration_sha256','manifest_sha256','teacher_B_sha256','taskbook_sha256','code_provenance']:
        bad=dict(binding)
        bad[field]={'source_bundle_hash':'stale-code'} if field=='code_provenance' else 'stale-identity'
        (root/'engineering_acceptance.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='does not bind'):
            training.train_group(cfg,'F-Mean')
    (root/'engineering_acceptance.json').write_text(json.dumps(binding))
    # Altering even a PASS test-count record after binding invalidates its hash.
    cpu_path=root/'engineering_cpu_tests.json'
    cpu_path.write_text(json.dumps(dict(status='PASS',passed_tests=48,changed=True)))
    with pytest.raises(ValueError,match='CPU control-flow'):
        training.train_group(cfg,'F-Mean')
    assert not called
    assert not (root/'runs').exists()


def test_check_inputs_rejects_old_output_and_cache_symlink_before_reading_assets(tmp_path):
    # These path gates run before manifest/weights or any GPU interaction.
    cfg=dict(output_root=str(tmp_path),run_root=str(tmp_path/'runs'),
             cache_dir='/home/featurize/rsi_data/ima/cache256')
    with pytest.raises(ValueError,match='Cache writes outside'):
        training.check_inputs(cfg)
    cache_link=tmp_path/'cache-link'
    cache_link.symlink_to('/home/featurize/rsi_data/ima/cache256',target_is_directory=True)
    cfg['cache_dir']=str(cache_link)
    with pytest.raises(ValueError,match='Cache writes outside'):
        training.check_inputs(cfg)
    for protected in ['/home/featurize/rsi_runs/P2-IMA-M-v2','/home/featurize/work/RSI_Implementation_20261003']:
        old=dict(output_root=protected,run_root=protected+'/runs',cache_dir=protected+'/cache')
        with pytest.raises(ValueError,match='historical protected assets'):
            training.check_inputs(old)


def test_test_evaluation_is_rejected_before_model_or_gpu(monkeypatch):
    dataset = TinyImageDataset('test')
    with pytest.raises(PermissionError, match='test scoring'):
        training.evaluate(None, dataset, {})


def test_provenance_covers_new_runner_tests_export_and_config(monkeypatch, tmp_path):
    files = ['rsi/tail_probe_model.py', 'tools/tail_probe_training.py', 'tools/tail_probe_analysis.py',
             'tools/run_tail_probe.py', 'tests/test_tail_probe_runner.py', 'configs/tail_probe_seed17.yaml']
    for file in files:
        p=tmp_path/file
        p.parent.mkdir(exist_ok=True)
        p.write_text(file)
    task=tmp_path/'task.txt';task.write_text('fixture task')
    monkeypatch.setattr(training, 'PROJECT', tmp_path)
    monkeypatch.setattr(training.subprocess, 'check_output', lambda *args, **kwargs: 'frozen-commit\n')
    result=training.provenance(dict(review_commit='review',taskbook=str(task)))
    assert set(result['source_sha256']) == set(files)
    assert all(result['source_sha256'][p] == training.sha256_file(tmp_path/p) for p in files)
    original_hash=result['source_bundle_hash']
    (tmp_path/'tools/tail_probe_analysis.py').write_text('changed exporter')
    assert training.provenance(dict(review_commit='review',taskbook=str(task)))['source_bundle_hash'] != original_hash
