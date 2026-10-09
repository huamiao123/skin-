"""Seed 43 stability runs for the three most informative architectures."""
import sys
import os
import faulthandler
import signal
import logging
from pathlib import Path
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config_setting import setting_config
from configs.config_ege_dual_difflr import ege_dual_difflr_config
from configs.config_ege_dual_difflr_scf import ege_dual_difflr_scf_config
from train import main
import torch

# CPU image rotations are small; large intra-op pools can starve the GPU.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def wave_config(seed):
    return type(f'WaveV1Seed{seed}', (setting_config,), {'network':'ege_wave_unet','wave_mode':'full','fusion_type':'scalar','seed':seed})

jobs=[]
for seed in (43,):
    jobs += [(f'wave_v1_seed{seed}', wave_config(seed), {'network':'ege_wave_unet','wave_mode':'full','fusion_type':'scalar','diff_lr':False}),
             (f'dual_difflr_seed{seed}', type(f'DualSeed{seed}', (ege_dual_difflr_config,), {'seed':seed}), {'network':'ege_dual','fusion_type':'scalar','diff_lr':True}),
             (f'dual_scf_seed{seed}', type(f'SCFSeed{seed}', (ege_dual_difflr_scf_config,), {'seed':seed}), {'network':'ege_dual','fusion_type':'scale_cond','diff_lr':True})]

start_at = os.environ.get('MEDSEG_STABILITY_START_AT')
if start_at:
    names = [name for name, _, _ in jobs]
    if start_at not in names:
        raise ValueError(f'unknown stability start job: {start_at}')
    jobs = jobs[names.index(start_at):]

for name, config, expected in jobs:
    actual={'network':config.network,'wave_mode':getattr(config,'wave_mode','full'),'fusion_type':getattr(config,'fusion_type','scalar' if config.network=='ege_dual' else 'spatial_gate'),'diff_lr':getattr(config,'diff_lr',False)}
    for k,v in expected.items(): assert actual[k]==v, f'{name}: {k}={actual[k]!r}, expected {v!r}'
    config.work_dir=f'results/stability_{name}_v1_20260914/'
    # Resume seed 43 from an epoch checkpoint with parallel data loading.
    # The worker-count change is recorded in PERFORMANCE_AUDIT_20260929.md.
    config.epochs=300; config.val_interval=10; config.save_interval=10; config.num_workers=4; config.batch_size=64; config.val_batch_size=8; config.gradient_accumulation_steps=1
    print(f'PREFLIGHT OK: {name} {actual} seed={config.seed} train_batch={config.batch_size} val_batch={config.val_batch_size}',flush=True)
    main(config)
    for handler in logging.getLogger('train').handlers[:]:
        logging.getLogger('train').removeHandler(handler)
        handler.close()
    print(f'FINISHED: {name}',flush=True)
