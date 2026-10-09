"""Run the EGE baseline with the seed-43 stability sweep settings."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config_setting import setting_config
from train import main

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class EGEBaselineSeed43(setting_config):
    network = 'egeunet'
    seed = 43
    work_dir = 'results/stability_ege_baseline_seed43_v1_20260929/'
    epochs = 300
    val_interval = 10
    save_interval = 10
    num_workers = 4
    batch_size = 64
    val_batch_size = 8
    gradient_accumulation_steps = 1


if __name__ == '__main__':
    assert EGEBaselineSeed43.network == 'egeunet'
    assert EGEBaselineSeed43.seed == 43
    print(f'PREFLIGHT OK: ege_baseline_seed43 train_batch={EGEBaselineSeed43.batch_size} val_batch={EGEBaselineSeed43.val_batch_size}', flush=True)
    main(EGEBaselineSeed43)
    print('FINISHED: ege_baseline_seed43', flush=True)
