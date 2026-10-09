import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config_ege_dual_difflr_scf import ege_dual_difflr_scf_config
from train import main

config = ege_dual_difflr_scf_config
assert config.network == 'ege_dual'
assert config.fusion_type == 'scale_cond'
assert config.diff_lr is True
config.work_dir = 'results/retrain_ege_dual_difflr_scf_v1_20260914/'
config.epochs = 300
config.val_interval = 10
config.save_interval = 10
config.num_workers = 4
config.batch_size = 64
config.gradient_accumulation_steps = 1
config.seed = 42
print('PREFLIGHT OK: EGE-Dual difflr SCF', flush=True)
main(config)
