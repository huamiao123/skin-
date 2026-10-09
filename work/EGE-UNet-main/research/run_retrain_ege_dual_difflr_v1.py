"""Guarded launcher: refuses to run if the requested config resolves to another model."""
from configs.config_ege_dual_difflr import ege_dual_difflr_config
from train import main


config = ege_dual_difflr_config
assert config.__name__ == "ege_dual_difflr_config"
assert config.network == "ege_dual"
assert getattr(config, "diff_lr", False) is True
assert getattr(config, "fusion_type", "scalar") == "scalar"
config.work_dir = "results/retrain_ege_dual_difflr_v2_20260912/"
config.epochs = 300
config.val_interval = 10
config.save_interval = 10
config.num_workers = 4
config.batch_size = 64
config.gradient_accumulation_steps = 1
config.seed = 42
print("PREFLIGHT OK:", config.__name__, config.network, "fusion=scalar", "diff_lr=True")
main(config)
