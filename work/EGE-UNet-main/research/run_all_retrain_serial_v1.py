"""Serial, guarded retraining of the five historical mainline models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.config_setting import setting_config
from configs.config_wave_ll_only import wave_ll_only_config
from configs.config_ege_dual import ege_dual_config
from configs.config_ege_dual_difflr import ege_dual_difflr_config
from train import main


def base_config(name, network, **attrs):
    cls = type(name, (setting_config,), {"network": network, **attrs})
    return cls


jobs = [
    ("EGE_baseline", base_config("EGEBaselineConfig", "egeunet"), {"network": "egeunet"}),
    ("EGE_Wave_v1", base_config("EGEWaveV1Config", "ege_wave_unet", wave_mode="full", fusion_type="scalar"), {"network": "ege_wave_unet", "wave_mode": "full", "fusion_type": "scalar"}),
    ("Wave_LL_only", wave_ll_only_config, {"network": "ege_wave_unet", "wave_mode": "ll_only", "fusion_type": "scalar"}),
    ("EGE_Dual_unified", ege_dual_config, {"network": "ege_dual", "fusion_type": "scalar", "diff_lr": False}),
    ("EGE_Dual_difflr", ege_dual_difflr_config, {"network": "ege_dual", "fusion_type": "scalar", "diff_lr": True}),
]

for name, config, expected in jobs:
    actual = {
        "network": config.network,
        "wave_mode": getattr(config, "wave_mode", "full"),
        "fusion_type": getattr(config, "fusion_type", "scalar" if config.network == "ege_dual" else "spatial_gate"),
        "diff_lr": getattr(config, "diff_lr", False),
    }
    for key, value in expected.items():
        assert actual[key] == value, f"{name} preflight failed: {key}={actual[key]!r}, expected {value!r}"
    version = 'v2' if name == 'EGE_Dual_difflr' else 'v1'
    config.work_dir = f"results/retrain_{name.lower()}_{version}_20260912/"
    if (Path(config.work_dir) / 'INVALID_RUN.md').exists():
        raise RuntimeError(f'Refusing invalid run directory: {config.work_dir}')
    config.epochs = 300
    config.val_interval = 10
    config.save_interval = 10
    config.num_workers = 4
    config.batch_size = 64
    config.gradient_accumulation_steps = 1
    config.seed = 42
    print(f"PREFLIGHT OK: {name} {actual}; output={config.work_dir}", flush=True)
    main(config)
    print(f"FINISHED: {name}", flush=True)
