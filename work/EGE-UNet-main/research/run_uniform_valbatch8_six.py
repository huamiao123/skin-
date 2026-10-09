"""Run one of six fresh, comparable seed-42/43 legacy-split experiments."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config_setting import setting_config
from configs.config_ege_dual_difflr import ege_dual_difflr_config
from configs.config_ege_dual_difflr_scf import ege_dual_difflr_scf_config
from train import main


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "rebatch8_20260930"
JOBS = (
    ("wave_v1_seed43", setting_config, 43, "ege_wave_unet", "scalar", False),
    ("dual_difflr_seed43", ege_dual_difflr_config, 43, "ege_dual", "scalar", True),
    ("ege_baseline_seed42", setting_config, 42, "egeunet", "spatial_gate", False),
    ("wave_v1_seed42", setting_config, 42, "ege_wave_unet", "scalar", False),
    ("dual_difflr_seed42", ege_dual_difflr_config, 42, "ege_dual", "scalar", True),
    ("dual_scf_seed42", ege_dual_difflr_scf_config, 42, "ege_dual", "scale_cond", True),
)


def run(name):
    spec = next((item for item in JOBS if item[0] == name), None)
    if spec is None:
        raise ValueError(f"Unknown job: {name}")
    _, parent, seed, network, fusion_type, diff_lr = spec
    attrs = {
        "network": network,
        "seed": seed,
        "wave_mode": "full",
        "fusion_type": fusion_type,
        "diff_lr": diff_lr,
        "work_dir": str(RUN_ROOT / name) + "/",
        "epochs": 300,
        "val_interval": 10,
        "save_interval": 10,
        "num_workers": 4,
        "batch_size": 64,
        "val_batch_size": 8,
        "gradient_accumulation_steps": 1,
    }
    config = type(f"UniformValBatch8_{name}", (parent,), attrs)
    for key in ("network", "seed", "wave_mode", "fusion_type", "diff_lr"):
        assert getattr(config, key) == attrs[key], (name, key)
    output = RUN_ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    done = output / "DONE.json"
    if done.exists():
        print(f"SKIPPED: {name} already complete", flush=True)
        return
    print(
        f"PREFLIGHT OK: {name} seed={seed} network={network} "
        f"wave_mode=full fusion_type={fusion_type} diff_lr={diff_lr} "
        "train_batch=64 val_batch=8 workers=4 epochs=300",
        flush=True,
    )
    main(config)
    latest = output / "checkpoints" / "latest.pth"
    if not latest.exists() or not list((output / "checkpoints").glob("best-epoch*.pth")):
        raise RuntimeError(f"Training or final evaluation did not complete: {name}")
    done.write_text(
        json.dumps({"job": name, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "latest_checkpoint": str(latest)}, indent=2) + "\n"
    )
    for handler in logging.getLogger("train").handlers[:]:
        logging.getLogger("train").removeHandler(handler)
        handler.close()
    print(f"FINISHED: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=[job[0] for job in JOBS])
    run(parser.parse_args().job)
