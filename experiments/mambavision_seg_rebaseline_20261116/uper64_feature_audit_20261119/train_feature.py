"""Matched UPer64 feature correction ablations on the historical validation protocol."""
import argparse
import fcntl
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from feature_decoder import ARMS, ROOT, build
sys.path.insert(0, str(ROOT.parent / "method1_f3_20261118"))
from train import reference, validate, write_rows
from train_stageB_color import accumulate, atomic_json, atomic_save, base, rng_state, restore_rng
from model import PROJECT, WEIGHT_SHA256


def optimizer(model):
    groups = [{"name": "decoder", "params": list(model.probe.parameters()), "lr": 1e-3},
              {"name": "encoder", "params": [p for p in model.encoder.parameters() if p.requires_grad],
               "lr": 1e-4}]
    for group in groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.AdamW(groups, weight_decay=.01, betas=(.9, .999), eps=1e-8)


def train(arm, run):
    reference.setup()
    model = build(arm)
    data, _ = reference.load_data()
    validation_ids = data.ids["val"]
    assert (len(data.ids["train"]), len(validation_ids)) == (1886, 762)
    src = ROOT.parent / "src"
    files = [Path(__file__), ROOT / "feature_decoder.py", src / "model.py",
             src / "train_stageB_color.py", ROOT.parent / "method1_f3_20261118/train.py",
             Path(reference.__file__), PROJECT / "stage45_upernet_decoder/identity_partition.json"]
    config = dict(arm=arm, seed=42, epochs=100, batch=64, image_size=256, threshold=.5,
                  training_images=1886, validation_images=762, auxiliary_weight=0,
                  selection="validation mean per-image foreground Dice, earliest tie",
                  encoder_trainable_stages=[3, 4], encoder_weight_sha256=WEIGHT_SHA256,
                  head_parameters=sum(p.numel() for p in model.probe.parameters()),
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    cfgfile = run / "config.json"
    if cfgfile.exists() and json.loads(cfgfile.read_text()) != config:
        raise RuntimeError("Source or training configuration changed since checkpoint")
    if (run / "complete.json").exists():
        return
    atomic_json(cfgfile, config)
    opt = optimizer(model)
    history = []
    latest = run / "latest.pth"
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        if state["config"] != config:
            raise RuntimeError("Resume configuration mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history = state["history"]
        del state
    val = reference.cache_new(len(validation_ids))
    reference.fill_cache(model, data, "val", val)
    cache = reference.cache_new(len(data.ids["train"]))
    best = max((r["val_macro_dice"] for r in history), default=-1.)
    for epoch in range(len(history) + 1, 101):
        order, geometry = data.geometry(epoch)
        reference.fill_cache(model, data, "train", cache, geometry)
        factor = .01 + .99 * (1 + math.cos(math.pi * (epoch - 1) / 100)) / 2
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * factor
        model.train()
        loss_sum = 0.
        for start in range(0, len(order), 64):
            ids = order[start:start + 64]
            loss, norms = accumulate(model, opt, cache, ids, 64)
            if epoch == 1 and start == 0:
                if not all(math.isfinite(v) and v > 0 for v in norms.values()):
                    raise FloatingPointError(f"Invalid first update gradients: {norms}")
                atomic_json(run / "first_update.json", dict(epoch=epoch, batch=len(ids), loss=loss,
                                                          gradient_norms=norms))
            loss_sum += loss * len(ids)
        scores, rows = validate(model, val, validation_ids, interiors=None)
        record = dict(epoch=epoch, train_loss=loss_sum / len(order), **scores)
        history.append(record)
        if record["val_macro_dice"] > best:
            best = record["val_macro_dice"]
            atomic_save(run / "best.pth", dict(epoch=epoch, model=base.cpu_tree(model.state_dict()),
                                              config=config, scores=scores))
            write_rows(run / "best_validation.csv", rows)
        atomic_save(latest, dict(model=base.cpu_tree(model.state_dict()), optimizer=base.cpu_tree(opt.state_dict()),
                                 rng=rng_state(), history=history, config=config))
        atomic_json(run / "history.json", history)
        print(json.dumps(record), flush=True)
    atomic_json(run / "complete.json", dict(best=max(history, key=lambda r: r["val_macro_dice"]),
                                           epochs=len(history), test_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=ARMS)
    arm = parser.parse_args().arm
    run = ROOT / "runs" / arm
    run.mkdir(parents=True, exist_ok=True)
    with (run / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(arm, run)
