import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from decoder import HERE, build
from train_stageB_color import accumulate, atomic_json, atomic_save, base, reference, rng_state, restore_rng
from model import PROJECT
from stage1_visual.analyze import metrics

FIELDS = ("tp", "fp", "fn", "coarse_tp", "coarse_fp", "coarse_fn", "saved_fn",
          "new_fp", "saved_fp", "new_fn", "conf_fn", "conf_fn_fixed",
          "conf_interior_fn", "conf_interior_fixed")


def dice(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 1.0


def write_rows(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def interior_masks(targets):
    masks = targets[:, 0].bool().cpu().numpy()
    return torch.from_numpy(np.stack([
        distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1] > 8 for m in masks
    ])).cuda()


@torch.inference_mode()
def validate(model, cache, subjects, interiors):
    model.eval()
    rows = []
    has_coarse = hasattr(model.probe, "guide")
    for first in range(0, len(subjects), 64):
        ix = slice(first, first + 64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.from_cache(cache["features"][0][ix], cache["features"][1][ix],
                                   cache["mid"][ix], (256, 256))
        pred = out["main"][:, 0].float().sigmoid() >= .5
        gt = cache["target"][ix, 0] >= .5
        if not torch.isfinite(out["main"]).all():
            raise FloatingPointError("validation logits are nonfinite")
        bits = [pred & gt, pred & ~gt, ~pred & gt]
        if has_coarse:
            prob = out["aux"][:, 0].float().sigmoid()
            if not torch.isfinite(out["aux"]).all():
                raise FloatingPointError("initial logits are nonfinite")
            coarse = prob >= .5
            hard = (prob < .1) & gt
            deep = hard & interiors[ix]
            bits += [coarse & gt, coarse & ~gt, ~coarse & gt, ~coarse & pred & gt,
                     ~coarse & pred & ~gt, coarse & ~pred & ~gt, coarse & ~pred & gt,
                     hard, hard & pred, deep, deep & pred]
        counts = torch.stack([x.sum((1, 2), dtype=torch.int32) for x in bits], dim=1).cpu().tolist()
        for n, values in enumerate(counts):
            row = dict(zip(FIELDS, values))
            row.update(id=subjects[first+n], dice=dice(*values[:3]))
            if has_coarse:
                row["coarse_dice"] = dice(*values[3:6])
            rows.append(row)
        if first == 0:
            expected = metrics(pred[0].cpu().numpy(), gt[0].cpu().numpy())["dice"]
            assert expected == rows[0]["dice"]
    assert len(rows) == len(subjects) == 762
    out = {"val_macro_dice": float(np.mean([r["dice"] for r in rows]))}
    if has_coarse:
        out["coarse_val_macro_dice"] = float(np.mean([r["coarse_dice"] for r in rows]))
        out["correction_counts"] = {k: sum(r[k] for r in rows) for k in FIELDS[6:]}
    return out, rows


def main(arm):
    run = HERE / arm
    run.mkdir(exist_ok=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "complete.json").exists():
        return
    reference.setup()
    model = build(arm)
    data, partition = reference.load_data()
    split_path = PROJECT / "stage45_upernet_decoder/identity_partition.json"
    config = {"arm": arm, "device": torch.cuda.get_device_name(0), "seed": 42,
              "physical_batch": 64, "effective_batch": 64, "epochs": 100, "threshold": .5,
              "optimizer": "AdamW decoder 0.001 / encoder 0.0001, wd .01, cosine100 floor .01; per-group clip 1",
              "train_n": len(data.ids["train"]), "val_n": len(data.ids["val"]),
              "stage3_4_tune": True, "shared_coarse_loss_weight": .4 if arm != "uper64" else 0.,
              "head_parameters": sum(p.numel() for p in model.probe.parameters()),
              "partition_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (HERE/"decoder.py", Path(__file__), HERE.parent/"src/model.py",
                                 HERE.parent/"src/train_stageB_color.py")},
              "selection": "highest validation per-image foreground Dice, earliest tie; no test access"}
    cfg = run / "config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise RuntimeError("Configuration changed; resume separately")
    atomic_json(cfg, config)
    opt = torch.optim.AdamW([
        {"params":list(model.probe.parameters()), "lr":1e-3, "initial_lr":1e-3, "name":"decoder"},
        {"params":[p for p in model.encoder.parameters() if p.requires_grad], "lr":1e-4,
         "initial_lr":1e-4, "name":"encoder"}], weight_decay=.01, betas=(.9,.999), eps=1e-8)
    val = reference.cache_new(len(data.ids["val"]))
    reference.fill_cache(model, data, "val", val)
    interiors = interior_masks(val["target"]) if arm != "uper64" else None
    train = reference.cache_new(len(data.ids["train"]))
    history, best = [], -1.
    latest = run/"latest.pth"
    if latest.exists():
        state = torch.load(latest, map_location="cpu")
        if state["config"] != config:
            raise RuntimeError("Resume state mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history, best = state["history"], state["best"]
        del state
    print(json.dumps({"arm":arm, "phase":"data_verified", "config":config}), flush=True)
    for epoch in range(len(history)+1, 101):
        tick = time.perf_counter()
        order, geometry = data.geometry(epoch)
        reference.fill_cache(model, data, "train", train, geometry)
        factor = .01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"]*factor
        model.train()
        total_loss = 0.
        for first in range(0, len(order), 64):
            ids = order[first:first+64]
            value, norms = accumulate(model, opt, train, ids, 64)
            if epoch == 1 and first == 0:
                assert all(v > 0 and math.isfinite(v) for v in norms.values())
                assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
                if arm != "uper64":
                    for block in [model.probe.f3, model.probe.f4, model.probe.coarse, *model.probe.delta]:
                        assert sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None) > 0
                atomic_json(run/"first_update.json", {"train_loss":value, "group_norms":norms,
                            "batch":len(ids), "peak_allocated_GiB":torch.cuda.max_memory_allocated()/2**30})
            total_loss += value * len(ids)
        scores, rows = validate(model, val, data.ids["val"], interiors)
        record = dict(epoch=epoch, train_loss=total_loss/len(order), seconds=time.perf_counter()-tick,
                      peak_GiB=torch.cuda.max_memory_allocated()/2**30, **scores)
        history.append(record)
        if scores["val_macro_dice"] > best:
            best = scores["val_macro_dice"]
            atomic_save(run/"best.pth", {"epoch":epoch, "model":base.cpu_tree(model.state_dict()),
                        "scores":scores, "config":config})
            write_rows(run/"best_validation.csv", rows)
        if epoch in (1, 100) or epoch % 5 == 0:
            atomic_save(latest, {"model":base.cpu_tree(model.state_dict()), "optimizer":base.cpu_tree(opt.state_dict()),
                        "rng":rng_state(), "config":config, "best":best, "history":history})
        atomic_json(run/"history.json", history)
        print(json.dumps(dict(arm=arm, **record)), flush=True)
    chosen = max(history, key=lambda r:r["val_macro_dice"])
    atomic_json(run/"complete.json", {"epochs":len(history), "best":chosen, "n_val":len(rows),
                                      "test_used":False})


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm", choices=["f3","f4","uper64"])
    main(parser.parse_args().arm)
