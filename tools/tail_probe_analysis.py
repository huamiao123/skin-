"""Validation-only readout for the fixed seed17 decoder-tail experiment.

This file never trains, selects a lambda, exposes test masks, or changes the
frozen RSI implementation. Complete-update interpolation is performed before
coordinate restoration. Source cohorts reuse the exact M prediction cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

GROUPS = ("F-Mean", "F-RSI", "U-Mean", "U-RSI", "U-NoMessage")
ALPHAS = (0., 1/3, 2/3, 1.)
COUNTS = {"M": (223, 471), "H": (47, 104), "T1": (20, 49)}
EPSILON = .005
BOOTSTRAP_REPEATS, BOOTSTRAP_SEED = 2000, 17
CORE_METRICS = ("dice", "G_plus", "H_minus", "H_epsilon", "mean_gain")
TAIL_METRICS = ("worst_tail_10pct", "mean_gain_tail_10pct")


def default_screening_rules():
    """Task section 11, fixed implementation choices; never fitted to results."""
    return {
        "route_A": {"dice_advantage_over_max_M_N_C": .002,
                    "H_epsilon_R_minus_M_max": .0005},
        "route_B": {"dice_deficit_max_M_N": .001, "dice_deficit_C": .001,
                    "G_retention_min": .8, "M_H_absolute_reduction": .001,
                    "M_H_relative_reduction": .2, "C_H_absolute_reduction": .0005,
                    "C_H_relative_reduction": .1},
        "simple_control": {"dice_or_G_practical_difference": .001,
                           "H_practical_difference": .0005},
        "source_dice_regression": .005,
        "H_minus_increase_max": .0005,
        "worst_tail_decrease_max": .005,
        "low_risk_headroom_M": .001,
        "low_risk_headroom_C": .0005,
        "epsilon_d": EPSILON,
        "fixed_logit_alphas": list(ALPHAS),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sensitivity_epsilons": [0., .005, .01, .02, .05],
        "decision_states": ["INVALID", "STOP_CURRENT_RSI", "INCONCLUSIVE",
                            "CANDIDATE_FOR_CONFIRMATION"],
    }


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(set(row) for row in rows))) if rows else ["status"]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def output_root(cfg):
    root = Path(cfg.get("output_root", cfg["run_root"])).resolve()
    require("RSI-TailProbe" in root.name or root.is_relative_to(PROJECT),
            "TailProbe outputs must use the isolated experiment directory")
    require(root != PROJECT and root != Path(cfg["manifest"]).parent,
            "Outputs cannot overwrite source or manifest directories")
    return root


def validate_scope(cfg, scope):
    if scope != "val":
        raise PermissionError("TailProbe scoring exposes validation only; test remains sealed")
    require(cfg.get("test_scoring_locked", True) is True, "Test scoring must remain locked")
    require(int(cfg.get("seed", 17)) == 17, "Only the registered seed17 experiment is authorized")


def manifest_cohorts(cfg):
    from rsi.datasets import read_manifest
    manifest = read_manifest(cfg["manifest"])
    result = {s: [r for r in manifest if r["split"] == "val" and r["subset"] == s]
              for s in COUNTS}
    for subset, refs in result.items():
        require((len({r["image_id"] for r in refs}), len(refs)) == COUNTS[subset],
                f"Incomplete {subset} validation cohort")
    train = [r for r in manifest if r["split"] == "train" and r["subset"] == "M"]
    require((len({r["image_id"] for r in train}), len(train)) == (1471, 3102),
            "Canonical M train cohort differs from the final audited protocol")
    return result, train


def implementation_identity():
    names = ("tools/tail_probe_analysis.py", "rsi/tail_probe_model.py", "rsi/datasets.py",
             "rsi/metrics.py", "rsi/objectives.py", "tools/fast_exact_metrics.py",
             "tools/cached_pilot_runner.py")
    return {name: file_record(PROJECT / name)["sha256"] for name in names}


def selected_checkpoint(cfg, group):
    value = cfg.get("selected_checkpoints", {}).get(group)
    if isinstance(value, dict):
        value = value.get("path") or value.get("checkpoint")
    return Path(value or Path(cfg["run_root"]) / "seed17" / group / "best.pth").resolve()


def load_model(cfg, group, checkpoint):
    import torch
    from rsi.models import ControlledModel
    from rsi.tail_probe_model import TailProbeModel
    base = ControlledModel(pretrained=False, input_size=cfg["input_size"][0])
    base.load_state_dict(torch.load(cfg["teacher_B"], map_location="cpu")["model"], strict=True)
    model = TailProbeModel(base, group)
    state = torch.load(checkpoint, map_location="cpu")
    if any(k.startswith("student_tail.") for k in state["model"]):
        require(state.get("group", group) == group, "Selected checkpoint belongs to another group")
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_legacy_state_dict(state["model"], strict=True)
    require(model.teacher_state_hash() == base.anchor_state_hash(), "Checkpoint changed fixed B teacher")
    require(model.independent_tail_storage(), "Student and teacher share tensor storage")
    return model.cuda().eval(), state


def cache_prediction(cfg, group, *, progress=None):
    """One M inference; bytes and geometry are bound to exact checkpoint/code."""
    import torch
    from rsi.datasets import IMAMultiReferenceDataset
    from rsi.train import loader, device_batch
    validate_scope(cfg, "val")
    checkpoint = selected_checkpoint(cfg, group)
    require(checkpoint.is_file(), f"Missing selected checkpoint: {checkpoint}")
    metadata = {
        "group": group, "checkpoint": file_record(checkpoint),
        "teacher_B": file_record(cfg["teacher_B"]), "manifest": file_record(cfg["manifest"]),
        "implementation": implementation_identity(), "size": cfg["input_size"],
        "amp": bool(cfg["amp"]), "eval_batch": int(cfg["eval_batch"]),
        "split": "val", "subset": "M", "augmentation": False,
        "coordinate_view": "canonical_letterbox_logits", "seed": 17,
    }
    key = json_hash(metadata)
    directory = output_root(cfg) / "canonical_logits" / group / key
    index_path = directory / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        require(index["metadata"] == metadata and index["key"] == key,
                "Prediction cache metadata differs")
        for item in index["images"].values():
            require(file_record(item["path"])["sha256"] == item["sha256"], "Prediction cache bytes changed")
        require(len(index["images"]) == 223, "Prediction cache is incomplete")
        return index
    model, _ = load_model(cfg, group, checkpoint)
    teacher_hash = model.teacher_state_hash()
    frozen_hash = model.frozen_state_hash()
    dataset = IMAMultiReferenceDataset(cfg["manifest"], split="val", subset="M", train=False,
                                      size=cfg["input_size"][0], cache_dir=cfg["cache_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    tick = time.monotonic()
    with torch.no_grad():
        for cpu in loader(dataset, cfg["eval_batch"], cfg["workers"]):
            batch = device_batch(cpu, torch.device("cuda"))
            with torch.autocast("cuda", enabled=cfg["amp"]):
                details = model.forward_details(batch["image"], batch["pixel_valid"])
                # Features are reused: this diagnostic is another tail decode,
                # not another encoder/Transformer inference.
                off = model.student_tail(details["f8"], details["c4"])
            for i, image_id in enumerate(batch["image_id"]):
                require(image_id not in images, "Repeated cached validation image")
                payload = dict(z_ref=details["z0"][i].float().cpu(),
                               z_on=details["z1"][i].float().cpu(), z_off=off[i].float().cpu(),
                               geometry=batch["geometry"][i], image_id=image_id,
                               pixel_valid=batch["pixel_valid"][i].bool().cpu(),
                               metadata_key=key, group_id=batch["group_id"][i],
                               message_rms=(float(details["message"][i].float().square().mean().sqrt())
                                            if details["message"] is not None else 0.))
                target = directory / (image_id + ".pt")
                temporary = target.with_suffix(".tmp.pt")
                torch.save(payload, temporary)
                temporary.replace(target)
                images[image_id] = {**file_record(target), "geometry": payload["geometry"],
                                    "logits_shape": list(payload["z_on"].shape)}
            if progress:
                progress(group=group, images=len(images), total=223)
            print(f"TailProbe cache {group}: {len(images)}/223 images", flush=True)
    require(model.teacher_state_hash() == teacher_hash and model.frozen_state_hash() == frozen_hash,
            "Frozen parameters or buffers drifted during export")
    index = dict(key=key, metadata=metadata, teacher_hash=teacher_hash, frozen_hash=frozen_hash,
                 images=images, seconds=time.monotonic()-tick, same_M_outputs_for_H_T1=True,
                 test_scoring_locked=True, complete=True)
    write_json(index_path, index)
    del model
    torch.cuda.empty_cache()
    return index


def _read_prediction(index, image_id):
    import torch
    value = torch.load(index["images"][image_id]["path"], map_location="cpu")
    require(value["image_id"] == image_id and value["metadata_key"] == index["key"],
            "Cached image/checkpoint identity mismatch")
    require(value["geometry"] == index["images"][image_id]["geometry"], "Cached geometry changed")
    return value


def _canonical_masks(refs, geometry):
    import torch
    from PIL import Image
    from rsi.datasets import _binary_mask
    size = geometry["size"]
    masks = np.zeros((len(refs), 1, size, size), np.uint8)
    originals = []
    t, l, h, w = (geometry[k] for k in ("top", "left", "resized_h", "resized_w"))
    for r, ref in enumerate(refs):
        original = _binary_mask(ref["mask_path"])
        require(original.shape == (geometry["original_h"], geometry["original_w"]),
                "Original mask geometry differs from cached RGB")
        originals.append(original.astype(bool))
        masks[r, 0, t:t+h, l:l+w] = np.asarray(Image.fromarray(original).resize((w,h), Image.Resampling.NEAREST))
    return torch.from_numpy(masks)[None].float(), originals


def canonical_components(cfg,index,subset,alpha=None):
    """Exact historical GPU batch/presence reduction, using the M logits.

    CPU per-image reductions differ in the final FP32 bits from the original
    GPU batch path. Keep the same loader order, batch size, maximum reference
    dimension, present flags, valid pixels and FP32 objective implementation.
    No image is inferred again, including source-subset last batches.
    """
    import torch
    from rsi.datasets import IMAMultiReferenceDataset
    from rsi.objectives import per_reference_loss,objective_from_losses
    from rsi.train import loader,device_batch
    from rsi.tail_probe_model import interpolate_update_logits
    require(subset in COUNTS,"Canonical loss scoring only exposes val source cohorts")
    dataset=IMAMultiReferenceDataset(cfg["manifest"],split="val",subset=subset,train=False,
                                    size=cfg["input_size"][0],cache_dir=cfg["cache_dir"])
    result={}
    with torch.no_grad():
        for cpu in loader(dataset,cfg["eval_batch"],cfg["workers"]):
            b=device_batch(cpu,torch.device("cuda"))
            cached=[_read_prediction(index,image_id) for image_id in cpu["image_id"]]
            for i,value in enumerate(cached):
                require(value["geometry"]==cpu["geometry"][i] and value["group_id"]==cpu["group_id"][i],
                        "Canonical subset batch does not share M geometry/case identity")
                require(torch.equal(value["pixel_valid"],cpu["pixel_valid"][i].bool()),
                        "Canonical subset valid pixels differ from M cache")
            zref=torch.stack([value["z_ref"] for value in cached])
            zon=torch.stack([value["z_on"] for value in cached])
            if alpha is not None:
                # Preserve the canonical CPU interpolation bytes also used by
                # original-coordinate restoration; only loss reductions move
                # to the historical GPU batch path.
                zon=interpolate_update_logits(zref,zon,alpha)
            zref,zon=zref.cuda(),zon.cuda()
            ell=per_reference_loss(zon,b["masks"],b["rater_present"],b["pixel_valid"],return_components=True)
            ellref=per_reference_loss(zref,b["masks"],b["rater_present"],b["pixel_valid"],return_components=True)
            diagnostics=objective_from_losses(ell["loss"],ellref["loss"],b["rater_present"],method="d0")
            for i,image_id in enumerate(cpu["image_id"]):
                require(image_id not in result,"Repeated canonical subset image")
                count=len(cpu["references"][i])
                present=b["rater_present"][i]
                require(int(present.sum())==count and bool(present[:count].all()),
                        "Reference order and presence differ from the original collate path")
                result[image_id]={"ell":{k:v[i,:count].cpu()[None] for k,v in ell.items()},
                                  "ellref":{k:v[i,:count].cpu()[None] for k,v in ellref.items()},
                                  "d":diagnostics["d"][i,present].cpu().numpy(),
                                  "rho":float((diagnostics["d"][i,present]>0).float().mean()),
                                  "J":float(diagnostics["J_per_image"][i]),
                                  "reference_ids":[r["reference_id"] for r in cpu["references"][i]],
                                  "loss_reduction_device":"cuda_FP32_original_batch_present"}
    require(len(result)==COUNTS[subset][0],"Canonical loss subset is incomplete")
    return result


def point_rows(cfg, index, *, run_id=None, alpha=None, amplitude=None, hard_only=False):
    """Read actual original masks; alpha applies to complete canonical logits."""
    import torch
    from rsi.datasets import restore_logits
    from rsi.metrics import evaluate_reference, update_diagnostics
    cohorts, _ = manifest_cohorts(cfg)
    rows = []
    run_id = run_id or index["metadata"]["group"]
    for subset, cohort in cohorts.items():
        canonical=canonical_components(cfg,index,subset,alpha)
        by_image = defaultdict(list)
        for ref in cohort:
            by_image[ref["image_id"]].append(ref)
        for image_id in sorted(by_image):
            refs = sorted(by_image[image_id], key=lambda r: r["reference_id"])
            cached = _read_prediction(index, image_id)
            zref, zon, zoff = [cached[k][None] for k in ("z_ref", "z_on", "z_off")]
            if alpha is not None:
                from rsi.tail_probe_model import interpolate_update_logits
                zon = interpolate_update_logits(zref, zon, alpha)
            geometry = cached["geometry"]
            _, originals = _canonical_masks(refs, geometry)
            valid = cached["pixel_valid"][None]
            require(canonical[image_id]["reference_ids"]==[r["reference_id"] for r in refs],
                    "Original mask loop and canonical loss reference orders differ")
            ell,ellref=canonical[image_id]["ell"],canonical[image_id]["ellref"]
            d,rho,J=[canonical[image_id][k] for k in ("d","rho","J")]
            orig = [restore_logits(z[0], geometry).numpy() for z in (zon,zref,zoff)]
            z, zr, zo = orig
            update = update_diagnostics(None, zon, zref, valid)
            update["changed_pixel_fraction_canonical"] = update.pop("changed_pixel_fraction")
            update["changed_pixel_fraction"] = float(np.mean((z >= 0) != (zr >= 0)))
            for r, (ref, gt) in enumerate(zip(refs, originals)):
                if hard_only:
                    from rsi.train import hard_overlap
                    ds, iou = hard_overlap(z >= 0, gt)
                    anchor, _ = hard_overlap(zr >= 0, gt)
                    metrics = dict(dice=ds, iou=iou, dice_anchor=anchor, gain_dice=ds-anchor)
                else:
                    metrics = evaluate_reference(z, gt, zr)
                    for name in ("loss", "bce", "soft_dice", "loss_anchor", "gain_loss"):
                        metrics[name + "_orig"] = metrics.pop(name)
                    pred, anchor = z >= 0, zr >= 0
                    pixels = gt.size
                    area, gtarea = int(pred.sum()), int(gt.sum())
                    offdice = (2 * np.count_nonzero((zo >= 0) & gt) /
                               ((zo >= 0).sum() + gtarea)) if ((zo >= 0).sum() + gtarea) else 1.
                    metrics.update(precision=(metrics["tp"] / area if area else 1.),
                                   recall=(metrics["tp"] / gtarea if gtarea else 1.),
                                   prediction_area=area, prediction_area_fraction=area/pixels,
                                   reference_area=gtarea, reference_area_fraction=gtarea/pixels,
                                   original_pixels=pixels, dice_off_current=float(offdice),
                                   gain_on_vs_off=float(metrics["dice"]-offdice),
                                   gain_off_vs_teacher=float(offdice-metrics["dice_anchor"]))
                metadata = dict(protocol_id=cfg.get("protocol_id", "RSI-TailProbe-20261004"), seed=17,
                                run_id=run_id, method=("complete_logit_interpolation" if alpha is not None
                                                       else "student_tail"),
                                comparison_family=("complete_update" if alpha is not None else "five_group"),
                                checkpoint_hash=index["metadata"]["checkpoint"]["sha256"],
                                teacher_B_hash=index["metadata"]["teacher_B"]["sha256"],
                                teacher_hash=index["teacher_hash"],
                                manifest_hash=index["metadata"]["manifest"]["sha256"],
                                cache_key=index["key"], prediction_cache_sha256=index["images"][image_id]["sha256"],
                                geometry_sha256=json_hash(geometry), image_id=image_id,
                                group_id=ref["group_id"], reference_id=ref["reference_id"],
                                image_sha256=ref.get("image_sha256", ""), mask_sha256=ref.get("mask_sha256", ""),
                                annotator_id=ref.get("annotator_id", ref.get("annotator", "")),
                                source=ref.get("source", ref.get("tool", "")),
                                tool=ref.get("tool", ""), skill_level=ref.get("skill_level", ""),
                                seg_filename=ref.get("seg_filename", ""), split="val", subset=subset,
                                reference_count=len(refs), alpha=(1. if alpha is None else float(alpha)),
                                action=(1. if alpha is None else float(alpha)),
                                **{"lambda": 3. if index["metadata"]["group"].endswith("RSI") else 0.},
                                amplitude_match_hash=(amplitude or ""),
                                canonical_loss=float(ell["loss"][0,r]), canonical_bce=float(ell["bce"][0,r]),
                                canonical_soft_dice=float(ell["soft_dice"][0,r]),
                                loss=float(ell["loss"][0,r]), bce=float(ell["bce"][0,r]),
                                soft_dice=float(ell["soft_dice"][0,r]),
                                loss_anchor=float(ellref["loss"][0,r]),
                                gain_loss=float(ellref["loss"][0,r]-ell["loss"][0,r]),
                                canonical_loss_teacher=float(ellref["loss"][0,r]),
                                canonical_loss_difference=float(d[r]), risk_activation=int(d[r] > 0),
                                risk_relu=float(max(d[r],0)), rho=rho, J=J,
                                loss_view="canonical_letterbox_valid",
                                loss_reduction_device=canonical[image_id]["loss_reduction_device"],
                                message_rms=cached["message_rms"])
                rows.append({**metadata, **metrics, **update})
            if len(rows) % 40 < len(refs):
                print(f"TailProbe metrics {run_id} {subset}: {len(rows)} reference rows", flush=True)
    require(len(rows) == 624, "A method must retain all 624 M/H/T1 references")
    return rows


def prove_f_reuse(cfg):
    """Full val hard/canonical readout before training; reuse in final export."""
    validate_scope(cfg, "val")
    import torch
    torch.set_num_threads(4)
    legacy = read_csv(cfg.get("legacy_reference_csv", PROJECT / "outputs/seed17/per_reference_gains.csv"))
    evidence = []
    for group, old in (("F-Mean", "D0"), ("F-RSI", "RSI-3")):
        index = cache_prediction(cfg, group)
        current = point_rows(cfg, index, hard_only=True)
        previous = {(r["subset"], r["image_id"], r["reference_id"]):r for r in legacy if r["run_id"] == old}
        require(len(previous) == 624, "Historical F export is not complete")
        deltas = defaultdict(list)
        for row in current:
            key = row["subset"], row["image_id"], row["reference_id"]
            require(key in previous, "F reuse reference cohort changed")
            prior = previous[key]
            require(prior["checkpoint_hash"] == row["checkpoint_hash"] and
                    prior["manifest_hash"] == row["manifest_hash"], "F reuse identities changed")
            for name in ("dice", "dice_anchor", "iou", "loss", "loss_anchor", "bce", "soft_dice"):
                deltas[name].append(abs(float(row[name])-float(prior[name])))
        maxima = {k: max(v) for k,v in deltas.items()}
        require(all(v == 0 for k,v in maxima.items() if k in ("dice", "dice_anchor", "iou")),
                f"Historical hard F evaluation differs: {group}: {maxima}")
        require(all(v <= 2e-7 for k,v in maxima.items() if k not in ("dice", "dice_anchor", "iou")),
                f"Historical canonical F loss differs: {group}: {maxima}")
        evidence.append(dict(group=group, historical_run=old, cache_key=index["key"],
                             references=624, maximum_absolute_errors=maxima,
                             hard_metrics_exact=True, canonical_loss_tolerance=2e-7,
                             checkpoint=index["metadata"]["checkpoint"], teacher_hash=index["teacher_hash"]))
        write_csv(output_root(cfg) / "reuse_validation" / (group + ".csv"), current)
    result = dict(status="PASS", groups=evidence, scoring_scope="M/H/T1 val", test_scoring_locked=True,
                  historical_source=file_record(cfg.get("legacy_reference_csv", PROJECT / "outputs/seed17/per_reference_gains.csv")),
                  note="Forward/objective/gradient/storage/runtime/budget acceptance must additionally pass preflight.")
    write_json(output_root(cfg) / "reuse_validation" / "readout.json", result)
    return result


def rms_control(S_mean, S_rsi):
    require(math.isfinite(S_mean) and math.isfinite(S_rsi) and S_mean >= 0 and S_rsi >= 0,
            "Train RMS statistics must be finite nonnegative means")
    raw = math.sqrt(S_rsi/S_mean) if S_mean else 0.
    clipped = min(1., max(0., raw))
    return dict(S_mean=S_mean, S_rsi=S_rsi, alpha_RMS_raw=raw, alpha_RMS=clipped,
                denominator_zero=S_mean == 0., raw_exceeds_one=raw > 1.,
                strict_amplitude_match=(S_mean > 0. and raw <= 1.),
                RMS_mean=math.sqrt(S_mean), RMS_rsi=math.sqrt(S_rsi),
                RMS_control=clipped*math.sqrt(S_mean),
                RMS_absolute_mismatch=abs(clipped*math.sqrt(S_mean)-math.sqrt(S_rsi)))


def calibrate_train_rms(cfg):
    """Canonical RGB only: no mask file, annotation target or GT metric is read."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from rsi.datasets import letterbox_geometry
    _, train_refs = manifest_cohorts(cfg)
    existing=output_root(cfg)/"amplitude_match_train.json"
    if existing.is_file():
        previous=json.loads(existing.read_text())
        identity=previous["identity"]
        require(identity["checkpoints"]=={g:file_record(selected_checkpoint(cfg,g)) for g in ("U-Mean","U-RSI")}
                and identity["teacher_B"]==file_record(cfg["teacher_B"])
                and identity["manifest"]==file_record(cfg["manifest"])
                and identity["implementation"]==implementation_identity()
                and identity["AMP"]==bool(cfg["amp"]),"Frozen train-side amplitude calibration inputs changed")
        require(previous["per_image_statistics"]["sha256"]==file_record(previous["per_image_statistics"]["path"])["sha256"],
                "Frozen train-side per-image amplitude statistics changed")
        require(previous["identity_hash"]==json_hash(identity),"Frozen amplitude identity hash changed")
        return previous
    unique = {}
    for ref in train_refs:
        unique.setdefault(ref["image_id"], ref)
    sources = sorted(unique)
    models = {g: load_model(cfg, g, selected_checkpoint(cfg,g))[0] for g in ("U-Mean", "U-RSI")}
    class RGBOnly(Dataset):
        def __len__(self):
            return len(sources)
        def __getitem__(self,i):
            image_id=sources[i]
            with Image.open(unique[image_id]["image_path"]) as handle:
                rgb=handle.convert("RGB")
                w,h=rgb.size
                geometry=letterbox_geometry(h,w,cfg["input_size"][0])
                shape=(geometry["resized_w"],geometry["resized_h"])
                small=np.asarray(rgb.resize(shape,Image.Resampling.BILINEAR))
            size=geometry["size"]
            image=np.zeros((size,size,3),np.uint8)
            valid=np.zeros((1,size,size),np.uint8)
            t,l=geometry["top"],geometry["left"]
            image[t:t+shape[1],l:l+shape[0]]=small
            valid[:,t:t+shape[1],l:l+shape[0]]=1
            return dict(image=torch.from_numpy(image).permute(2,0,1).float()/255,
                        pixel_valid=torch.from_numpy(valid).bool(),image_id=image_id)
    per_image=[]
    tick=time.monotonic()
    hashes={g:m.teacher_state_hash() for g,m in models.items()}
    require(len(set(hashes.values())) == 1,"RMS groups have different teachers")
    with torch.no_grad():
        for cpu in DataLoader(RGBOnly(),batch_size=cfg["eval_batch"],num_workers=cfg["workers"],shuffle=False):
            rgb,valid=cpu["image"].cuda(),cpu["pixel_valid"].cuda()
            updates={}
            teacher=None
            with torch.autocast("cuda",enabled=cfg["amp"]):
                for group,model in models.items():
                    ref,on=model.forward_pair(rgb,valid)
                    if teacher is None:
                        teacher=ref
                    else:
                        require(torch.equal(teacher,ref),"RMS groups disagree on fixed teacher logits")
                    delta=on.float()-ref.float()
                    updates[group]=[(float(delta[i][valid[i]].double().square().mean())) for i in range(len(rgb))]
            for i,image_id in enumerate(cpu["image_id"]):
                per_image.append(dict(image_id=image_id,S_mean=updates["U-Mean"][i],S_rsi=updates["U-RSI"][i]))
            print(f"TailProbe train-side RMS: {len(per_image)}/1471 RGB-only images",flush=True)
    require([r["image_id"] for r in per_image] == sources,"RMS calibration lost or reordered train images")
    control=rms_control(float(np.mean([r["S_mean"] for r in per_image])),
                        float(np.mean([r["S_rsi"] for r in per_image])))
    # Alpha is already locked. A second label-free pass measures the update
    # actually produced by FP32 interpolation, including rounding; it cannot
    # refit alpha or use validation performance to correct the match.
    from rsi.tail_probe_model import interpolate_update_logits
    match_tick=time.monotonic()
    actual_control=[]
    with torch.no_grad():
        for cpu in DataLoader(RGBOnly(),batch_size=cfg["eval_batch"],num_workers=cfg["workers"],shuffle=False):
            rgb,valid=cpu["image"].cuda(),cpu["pixel_valid"].cuda()
            with torch.autocast("cuda",enabled=cfg["amp"]):
                ref,on=models["U-Mean"].forward_pair(rgb,valid)
            scaled=interpolate_update_logits(ref,on,control["alpha_RMS"])
            delta=scaled-ref.float()
            for i,image_id in enumerate(cpu["image_id"]):
                actual_control.append((image_id,float(delta[i][valid[i]].double().square().mean())))
            print(f"TailProbe fixed-alpha actual RMS check: {len(actual_control)}/1471 RGB-only images",flush=True)
    require([r[0] for r in actual_control]==sources,"Fixed-alpha actual RMS check lost train images")
    control["RMS_control_target"]=control["RMS_control"]
    control["S_control_actual"]=float(np.mean([r[1] for r in actual_control]))
    control["RMS_control"]=math.sqrt(control["S_control_actual"])
    control["RMS_interpolation_roundoff_difference"]=control["RMS_control"]-control["RMS_control_target"]
    control["RMS_absolute_mismatch"]=abs(control["RMS_control"]-control["RMS_rsi"])
    control["actual_RMS_check_seconds"]=time.monotonic()-match_tick
    control["actual_RMS_check_refitted_alpha"]=False
    for row,actual in zip(per_image,actual_control):
        row["S_control_actual"]=actual[1]
    identity={"checkpoints":{g:file_record(selected_checkpoint(cfg,g)) for g in models},
              "teacher_B":file_record(cfg["teacher_B"]),"teacher_hash":next(iter(hashes.values())),
              "manifest":file_record(cfg["manifest"]),"implementation":implementation_identity(),
              "image_ids_sha256":json_hash(sources),"train_images":1471,"seed":17,
              "augmentation":False,"GT_used":False,"mask_files_opened":0,
              "reference_masks_materialized":False,"AMP":bool(cfg["amp"]),
              "view":"canonical_train_valid_pixels"}
    for group,model in models.items():
        require(model.teacher_state_hash() == hashes[group],"Teacher drifted during train-side calibration")
    statistics_path=output_root(cfg)/"amplitude_match_train_per_image.csv"
    write_csv(statistics_path,per_image)
    result={**control,"identity":identity,"identity_hash":json_hash(identity),
            "per_image_statistics":file_record(statistics_path),"seconds":time.monotonic()-tick,
            "GT_used":False,"fitted_once_on_train":True,"validation_used_for_fit":False,
            "test_used_for_fit":False,"alpha_selection":"predeclared sqrt(mean_image_MSE_RSI/mean_image_MSE_Mean), clip[0,1]"}
    write_json(output_root(cfg)/"amplitude_match_train.json",result)
    del models
    torch.cuda.empty_cache()
    return result


def historical_rows(cfg):
    legacy_path=cfg.get("legacy_reference_csv",PROJECT/"outputs/seed17/per_reference_gains.csv")
    legacy_hash=file_record(legacy_path)["sha256"]
    mapping={"D0":"Historical-D0","RSI-3":"Historical-RSI-3",
             "Shrink-D0-0.333333":"Historical-feature-Shrink-1of3",
             "Shrink-D0-0.666667":"Historical-feature-Shrink-2of3"}
    result=[]
    for old in read_csv(legacy_path):
        if old["run_id"] not in mapping:
            continue
        require(old["split"] == "val","Historical test scores are not authorized")
        row=dict(old,historical_run_id=old["run_id"],run_id=mapping[old["run_id"]],
                 comparison_family="historical",historical_source_sha256=legacy_hash,
                 same_condition_comparison="not_assumed")
        result.append(row)
    require(len(result)==4*624,"Required historical references are incomplete")
    return result


def run_export(cfg, scope="val"):
    validate_scope(cfg,scope)
    import torch
    torch.set_num_threads(4)
    root=output_root(cfg)
    root.mkdir(parents=True,exist_ok=True)
    manifest_cohorts(cfg)
    indices={}
    for group in GROUPS:
        indices[group]=cache_prediction(cfg,group)
    # Compare exact canonical teacher predictions and geometry image by image.
    import torch
    for image_id in sorted(indices["U-Mean"]["images"]):
        base=_read_prediction(indices["U-Mean"],image_id)
        for group,index in indices.items():
            other=_read_prediction(index,image_id)
            require(torch.equal(base["z_ref"],other["z_ref"]) and base["geometry"]==other["geometry"],
                    f"Fixed teacher or same-image geometry disagrees: {group}/{image_id}")
    amplitude_path=root/"amplitude_match_train.json"
    amplitude=calibrate_train_rms(cfg)
    from tools.cached_pilot_runner import ReferenceMetricCache, installed_metric_cache
    cache=ReferenceMetricCache(max_entries=60000)
    rows=[]
    with installed_metric_cache(cache,fast_exact=True):
        for group,index in indices.items():
            # rsi.metrics global lookup is patched by the context; import in
            # point_rows occurs inside this block and receives the exact cache.
            current=point_rows(cfg,index)
            write_csv(root/"per_reference"/(group+".csv"),current)
            rows.extend(current)
        for alpha in ALPHAS:
            run=f"U-Mean-logit-alpha-{alpha:.6f}"
            current=point_rows(cfg,indices["U-Mean"],run_id=run,alpha=alpha)
            write_csv(root/"per_reference"/(run+".csv"),current)
            rows.extend(current)
        current=point_rows(cfg,indices["U-Mean"],run_id="U-Mean-logit-RMS",alpha=amplitude["alpha_RMS"],
                           amplitude=file_record(amplitude_path)["sha256"])
        write_csv(root/"per_reference/U-Mean-logit-RMS.csv",current)
        rows.extend(current)
    rows.extend(historical_rows(cfg))
    write_csv(root/"per_reference_results.csv",rows)
    timing=benchmark_deployment(cfg)
    audit=dict(status="complete",reference_rows=len(rows),current_points=10,historical_points=4,
               source_cohorts=dict(COUNTS),test_scoring_locked=True,scope="val",seed=17,
               teacher_predictions_exact_same_all_groups=True,H_T1_reuse_M_logits=True,
               interpolation_view="canonical logits before restore/threshold",fixed_alphas=list(ALPHAS),
               amplitude_match=file_record(amplitude_path),checkpoint_indices={g:{"key":i["key"],
                   "checkpoint":i["metadata"]["checkpoint"],"teacher_hash":i["teacher_hash"]} for g,i in indices.items()},
               implementation=implementation_identity(),metric_cache_statistics=cache.statistics(),
               csv=file_record(root/"per_reference_results.csv"),deployment_timing=timing,
               finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/"export_audit.json",audit)
    return audit


def benchmark_deployment(cfg):
    """Time actual label-free deployment forwards; no forced NoMessage work."""
    import torch
    from rsi.datasets import IMAMultiReferenceDataset
    from rsi.train import loader,device_batch
    from rsi.tail_probe_model import interpolate_update_logits
    dataset=IMAMultiReferenceDataset(cfg["manifest"],split="val",subset="M",train=False,
                                    size=cfg["input_size"][0],cache_dir=cfg["cache_dir"])
    batch=device_batch(next(iter(loader(dataset,cfg["eval_batch"],0))),torch.device("cuda"))
    rgb,valid=batch["image"],batch["pixel_valid"]
    records=[]
    for group in GROUPS:
        model,_=load_model(cfg,group,selected_checkpoint(cfg,group))
        modes=["student_once"]+( ["complete_logit_interpolation"] if group=="U-Mean" else [])
        for mode in modes:
            def forward():
                with torch.autocast("cuda",enabled=cfg["amp"]):
                    if mode=="student_once":
                        return model(rgb,valid)
                    ref,on=model.forward_pair(rgb,valid)
                    return interpolate_update_logits(ref,on,1/3)
            with torch.no_grad():
                for _ in range(10):
                    forward()
                torch.cuda.synchronize()
                tick=time.perf_counter()
                for _ in range(50):
                    forward()
                torch.cuda.synchronize()
                seconds=time.perf_counter()-tick
            records.append(dict(group=group,mode=mode,images_per_batch=len(rgb),timed_batches=50,
                                warmup_batches=10,seconds=seconds,milliseconds_per_batch=1000*seconds/50,
                                milliseconds_per_image=1000*seconds/(50*len(rgb)),
                                transformer_forwards_per_batch=int(group!="U-NoMessage"),
                                tail_decodes_per_batch=1 if mode=="student_once" else 2,
                                teacher_forward=mode!="student_once",GT_used=False,AMP=bool(cfg["amp"]),
                                includes_input_pipeline=False,checkpoint_sha256=file_record(selected_checkpoint(cfg,group))["sha256"]))
        del model
        torch.cuda.empty_cache()
    result=dict(status="complete",records=records,GPU=torch.cuda.get_device_name(),
                image_ids=batch["image_id"],GT_model_input=False,no_message_forced_transformer=False,
                scope="fixed real val batch forward latency; not end-to-end clinical throughput")
    write_json(output_root(cfg)/"deployment_timing.json",result)
    write_csv(output_root(cfg)/"deployment_timing.csv",records)
    return result


def finite(value, label):
    value=float(value)
    require(math.isfinite(value),f"Nonfinite {label}")
    return value


def tail_mean(values):
    values=np.asarray(values,float)
    require(values.ndim==1 and len(values)>0 and np.isfinite(values).all(),"Invalid tail vector")
    return float(np.sort(values)[:max(1,math.ceil(.1*len(values)))].mean())


def summarize_point(rows):
    """All additive statistics use image then reference averaging."""
    by_image=defaultdict(list)
    seen=set()
    for row in rows:
        require(row.get("split")=="val","Only validation exports may be summarized")
        key=row["image_id"],row["reference_id"]
        require(key not in seen,"Duplicate per-reference row in one point")
        seen.add(key)
        by_image[key[0]].append(row)
    require(bool(by_image),"Empty method point")
    ids=sorted(by_image)
    group_ids=[]
    arrays=defaultdict(list)
    anchors={}
    gains={}
    metric_columns=("dice","iou","thresholded_jaccard","bf1","bf1_025","bf1_1pct",
                    "hd95","hd95_normalized","loss","bce","soft_dice","loss_orig",
                    "bce_orig","soft_dice_orig","precision","recall","prediction_area",
                    "prediction_area_fraction","reference_area","fp","fn","tp","fp_added",
                    "fn_added","fp_removed","fn_removed","logit_delta_rms","changed_pixel_fraction",
                    "changed_pixel_fraction_canonical","rho","J","message_rms","dice_off_current",
                    "gain_on_vs_off","gain_off_vs_teacher","dice_anchor")
    for image_id in ids:
        refs=sorted(by_image[image_id],key=lambda r:r["reference_id"])
        groups={str(r.get("group_id") or "") for r in refs}
        require(len(groups)==1,"Case group changes within an image")
        group_ids.append(groups.pop())
        g=np.asarray([finite(r["dice"],"dice")-finite(r["dice_anchor"],"teacher dice") for r in refs])
        require(np.isfinite(g).all(),"Invalid gains")
        gains[image_id]=g
        for ref in refs:
            anchors[(image_id,ref["reference_id"])]=finite(ref["dice_anchor"],"teacher Dice")
            if "gain_dice" in ref:
                require(abs(finite(ref["gain_dice"],"gain")-(finite(ref["dice"],"Dice")-
                    finite(ref["dice_anchor"],"teacher")))<1e-12,"Gain inconsistent with fixed teacher")
        arrays["G_plus"].append(float(np.maximum(g,0).mean()))
        arrays["H_minus"].append(float(np.maximum(-g,0).mean()))
        arrays["H_epsilon"].append(float(np.maximum(-g-EPSILON,0).mean()))
        arrays["mean_gain"].append(float(g.mean()))
        arrays["worst_reference_gain"].append(float(g.min()))
        arrays["mean_harm"].append(float(g.mean()<-EPSILON))
        arrays["any_harm"].append(float(g.min()<-EPSILON))
        arrays["all_harm"].append(float(g.max()<-EPSILON))
        arrays["sign_conflict"].append(float(g.min()<-EPSILON and g.max()>EPSILON))
        for eps in (0.,.005,.01,.02,.05):
            suffix=f"{eps:.3f}"
            arrays[f"H_epsilon_{suffix}"].append(float(np.maximum(-g-eps,0).mean()))
            arrays[f"mean_harm_{suffix}"].append(float(g.mean()<-eps))
            arrays[f"any_harm_{suffix}"].append(float(g.min()<-eps))
            arrays[f"all_harm_{suffix}"].append(float(g.max()<-eps))
            arrays[f"sign_conflict_{suffix}"].append(float(g.min()<-eps and g.max()>eps))
        for name in metric_columns:
            values=[r.get(name) for r in refs]
            if all(v not in (None,"") for v in values):
                arrays[name].append(float(np.mean([finite(v,name) for v in values])))
            elif any(v not in (None,"") for v in values):
                raise ValueError(f"Partially exported metric {name}")
        # Original historical exports have per-reference loss, from which rho
        # and Jensen gap can be reconstructed without changing those files.
        if not all(r.get("rho") not in (None,"") for r in refs):
            d=np.asarray([finite(r["loss"],"loss")-finite(r["loss_anchor"],"teacher loss") for r in refs])
            arrays["rho"].append(float((d>0).mean()))
            arrays["J"].append(float(np.maximum(d,0).mean()-max(d.mean(),0.)))
    vectors={k:np.asarray(v,float) for k,v in arrays.items()}
    require(all(len(v)==len(ids) for v in vectors.values()),"Metric is present for only some images")
    values={k:float(v.mean()) for k,v in vectors.items()}
    require(abs(values["mean_gain"]-(values["G_plus"]-values["H_minus"]))<1e-12,
            "Mean gain must equal G+ minus unadjusted H-")
    values.update(worst_tail_10pct=tail_mean(vectors["worst_reference_gain"]),
                  mean_gain_tail_10pct=tail_mean(vectors["mean_gain"]),
                  worst_tail_n=max(1,math.ceil(.1*len(ids))),n_images=len(ids),n_references=len(rows))
    return dict(image_ids=ids,group_ids=group_ids,reference_keys=seen,anchors=anchors,
                vectors=vectors,values=values,gains=gains)


def bootstrap_units(ids, groups, image_only=False):
    units=defaultdict(list)
    missing={"", "unknown", "none", "null", "nan", "na", "n/a"}
    for i,(image,group) in enumerate(zip(ids,groups)):
        key=("image",image) if image_only or str(group).strip().lower() in missing else ("known",str(group))
        units[key].append(i)
    return [np.asarray(v,dtype=int) for v in units.values()]


def paired_comparison(current, control, *, image_only=False, draws=None, repeats=2000, seed=17):
    require(current["image_ids"]==control["image_ids"] and current["group_ids"]==control["group_ids"],
            "Paired methods have different image/case cohorts")
    require(current["reference_keys"]==control["reference_keys"] and current["anchors"]==control["anchors"],
            "Paired methods have different reference or teacher identities")
    units=bootstrap_units(current["image_ids"],current["group_ids"],image_only)
    if draws is None:
        draws=np.random.default_rng(seed).integers(len(units),size=(repeats,len(units)))
    draws=np.asarray(draws)
    require(draws.shape==(repeats,len(units)) and np.issubdtype(draws.dtype,np.integer) and
            draws.min()>=0 and draws.max()<len(units),"Invalid paired bootstrap indices")
    names=[k for k in current["vectors"] if k in control["vectors"] and k!="worst_reference_gain"]
    counts=np.asarray([len(u) for u in units])
    diffs=np.column_stack([current["vectors"][k]-control["vectors"][k] for k in names])
    sums=np.asarray([diffs[u].sum(axis=0) for u in units])
    samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
    tail_samples=np.empty((repeats,2))
    for j,draw in enumerate(draws):
        indices=np.concatenate([units[int(d)] for d in draw])
        for t,name in enumerate(("worst_reference_gain","mean_gain")):
            tail_samples[j,t]=tail_mean(current["vectors"][name][indices])-tail_mean(control["vectors"][name][indices])
    all_names=[*names,*TAIL_METRICS]
    intervals=np.percentile(np.column_stack((samples,tail_samples)),[2.5,97.5],axis=0)
    result=dict(bootstrap_unit="image" if image_only else "known_group_else_image",bootstrap_units=len(units),
                bootstrap_repeats=repeats,bootstrap_seed=seed,n_images=len(current["image_ids"]),
                n_references=len(current["reference_keys"]),tail_reranked_each_draw=True,
                interval_scope="exploratory single-seed selected development validation; not equivalence or confirmation")
    for i,name in enumerate(all_names):
        result[name+"_difference"]=current["values"][name]-control["values"][name]
        result[name+"_ci_low"]=float(intervals[0,i])
        result[name+"_ci_high"]=float(intervals[1,i])
    return result


def fixed_categories(umean_point):
    result={}
    for image,gains in umean_point["gains"].items():
        result[image]=("all-benefit" if gains.min()>EPSILON else "all-harm" if gains.max()<-EPSILON
                       else "conflict" if gains.min()<-EPSILON and gains.max()>EPSILON else "other")
    return result


def factor_differences(points, same_conditions):
    expressions={"tail_Mean":{"U-Mean":1,"F-Mean":-1},
                 "tail_RSI":{"U-RSI":1,"F-RSI":-1},
                 "RSI_frozen":{"F-RSI":1,"F-Mean":-1},
                 "RSI_unfrozen":{"U-RSI":1,"U-Mean":-1},
                 "difference_in_differences":{"U-RSI":1,"U-Mean":-1,"F-RSI":-1,"F-Mean":1}}
    rows=[]
    for subset in COUNTS:
        baseline=points[("U-Mean",subset)]
        for expression,weights in expressions.items():
            participants=[points[(g,subset)] for g in weights]
            require(all(p["image_ids"]==baseline["image_ids"] and p["reference_keys"]==baseline["reference_keys"]
                        and p["anchors"]==baseline["anchors"] for p in participants),"Factor cohorts mismatch")
            for image_only in (False,True):
                units=bootstrap_units(baseline["image_ids"],baseline["group_ids"],image_only)
                draws=np.random.default_rng(17).integers(len(units),size=(2000,len(units)))
                counts=np.asarray([len(u) for u in units])
                for metric in ("dice","G_plus","H_minus","H_epsilon"):
                    vector=sum(w*points[(g,subset)]["vectors"][metric] for g,w in weights.items())
                    sums=np.asarray([vector[u].sum() for u in units])
                    samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
                    ci=np.percentile(samples,[2.5,97.5])
                    rows.append(dict(subset=subset,expression=expression,metric=metric,
                                     difference=float(vector.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                                     weights=json.dumps(weights,sort_keys=True),bootstrap_units=len(units),
                                     bootstrap_unit="image" if image_only else "known_group_else_image",
                                     bootstrap_repeats=2000,bootstrap_seed=17,
                                     same_condition_mechanism_comparison=bool(same_conditions),
                                     interpretation="same-condition exploratory factor" if same_conditions else "descriptive historical factor only"))
    return rows


def screening_decision(main, amplitude, *, engineering_ok, same_conditions, rules=None):
    """Deterministic resource screen; point estimates are not formal tests."""
    rules=rules or default_screening_rules()
    require(rules==default_screening_rules(),"Task thresholds changed after preregistration")
    M,R,N,C=[main[(name,"M")] for name in ("U-Mean","U-RSI","U-NoMessage","U-Mean-logit-RMS")]
    A=rules["route_A"];B=rules["route_B"]
    route_A_checks={"dice_advantage":R["dice"]-max(M["dice"],N["dice"],C["dice"])>=A["dice_advantage_over_max_M_N_C"],
                    "H_epsilon_not_increased":R["H_epsilon"]<=M["H_epsilon"]+A["H_epsilon_R_minus_M_max"]}
    route_B_checks={"dice_vs_M_N":R["dice"]>=max(M["dice"],N["dice"])-B["dice_deficit_max_M_N"],
                    "dice_vs_C":R["dice"]>=C["dice"]-B["dice_deficit_C"],
                    "G_positive_M":M["G_plus"]>0.,
                    "G_retained":M["G_plus"]>0. and R["G_plus"]>=B["G_retention_min"]*M["G_plus"],
                    "H_M_positive":M["H_epsilon"]>0.,"H_C_positive":C["H_epsilon"]>0.,
                    "H_M_absolute":M["H_epsilon"]-R["H_epsilon"]>=B["M_H_absolute_reduction"],
                    "H_M_relative":M["H_epsilon"]>0. and (M["H_epsilon"]-R["H_epsilon"])/M["H_epsilon"]>=B["M_H_relative_reduction"],
                    "H_C_absolute":C["H_epsilon"]-R["H_epsilon"]>=B["C_H_absolute_reduction"],
                    "H_C_relative":C["H_epsilon"]>0. and (C["H_epsilon"]-R["H_epsilon"])/C["H_epsilon"]>=B["C_H_relative_reduction"]}
    route_A,route_B=all(route_A_checks.values()),all(route_B_checks.values())
    flags=[]
    dominant=[]
    for alpha in ALPHAS:
        name=f"U-Mean-logit-alpha-{alpha:.6f}"
        value=main[(name,"M")]
        no_worse=value["dice"]>=R["dice"] and value["G_plus"]>=R["G_plus"] and value["H_epsilon"]<=R["H_epsilon"]
        practical=(value["dice"]-R["dice"]>=rules["simple_control"]["dice_or_G_practical_difference"] or
                   value["G_plus"]-R["G_plus"]>=rules["simple_control"]["dice_or_G_practical_difference"] or
                   R["H_epsilon"]-value["H_epsilon"]>=rules["simple_control"]["H_practical_difference"])
        if no_worse and practical:
            dominant.append(name)
    if dominant:
        flags.append("SIMPLE_CONTROL_EXPLAINS_OR_EXCEEDS")
    source_regressions=[]
    for subset in ("H","T1"):
        for control in ("U-Mean","U-Mean-logit-RMS","U-NoMessage"):
            difference=main[("U-RSI",subset)]["dice"]-main[(control,subset)]["dice"]
            if difference < -rules["source_dice_regression"]:
                source_regressions.append(dict(subset=subset,comparator=control,dice_difference=difference))
    if source_regressions:
        flags.append("SOURCE_REGRESSION")
    harm_regressions=[]
    for name,value in (("U-Mean",M),("U-Mean-logit-RMS",C)):
        if R["H_minus"]-value["H_minus"]>rules["H_minus_increase_max"]:
            harm_regressions.append(dict(comparator=name,metric="H_minus",difference=R["H_minus"]-value["H_minus"]))
        if R["worst_tail_10pct"]-value["worst_tail_10pct"] < -rules["worst_tail_decrease_max"]:
            harm_regressions.append(dict(comparator=name,metric="worst_tail_10pct",difference=R["worst_tail_10pct"]-value["worst_tail_10pct"]))
    if harm_regressions:
        flags.append("TOTAL_HARM_OR_TAIL_REGRESSION")
    if M["H_epsilon"]<rules["low_risk_headroom_M"] or C["H_epsilon"]<rules["low_risk_headroom_C"]:
        flags.append("LOW_RISK_HEADROOM")
    degenerate=amplitude["denominator_zero"] or amplitude["raw_exceeds_one"]
    if degenerate:
        flags.append("AMPLITUDE_CONTROL_DEGENERATE")
    if not same_conditions:
        flags.append("F_FACTOR_CONDITIONS_NOT_EQUIVALENT")
    if not engineering_ok:
        status="INVALID"
    elif dominant:
        status="STOP_CURRENT_RSI"
    elif not (route_A or route_B):
        status="STOP_CURRENT_RSI"
    elif source_regressions or harm_regressions or degenerate or not same_conditions:
        status="INCONCLUSIVE"
    else:
        status="CANDIDATE_FOR_CONFIRMATION"
    return dict(status=status,route_A_pass=route_A,route_B_pass=route_B,route_A_checks=route_A_checks,
                route_B_checks=route_B_checks,flags=flags,simple_controls_dominating=dominant,
                source_regressions=source_regressions,total_harm_or_tail_regressions=harm_regressions,
                engineering_compliant=bool(engineering_ok),same_condition_factors=bool(same_conditions),
                rules=rules,rules_sha256=json_hash(rules),main_metrics={k:v for k,v in zip(("M","R","N","C"),(M,R,N,C))},
                inference_scope="single seed17 selected development validation; no clinical threshold or formal test",
                authorized_round_finished=True,next_seed_started=False,test_scoring_locked=True)


def _acceptance_complete(value):
    status=str(value.get("status","")).upper()
    if status in {"FAIL","FAILED","INVALID","PENDING","INCOMPLETE"}:
        return False
    if status not in {"PASS","PASSED","COMPLETE","COMPLETED"}:
        return False
    def failed(v):
        if isinstance(v,dict):
            return any((str(k).lower() in {"status","result"} and str(x).upper() in {"FAIL","FAILED"})
                       or failed(x) for k,x in v.items())
        if isinstance(v,list):
            return any(failed(x) for x in v)
        return False
    return not failed(value)


def validate_deliverables(cfg, rows):
    root=output_root(cfg)
    prereg_path=root/"preregistration.json"
    require(prereg_path.is_file(),"Preregistration missing")
    prereg=json.loads(prereg_path.read_text())
    task_hash=file_record(cfg["taskbook"])["sha256"]
    require(task_hash==cfg["taskbook_sha256"],"Taskbook bytes changed")
    require(task_hash in json.dumps(prereg),"Preregistration does not bind taskbook SHA")
    rules=prereg.get("screening_rules",prereg.get("rules"))
    require(rules==default_screening_rules(),"Preregistered thresholds differ from the fixed analysis rules")
    cohorts,_=manifest_cohorts(cfg)
    expected={s:{(r["image_id"],r["reference_id"]) for r in refs} for s,refs in cohorts.items()}
    groups=defaultdict(list)
    for row in rows:
        require(row.get("split")=="val" and str(row.get("seed"))=="17","Only seed17 val rows allowed")
        groups[(row["run_id"],row["subset"])].append(row)
    current_names={*GROUPS,*[f"U-Mean-logit-alpha-{a:.6f}" for a in ALPHAS],"U-Mean-logit-RMS"}
    for name in current_names:
        for subset in COUNTS:
            refs=groups[(name,subset)]
            keys={(r["image_id"],r["reference_id"]) for r in refs}
            require(len(refs)==len(keys) and keys==expected[subset],"Incomplete or changed exported reference cohort")
    audit_path=root/"export_audit.json"
    export_audit=json.loads(audit_path.read_text())
    require(export_audit.get("status")=="complete" and export_audit["csv"]["sha256"]==file_record(root/"per_reference_results.csv")["sha256"],
            "Final export audit is absent or does not bind CSV bytes")
    require(export_audit["implementation"]==implementation_identity(),"Analysis or model code changed after inference")
    acceptance=json.loads((root/"engineering_acceptance.json").read_text())
    reuse=json.loads((root/"reuse_decision.json").read_text())
    engineering_ok=_acceptance_complete(acceptance)
    same_conditions=bool(reuse.get("same_condition_factors",reuse.get("same_conditions",True)))
    budgets=[]
    for group in GROUPS:
        checkpoint=selected_checkpoint(cfg,group)
        done_path=checkpoint.parent/"DONE.json"
        done=json.loads(done_path.read_text())
        require(int(done.get("epochs",0))==40 and int(done.get("seed",0))==17,"A group lacks the registered 40-epoch seed17 budget")
        actual=file_record(checkpoint)
        expected_hash=done.get("best_sha256",done.get("checkpoint_sha256",done.get("best_checkpoint_sha256")))
        if not expected_hash:
            expected_hash=done.get("selected_checkpoint_sha256")
        require(expected_hash==actual["sha256"],f"DONE does not bind selected checkpoint {group}")
        epoch_path=checkpoint.parent/"epochs.csv"
        if not epoch_path.is_file():
            epoch_path=checkpoint.parent/"epoch_diagnostics.csv"
        epoch_rows=read_csv(epoch_path)
        epochs=[int(r["epoch"]) for r in epoch_rows]
        require(len(epochs)==40 and epochs==list(range(1,41)),f"40 committed epoch logs missing: {group}")
        budgets.append(dict(group=group,checkpoint=actual,DONE=file_record(done_path),epochs=file_record(epoch_path),
                            best_epoch=done["best_epoch"],best_val_dice=done["best_val_dice"],
                            reuse=group.startswith("F-") and not checkpoint.is_relative_to(Path(cfg["run_root"]).resolve())))
    return dict(engineering_ok=engineering_ok,same_conditions=same_conditions,budgets=budgets,
                rules=rules,preregistration=file_record(prereg_path),groups=groups)


def run_summarize(cfg):
    validate_scope(cfg,"val")
    root=output_root(cfg)
    rows=read_csv(root/"per_reference_results.csv")
    evidence=validate_deliverables(cfg,rows)
    grouped=evidence.pop("groups")
    points={key:summarize_point(refs) for key,refs in grouped.items()}
    for subset in COUNTS:
        baseline=points[("U-Mean",subset)]
        for (run,s),point in points.items():
            if s!=subset or run.startswith("Historical-"):
                continue
            require(point["anchors"]==baseline["anchors"] and point["reference_keys"]==baseline["reference_keys"],
                    "Methods have different reference anchors")
    main=[]
    values={}
    for (name,subset),point in sorted(points.items()):
        metadata=grouped[(name,subset)][0]
        row=dict(run_id=name,subset=subset,seed=17,split="val",checkpoint_hash=metadata["checkpoint_hash"],
                 manifest_hash=metadata["manifest_hash"],alpha=float(metadata.get("alpha",1)),
                 method=metadata.get("method",""),comparison_family=metadata.get("comparison_family",""),
                 **point["values"])
        main.append(row)
        values[(name,subset)]=point["values"]
    require(len(bootstrap_units(points[("U-Mean","M")]["image_ids"],points[("U-Mean","M")]["group_ids"]))==222,
            "Known M resampling units differ from the registered 222 groups")
    write_csv(root/"main_table.csv",main)
    pairs=set()
    for i,a in enumerate(GROUPS):
        for b in GROUPS[i+1:]:
            pairs.add((a,b))
    for comparator in ("U-Mean","U-NoMessage","U-Mean-logit-RMS",*[f"U-Mean-logit-alpha-{a:.6f}" for a in ALPHAS]):
        pairs.add(("U-RSI",comparator))
    comparisons=[]
    for subset in COUNTS:
        baseline=points[("U-Mean",subset)]
        for image_only in (False,True):
            units=bootstrap_units(baseline["image_ids"],baseline["group_ids"],image_only)
            draws=np.random.default_rng(17).integers(len(units),size=(2000,len(units)))
            for a,b in sorted(pairs):
                comparison=paired_comparison(points[(a,subset)],points[(b,subset)],image_only=image_only,draws=draws)
                comparisons.append(dict(current=a,comparator=b,subset=subset,**comparison))
    write_csv(root/"paired_comparisons.csv",comparisons)
    write_csv(root/"factor_differences.csv",factor_differences(points,evidence["same_conditions"]))
    categories=fixed_categories(points[("U-Mean","M")])
    write_json(root/"fixed_U_Mean_case_categories.json",dict(definition="M U-Mean selected checkpoint hard gain; GT diagnostic only",
                                                           epsilon=.005,categories=categories,inference_use=False))
    decomposition=[]
    for (run,subset),point in sorted(points.items()):
        for category in ("all-benefit","all-harm","conflict","other"):
            indices=np.asarray([i for i,image in enumerate(point["image_ids"]) if categories[image]==category],int)
            if not len(indices):
                decomposition.append(dict(run_id=run,subset=subset,category=category,n_images=0,image_fraction=0.))
                continue
            record=dict(run_id=run,subset=subset,category=category,n_images=len(indices),
                        image_fraction=len(indices)/len(point["image_ids"]),fixed_grouping="selected_U_Mean_M_GT_diagnostic_only")
            for metric in CORE_METRICS:
                record[metric]=float(point["vectors"][metric][indices].mean())
                record[metric+"_overall_contribution"]=float(point["vectors"][metric][indices].sum()/len(point["image_ids"]))
                control=points[("U-Mean",subset)]
                record[metric+"_change_vs_U_Mean"]=float((point["vectors"][metric]-control["vectors"][metric])[indices].mean())
            decomposition.append(record)
    write_csv(root/"group_decomposition.csv",decomposition)
    amplitude=json.loads((root/"amplitude_match_train.json").read_text())
    decision=screening_decision(values,amplitude,engineering_ok=evidence["engineering_ok"],
                                same_conditions=evidence["same_conditions"],rules=evidence["rules"])
    decision["evidence"]=evidence
    decision["paired_intervals"]=[r for r in comparisons if r["current"]=="U-RSI" and
                                  r["comparator"] in {"U-Mean","U-NoMessage","U-Mean-logit-RMS"}]
    # A threshold screen is a resource decision. CI crossing zero never implies
    # equivalence; coherent independent confirmation remains a separate task.
    decision["CI_interpretation"]="All 2000 paired intervals are exploratory; crossing 0 is not equivalence or noninferiority."
    write_json(root/"decision.json",decision)
    epoch_rows=[]
    for budget in evidence["budgets"]:
        for row in read_csv(budget["epochs"]["path"]):
            epoch_rows.append(dict(row,group=budget["group"],historical_reuse=budget["reuse"]))
    write_csv(root/"epoch_diagnostics.csv",epoch_rows)
    plot_results(root,main,epoch_rows)
    panels=case_panels(cfg,points,categories,amplitude)
    report=report_text(cfg,decision,main,comparisons,decomposition,amplitude,evidence,panels)
    (root/"experiment_report.md").write_text(report,encoding="utf-8")
    (root/"experiment_report.txt").write_text(report,encoding="utf-8")
    delivery={"status":"complete" if evidence["engineering_ok"] else "INVALID","decision":decision["status"],
              "seed":17,"groups":5,"new_training_groups":sum(not b["reuse"] for b in evidence["budgets"]),
              "test_scoring_locked":True,"weight_upload":False,"weights_location":"local training run directories",
              "input_evidence":evidence,"taskbook":file_record(cfg["taskbook"]),
              "outputs":[],"complete_scope":"five-group seed17 mechanism screen; no automatic extensions"}
    names=("per_reference_results.csv","export_audit.json","main_table.csv","paired_comparisons.csv",
           "factor_differences.csv","group_decomposition.csv","amplitude_match_train.json",
           "amplitude_match_train_per_image.csv","fixed_U_Mean_case_categories.json","epoch_diagnostics.csv",
           "decision.json","experiment_report.md","experiment_report.txt","gain_harm_tradeoff.png",
           "learning_curves.png","case_panels/selection.json","deployment_timing.json","deployment_timing.csv")
    delivery["outputs"]=[file_record(root/name) for name in names]
    write_json(root/"delivery_manifest.json",delivery)
    return delivery


def plot_results(root, main, epoch_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points=[r for r in main if r["subset"]=="M"]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for row in points:
        historical=row["run_id"].startswith("Historical-")
        marker="x" if historical else "o"
        axes[0].scatter(row["G_plus"],row["H_epsilon"],marker=marker,s=42)
        axes[1].scatter(row["H_epsilon"],row["dice"],marker=marker,s=42)
        label=row["run_id"].replace("U-Mean-logit-alpha-","logit a=")
        axes[0].annotate(label,(row["G_plus"],row["H_epsilon"]),fontsize=6,xytext=(3,3),textcoords="offset points")
        axes[1].annotate(label,(row["H_epsilon"],row["dice"]),fontsize=6,xytext=(3,3),textcoords="offset points")
    axes[0].set(xlabel="G+ (larger better)",ylabel="H epsilon (smaller better)",title="M validation gain / harm")
    axes[1].set(xlabel="H epsilon (smaller better)",ylabel="Dice",title="Selected development checkpoints")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(root/"gain_harm_tradeoff.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for group in GROUPS:
        history=sorted([r for r in epoch_rows if r["group"]==group],key=lambda r:int(r["epoch"]))
        for ax,name,title in zip(axes.ravel(),("val_dice","val_G_plus","val_H_epsilon","val_H_minus"),
                                 ("M Dice","M G+","M H epsilon","M total H-")):
            xs=[];ys=[]
            for row in history:
                if row.get(name) not in (None,""):
                    xs.append(int(row["epoch"]));ys.append(finite(row[name],name))
            if ys:
                ax.plot(xs,ys,label=group,linewidth=1)
            ax.set(xlabel="epoch (all 40 retained)",ylabel=title)
            ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"learning_curves.png",dpi=160)
    plt.close(fig)


def case_panels(cfg, points, categories, amplitude):
    """Fixed U-Mean harm/benefit/conflict and seed17 random; local pixels only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from rsi.datasets import _binary_mask,restore_logits
    from rsi.tail_probe_model import interpolate_update_logits
    root=output_root(cfg)/"case_panels"
    root.mkdir(parents=True,exist_ok=True)
    point=points[("U-Mean","M")]
    selectors=[]
    harmed=[i for i,image in enumerate(point["image_ids"]) if categories[image]=="all-harm"]
    improved=[i for i,image in enumerate(point["image_ids"]) if categories[image]=="all-benefit"]
    conflict=[i for i,image in enumerate(point["image_ids"]) if categories[image]=="conflict"]
    if harmed:
        chosen=min(harmed,key=lambda i:(point["vectors"]["mean_gain"][i],point["image_ids"][i]))
        selectors.append(("harmed",point["image_ids"][chosen]))
    if improved:
        chosen=max(improved,key=lambda i:(point["vectors"]["mean_gain"][i],point["image_ids"][i]))
        selectors.append(("improved",point["image_ids"][chosen]))
    if conflict:
        chosen=min(conflict,key=lambda i:(point["vectors"]["worst_reference_gain"][i],point["image_ids"][i]))
        selectors.append(("conflict",point["image_ids"][chosen]))
    rng=np.random.default_rng(17)
    selectors.append(("random",point["image_ids"][int(rng.integers(len(point["image_ids"]))) ]))
    cohorts,_=manifest_cohorts(cfg)
    by_image=defaultdict(list)
    for ref in cohorts["M"]:
        by_image[ref["image_id"]].append(ref)
    indices={group:cache_prediction(cfg,group) for group in ("U-Mean","U-RSI","U-NoMessage")}
    outputs=[]
    for rule,image_id in selectors:
        refs=sorted(by_image[image_id],key=lambda r:r["reference_id"])
        cached={g:_read_prediction(index,image_id) for g,index in indices.items()}
        base=cached["U-Mean"]
        geometry=base["geometry"]
        predictions={"fixed teacher":base["z_ref"],"U-Mean":base["z_on"],
                     "U-RSI":cached["U-RSI"]["z_on"],"U-NoMessage":cached["U-NoMessage"]["z_on"],
                     "RMS":interpolate_update_logits(base["z_ref"][None],base["z_on"][None],amplitude["alpha_RMS"])[0]}
        masks={name:restore_logits(z,geometry).numpy()>=0 for name,z in predictions.items()}
        with Image.open(refs[0]["image_path"]) as handle:
            rgb=np.asarray(handle.convert("RGB"))
        panes=[("source RGB",rgb,"rgb")]
        for ref in refs:
            panes.append((f"{ref['reference_id']}\n{ref.get('annotator_id','')} / {ref.get('tool','')}",
                          _binary_mask(ref["mask_path"]),"mask"))
        panes.extend((name,mask,"mask") for name,mask in masks.items())
        for name in ("U-Mean","U-RSI","RMS","U-NoMessage"):
            difference=masks[name].astype(np.int8)-masks["fixed teacher"].astype(np.int8)
            panes.append((name+" changes vs teacher\nred removed / blue added",difference,"diff"))
        ncol=5;nrow=math.ceil(len(panes)/ncol)
        fig,axes=plt.subplots(nrow,ncol,figsize=(20,4*nrow))
        for ax,(title,pixels,kind) in zip(axes.ravel(),panes):
            if kind=="rgb":
                ax.imshow(pixels)
            elif kind=="diff":
                ax.imshow(pixels,cmap="bwr",vmin=-1,vmax=1,interpolation="nearest")
            else:
                ax.imshow(pixels,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            ax.set_title(title,fontsize=8)
            ax.axis("off")
        for ax in axes.ravel()[len(panes):]:
            ax.axis("off")
        fig.suptitle(f"{rule}: {image_id}; fixed U-Mean category={categories[image_id]}; annotation differences are diagnostic",fontsize=11)
        fig.tight_layout()
        target=root/(rule+"_"+image_id+".png")
        fig.savefig(target,dpi=100)
        plt.close(fig)
        outputs.append(dict(rule=rule,image_id=image_id,category=categories[image_id],
                            panel=file_record(target),reference_ids=[r["reference_id"] for r in refs],
                            image_sha256=refs[0].get("image_sha256",""),
                            prediction_cache_keys={g:i["key"] for g,i in indices.items()}))
    result=dict(selection_rules={"harmed":"minimum mean gain among fixed U-Mean all-harm",
                                 "improved":"maximum mean gain among fixed U-Mean all-benefit",
                                 "conflict":"minimum worst-reference gain among fixed U-Mean conflict",
                                 "random":"one uniform M image, NumPy default_rng(17)"},
                available_category_counts={k:sum(v==k for v in categories.values()) for k in ("all-benefit","all-harm","conflict","other")},
                missing_category_panels=[k for k,v in (("harmed",harmed),("improved",improved),("conflict",conflict)) if not v],
                examples=outputs,local_only=True,public_upload_allowed=False,GT_inference_use=False)
    write_json(root/"selection.json",result)
    return result


def report_text(cfg,decision,main,comparisons,decomposition,amplitude,evidence,panels):
    by={(r["run_id"],r["subset"]):r for r in main}
    R,M,N,C=[by[(name,"M")] for name in ("U-RSI","U-Mean","U-NoMessage","U-Mean-logit-RMS")]
    def number(x):
        return f"{x:.9f}"
    lines=[f"# RSI 解码器去留验证：{decision['status']}","",
           "本轮只完成固定 seed17、共同 B 初始化的五组机制筛查。以下阈值是预登记的资源筛查选择，不能解释为临床或正式统计标准。",
           "",
           f"路线 A：{'通过' if decision['route_A_pass'] else '未通过'}；路线 B：{'通过' if decision['route_B_pass'] else '未通过'}。附加标记：{', '.join(decision['flags']) or '无'}。",
           "",
           "## 实测事实：五组与完整更新控制", "",
           "| 组 / 控制 | M Dice | G+ | H- | Hε | 最差 10% 最小参考收益 |", "|---|---:|---:|---:|---:|---:|"]
    names=[*GROUPS,*[f"U-Mean-logit-alpha-{a:.6f}" for a in ALPHAS],"U-Mean-logit-RMS"]
    for name in names:
        row=by[(name,"M")]
        lines.append("| "+name+" | "+" | ".join(number(row[k]) for k in ("dice","G_plus","H_minus","H_epsilon","worst_tail_10pct"))+" |")
    lines.extend(["",f"训练侧 RMS α={number(amplitude['alpha_RMS'])}，raw={number(amplitude['alpha_RMS_raw'])}；S_mean={number(amplitude['S_mean'])}、S_rsi={number(amplitude['S_rsi'])}。",
                  f"严格等幅：{amplitude['strict_amplitude_match']}；实际 RMS 差={number(amplitude['RMS_absolute_mismatch'])}。只读取1471张canonical train RGB和pixel_valid，没有打开mask文件，标定额外耗时{amplitude['seconds']:.2f}秒。",
                  "",
                  "完整更新 α=0/1/3/2/3/1 在canonical logits插值后恢复原尺寸，再以logit>=0评分。旧feature Shrink在main_table单独标为historical；没有新增α网格，没有根据val分数重选α。",
                  "", "## 三个问题", "",
                  f"1. tail放开是否扩大有效纠错：普通目标 U-Mean−F-Mean 的M Dice差={number(M['dice']-by[('F-Mean','M')]['dice'])}、G+差={number(M['G_plus']-by[('F-Mean','M')]['G_plus'])}、Hε差={number(M['H_epsilon']-by[('F-Mean','M')]['H_epsilon'])}。这直接描述tail与message共同适应的变化；{'同条件复用通过，可作本轮因子比较。' if evidence['same_conditions'] else 'F条件不等价，只作历史描述。'}",
                  f"2. RSI是否仍牺牲有用更新：U-RSI−U-Mean 的G+差={number(R['G_plus']-M['G_plus'])}、Hε差={number(R['H_epsilon']-M['H_epsilon'])}；收益保留比={number(R['G_plus']/M['G_plus']) if M['G_plus']>0 else '分母为0'}。固定U-Mean的all-benefit/all-harm/conflict/other分类在group_decomposition.csv给出同病例变化与总体贡献，不把GT诊断作为推理策略。",
                  f"3. 简单微调或缩放是否足以解释：U-RSI−U-NoMessage 的M Dice差={number(R['dice']-N['dice'])}；U-RSI−训练侧RMS 的Dice差={number(R['dice']-C['dice'])}、G+差={number(R['G_plus']-C['G_plus'])}、Hε差={number(R['H_epsilon']-C['H_epsilon'])}。预登记实用幅度的简单控制支配点：{', '.join(decision['simple_controls_dominating']) or '未出现'}。",
                  "", "这些是实测事实对应的局部机制证据。U组相对U-NoMessage同时改变异构信息和message–tail共同适应，不能单独证明异构特异性；same-tail off输出只是诊断，固定B才是收益anchor。",
                  "", "## 来源、总损害和尾部限制", "",
                  "| 来源 | 比较 | Dice差（U-RSI−对照） |", "|---|---|---:|"])
    for subset in ("H","T1"):
        for comparator in ("U-Mean","U-Mean-logit-RMS","U-NoMessage"):
            lines.append(f"| {subset} | {comparator} | {number(by[('U-RSI',subset)]['dice']-by[(comparator,subset)]['dice'])} |")
    lines.extend(["", "H 47图/104参考与T1 20图/49参考为M来源子集，预测完全复用M缓存，不能累加样本。尾部M为23图，H为5图，T1仅2图；所有bootstrap重采样后重新排序。",
                  "",
                  f"未扣容差H-及尾部回退记录：{json.dumps(decision['total_harm_or_tail_regressions'],ensure_ascii=False)}。mean_gain=G+−H-逐点数值核验；mean_harm、any_harm、all_harm、sign_conflict分开报告，ε=0/.005/.01/.02/.05敏感性不参与重选。",
                  "", "## 配对区间与40轮趋势", "",
                  "2000次、固定rng17，已知病例按组、缺失身份按独立图像重采样；M主口径222单位，223图口径另列。所有参考保留同图关联，不把像素当独立样本。区间是单seed、选中开发集探索性证据，跨0不等于等价或非劣。",
                  "", "| M比较 | Dice差 | 95%探索区间 | Hε差 | 95%探索区间 |", "|---|---:|---|---:|---|"])
    for row in comparisons:
        if row["subset"]=="M" and row["bootstrap_unit"]=="known_group_else_image" and row["current"]=="U-RSI" and row["comparator"] in {"U-Mean","U-Mean-logit-RMS","U-NoMessage"}:
            lines.append(f"| {row['comparator']} | {number(row['dice_difference'])} | [{number(row['dice_ci_low'])}, {number(row['dice_ci_high'])}] | {number(row['H_epsilon_difference'])} | [{number(row['H_epsilon_ci_low'])}, {number(row['H_epsilon_ci_high'])}] |")
    lines.extend(["", "40轮完整曲线见learning_curves.png/epoch_diagnostics.csv；主checkpoint按M原尺寸macro mean-rater Dice最大选出，精确并列选更早epoch，epoch40保留独立记录。", "",
                  "| 组 | best epoch | best Dice | epoch40 Dice | 本地权重SHA |", "|---|---:|---:|---:|---|"])
    for budget in evidence["budgets"]:
        history=read_csv(budget["epochs"]["path"])
        final=history[-1]
        lines.append(f"| {budget['group']} | {budget['best_epoch']} | {number(budget['best_val_dice'])} | {number(float(final['val_dice']))} | {budget['checkpoint']['sha256']} |")
    lines.extend(["", "## 工程与复核边界", "",
                  f"工程验收合规：{decision['engineering_compliant']}。固定teacher来自B/best、参数buffers和输出不漂移；student tail独立storage。逐参考CSV绑定checkpoint、teacher、manifest、canonical cache与geometry SHA。原尺寸logits恢复后阈值，BF1与HD95沿用旧实现及已验证精确KD计算，没有TTA/后处理/阈值搜索。",
                  "",
                  "U-NoMessage部署不运行Transformer/message，U-RSI部署只需student一次tail；完整logit插值通常需要teacher和student两次tail。实际训练参数、前向次数、GPU时间和峰值显存在environment/engineering_acceptance/DONE/epoch日志保留，不能把U-NoMessage空算成相同时间。",
                  "",
                  "画册按固定U-Mean受损/改善/冲突及seed17随机规则选择，selection.json列出缺失类别和文件SHA。原图/各真实参考/teacher/各对照及变化同时展示，仅保留用户本地；不同标注和工具流程不等于明确医学错误。",
                  "",
                  "未验证项：独立seed的共同A/B复建、正式独立病例test、跨来源泛化及CNN自消息/近邻算法归因。没有自动启动seed29、101/202/303、额外λ/模块/α网格或ISIC训练，没有上传权重。",
                  "",
                  "本轮结果与所有检查的SHA见delivery_manifest.json；preregistration.json绑定任务书与固定筛查规则。若状态CANDIDATE，只建议独立确认；其余状态按decision.json结束当前自动预算，不扩展实验。", ""])
    return "\n".join(lines)


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("reuse","export","summarize"))
    parser.add_argument("--config",type=Path,default=PROJECT/"configs/tail_probe_seed17.yaml")
    parser.add_argument("--scope",choices=("val",),default="val")
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    result=(prove_f_reuse(cfg) if args.command=="reuse" else run_export(cfg,args.scope) if args.command=="export" else run_summarize(cfg))
    print(json.dumps({"status":result["status"],"test_scoring_locked":True},ensure_ascii=False))


if __name__=="__main__":
    main()
