"""Profile uncached model/candidate compute on ten fixed cal50 images; exclude disk I/O."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from scipy import ndimage

from project_paths import PHASE1
from phase3_calibrate import ROOT, CACHE
from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact
from phase3_train import make_model

sys.path.insert(0, str(PHASE1))
from contour.geometry import (extract_geometry, build_prediction_context,
                              compose_prediction_context, mask_contours, polygon_area)
from contour.models import FrozenMSGUNet
from radius_sensitivity import dense
from tools.train_boundary import load_selected_head


def node_context(features, rgb, probability, boundary, source, normals):
    p = np.asarray(source, np.float32)
    normal = np.asarray(normals, np.float32)
    field = np.concatenate((features, rgb, probability, boundary), axis=0).astype(np.float32)
    sampled = np.stack([ndimage.map_coordinates(channel, [p[:, 1], p[:, 0]],
                                                order=1, mode="nearest", prefilter=False)
                        for channel in field], axis=-1)
    curvature = np.linalg.norm(np.roll(normal, -1, axis=0)-np.roll(normal, 1, axis=0), axis=-1, keepdims=True)
    phase = np.arange(256)*2*np.pi/256
    additional = np.concatenate((p/255.0, normal, curvature,
                                 np.stack((np.sin(phase), np.cos(phase)), axis=-1)), axis=-1)
    return np.concatenate((sampled, additional), axis=-1).astype(np.float16)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Profile requires CUDA for comparable timings")
    torch.backends.cudnn.benchmark = True
    cnn = FrozenMSGUNet(PHASE1 / "third_party/msgu_net").to(device).eval()
    edge = load_selected_head(root=PHASE1, device=device).eval()
    models = {}
    for family in ("S64", "S96", "N0", "TG"):
        model = make_model(family).to(device).eval()
        checkpoint = torch.load(ROOT / "models_phase3" / f"{family}_seed17" / "best.pth",
                                map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[family] = model
    configs = {family: json.loads((ROOT / "results_phase3/controlled" /
                                   f"{family}_seed17_chosen_config.json").read_text())["chosen"]
               for family in models}
    profile_ids = json.loads((ROOT / "results_phase3/full_dp_profile.json").read_text())["profile_ids"]
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    by_id = {row["image_id"]: row["index"] for row in records}
    rgb_cache = np.load(CACHE / "rgb.npy", mmap_mode="r")
    steps = []

    def process(image_id, record):
        index = by_id[image_id]
        timing = {"image_id": image_id}
        x = torch.from_numpy(np.array(rgb_cache[index], copy=True))[None].float().to(device)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = cnn.forward_with_features(x)
            boundary = torch.sigmoid(edge(output["features"]))
        torch.cuda.synchronize(); timing["frozen_CNN_boundary_ms"] = 1000*(time.perf_counter()-start)
        pred = output["mask"][0, 0].byte().cpu().numpy().astype(bool)
        q = boundary[0, 0].float().cpu().numpy()
        torch.cuda.synchronize(); start = time.perf_counter()
        geo = extract_geometry(pred, 256, 16)
        if geo is None: return None
        field = dense(q, replace(geo, search_radius=32.0))
        timing["candidate_geometry_ms"] = 1000*(time.perf_counter()-start)
        point = torch.from_numpy(np.asarray(field.points, np.float32))[None].to(device)
        source = torch.from_numpy(np.asarray(geo.points, np.float32))[None].to(device)
        normals = torch.from_numpy(np.asarray(geo.normals, np.float32))[None].to(device)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score64 = models["S64"](output["features"].float(), x, output["probability"].float(),
                                     boundary.float(), point, source, normals)
        torch.cuda.synchronize(); timing["S64_ms"] = 1000*(time.perf_counter()-start)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            score96 = models["S96"](output["features"].float(), x, output["probability"].float(),
                                     boundary.float(), point, source, normals)
        torch.cuda.synchronize(); timing["S96_ms"] = 1000*(time.perf_counter()-start)
        maps = (output["features"][0].float().cpu().numpy(), x[0].float().cpu().numpy(),
                output["probability"][0].float().cpu().numpy(), boundary[0].float().cpu().numpy())
        torch.cuda.synchronize(); start = time.perf_counter()
        context = node_context(*maps, geo.points, geo.normals)
        context_tensor = torch.from_numpy(context)[None].to(device)
        validity = torch.from_numpy(np.asarray(field.valid, np.uint8))[None].to(device)
        torch.cuda.synchronize(); timing["node_context_ms"] = 1000*(time.perf_counter()-start)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            n0 = models["N0"](context_tensor, score64.float(), validity)
        torch.cuda.synchronize(); timing["N0_ms"] = 1000*(time.perf_counter()-start)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tg = models["TG"](context_tensor, score64.float(), validity)
        torch.cuda.synchronize(); timing["TG_ms"] = 1000*(time.perf_counter()-start)
        scores = {"N0": n0[0].float().cpu().numpy(), "TG": tg[0].float().cpu().numpy(),
                  "S96": score96[0].float().cpu().numpy()}
        valid_numpy = np.asarray(field.valid, bool)
        torch.cuda.synchronize(); start = time.perf_counter()
        context_pred = build_prediction_context(pred, max(mask_contours(pred), key=lambda p: abs(polygon_area(p))))
        timing["prediction_context_ms"] = 1000*(time.perf_counter()-start)
        for family, mode in (("N0", "A"), ("TG", "A"), ("TG", "D"), ("S96", "D")):
            key = f"{family}_{mode}"
            config = configs[family][mode]
            torch.cuda.synchronize(); start = time.perf_counter()
            selected = (choose_argmax(scores[family], valid_numpy, config["alpha"]) if mode == "A"
                        else choose_closed_dp(scores[family], valid_numpy,
                                              config["alpha"], config["smooth_lambda"]))
            polygon = field.points[np.arange(256), selected]
            compose_prediction_context(rasterize_exact(polygon, pred.shape), context_pred)
            torch.cuda.synchronize(); timing[f"{key}_post_ms"] = 1000*(time.perf_counter()-start)
        if record: steps.append(timing)
        return timing

    process(profile_ids[0], False)  # Warm CUDA kernels and Numba without reporting startup.
    torch.cuda.reset_peak_memory_stats()
    for image_id in profile_ids:
        result = process(image_id, True)
        if result is None: raise RuntimeError(f"No CNN contour for runtime profile {image_id}")
    aggregate = {}
    for key in steps[0]:
        if key == "image_id": continue
        values = [row[key] for row in steps]
        aggregate[key] = {"mean_ms": float(np.mean(values)), "p50_ms": float(np.median(values)),
                          "p95_ms": float(np.quantile(values, .95))}
    common = sum(aggregate[key]["mean_ms"] for key in ("frozen_CNN_boundary_ms", "candidate_geometry_ms",
                                                        "prediction_context_ms"))
    aggregate["N0_A_total_mean_ms"] = common+sum(aggregate[key]["mean_ms"] for key in ("S64_ms", "node_context_ms", "N0_ms", "N0_A_post_ms"))
    aggregate["TG_A_total_mean_ms"] = common+sum(aggregate[key]["mean_ms"] for key in ("S64_ms", "node_context_ms", "TG_ms", "TG_A_post_ms"))
    aggregate["TG_D_total_mean_ms"] = common+sum(aggregate[key]["mean_ms"] for key in ("S64_ms", "node_context_ms", "TG_ms", "TG_D_post_ms"))
    aggregate["S96_D_total_mean_ms"] = common+aggregate["S96_ms"]["mean_ms"]+aggregate["S96_D_post_ms"]["mean_ms"]
    for family, components in {
        "N0_A": ("S64_ms", "node_context_ms", "N0_ms", "N0_A_post_ms"),
        "TG_A": ("S64_ms", "node_context_ms", "TG_ms", "TG_A_post_ms"),
        "TG_D": ("S64_ms", "node_context_ms", "TG_ms", "TG_D_post_ms"),
        "S96_D": ("S96_ms", "S96_D_post_ms"),
    }.items():
        values = [sum(row[key] for key in ("frozen_CNN_boundary_ms", "candidate_geometry_ms",
                                               "prediction_context_ms", *components)) for row in steps]
        aggregate[f"{family}_total"] = {"mean_ms": float(np.mean(values)),
                                        "p50_ms": float(np.median(values)),
                                        "p95_ms": float(np.quantile(values, .95))}
    result = dict(scope="10 fixed hashed cal50 images, seed17, pure inference compute plus host-device transfer; startup and disk I/O excluded",
                  image_ids=profile_ids, images=len(steps), device=torch.cuda.get_device_name(0),
                  candidate_count=65, dp_exact=True, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  measurements=aggregate, per_image=steps)
    path = ROOT / "results_phase3/controlled/runtime_profile.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"measurements": aggregate, "peak_cuda_bytes": result["peak_cuda_bytes"]}, indent=2))


if __name__ == "__main__": main()
