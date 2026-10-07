"""Fixed seed17 and hash-random dev100 panels; qualitative review follows main readout."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation import (boundary_pixels, mask_contours, polygon_area,
                        build_prediction_context, compose_prediction_context)
from phase3_calibrate import ROOT, CACHE
from phase3_dp import choose_argmax, choose_closed_dp
from phase3_fast_raster import rasterize_exact

SEED = 17
SIZE = 256


def rgb_image(index, rgb_cache):
    normalized = np.asarray(rgb_cache[index], np.float32).transpose(1, 2, 0)
    mean = np.array([.485, .456, .406], np.float32)*255
    scale = np.array([.229, .224, .225], np.float32)*255
    return np.clip(normalized*scale+mean, 0, 255).astype(np.uint8)


def overlay(rgb, pred=None, gt=None):
    value = np.array(rgb, copy=True)
    if gt is not None:
        edge = boundary_pixels(gt)
        value[edge] = [30, 245, 60]
    if pred is not None:
        edge = boundary_pixels(pred)
        value[edge] = [255, 30, 25]
    return Image.fromarray(value)


def draw_panel(base, title, x, y, canvas):
    canvas.paste(base, (x, y+22))
    ImageDraw.Draw(canvas).text((x+4, y+4), title, fill=(255, 255, 255))


def main():
    target = ROOT / "results_phase3/controlled"
    panel_dir = target / "case_panels"; panel_dir.mkdir(exist_ok=True)
    rows = list(csv.DictReader((target / "dev100_per_image.csv").open()))
    keyed = {(row["image_id"], row["method"], int(row["seed"])): row for row in rows}
    oracle = {row["image_id"]: row for row in csv.DictReader((target / "oracle_GT_distance_per_image.csv").open())}
    ids = (ROOT / "splits/development_readout.txt").read_text().splitlines()
    def dice(image_id, method): return float(keyed[(image_id, method, SEED)]["dice"])
    def category(name, scored):
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:5]
        return [dict(category=name, image_id=image_id, criterion_value=value, rank=j+1)
                for j, (image_id, value) in enumerate(ranked)]
    selected = []
    selected += category("N0_fail_TG_success", [(image_id, dice(image_id, "TG_A")-dice(image_id, "N0_A"))
                                                for image_id in ids if dice(image_id, "TG_A") > dice(image_id, "N0_A")
                                                and dice(image_id, "N0_A") <= dice(image_id, "CNN")])
    selected += category("simple_DP_success_TG_failure", [(image_id, dice(image_id, "S96_D")-dice(image_id, "TG_D"))
                                                        for image_id in ids if dice(image_id, "S96_D") > dice(image_id, "TG_D")
                                                        and dice(image_id, "S96_D") > dice(image_id, "CNN")])
    selected += category("TG_harmful_change", [(image_id, float(keyed[(image_id, "TG_A", SEED)]["correct_to_incorrect_rate"]))
                                               for image_id in ids if keyed[(image_id, "TG_A", SEED)].get("correct_to_incorrect_rate") not in ("", None)])
    selected += category("all_learned_fail_oracle_gain", [(image_id, float(oracle[image_id]["dice"])-dice(image_id, "CNN"))
                                                         for image_id in ids if all(dice(image_id, method) <= dice(image_id, "CNN")
                                                                                     for method in ("S64_A", "N0_A", "T32_A", "TG_A"))
                                                         and float(oracle[image_id]["dice"]) > dice(image_id, "CNN")+.02])
    selected += category("candidate_unreachable", [(image_id, 1.0-float(oracle[image_id]["reachable_nodes"])/
                                                    max(1.0, float(oracle[image_id]["candidate_nodes"])))
                                                   for image_id in ids if float(oracle[image_id]["candidate_nodes"]) > 0])
    random_ids = sorted(ids, key=lambda value: hashlib.sha256(f"LC-P3-RANDOM|{value}".encode()).hexdigest())[:10]
    selected += [dict(category="hash_random", image_id=image_id,
                      criterion_value=hashlib.sha256(f"LC-P3-RANDOM|{image_id}".encode()).hexdigest(), rank=j+1)
                 for j, image_id in enumerate(random_ids)]
    (panel_dir / "selection.json").write_text(json.dumps(dict(seed=SEED, entries=selected,
       definitions={"N0_fail_TG_success": "TG_A Dice>N0_A and N0_A<=CNN; rank by TG_A-N0_A",
                    "simple_DP_success_TG_failure": "S96_D Dice>TG_D and S96_D>CNN; rank by S96_D-TG_D",
                    "TG_harmful_change": "rank by TG_A correct-to-incorrect rate",
                    "all_learned_fail_oracle_gain": "four A models <= CNN and oracle>CNN+0.02; rank oracle-CNN",
                    "candidate_unreachable": "rank by 1 - reachable candidate node fraction",
                    "hash_random": "SHA256('LC-P3-RANDOM|' + image_id), first 10"}), indent=2) + "\n")
    records = json.loads((CACHE / "COMPLETE.json").read_text())["records"]
    index_by_id = {row["image_id"]: row["index"] for row in records}
    rgb = np.load(CACHE / "rgb.npy", mmap_mode="r")
    prediction = np.load(CACHE / "prediction.npy", mmap_mode="r")
    truth = np.load(CACHE / "gt.npy", mmap_mode="r")
    candidate = CACHE / "candidates_R32"
    points = np.load(candidate / "points.npy", mmap_mode="r")
    valid = np.load(candidate / "valid.npy", mmap_mode="r")
    distance = np.load(candidate / "gt_distance.npy", mmap_mode="r")
    has = np.load(candidate / "has_contour.npy", mmap_mode="r")
    scores = {family: np.load(ROOT / "models_phase3" / f"{family}_seed{SEED}" / "candidate_scores.npy", mmap_mode="r")
              for family in ("S64", "S96", "N0", "T32", "TG")}
    configs = {family: json.loads((target / f"{family}_seed{SEED}_chosen_config.json").read_text())["chosen"]
               for family in scores}
    for image_id in dict.fromkeys(row["image_id"] for row in selected):
        i = index_by_id[image_id]
        base = rgb_image(i, rgb)
        pred = np.asarray(prediction[i], bool)
        gt = np.asarray(truth[i], bool)
        paths = {}
        masks = {"CNN": pred}
        if has[i]:
            validity = np.asarray(valid[i], bool)
            point = np.asarray(points[i], np.float32)
            exterior = max(mask_contours(pred), key=lambda p: abs(polygon_area(p)))
            context = build_prediction_context(pred, exterior)
            for family, mode in (("S64", "A"), ("N0", "A"), ("T32", "A"),
                                 ("TG", "A"), ("TG", "D"), ("S96", "D")):
                config = configs[family][mode]
                indices = (choose_argmax(scores[family][i], validity, config["alpha"]) if mode == "A"
                           else choose_closed_dp(scores[family][i], validity, config["alpha"], config["smooth_lambda"]))
                polygon = point[np.arange(256), indices]
                method = f"{family}_{mode}"
                paths[method] = indices
                masks[method] = compose_prediction_context(rasterize_exact(polygon, pred.shape), context)
            oracle_index = choose_closed_dp(-np.asarray(distance[i], np.float32), validity, 0.0, .05)
            oracle_polygon = point[np.arange(256), oracle_index]
            masks["Oracle"] = compose_prediction_context(rasterize_exact(oracle_polygon, pred.shape), context)
        else:
            for method in ("S64_A", "N0_A", "T32_A", "TG_A", "TG_D", "S96_D", "Oracle"):
                masks[method] = pred
        displacement = Image.fromarray(base.copy())
        candidates = Image.fromarray(base.copy())
        if has[i]:
            draw = ImageDraw.Draw(displacement)
            positions = point[np.arange(256), paths["TG_A"]]
            for node in range(0, 256, 2):
                x, y = positions[node]
                offset = abs(int(paths["TG_A"][node])-32)
                color = (255, 30, 30) if offset > 8 else (255, 220, 40) if offset > 2 else (40, 240, 60)
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            draw = ImageDraw.Draw(candidates)
            for node in range(0, 256, 16):
                for candidate_index in range(0, 65, 4):
                    if valid[i, node, candidate_index]:
                        x, y = point[node, candidate_index]
                        draw.ellipse((x-1, y-1, x+1, y+1), fill=(30, 120, 255))
                x, y = point[node, 32]
                draw.ellipse((x-2, y-2, x+2, y+2), fill=(255, 220, 40))
        content = [("RGB", Image.fromarray(base)), ("GT", overlay(base, gt=gt)),
                   ("CNN", overlay(base, pred=masks["CNN"], gt=gt))]
        for method in ("S64_A", "N0_A", "T32_A", "TG_A", "TG_D", "S96_D", "Oracle"):
            score = float(oracle[image_id]["dice"]) if method == "Oracle" else dice(image_id, method)
            content.append((f"{method} D={score:.3f}", overlay(base, pred=masks[method], gt=gt)))
        content += [("TG displacement", displacement), ("candidate tracks", candidates)]
        canvas = Image.new("RGB", (4*SIZE, 3*(SIZE+22)), color=(18, 18, 18))
        for j, (title, panel) in enumerate(content):
            draw_panel(panel, title, (j%4)*SIZE, (j//4)*(SIZE+22), canvas)
        canvas.save(panel_dir / f"{image_id}.png", optimize=True)
    print(f"Rendered {len(set(row['image_id'] for row in selected))} case panels", flush=True)


if __name__ == "__main__": main()
