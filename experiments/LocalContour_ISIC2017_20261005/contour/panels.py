"""Local-only fixed-case panels from already evaluated contour NPZ artifacts.

There is no inference, candidate generation or GT-based case removal here.
All geometry is drawn in 256x256 model-input coordinates. Source RGB and GT
are read locally; raw source files are not copied or published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import boundary_pixels


GT_RED = (255, 48, 48)
PREDICTION_BLUE = (35, 145, 255)
CANDIDATE_GREEN = (35, 255, 120)
ZERO_WHITE = (250, 250, 250)
BAND_YELLOW = (255, 210, 60)
NORMAL_CYAN = (40, 220, 255)
METHODS = ("G0", "G1", "G2", "G3", "G4")
INPUT_SHAPE = (256, 256)


def _role(value: Any) -> str | None:
    if str(value) in {"cal", "cal50", "calibration", "calibration50"}:
        return "cal50"
    if str(value) in {"locked", "locked100", "locked_verification", "locked_verification100", "verification", "verify100"}:
        return "locked100"
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def overlay_boundaries(rgb: np.ndarray, gt_mask: np.ndarray, prediction_mask: np.ndarray | None = None) -> np.ndarray:
    """Red always denotes true GT; blue denotes prediction, GT drawn last."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    gt = np.asarray(gt_mask, dtype=bool)
    if rgb.shape[:2] != gt.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB/GT must share input-coordinate dimensions")
    result = rgb.copy()
    if prediction_mask is not None:
        prediction = np.asarray(prediction_mask, dtype=bool)
        if prediction.shape != gt.shape:
            raise ValueError("Prediction/GT must share input-coordinate dimensions")
        result[boundary_pixels(prediction)] = PREDICTION_BLUE
    result[boundary_pixels(gt)] = GT_RED
    return result


def select_cases(records: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exactly the fixed random2/min-G4-minus-G0/min-coverage rules, deduped."""
    if len(records) != 150 or len({str(record["image_id"]) for record in records}) != 150:
        raise ValueError("Panels require metadata for all 150 official validation images")
    record_roles = {str(record["image_id"]): _role(record.get("role")) for record in records}
    if sum(value == "cal50" for value in record_roles.values()) != 50 or sum(value == "locked100" for value in record_roles.values()) != 100:
        raise ValueError("Panels require the frozen calibration50/locked100 partition")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        image_id, method = str(row.get("image_id", "")), str(row.get("method", ""))
        if image_id not in record_roles or method not in METHODS or _role(row.get("role")) != record_roles[image_id]:
            raise ValueError("Panel results have an unknown image, method or partition")
        if (image_id, method) in lookup:
            raise ValueError("Duplicate image/method panel result")
        if not np.isfinite(float(row["dice"])):
            raise ValueError("Panel case selection requires actual finite per-image Dice")
        lookup[(image_id, method)] = row
    if len(lookup) != 750:
        raise ValueError("Retain all 750 method rows, including difficult cases, before panels")
    locked = sorted(image_id for image_id, role in record_roles.items() if role == "locked100")
    random_ids = [str(value) for value in np.random.default_rng(17).choice(locked, size=2, replace=False)]
    delta = {image_id: float(lookup[(image_id, "G4")]["dice"]) - float(lookup[(image_id, "G0")]["dice"]) for image_id in locked}
    coverage = {image_id: lookup[(image_id, "G0")].get("search_band_coverage") for image_id in locked}
    worst_delta = min(locked, key=lambda image_id: (delta[image_id], image_id))
    # Undefined coverage is shown first, explicitly marked unavailable, rather
    # than hidden. No scalar is replaced by zero in records or results.
    lowest_coverage = min(locked, key=lambda image_id: (float(coverage[image_id]) if coverage[image_id] is not None else -np.inf, image_id))
    labels: dict[str, list[str]] = {}
    for image_id in random_ids:
        labels.setdefault(image_id, []).append("seed17_random_locked")
    labels.setdefault(worst_delta, []).append("minimum_locked_g4_minus_g0_dice")
    labels.setdefault(lowest_coverage, []).append("minimum_locked_search_band_coverage" if coverage[lowest_coverage] is not None else "unavailable_locked_search_band_coverage")
    selected = [{"image_id": image_id, "labels": value, "g4_minus_g0_dice": delta[image_id], "search_band_coverage": None if coverage[image_id] is None else float(coverage[image_id])} for image_id, value in labels.items()]
    return {"seed": 17, "scope": "locked100", "rules": {"random": "two IDs without replacement from lexicographically sorted locked100 IDs using numpy.default_rng(17)", "failure": "minimum per-image G4-minus-G0 Dice; tie uses image_id", "coverage": "minimum search-band coverage; undefined shown explicitly first; tie uses image_id", "deduplication": "one PNG per unique ID, preserving all selection labels", "selection_does_not_remove_other_images_from_metrics": True}, "locked_candidate_image_ids": locked, "random_image_ids": random_ids, "failure_image_id": worst_delta, "lowest_coverage_image_id": lowest_coverage, "selected": selected, "local_only_source_visuals": True, "g4_deployable": False, "g4_strict_dice_upper_bound": False}


def _input_assets(record: dict[str, Any], artifact: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_path, mask_path = Path(record["image_path"]), Path(record["mask_path"])
    with Image.open(image_path) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    with Image.open(mask_path) as mask:
        source_gt = np.asarray(mask.convert("L"))
    if source_rgb.shape[:2] != source_gt.shape:
        raise ValueError(f"Actual RGB and GT dimensions differ for {record['image_id']}")
    rgb = cv2.resize(source_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    gt_from_source = cv2.resize(source_gt, (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    artifact_gt = next((artifact[key] for key in ("gt_mask", "mask_gt", "gt") if key in artifact), None)
    if artifact_gt is not None:
        gt = np.asarray(artifact_gt).squeeze().astype(bool)
        if gt.shape != INPUT_SHAPE or not np.array_equal(gt, gt_from_source):
            raise ValueError(f"Evaluated artifact GT differs from actual source/preprocessing for {record['image_id']}")
        gt_source = "verified_evaluated_artifact_and_actual_mask"
    else:
        gt, gt_source = gt_from_source, "actual_mask_author_cv2_inter_nearest_input_resize"
    info = {"source_image_path": str(image_path), "source_mask_path": str(mask_path), "source_image_sha256": _sha256(image_path), "source_mask_sha256": _sha256(mask_path), "source_shape": list(source_gt.shape), "display_shape": list(INPUT_SHAPE), "rgb_decode": "PIL RGB", "rgb_resize": "cv2 INTER_LINEAR to 256x256, matching CNN input transform", "gt_resize": "cv2 INTER_NEAREST to 256x256, matching CNN evaluation", "gt_display_source": gt_source}
    return rgb, gt, info


def _candidate_panel(rgb: np.ndarray, gt: np.ndarray, artifact: dict[str, np.ndarray]) -> Image.Image:
    points = np.asarray(artifact["geometry_points"], dtype=np.float64).reshape(-1, 2)
    normals = np.asarray(artifact["normals"], dtype=np.float64).reshape(-1, 2)
    candidates = np.asarray(artifact["candidate_points"], dtype=np.float64)
    valid = np.asarray(artifact["candidate_valid"], dtype=bool)
    if points.shape != normals.shape or candidates.shape != (len(points), 4, 2) or valid.shape != (len(points), 4):
        raise ValueError("Candidate geometry artifact dimensions differ from fixed N-by-4 interface")
    if not np.isfinite(points).all() or not np.isfinite(normals).all() or not np.isfinite(candidates[valid]).all():
        raise ValueError("Valid geometry/candidates must be finite")
    image = Image.fromarray(rgb).convert("RGBA")
    fill = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(fill)
    radius = float(np.asarray(artifact.get("geometry_search_radius", artifact.get("search_radius", 16))).item())
    if len(points):
        lo, hi = points - radius * normals, points + radius * normals
        for index in range(len(points)):
            following = (index + 1) % len(points)
            polygon = [tuple(lo[index]), tuple(lo[following]), tuple(hi[following]), tuple(hi[index])]
            draw.polygon(polygon, fill=(*BAND_YELLOW, 38))
        image = Image.alpha_composite(image, fill)
        draw = ImageDraw.Draw(image)
        for index in range(0, len(points), 8):
            draw.line((tuple(lo[index]), tuple(hi[index])), fill=(*NORMAL_CYAN, 190), width=1)
        for point in candidates[valid]:
            x, y = point
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(*CANDIDATE_GREEN, 255))
        for point in candidates[:, 0]:
            x, y = point
            draw.ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=(*ZERO_WHITE, 255))
        draw.line([tuple(point) for point in points] + [tuple(points[0])], fill=(*ZERO_WHITE, 200), width=1)
    else:
        ImageDraw.Draw(image).text((10, 110), "Empty prediction: no candidates", font=_font(12), fill=(255, 255, 255))
    result = np.asarray(image.convert("RGB")).copy()
    result[boundary_pixels(gt)] = GT_RED
    return Image.fromarray(result)


def render_panels(records: list[dict[str, Any]], result_rows: list[dict[str, Any]], artifact_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Render at most four fixed selected images from existing evaluated NPZs.

    Artifact keys: masks_G0..masks_G4 [256,256], candidate_points [N,4,2],
    candidate_valid [N,4], geometry_points/normals [N,2]; optional gt_mask and
    geometry_search_radius. Empty geometry must use correctly shaped zero rows.
    """
    selection = select_cases(records, result_rows)
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_lookup = {str(record["image_id"]): record for record in records}
    row_lookup = {(str(row["image_id"]), str(row["method"])): row for row in result_rows}
    for selected in selection["selected"]:
        image_id = selected["image_id"]
        artifact_path = artifact_dir / f"{image_id}.npz"
        with np.load(artifact_path, allow_pickle=False) as archive:
            artifact = {name: archive[name] for name in archive.files}
        required = [f"masks_{method}" for method in METHODS] + ["candidate_points", "candidate_valid", "geometry_points", "normals"]
        if any(name not in artifact for name in required):
            raise ValueError(f"Missing required evaluated artifact keys for {image_id}")
        masks = {method: np.asarray(artifact[f"masks_{method}"]).astype(bool) for method in METHODS}
        if any(mask.shape != INPUT_SHAPE for mask in masks.values()):
            raise ValueError("All displayed selectors must have the same evaluated 256x256 masks")
        rgb, gt, source_info = _input_assets(record_lookup[image_id], artifact)
        gt_panel = np.repeat((gt.astype(np.uint8) * 255)[..., None], 3, axis=2)
        gt_panel[boundary_pixels(gt)] = GT_RED
        panels: list[tuple[str, str, Image.Image]] = [
            ("Source RGB", "PIL RGB -> input 256x256", Image.fromarray(rgb)),
            ("True GT", "Red boundary; real annotation", Image.fromarray(gt_panel)),
            ("Fixed candidates", "Yellow band / green peaks / white s=0", _candidate_panel(rgb, gt, artifact)),
        ]
        descriptions = {"G0": "Original CNN mask", "G1": "Zero-offset reconstruction", "G2": "Independent local selection", "G3": "Closed DP, calibrated lambda", "G4": "Offline GT-assisted; NOT DEPLOYABLE"}
        for method in METHODS:
            row = row_lookup[(image_id, method)]
            title = f"{method}: {descriptions[method]}"
            subtitle = f"Dice {float(row['dice']):.4f}"
            if "bf1" in row:
                subtitle += f" / BF1 {float(row['bf1']):.4f}"
            panels.append((title, subtitle, Image.fromarray(overlay_boundaries(rgb, gt, masks[method]))))
        gap, title_height, header, footer = 14, 60, 95, 90
        width = 4 * 256 + 5 * gap
        height = 2 * (256 + title_height) + 3 * gap + header + footer
        canvas = Image.new("RGB", (width, height), color=(24, 27, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 12), f"{image_id} | model input coordinates (256x256)", font=_font(23), fill=(245, 245, 245))
        draw.text((gap, 46), "Selection: " + "; ".join(selected["labels"]), font=_font(12), fill=(235, 235, 235))
        coverage_text = "undefined" if selected["search_band_coverage"] is None else f"{selected['search_band_coverage']:.4f}"
        draw.text((gap, 67), f"G4-G0 Dice={selected['g4_minus_g0_dice']:+.5f}; search-band coverage={coverage_text}", font=_font(14), fill=(220, 220, 220))
        for index, (title, subtitle, panel) in enumerate(panels):
            column, row_index = index % 4, index // 4
            x = gap + column * (256 + gap)
            y = header + gap + row_index * (256 + title_height + gap)
            if title.startswith("G4:"):
                draw.text((x, y), "G4: Offline GT-assisted", font=_font(14), fill=(255, 205, 80))
                draw.text((x, y + 18), "NOT DEPLOYABLE | " + subtitle, font=_font(11), fill=(255, 205, 80))
            else:
                draw.text((x, y), title, font=_font(13), fill=(245, 245, 245))
                draw.text((x, y + 21), subtitle, font=_font(11), fill=(210, 210, 210))
            canvas.paste(panel, (x, y + title_height))
        legend_y = height - footer + 10
        draw.text((gap, legend_y), "Red = true GT boundary", font=_font(16), fill=GT_RED)
        draw.text((gap + 290, legend_y), "Blue = predicted boundary", font=_font(16), fill=PREDICTION_BLUE)
        draw.text((gap, legend_y + 25), "G4 chooses existing candidates using GT distance with the same closed DP; it is not a strict Dice upper bound.", font=_font(14), fill=(220, 220, 220))
        draw.text((gap, legend_y + 49), "Random and fixed failure rules are shown together. All 150 images remain in the numeric results. Local diagnostic only.", font=_font(13), fill=(190, 190, 190))
        output_path = output_dir / f"{image_id}.png"
        canvas.save(output_path)
        selected.update({"panel_path": str(output_path), "panel_sha256": _sha256(output_path), "artifact_path": str(artifact_path), "artifact_sha256": _sha256(artifact_path), "source": source_info})
    selection.update({"legend": {"GT_boundary": list(GT_RED), "prediction_boundary": list(PREDICTION_BLUE), "candidate_peaks": list(CANDIDATE_GREEN), "original_zero_offset": list(ZERO_WHITE), "search_band": list(BAND_YELLOW)}, "npz_masks_are_not_recomputed": True, "source_RGB_and_GT_are_not_published": True})
    (output_dir / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return selection
