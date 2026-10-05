"""CPU display/selection checks using synthetic files only."""

import json

import numpy as np
from PIL import Image

from contour.panels import GT_RED, PREDICTION_BLUE, overlay_boundaries, render_panels, select_cases


def panel_inputs(tmp_path):
    rgb = np.full((256, 256, 3), 85, dtype=np.uint8)
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[70:180, 70:180] = 255
    image_path, mask_path = tmp_path / "synthetic_rgb.png", tmp_path / "synthetic_gt.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(gt).save(mask_path)
    records, rows = [], []
    for index in range(150):
        image_id = f"fixture_{index:03d}"
        role = "calibration50" if index < 50 else "locked_verification100"
        records.append({"image_id": image_id, "role": role, "image_path": str(image_path), "mask_path": str(mask_path)})
        for method in ("G0", "G1", "G2", "G3", "G4"):
            rows.append({"image_id": image_id, "role": role, "method": method, "dice": 0.5 if method == "G4" and index == 81 else 0.8, "bf1": 0.7, "search_band_coverage": 0.2 if index == 98 else 0.95})
    return records, rows, gt > 0


def test_gt_red_and_prediction_blue_have_correct_semantics():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    gt, prediction = np.zeros((40, 40), dtype=bool), np.zeros((40, 40), dtype=bool)
    gt[4:12, 4:12] = True
    prediction[20:30, 20:30] = True
    overlay = overlay_boundaries(rgb, gt, prediction)
    assert tuple(overlay[4, 4]) == GT_RED
    assert tuple(overlay[20, 20]) == PREDICTION_BLUE
    coincident = overlay_boundaries(rgb, gt, gt)
    assert tuple(coincident[4, 4]) == GT_RED


def test_fixed_selection_is_order_invariant_and_preserves_each_rule(tmp_path):
    records, rows, _ = panel_inputs(tmp_path)
    first = select_cases(records, rows)
    second = select_cases(records[::-1], rows[::-1])
    assert first == second
    assert first["failure_image_id"] == "fixture_081"
    assert first["lowest_coverage_image_id"] == "fixture_098"
    assert len(first["selected"]) <= 4
    assert len(first["random_image_ids"]) == 2
    assert all(int(value.split("_")[-1]) >= 50 for value in first["random_image_ids"])


def test_renderer_uses_existing_masks_and_writes_explicit_local_selection(tmp_path):
    records, rows, gt = panel_inputs(tmp_path)
    chosen = select_cases(records, rows)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    angle = np.arange(128) * 2 * np.pi / 128
    normals = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    points = 125 + 55 * normals
    candidate_points = np.repeat(points[:, None, :], 4, axis=1)
    valid = np.zeros((128, 4), dtype=bool)
    valid[:, 0] = True
    for item in chosen["selected"]:
        np.savez_compressed(artifacts / (item["image_id"] + ".npz"), **{f"masks_G{i}": gt for i in range(5)}, gt_mask=gt, geometry_points=points, normals=normals, candidate_points=candidate_points, candidate_valid=valid)
    saved = render_panels(records, rows, artifacts, tmp_path / "panels")
    assert len(list((tmp_path / "panels").glob("*.png"))) == len(chosen["selected"])
    assert not saved["g4_deployable"]
    assert not saved["g4_strict_dice_upper_bound"]
    assert saved["npz_masks_are_not_recomputed"]
    assert saved["source_RGB_and_GT_are_not_published"]
    persisted = json.loads((tmp_path / "panels" / "selection.json").read_text())
    assert persisted["legend"]["GT_boundary"] == list(GT_RED)
    assert persisted["legend"]["prediction_boundary"] == list(PREDICTION_BLUE)
    with Image.open(saved["selected"][0]["panel_path"]) as image:
        assert image.width > 4 * 256
        assert image.height > 2 * 256
