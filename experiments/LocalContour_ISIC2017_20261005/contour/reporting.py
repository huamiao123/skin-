"""Postprocessing only for the frozen 50/100-image contour feasibility probe.

No images, masks, checkpoints or candidates are loaded here. One row per image
and selector is required, with all five selectors retained for difficult cases.
This is a selected-checkpoint development diagnosis, not independent testing.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("G0", "G1", "G2", "G3", "G4")
METRICS = ("dice", "iou", "bf1")
COMPARISONS = (("G1", "G0"), ("G2", "G0"), ("G3", "G0"), ("G4", "G0"), ("G4", "G1"), ("G4", "G2"), ("G4", "G3"), ("G3", "G2"))
COVERAGE_FIELDS = (
    "search_band_coverage", "effective_normal_nodes", "total_nodes",
    "candidate_covered_effective_nodes", "local_wrong_among_covered_nodes",
    "no_intersection_nodes", "multiple_intersection_nodes",
    "collinear_intersection_nodes", "edge_clipped_nodes",
    "prediction_components", "prediction_holes", "gt_components", "gt_holes",
    "contour_self_intersections", "strip_self_intersections", "degenerate_strip_count",
)


def default_reporting_rules() -> dict[str, Any]:
    """Copy this complete object into preregistration before viewing results."""
    return {
        "primary_scope": "locked100",
        "calibration_image_count": 50,
        "locked_image_count": 100,
        "search_band_aggregation": "macro_image",
        "search_band_min": 0.90,
        "candidate_coverage_aggregation": "pooled_covered_over_effective_nodes",
        "candidate_coverage_min": 0.80,
        "wrong_rate_aggregation": "pooled_local_wrong_over_candidate_covered_nodes",
        "local_wrong_rate_min": 0.10,
        "oracle_dice_gain_min": 0.005,
        "oracle_bf1_gain_min": 0.02,
        "oracle_bf1_requires_dice_nonnegative": True,
        "representation_cal_dice_drop_trigger": -0.001,
        "representation_sensitivity_budget": 1,
        "residual_dice_gain_min": 0.0025,
        "residual_bf1_gain_min": 0.01,
        "residual_bf1_requires_dice_nonnegative": True,
        "residual_corresponding_ci_low_must_be_positive": True,
        "dp_recovery_fraction_keep_min": 0.80,
        "G1_reconstruction_explained_fraction_keep_min": 0.80,
        "dp_recovery_metric": "dice_if_oracle_dice_gate_else_bf1",
        "decision_precedence": ["representation", "generator", "oracle_space", "representation_gain", "local_ambiguity", "dp_recovery", "residual"],
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 17,
        "bootstrap_unit": "paired_image",
        "bootstrap_shared_draws": "all_methods_metrics_comparisons_within_scope",
        "confidence_percentiles": [2.5, 97.5],
        "diagnosis_only": True,
        "origins": {
            "taskbook": ["search_band_min", "candidate_coverage_min", "oracle_dice_gain_min", "oracle_bf1_gain_min", "representation_sensitivity_budget"],
            "run_before_results_operationalization": ["local_wrong_rate_min", "representation_cal_dice_drop_trigger", "residual_dice_gain_min", "residual_bf1_gain_min", "dp_recovery_fraction_keep_min", "G1_reconstruction_explained_fraction_keep_min", "bootstrap_iterations", "bootstrap_seed", "decision_precedence", "search_band_aggregation"],
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = _json_value(row)
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in clean.items()})
    temporary.replace(path)


def paired_bootstrap(values: np.ndarray, image_ids: list[str], iterations: int = 2000, seed: int = 17) -> dict[str, Any]:
    """Input shape [images, five methods, three metrics], sorted paired IDs.

    Every comparison and metric uses the identical sampled image index matrix.
    This preserves within-image differences; pixels and contour nodes are never
    bootstrap observations.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(image_ids), 5, 3) or not len(image_ids) or not np.isfinite(values).all():
        raise ValueError("Bootstrap expects complete finite [N,5,3] paired image metrics")
    if len(set(image_ids)) != len(image_ids) or iterations != 2000 or seed != 17:
        raise ValueError("Use unique image IDs and preregistered 2000 bootstrap draws with seed17")
    draws = np.random.default_rng(seed).integers(0, len(image_ids), size=(iterations, len(image_ids)), dtype=np.int64)
    comparisons = []
    for method, baseline in COMPARISONS:
        delta = values[:, METHODS.index(method)] - values[:, METHODS.index(baseline)]
        resampled = delta[draws].mean(axis=1)
        for metric_index, metric in enumerate(METRICS):
            low, high = np.percentile(resampled[:, metric_index], [2.5, 97.5])
            comparisons.append({"method": method, "baseline": baseline, "metric": metric, "n_images": len(image_ids), "difference": float(delta[:, metric_index].mean()), "ci_low": float(low), "ci_high": float(high), "improved_images": int(np.sum(delta[:, metric_index] > 1e-12)), "harmed_images": int(np.sum(delta[:, metric_index] < -1e-12)), "equal_images": int(np.sum(np.abs(delta[:, metric_index]) <= 1e-12)), "improved_fraction": float(np.mean(delta[:, metric_index] > 1e-12)), "harmed_fraction": float(np.mean(delta[:, metric_index] < -1e-12))})
    return {"unit": "paired_image", "iterations": iterations, "seed": seed, "same_draws_for_all_comparisons_and_metrics": True, "draw_indices_sha256": hashlib.sha256(draws.tobytes()).hexdigest(), "ordered_image_ids_sha256": hashlib.sha256("\n".join(image_ids).encode()).hexdigest(), "interpretation": "exploratory_selected_checkpoint_development_diagnosis", "comparisons": comparisons}


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = {"cal": "cal50", "cal50": "cal50", "calibration": "cal50", "calibration50": "cal50", "locked": "locked100", "locked100": "locked100", "verification": "locked100", "verify100": "locked100", "locked_verification": "locked100", "locked_verification100": "locked100"}
    metric_aliases = {"dice": ("dice", "input_dice", "inputDice", "Dice"), "iou": ("iou", "input_iou", "inputIoU", "IoU"), "bf1": ("bf1", "input_bf1", "BF1")}
    result = []
    keys: set[tuple[str, str]] = set()
    image_roles: dict[str, str] = {}
    for supplied in rows:
        row = dict(supplied)
        image_id = str(row.get("image_id", "")).strip()
        method = row.get("method")
        role = aliases.get(str(row.get("role", "")))
        if not image_id or method not in METHODS or role is None:
            raise ValueError("Every row needs image_id, role cal50/locked100, and method G0..G4")
        key = image_id, method
        if key in keys:
            raise ValueError(f"Duplicate image/method row {key}")
        keys.add(key)
        if image_id in image_roles and image_roles[image_id] != role:
            raise ValueError(f"Image {image_id} changes role across methods")
        image_roles[image_id] = role
        row.update(image_id=image_id, role=role, method=method)
        for metric, choices in metric_aliases.items():
            found = next((row[name] for name in choices if name in row), None)
            if found is None or not np.isfinite(float(found)) or not 0 <= float(found) <= 1:
                raise ValueError(f"Invalid/missing {metric} for {key}")
            row[metric] = float(found)
        for field in COVERAGE_FIELDS:
            if field not in row:
                raise ValueError(f"Missing shared image diagnostic field {field} for {key}")
        search = row["search_band_coverage"]
        if search is not None and (not np.isfinite(float(search)) or not 0 <= float(search) <= 1):
            raise ValueError(f"Invalid search-band coverage for {key}")
        counts = (row["local_wrong_among_covered_nodes"], row["candidate_covered_effective_nodes"], row["effective_normal_nodes"], row["total_nodes"])
        if any(not np.isfinite(float(c)) or float(c) < 0 or float(c) != int(c) for c in counts) or not 0 <= counts[0] <= counts[1] <= counts[2] <= counts[3]:
            raise ValueError(f"Inconsistent covered/effective/wrong node counts for {key}")
        result.append(row)
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in result:
        by_image.setdefault(row["image_id"], []).append(row)
    for image_id, group in by_image.items():
        if {row["method"] for row in group} != set(METHODS) or len(group) != 5:
            raise ValueError(f"Every image must retain all five selectors, including difficult image {image_id}")
        base = next(row for row in group if row["method"] == "G0")
        for row in group:
            for field in COVERAGE_FIELDS:
                if _json_value(row[field]) != _json_value(base[field]):
                    raise ValueError(f"GT-free shared candidate diagnostic {field} differs across selectors for {image_id}")
    role_counts = {role: sum(value == role for value in image_roles.values()) for role in ("cal50", "locked100")}
    if role_counts != {"cal50": 50, "locked100": 100}:
        raise ValueError(f"The official validation partition must be exactly 50/100 images: {role_counts}")
    return sorted(result, key=lambda row: (row["role"], row["image_id"], row["method"]))


def _coverage_summary(image_rows: list[dict[str, Any]], cfg_metadata: dict[str, Any]) -> dict[str, Any]:
    n = len(image_rows)
    coverages = [float(row["search_band_coverage"]) for row in image_rows if row["search_band_coverage"] is not None]
    effective = sum(int(row["effective_normal_nodes"]) for row in image_rows)
    total = sum(int(row["total_nodes"]) for row in image_rows)
    covered = sum(int(row["candidate_covered_effective_nodes"]) for row in image_rows)
    wrong = sum(int(row["local_wrong_among_covered_nodes"]) for row in image_rows)
    planned_nodes = int(cfg_metadata.get("nodes", cfg_metadata.get("node_count", 128)))
    covered_lengths = [row.get("search_band_covered_length_input_px") for row in image_rows]
    total_lengths = [row.get("search_band_total_length_input_px") for row in image_rows]
    complete_lengths = all(v is not None and np.isfinite(float(v)) for v in covered_lengths + total_lengths)
    pooled_length = float(sum(total_lengths)) if complete_lengths else None
    result = {"n_images": n, "search_band_macro_image": float(np.mean(coverages)) if coverages else None, "search_band_defined_images": len(coverages), "search_band_undefined_images": n - len(coverages), "search_band_pooled_arc_length": float(sum(covered_lengths) / pooled_length) if complete_lengths and pooled_length else None, "effective_normal_nodes": effective, "total_existing_contour_nodes": total, "total_planned_nodes_including_empty_predictions": n * planned_nodes, "effective_normal_fraction_existing": effective / total if total else 0.0, "effective_normal_fraction_all_planned": effective / (n * planned_nodes), "candidate_covered_effective_nodes": covered, "candidate_coverage_pooled_effective": covered / effective if effective else None, "local_wrong_among_covered_nodes": wrong, "local_wrong_rate_pooled_given_covered": wrong / covered if covered else None}
    for field in ("no_intersection_nodes", "multiple_intersection_nodes", "collinear_intersection_nodes", "edge_clipped_nodes", "contour_self_intersections", "strip_self_intersections", "degenerate_strip_count"):
        result[field] = sum(int(row[field]) for row in image_rows)
        result[field.replace("_nodes", "") + "_images"] = sum(int(row[field]) > 0 for row in image_rows)
    result["multiple_prediction_component_images"] = sum(int(row["prediction_components"]) > 1 for row in image_rows)
    result["multiple_gt_component_images"] = sum(int(row["gt_components"]) > 1 for row in image_rows)
    result["prediction_hole_images"] = sum(int(row["prediction_holes"]) > 0 for row in image_rows)
    result["gt_hole_images"] = sum(int(row["gt_holes"]) > 0 for row in image_rows)
    result["empty_prediction_images"] = sum(int(row["total_nodes"]) == 0 for row in image_rows)
    statuses: dict[str, int] = {}
    for row in image_rows:
        status = str(row.get("status", "not_supplied"))
        statuses[status] = statuses.get(status, 0) + 1
    result["status_image_counts"] = statuses
    return result


def decide(coverage: dict[str, Any], comparison_rows: list[dict[str, Any]], cal_reconstruction_delta: float, cfg_metadata: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Frozen practical screening, scoped to this specific candidate interface."""
    def difference(method: str, baseline: str, metric: str) -> dict[str, Any]:
        return next(row for row in comparison_rows if (row["method"], row["baseline"], row["metric"]) == (method, baseline, metric))
    dg = difference("G4", "G0", "dice")
    bg = difference("G4", "G0", "bf1")
    dr = difference("G4", "G3", "dice")
    br = difference("G4", "G3", "bf1")
    search = coverage["search_band_macro_image"]
    candidate = coverage["candidate_coverage_pooled_effective"]
    wrong = coverage["local_wrong_rate_pooled_given_covered"]
    generator_ok = search is not None and candidate is not None and search >= rules["search_band_min"] and candidate >= rules["candidate_coverage_min"]
    wrong_ok = wrong is not None and wrong >= rules["local_wrong_rate_min"]
    dice_gate = dg["difference"] >= rules["oracle_dice_gain_min"]
    bf1_gate = bg["difference"] >= rules["oracle_bf1_gain_min"] and dg["difference"] >= 0
    oracle_ok = dice_gate or bf1_gate
    residual_dice = dr["difference"] >= rules["residual_dice_gain_min"] and dr["ci_low"] > 0
    residual_bf1 = br["difference"] >= rules["residual_bf1_gain_min"] and dr["difference"] >= 0 and br["ci_low"] > 0
    recovery_metric = "dice" if dice_gate else "bf1"
    reconstruction_denominator = difference("G4", "G0", recovery_metric)["difference"]
    reconstruction_numerator = difference("G1", "G0", recovery_metric)["difference"]
    reconstruction_explained = reconstruction_numerator / reconstruction_denominator if reconstruction_denominator > 0 else None
    keep_reconstruction = reconstruction_explained is not None and reconstruction_explained >= rules["G1_reconstruction_explained_fraction_keep_min"]
    oracle_denominator = difference("G4", "G2", recovery_metric)["difference"]
    dp_numerator = difference("G3", "G2", recovery_metric)["difference"]
    recovery = dp_numerator / oracle_denominator if oracle_denominator > 0 else None
    keep_dp = recovery is not None and recovery >= rules["dp_recovery_fraction_keep_min"]
    representation_problem = cal_reconstruction_delta < rules["representation_cal_dice_drop_trigger"]
    sensitivity_used = bool(cfg_metadata.get("sensitivity_performed", False)) or int(cfg_metadata.get("sensitivity_count", 0)) > 0 or int(cfg_metadata.get("nodes", cfg_metadata.get("node_count", 128))) == 256 or float(cfg_metadata.get("search_radius", 16)) == 32
    needs_sensitivity = representation_problem and not sensitivity_used
    if representation_problem:
        status, reason = "INCONCLUSIVE", "calibration_reconstruction_loss_requires_the_single_allowed_sensitivity" if needs_sensitivity else "reconstruction_loss_remains_after_the_sensitivity_budget"
    elif not generator_ok:
        status, reason = "GENERATOR_INADEQUATE", "current_search_range_or_candidate_generator_below_working_threshold"
    elif not oracle_ok:
        status, reason = "NO_ORACLE_SPACE", "current_fixed_candidate_interface_has_no_practical_gt_assisted_remaining_space"
    elif keep_reconstruction:
        status, reason = "KEEP_RECONSTRUCTION_SIMPLE", "zero_offset_reconstruction_explains_at_least_80_percent_of_positive_g4_to_g0_gain"
    elif not wrong_ok:
        status, reason = "INCONCLUSIVE", "candidate_covered_local_wrong_rate_below_preregistered_10_percent"
    elif keep_dp:
        status, reason = "KEEP_SIMPLE_DP", "simple_dp_recovers_at_least_80_percent_of_positive_g2_to_g4_gain"
    elif residual_dice or residual_bf1:
        status, reason = "CANDIDATE_NEXT_SMALL_COMPARISON", "practical_residual_after_dp_has_positive_paired_bootstrap_lower_limit"
    else:
        status, reason = "INCONCLUSIVE", "remaining_gain_after_dp_is_small_or_not_stable_in_the_exploratory_interval"
    return {"status": status, "reason": reason, "scope": "current_candidate_interface_only", "primary_scope": "locked100", "independent_test_claim": False, "g4_deployable": False, "g4_strict_dice_upper_bound": False, "needs_single_representation_sensitivity": needs_sensitivity, "sensitivity_performed": sensitivity_used, "calibration_g1_minus_g0_dice": cal_reconstruction_delta, "generator_working_thresholds_met": generator_ok, "local_ambiguity_threshold_met": wrong_ok, "gt_assisted_practical_space": oracle_ok, "g4_minus_g0_dice": dg, "g4_minus_g0_bf1": bg, "g4_minus_g1_dice": difference("G4", "G1", "dice"), "g4_minus_g1_bf1": difference("G4", "G1", "bf1"), "g4_minus_g3_dice": dr, "g4_minus_g3_bf1": br, "reconstruction_explained_metric": recovery_metric, "reconstruction_explained_denominator_g4_minus_g0": reconstruction_denominator, "reconstruction_explained_numerator_g1_minus_g0": reconstruction_numerator, "reconstruction_explained_fraction": reconstruction_explained, "reconstruction_explains_at_least_80_percent": keep_reconstruction, "residual_dice_screen": residual_dice, "residual_bf1_screen": residual_bf1, "dp_recovery_metric": recovery_metric, "dp_recovery_denominator_g4_minus_g2": oracle_denominator, "dp_recovery_numerator_g3_minus_g2": dp_numerator, "dp_recovery_fraction": recovery, "dp_recovers_at_least_80_percent": keep_dp, "rules": rules, "limitations": ["selected public CNN checkpoint and development-validation diagnostic", "GT-assisted G4 cannot be deployed and is not a strict Dice upper bound", "this screen does not establish method novelty or invalidate other contour methods", "paired bootstrap intervals are exploratory and do not replace independent confirmation"]}


def summarize(rows: list[dict[str, Any]], prereg: dict[str, Any], cfg_metadata: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Validate 750 complete rows and write CSV/JSON plus a one-page txt report.

    prereg['reporting_rules'] must equal default_reporting_rules(), recorded
    before results were inspected. cfg_metadata is copied as provenance only.
    """
    rules = prereg.get("reporting_rules")
    if rules != default_reporting_rules():
        raise ValueError("Record the complete default_reporting_rules() in preregistration before reading results")
    if "recorded_before_results" in prereg and prereg["recorded_before_results"] is not True:
        raise ValueError("Post-hoc rule registration cannot support a preregistered screen")
    normalized = _normalize_rows(rows)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    macro_rows, bootstrap_rows, scoped = [], [], {}
    for role in ("cal50", "locked100"):
        scope_rows = [row for row in normalized if row["role"] == role]
        image_ids = sorted({row["image_id"] for row in scope_rows})
        lookup = {(row["image_id"], row["method"]): row for row in scope_rows}
        values = np.array([[[lookup[(image_id, method)][metric] for metric in METRICS] for method in METHODS] for image_id in image_ids], dtype=np.float64)
        bootstrap = paired_bootstrap(values, image_ids)
        for row in bootstrap["comparisons"]:
            bootstrap_rows.append({"role": role, **row})
        for method_index, method in enumerate(METHODS):
            row = {"role": role, "method": method, "n_images": len(image_ids)}
            for metric_index, metric in enumerate(METRICS):
                vector = values[:, method_index, metric_index]
                delta = vector - values[:, 0, metric_index]
                row[metric] = float(vector.mean())
                row[metric + "_minus_g0"] = float(delta.mean())
                row[metric + "_improved_fraction_vs_g0"] = float(np.mean(delta > 1e-12))
                row[metric + "_harmed_fraction_vs_g0"] = float(np.mean(delta < -1e-12))
            macro_rows.append(row)
        coverage = _coverage_summary([lookup[(image_id, "G0")] for image_id in image_ids], cfg_metadata)
        scoped[role] = {"ordered_image_ids": image_ids, "coverage": coverage, "bootstrap": bootstrap}
    cal_reconstruction = next(row["difference"] for row in scoped["cal50"]["bootstrap"]["comparisons"] if (row["method"], row["baseline"], row["metric"]) == ("G1", "G0", "dice"))
    decision = decide(scoped["locked100"]["coverage"], scoped["locked100"]["bootstrap"]["comparisons"], cal_reconstruction, cfg_metadata, rules)
    payload = {"status": "COMPLETE_DEVELOPMENT_DIAGNOSIS", "n_images": 150, "n_rows": len(normalized), "all_five_selectors_all_images_retained": True, "no_gt_based_case_exclusions": True, "metric_coordinates": "CNN original evaluation at model input coordinates; geometry/BF1 in the same input coordinates, BF1 tolerance 2", "scope_results": scoped, "macro_table": macro_rows, "decision": decision, "preregistration": prereg, "configuration_metadata": cfg_metadata}
    _write_csv(root / "per_image_results.csv", normalized)
    _write_csv(root / "main_table.csv", macro_rows)
    _write_csv(root / "paired_bootstrap.csv", bootstrap_rows)
    _write_json(root / "summary.json", payload)
    _write_json(root / "decision.json", decision)
    cov = scoped["locked100"]["coverage"]
    def fmt(value: Any, percentage: bool = False) -> str:
        if value is None:
            return "未定义"
        return f"{100 * value:.2f}%" if percentage else f"{value:.6f}"
    lines = [
        "局部候选—整体轮廓：首轮开发诊断",
        f"结论：{decision['status']}；原因：{decision['reason']}",
        "范围：当前固定候选接口；150张官方验证图像，seed17划分50张校准/100张锁定核验，五组共750行。",
        "这属于选定公共CNN权重上的开发诊断，不声明独立测试；未使用官方test。",
        "G0原CNN；G1零位重建；G2局部评分；G3闭合DP；G4固定候选的GT距离代价，同一DP/平滑/栅格化。",
        "G4仅供离线诊断，不可部署，不是严格Dice上界；GT未参与候选生成或移动。",
        "主表（各组是逐图宏平均；BF1容差固定2输入像素）：",
    ]
    for role in ("cal50", "locked100"):
        for row in macro_rows:
            if row["role"] == role:
                lines.append(f"{role} {row['method']}  Dice={row['dice']:.6f}  IoU={row['iou']:.6f}  BF1={row['bf1']:.6f}")
    lines.extend([
        f"锁定核验：搜索带逐图覆盖={fmt(cov['search_band_macro_image'], True)}；候选覆盖/有效法线={fmt(cov['candidate_coverage_pooled_effective'], True)}；已覆盖时局部错选={fmt(cov['local_wrong_rate_pooled_given_covered'], True)}。",
        f"有效法线={cov['effective_normal_nodes']}/{cov['total_existing_contour_nodes']}现有节点；相对全部计划节点比例={fmt(cov['effective_normal_fraction_all_planned'], True)}；空预测图像={cov['empty_prediction_images']}。",
        f"无/多/共线交点节点={cov['no_intersection_nodes']}/{cov['multiple_intersection_nodes']}/{cov['collinear_intersection_nodes']}；贴边节点={cov['edge_clipped_nodes']}；多预测组件图像={cov['multiple_prediction_component_images']}；条带自交图像={cov['strip_self_intersections_images']}。",
    ])
    for name, comparison in (("G4−G0 Dice", decision["g4_minus_g0_dice"]), ("G4−G0 BF1", decision["g4_minus_g0_bf1"]), ("G4−G1 Dice", decision["g4_minus_g1_dice"]), ("G4−G1 BF1", decision["g4_minus_g1_bf1"]), ("G4−G3 Dice", decision["g4_minus_g3_dice"]), ("G4−G3 BF1", decision["g4_minus_g3_bf1"])):
        lines.append(f"{name}={comparison['difference']:.6f}，配对图像bootstrap95%区间[{comparison['ci_low']:.6f}, {comparison['ci_high']:.6f}]；受损图像={comparison['harmed_images']}/{comparison['n_images']}。")
    lines.extend([
        f"校准G1−G0 Dice={cal_reconstruction:.6f}；是否需唯一一次重建敏感性检查={decision['needs_single_representation_sensitivity']}；G1解释G4−G0收益比例={fmt(decision['reconstruction_explained_fraction'], True)}；DP恢复比例={fmt(decision['dp_recovery_fraction'], True)}（维度={decision['dp_recovery_metric']}）。",
        "工作门槛：搜索带≥90%，池化有效法线候选覆盖≥80%，G4−G0 Dice≥0.005或BF1≥0.02且Dice不降。",
        "运行前额外操作化：错选≥10%；校准重建损失<−0.001触发一次敏感性；G1解释≥80%收益时归因于简单重建；残余Dice≥0.0025或BF1≥0.01且Dice不降，并要求相应区间下限>0；DP恢复≥80%优先保留DP。",
        "区间使用2000次seed17配对图像重采样，各方法/指标/比较共享抽样。它们是开发诊断的探索区间。",
        "本轮仅筛查问题与剩余空间，不完成新颖性证明，不否定其他轮廓方法；困难图像保留并单列。",
        "详见同目录逐图CSV、主表、配对bootstrap、summary.json和decision.json。",
    ])
    (root / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
