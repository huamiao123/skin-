"""Run only the six registered symmetric seed17 lambda0.3/lambda3 jobs.

Require completed lambda1 training and the four registered Fixed-Shrink core
checks first. Delegate every D40 job to the unmodified train_stage using the
same selected B, fixed canonical M-train q, seed and scientific configuration.
No export, model inference, signal, source patch or additional grid is added.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rsi.runtime import atomic_json, code_provenance, load_config, sha256_file
from rsi.train import train_stage
from tools.train_remaining_pilot import completed_stage, preflight

HELPER_SHA256 = "9b8f348e412d7c339c567c40997a424c1f0b423415d0d60bd1594107f0a8912e"
METHODS = (("rsi", "RSI"), ("mean_hinge", "MeanHinge"), ("abs_hard", "AbsHard"))
PLAN = tuple((method, f"{name}-{weight:g}", weight) for weight in (.3, 3.) for method, name in METHODS)


def grid_preflight(cfg, sources):
    helper = PROJECT / "tools/train_remaining_pilot.py"
    if sha256_file(helper) != HELPER_SHA256:
        raise ValueError("The borrowed frozen training guard changed")
    if cfg["conditional_weight_grid"] != [.3, 1., 3.] or cfg["pilot_weight"] != 1.:
        raise ValueError("Require exactly the registered symmetric finite grid0.3/1/3")
    checks = preflight(cfg, sources["source_bundle_hash"])
    if not checks["ready"]:
        raise RuntimeError("Complete D0 full-budget DONE is required before grid training")
    base_hash = checks["B"]["selected_checkpoint_sha256"]
    common = dict(manifest_hash=checks["manifest_hash"], source_hash=sources["source_bundle_hash"])
    checks["lambda1"] = [completed_stage(cfg, f"{name}-1", "D", method=method, weight=1.,
                                        q=checks["q"], init_hash=base_hash, **common)
                         for method, name in METHODS]
    cnn = completed_stage(cfg, "A_CNN", "A_CNN", method="d0", weight=0., q=None, **common)
    transformer = completed_stage(cfg, "A_T", "A_T", method="d0", weight=0., q=None,
                                  init_hash=cnn["selected_checkpoint_sha256"], **common)
    b_done = json.loads(Path(checks["B"]["DONE_path"]).read_text())
    if b_done["signature"]["init_checkpoint_hash"] != transformer["selected_checkpoint_sha256"]:
        raise ValueError("Shared B is not initialized from the selected completed A_T")
    checks["A_CNN"], checks["A_T"] = cnn, transformer
    path = PROJECT / "outputs/fixed_shrink_core_seed17.json"
    core = json.loads(path.read_text())
    required_flags = (core.get("status") == "complete", core.get("seed") == 17,
                      core.get("test_scoring_locked") is True, core.get("training_performed") is False,
                      core.get("additional_weights_or_alphas") is False,
                      core.get("endpoint_consistency_passed") is True,
                      core.get("weights_unchanged_after_inference") is True,
                      core.get("frozen_source_unchanged_after_inference") is True,
                      core.get("state_sha256_before_inference") == core.get("state_sha256_after_inference"))
    if not all(required_flags):
        raise ValueError("Fixed-Shrink core is incomplete, changed, unlocked or has failed endpoints")
    if (core["config"] != cfg or core["source"]["source_sha256"] != sources["source_sha256"]
            or core["source"]["source_bundle_hash"] != sources["source_bundle_hash"]
            or core["manifest_sha256"] != checks["manifest_hash"]
            or core["shared_B_checkpoint_sha256"] != base_hash
            or core["model_checkpoint_sha256"] != checks["D0"]["selected_checkpoint_sha256"]
            or core["selected_D0_epoch"] != checks["D0"]["best_epoch"]
            or core["fixed_abs_hard_q_file_sha256"] != checks["q_file_sha256"]
            or core["fixed_q_record"] != checks["q_source"]):
        raise ValueError("Fixed-Shrink core differs from the frozen selected B/D0/q/config/manifest")
    if (Path(core["shared_B_checkpoint"]).resolve() != Path(checks["B"]["selected_checkpoint"]).resolve()
            or Path(core["model_checkpoint"]).resolve() != Path(checks["D0"]["selected_checkpoint"]).resolve()
            or Path(core["fixed_abs_hard_q_file"]).resolve() != Path(checks["q_file"]).resolve()):
        raise ValueError("Fixed-Shrink core paths differ from the selected training provenance")
    expected_alphas = [0., 1/3, 2/3, 1.]
    calculation = core["calculation"]
    if (cfg["fixed_shrink"]["alphas"] != expected_alphas or calculation["alphas"] != expected_alphas
            or calculation["q"] != checks["q"] or calculation["weight"] != 0.
            or calculation["method"] != "d0" or calculation["stage"] != "D"
            or calculation["original_metrics"] is not True
            or calculation["metric_coordinates"] != "original image coordinates"
            or calculation["hard_logit_threshold"] != 0.):
        raise ValueError("Fixed-Shrink core calculation is not the registered original-coordinate intervention")
    rows = core["rows"]
    if [row["alpha"] for row in rows] != expected_alphas:
        raise ValueError("Fixed-Shrink core does not contain exactly the four registered alphas")
    for row in rows:
        if (row["images"] != cfg["audited_counts"]["M"]["images"]["val"]
                or row["references"] != cfg["audited_counts"]["M"]["references"]["val"]
                or any(not math.isfinite(value) for value in row.values())):
            raise ValueError("Fixed-Shrink core lacks finite full M validation values")
    csv_path = Path(core["output_csv_path"])
    if sha256_file(csv_path) != core["output_csv_sha256"]:
        raise ValueError("Fixed-Shrink core CSV bytes changed")
    with csv_path.open(newline="") as stream:
        csv_rows = [{name: float(value) for name, value in row.items()} for row in csv.DictReader(stream)]
    if csv_rows != rows:
        raise ValueError("Fixed-Shrink core CSV and JSON numeric rows differ")
    endpoints = core["endpoint_checks"]
    zero = endpoints["alpha0_vs_selected_A_CNN"]
    if (not zero["exact_match"] or not zero["G_plus_exactly_zero"] or not zero["H_epsilon_exactly_zero"]
            or zero["expected_Dice"] != cnn["best_val_dice"] or zero["observed_Dice"] != rows[0]["dice"]
            or rows[0]["dice"] != cnn["best_val_dice"] or rows[0]["G_plus"] != 0. or rows[0]["H_epsilon"] != 0.):
        raise ValueError("Fixed-Shrink alpha0 is not exactly the selected CNN anchor")
    with (Path(checks["D0"]["DONE_path"]).parent / "epoch_diagnostics.csv").open(newline="") as stream:
        selected_d0 = next(row for row in csv.DictReader(stream) if int(row["epoch"]) == checks["D0"]["best_epoch"])
    for name in ("dice", "G_plus", "H_epsilon"):
        check = endpoints["alpha1_vs_D0_selected_validation"][name]
        expected = float(selected_d0["val_" + name])
        if not check["exact_match"] or check["expected"] != expected or check["observed"] != expected or rows[-1][name] != expected:
            raise ValueError(f"Fixed-Shrink alpha1 differs from selected D0: {name}")
    checks["FixedShrink_core"] = dict(path=str(path), sha256=sha256_file(path),
                                     CSV_path=str(csv_path), CSV_sha256=core["output_csv_sha256"],
                                     registered_alphas=expected_alphas, endpoint_consistency_passed=True,
                                     scope="M val core only; full tail, boundary, source and paired intervals remain pending")
    return checks


def run_grid(cfg, *, audit_path=None, validate_only=False):
    audit_path = Path(audit_path or PROJECT / "outputs/weight_grid_training_seed17.json").resolve()
    protected = [PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected += [Path(cfg["run_root"]).resolve(), Path(cfg["manifest"]).resolve(), Path(cfg["audit"]).resolve()]
    core_path = PROJECT / "outputs/fixed_shrink_core_seed17.json"
    protected += [core_path, PROJECT / "outputs/fixed_shrink_core_seed17.csv", PROJECT / "outputs/per_reference"]
    if core_path.is_file():
        protected.append(Path(json.loads(core_path.read_text())["output_csv_path"]).resolve())
    if any(audit_path == path or audit_path.is_relative_to(path) or path.is_relative_to(audit_path) for path in protected):
        raise ValueError("Audit output overlaps protected sources, training or scientific inputs")
    sources = code_provenance()
    frozen_cfg = json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False)
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="preflight", active="preflight",
                 seed=17, tool_path=str(Path(__file__).resolve()), tool_sha256=sha256_file(__file__),
                 invocation=list(sys.argv), scientific_config=json.loads(frozen_cfg), frozen_sources_before=sources,
                 frozen_helper_sha256=HELPER_SHA256, original_training_authority="unmodified rsi.train.train_stage",
                 original_rng_reset="unconditional set_seed(17) at every original train_stage call",
                 plan=[dict(method=method, run_name=name, weight=weight, stage="D", epochs=40) for method, name, weight in PLAN],
                 completed=[], maximum_D_jobs_including_D0_and_lambda1=10, test_scoring_locked=True,
                 exports_performed=False, training_budget_or_weights_migrated=False)
    atomic_json(audit_path, audit)
    tick = time.monotonic()
    def unchanged():
        if code_provenance()["source_sha256"] != sources["source_sha256"] or json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) != frozen_cfg:
            raise RuntimeError("Frozen source or scientific configuration changed")
    try:
        checks = grid_preflight(cfg, sources)
        audit["preflight"] = checks
        if validate_only:
            audit.update(status="validated_only", active="validated_only")
            return audit
        for method, name, weight in PLAN:
            unchanged()
            audit.update(status="running", active=name, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(audit_path, audit)
            selected = train_stage(cfg, 17, "D", init=Path(checks["B"]["selected_checkpoint"]),
                                   method=method, weight=weight, q=checks["q"], run_name=name)
            result = completed_stage(cfg, name, "D", method=method, weight=weight, q=checks["q"],
                                     manifest_hash=checks["manifest_hash"], source_hash=sources["source_bundle_hash"],
                                     init_hash=checks["B"]["selected_checkpoint_sha256"])
            if Path(selected).resolve() != Path(result["selected_checkpoint"]).resolve():
                raise RuntimeError("Original train_stage returned an unexpected selected checkpoint")
            unchanged()
            audit["completed"].append(result)
            atomic_json(audit_path, audit)
        audit.update(status="complete", active="symmetric_registered_weight_grid_complete")
        return audit
    except BaseException as error:
        audit.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        after = code_provenance()
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-tick,
                     frozen_sources_after=after, source_bundle_unchanged=after["source_sha256"] == sources["source_sha256"],
                     scientific_config_unchanged=json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) == frozen_cfg)
        atomic_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_grid(load_config(args.config), audit_path=args.audit_file, validate_only=args.validate_only)
    print(json.dumps(dict(status=result["status"], active=result["active"], completed=len(result["completed"]))))


if __name__ == "__main__":
    main()
