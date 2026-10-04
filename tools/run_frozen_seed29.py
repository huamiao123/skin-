"""Prepare or run the root-authorized, frozen seed29 confirmation protocol.

No command trains by default. --execute additionally requires an explicit root
selection JSON, complete symmetric seed17 exports, and their paired provenance.
The same original scientific config is used for seed29 A_CNN100/A_T100/B40,
its own canonical-train q, D0 and the fixed selected D40 objectives. Shrink
uses seed29's selected D0 at the frozen alpha and adds no training. No export,
H/T1/test selection, lambda/alpha search, or validation workpoint reselection
is performed here. Later export must preserve this frozen selection even if a
seed29 point falls outside the seed17 working-point eligibility rule.

Root selection schema (all paths identify already existing immutable evidence):
  schema_version: 1; decision_author: root; run_seed29: true
  scientific_status: unresolved_after_full_seed17_grid
  decision_reason: nonempty root explanation
  protocol_id: P2-IMA-M-v2; source_seed: 17; target_seed: 29
  selection_subset: M; selection_split: val; no_H_T1_test_selection: true
  fixed_workpoints: true; test_scoring_locked: true
  source_bundle_hash, manifest_sha256, config_sha256, seed29_driver_sha256
  evidence: {finalization_audit, main_table, per_reference_gains,
             paired_control_csv, paired_control_json}
    Each evidence entry is {path: ..., sha256: ...}.
  chosen_RSI / chosen_simple_control:
    {run_id: ..., checkpoint_hash: ..., method: ..., lambda: ..., alpha: ...}
  RSI lambda is 0.3/1/3. Simple control is the selected mean_hinge/abs_hard
  point at 0.3/1/3, a selected registered shrink alpha, or existing D0.

CLI preparation (missing decision exits 2 with status pending_decision):
  python tools/run_frozen_seed29.py --validate-only
Root-only conditional execution, after authoring the actual evidence contract:
  python tools/run_frozen_seed29.py --execute --selection-file PATH
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
from rsi.train import prepare_q, train_stage
from tools.finalize_pilot import review_matrix, review_completed_training
from tools.paired_control_comparison import method_name, read_existing_points, selected
from tools.train_remaining_pilot import completed_stage
from tools.train_weight_grid import PLAN as ADDITIONAL_GRID, grid_preflight

EXPECTED_SOURCE = "36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb"
BORROWED_TOOLS = {
    "tools/train_weight_grid.py": "67e1433c034d8c5efedb7d881fbb9c8e4d72fbf3b06a4d96b04e3ced6ded0e24",
    "tools/train_remaining_pilot.py": "9b8f348e412d7c339c567c40997a424c1f0b423415d0d60bd1594107f0a8912e",
    "tools/paired_control_comparison.py": "738f854195ee119fee387dc655c91a2a509387349dc0210d9169abaf02914ddb",
    "tools/finalize_pilot.py": "95de6989fc16e1de9ae477ab49c5252137907856a8f1f894c3c09d032bf766d8",
}
WEIGHTS = (0.3, 1.0, 3.0)
ALPHAS = (0.0, 1 / 3, 2 / 3, 1.0)
EVIDENCE_KEYS = ("finalization_audit", "main_table", "per_reference_gains",
                 "paired_control_csv", "paired_control_json")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _evidence_files(freeze):
    evidence = freeze["evidence"]
    if set(evidence) != set(EVIDENCE_KEYS):
        raise ValueError("Require all five explicit full-matrix/paired evidence artifacts")
    checked = {}
    for name in EVIDENCE_KEYS:
        item = evidence[name]
        path = Path(item["path"]).resolve()
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen evidence is absent or changed: {name}")
        checked[name] = dict(path=str(path), sha256=item["sha256"])
    if len({item["path"] for item in checked.values()}) != len(checked):
        raise ValueError("Frozen evidence artifacts must be distinct files")
    return checked


def _point_tuple(row, *, prefixed=""):
    get = lambda key: row[prefixed + key]
    return (str(get("run_id")), str(get("checkpoint_hash")), method_name(get("method")),
            float(get("lambda")), float(get("alpha")))


def _registered_choice(choice, main):
    point = _point_tuple(choice)
    matches = [row for row in main if row["subset"] == "M" and _point_tuple(row) == point]
    if len(matches) != 1 or not selected(matches[0]["workpoint_selected"]):
        raise ValueError("Choice is not the actual selected seed17 M working point")
    method, weight, alpha = point[2:]
    if method == "rsi":
        expected = f"RSI-{weight:g}"
        valid = weight in WEIGHTS and alpha == 1.0
    elif method in {"mean_hinge", "abs_hard"}:
        name = "MeanHinge" if method == "mean_hinge" else "AbsHard"
        expected = f"{name}-{weight:g}"
        valid = weight in WEIGHTS and alpha == 1.0
    elif method == "shrink":
        expected = f"Shrink-D0-{alpha:.6f}"
        valid = weight == 0.0 and alpha in ALPHAS
    elif method == "d0":
        expected = "D0"
        valid = weight == 0.0 and alpha == 1.0
    else:
        raise ValueError("Only the registered RSI and simple control families are allowed")
    if not valid or point[0] != expected:
        raise ValueError("Frozen working point is outside the registered run/weight/alpha grid")
    return dict(run_id=point[0], checkpoint_hash=point[1], method=method,
                weight=weight, alpha=alpha, source_subset="M", source_split="val")


def validate_selection(cfg, selection_path, sources):
    """Read-only CPU validation; never construct a model or call seed/CUDA APIs."""
    selection_path = Path(selection_path).resolve()
    freeze = json.loads(selection_path.read_text())
    flags = (freeze.get("schema_version") == 1, freeze.get("decision_author") == "root",
             freeze.get("run_seed29") is True,
             freeze.get("scientific_status") == "unresolved_after_full_seed17_grid",
             freeze.get("source_seed") == 17, freeze.get("target_seed") == 29,
             freeze.get("selection_subset") == "M", freeze.get("selection_split") == "val",
             freeze.get("no_H_T1_test_selection") is True, freeze.get("fixed_workpoints") is True,
             freeze.get("test_scoring_locked") is True)
    if not all(flags) or not isinstance(freeze.get("decision_reason"), str) or not freeze["decision_reason"].strip():
        raise ValueError("Require an explicit root-authored unresolved-grid decision with frozen M-only workpoints")
    if (not cfg["test_scoring_locked"] or cfg["epochs"] != dict(A_CNN=100, A_T=100, B=40, D=40)
            or cfg["conditional_weight_grid"] != list(WEIGHTS)
            or cfg["fixed_shrink"]["alphas"] != list(ALPHAS)
            or 29 not in cfg["development_seeds"]):
        raise ValueError("Scientific configuration is not the registered seed29 protocol")
    manifest_hash = sha256_file(cfg["manifest"])
    if (sources["source_bundle_hash"] != EXPECTED_SOURCE
            or freeze["source_bundle_hash"] != sources["source_bundle_hash"]
            or freeze["manifest_sha256"] != manifest_hash
            or freeze["protocol_id"] != cfg["protocol_id"]
            or freeze["config_sha256"] != sources["source_sha256"]["configs/p2_ima_m_v2.yaml"]
            or freeze["seed29_driver_sha256"] != sha256_file(__file__)):
        raise ValueError("Root choice is not bound to the frozen driver/source/config/manifest")
    for name, expected in BORROWED_TOOLS.items():
        if sha256_file(PROJECT / name) != expected:
            raise ValueError(f"Borrowed frozen validator changed: {name}")
    evidence = _evidence_files(freeze)
    final = json.loads(Path(evidence["finalization_audit"]["path"]).read_text())
    if (final["seed"] != 17 or final["scope"] != "T2" or not final["test_scoring_locked"]
            or not final["original_csv_bytes_unchanged"]
            or final["tool_sha256"] != BORROWED_TOOLS["tools/finalize_pilot.py"]
            or final["original_training_source"]["source_bundle_hash"] != sources["source_bundle_hash"]
            or final["original_training_source"]["source_sha256"] != sources["source_sha256"]):
        raise ValueError("Full seed17 finalization provenance differs from the frozen science")
    for item in final["original_csvs"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("An original seed17 export changed after finalization")
    checks = grid_preflight(cfg, sources)
    completions = {item["run_name"]: item for item in
                   [checks["A_CNN"], checks["A_T"], checks["B"], checks["D0"], *checks["lambda1"]]}
    for method, name, weight in ADDITIONAL_GRID:
        completions[name] = completed_stage(cfg, name, "D", method=method, weight=weight, q=checks["q"],
                                          manifest_hash=manifest_hash, source_hash=sources["source_bundle_hash"],
                                          init_hash=checks["B"]["selected_checkpoint_sha256"])
    final_done = {item["run_id"]: item for item in final["training_completion_evidence"]}
    if set(final_done) != set(completions):
        raise ValueError("Finalization does not bind all thirteen full-budget seed17 training stages")
    for name, actual in completions.items():
        proof = final_done[name]
        if (proof["done_sha256"] != actual["DONE_sha256"]
                or proof["best_checkpoint_sha256"] != actual["selected_checkpoint_sha256"]
                or proof["epochs"] != actual["epochs"] or proof["best_epoch"] != actual["best_epoch"]
                or sha256_file(proof["epoch_csv"]) != proof["epoch_csv_sha256"]):
            raise ValueError(f"Finalization does not match actual selected/full-budget training: {name}")
    main = _read_csv(evidence["main_table"]["path"])
    if any(row["split"] != "val" or row["seed"] != "17" or row["protocol_id"] != cfg["protocol_id"] for row in main):
        raise ValueError("Selection table is not the seed17 development-val protocol")
    if any(row["subset"] != "M" and selected(row["workpoint_selected"]) for row in main):
        raise ValueError("Source cohorts must not select working points")
    methods = {method: set() for method in ("rsi", "mean_hinge", "abs_hard")}
    shrink, ordinary = set(), set()
    mrows = [row for row in main if row["subset"] == "M"]
    if len(mrows) != 16 or len({_point_tuple(row) for row in mrows}) != 16:
        raise ValueError("Complete symmetric grid requires sixteen distinct M points including B/Anchor/four shrink points")
    for row in mrows:
        method, weight, alpha = _point_tuple(row)[2:]
        if method in methods:
            methods[method].add(weight)
        elif method == "shrink":
            shrink.add(alpha)
        else:
            ordinary.add(method)
    if any(weights != set(WEIGHTS) for weights in methods.values()) or shrink != set(ALPHAS) or ordinary != {"b", "anchor", "d0"}:
        raise ValueError("Selection evidence is not the complete predeclared symmetric grid")
    # This already-verified CPU reader recomputes image/reference aggregation
    # and checks shared anchors, main/ref identities, finite metrics and counts.
    existing = read_existing_points(Path(evidence["per_reference_gains"]["path"]),
                                    Path(evidence["main_table"]["path"]))
    if len(existing) != 16 or any(point["n_images"] != 223 or point["n_references"] != 471 for point in existing):
        raise ValueError("Every full-matrix M point must use the exact 223-image/471-reference cohort")
    reference_rows = _read_csv(evidence["per_reference_gains"]["path"])
    scope, stages, weights = review_matrix(reference_rows, "T2")
    if scope != "T2" or weights != list(WEIGHTS):
        raise ValueError("Reference evidence is not the actual complete symmetric seed17 grid")
    # The frozen finalizer also binds every exported method/lambda/action/alpha
    # to its DONE signature and checks all M/H/T1 identities. These source
    # cohorts establish provenance only; their values never select a point.
    review_completed_training(Path(cfg["run_root"]) / "seed17", stages, reference_rows, sources)
    for row in mrows:
        run = row["run_id"]
        parent = "B" if run == "Anchor" else "D0" if method_name(row["method"]) == "shrink" else run
        if parent not in completions or row["checkpoint_hash"] != completions[parent]["selected_checkpoint_sha256"]:
            raise ValueError("Full matrix references a checkpoint outside the actual selected training chain")
    chosen_rsi = _registered_choice(freeze["chosen_RSI"], main)
    chosen_control = _registered_choice(freeze["chosen_simple_control"], main)
    if chosen_rsi["method"] != "rsi" or chosen_control["method"] not in {"d0", "mean_hinge", "abs_hard", "shrink"}:
        raise ValueError("Require exactly one frozen RSI and one strongest simple control")
    paired = json.loads(Path(evidence["paired_control_json"]["path"]).read_text())
    hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in paired["input_files"]}
    if (paired["status"] != "complete" or paired["subset"] != "M" or paired["split"] != "val"
            or not paired["test_scoring_locked"] or paired["bootstrap_seed"] != 17 or paired["bootstrap_repeats"] != 2000
            or paired["existing_point_count"] != 16 or paired["tool_sha256"] != BORROWED_TOOLS["tools/paired_control_comparison.py"]
            or paired["output_csv_sha256"] != evidence["paired_control_csv"]["sha256"]
            or any(hashes.get(evidence[name]["path"]) != evidence[name]["sha256"]
                   for name in ("main_table", "per_reference_gains"))):
        raise ValueError("Direct paired-control evidence is not the bound full seed17 M matrix")
    pair_rows = _read_csv(evidence["paired_control_csv"]["path"])
    target = (chosen_rsi["run_id"], chosen_rsi["checkpoint_hash"], chosen_rsi["method"], chosen_rsi["weight"], chosen_rsi["alpha"])
    control = (chosen_control["run_id"], chosen_control["checkpoint_hash"], chosen_control["method"], chosen_control["weight"], chosen_control["alpha"])
    match = [row for row in pair_rows if _point_tuple(row, prefixed="rsi_") == target
             and _point_tuple(row, prefixed="control_") == control]
    if len(match) != 1 or match[0]["bootstrap_repeats"] != "2000" or match[0]["bootstrap_seed"] != "17" or not selected(match[0]["worst_tail_reranked_each_draw"]):
        raise ValueError("Frozen RSI/control lack their actual direct paired full-tail interval")
    return dict(selection_file=str(selection_path), selection_file_sha256=sha256_file(selection_path),
                root_decision=freeze, evidence=evidence, seed17_training=completions,
                original_seed17_csvs=final["original_csvs"],
                seed17_q=checks["q"], frozen_RSI=chosen_rsi, frozen_simple_control=chosen_control,
                manifest_sha256=manifest_hash, validation_scope="complete actual seed17 matrix and explicit root decision")


def training_plan(checked):
    rsi, control = checked["frozen_RSI"], checked["frozen_simple_control"]
    plan = [dict(run_name=name, stage=name, method="d0", weight=0.0, epochs=epochs)
            for name, epochs in (("A_CNN", 100), ("A_T", 100), ("B", 40))]
    plan += [dict(run_name="D0", stage="D", method="d0", weight=0.0, epochs=40),
             dict(run_name=rsi["run_id"], stage="D", method="rsi", weight=rsi["weight"], epochs=40)]
    if control["method"] in {"mean_hinge", "abs_hard"}:
        plan.append(dict(run_name=control["run_id"], stage="D", method=control["method"],
                         weight=control["weight"], epochs=40))
    return plan


def _verify_seed29_stage(cfg, point, source_hash, manifest_hash, *, init_hash, q):
    folder = Path(cfg["run_root"]) / "seed29" / point["run_name"]
    done_path = folder / "DONE.json"
    done = json.loads(done_path.read_text())
    expected = dict(config=cfg, seed=29, stage=point["stage"], method=point["method"], weight=point["weight"],
                    q=q, init_checkpoint_hash=init_hash, manifest_hash=manifest_hash, source_bundle_hash=source_hash)
    if done["signature"] != expected or done["seed"] != 29 or done["stage"] != point["stage"] or done["epochs"] != point["epochs"] or not done["test_scoring_locked"]:
        raise ValueError("Completed seed29 stage differs from its frozen budget/initialization/objective")
    rows = _read_csv(folder / "epoch_diagnostics.csv")
    if [int(row["epoch"]) for row in rows] != list(range(1, point["epochs"] + 1)):
        raise ValueError("Seed29 training log is not exactly the fixed epoch budget")
    best = max(float(row["val_dice"]) for row in rows)
    best_epoch = next(int(row["epoch"]) for row in rows if float(row["val_dice"]) == best)
    if done["best_val_dice"] != best or done["best_epoch"] != best_epoch:
        raise ValueError("Seed29 selected checkpoint does not match hard-Dice/earliest-tie history")
    selected_path = folder / "best.pth"
    if sha256_file(selected_path) != done["best_sha256"]:
        raise ValueError("Seed29 selected checkpoint bytes differ from DONE")
    return dict(point, selected_checkpoint=str(selected_path), selected_checkpoint_sha256=done["best_sha256"],
                best_epoch=best_epoch, best_val_dice=best, DONE_path=str(done_path), DONE_sha256=sha256_file(done_path))


def _own_q(path, cfg, base_hash):
    record = json.loads(Path(path).read_text())
    q = record["q"]
    if (not isinstance(q, (float, int)) or not math.isfinite(q) or record["checkpoint_hash"] != base_hash
            or record["manifest_hash"] != sha256_file(cfg["manifest"]) or record["source"] != "B_canonical_M_train"
            or record["quantile"] != 0.75 or record["fixed"] is not True or record["gradient"] is not False):
        raise ValueError("Seed29 q must come from its own selected B canonical M train")
    return dict(q=q, path=str(Path(path).resolve()), sha256=sha256_file(path), source=record)


def run_frozen(cfg, *, selection_path=None, audit_path=None, execute=False):
    selection_path = Path(selection_path or PROJECT / "outputs/seed17/frozen_seed29_selection.json").resolve()
    default_audit = "frozen_seed29_training.json" if execute else "frozen_seed29_preflight.json"
    audit_path = Path(audit_path or PROJECT / "outputs" / default_audit).resolve()
    # Restrict destinations before any read or write of a selection contract.
    # This owns only seed29-driver audits, never seed17 scientific audits,
    # raw/per-reference CSVs, checkpoints, data, or completion-supervisor logs.
    if (audit_path.parent != PROJECT / "outputs" or not audit_path.name.startswith("frozen_seed29_")
            or audit_path.suffix != ".json"):
        raise ValueError("Seed29 audit must use its own outputs/frozen_seed29_*.json namespace")
    protected = [(PROJECT / name).resolve() for name in
                 ("rsi", "configs", "tools", "tests", ".git", "data", "data_manifests", "runs",
                  "outputs/seed17", "outputs/per_reference")]
    protected += [Path(cfg[name]).resolve() for name in ("run_root", "manifest", "audit", "data_root", "cache_dir")]
    if cfg.get("audit_input_manifest"):
        protected.append(Path(cfg["audit_input_manifest"]).resolve())
    protected += [selection_path, PROJECT / "outputs/fixed_shrink_core_seed17.json", PROJECT / "outputs/fixed_shrink_core_seed17.csv"]
    if selection_path.is_file():
        raw = json.loads(selection_path.read_text())
        protected += [Path(item["path"]).resolve() for item in raw.get("evidence", {}).values()]
    if any(audit_path == path or audit_path.is_relative_to(path) or path.is_relative_to(audit_path) for path in protected):
        raise ValueError("Audit output overlaps a frozen source, input, selection or training artifact")
    sources = code_provenance()
    frozen_cfg = json.dumps(cfg, sort_keys=True, allow_nan=False)
    driver_hash = sha256_file(__file__)
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="preflight", active="root_selection_check",
                 tool_path=str(Path(__file__).resolve()), tool_sha256=driver_hash, invocation=list(sys.argv),
                 selection_file=str(selection_path), source_seed=17, target_seed=29,
                 execute_requested=execute, training_started=False, q_preparation_inference_started=False,
                 model_exports_performed=False, CSV_writes_started=False,
                 training_diagnostic_CSV_writes_started=False, export_CSV_writes_started=False,
                 test_scoring_locked=True,
                 lambda_or_alpha_reselection=False, H_T1_test_selection=False, completed=[],
                 frozen_sources_before=sources, scientific_config=cfg)
    atomic_json(audit_path, audit)
    started = time.monotonic()
    try:
        if not selection_path.is_file():
            audit.update(status="pending_decision", active="waiting_for_explicit_root_selection",
                         reason="No root-authored full-grid decision; no seed29 work is triggered")
            return audit
        if json.loads(selection_path.read_text()).get("run_seed29") is not True:
            audit.update(status="pending_decision", active="root_did_not_authorize_seed29",
                         reason="Explicit run_seed29=true is absent; no seed29 work is triggered")
            return audit
        checked = validate_selection(cfg, selection_path, sources)
        audit["preflight"] = checked
        plan = training_plan(checked)
        audit["plan"] = plan
        audit["total_training_epochs_if_fresh"] = sum(point["epochs"] for point in plan)
        def unchanged():
            if (code_provenance()["source_sha256"] != sources["source_sha256"]
                    or sha256_file(__file__) != driver_hash or json.dumps(cfg, sort_keys=True, allow_nan=False) != frozen_cfg
                    or sha256_file(selection_path) != checked["selection_file_sha256"]
                    or sha256_file(cfg["manifest"]) != checked["manifest_sha256"]
                    or any(sha256_file(item["path"]) != item["sha256"] for item in checked["evidence"].values())
                    or any(sha256_file(item["path"]) != item["sha256"] for item in checked["original_seed17_csvs"])
                    or any(sha256_file(PROJECT / name) != expected for name, expected in BORROWED_TOOLS.items())):
                raise RuntimeError("Frozen decision/evidence/source/config/manifest changed")
        unchanged()
        if not execute:
            audit.update(status="validated_only", active="frozen_seed29_plan_validated_no_training")
            return audit
        run_root = Path(cfg["run_root"]) / "seed29"
        marker_path = run_root / "FROZEN_SELECTION.json"
        marker = dict(selection_file_sha256=checked["selection_file_sha256"], driver_sha256=driver_hash,
                      source_bundle_hash=sources["source_bundle_hash"], manifest_sha256=checked["manifest_sha256"],
                      frozen_RSI=checked["frozen_RSI"], frozen_simple_control=checked["frozen_simple_control"], plan=plan)
        if marker_path.is_file():
            if json.loads(marker_path.read_text()) != marker:
                raise ValueError("Existing seed29 training is bound to a different root decision")
        else:
            if run_root.exists() and any(path.name in {"DONE.json", "latest.pth", "best.pth"} for path in run_root.rglob("*")):
                raise ValueError("Untracked earlier seed29 training exists; preserve and audit it before this frozen protocol")
            atomic_json(marker_path, marker)
        allowed = {point["run_name"] for point in plan}
        if any(path.parent.name not in allowed for pattern in ("*/DONE.json", "*/latest.pth", "*/best.pth")
               for path in run_root.glob(pattern)):
            raise ValueError("Seed29 contains an objective outside the frozen selected plan")
        audit["frozen_run_marker"] = str(marker_path)
        results, own_q = {}, None
        for point in plan:
            unchanged()
            stage = point["stage"]
            parent = {"A_T": "A_CNN", "B": "A_T", "D": "B"}.get(stage)
            init = Path(results[parent]["selected_checkpoint"]) if parent else None
            init_hash = results[parent]["selected_checkpoint_sha256"] if parent else None
            if stage == "D" and own_q is None:
                q_path = Path(results["B"]["selected_checkpoint"]).parent / "abs_hard_q.json"
                if not q_path.exists():
                    audit.update(active="seed29_B_canonical_train_q", q_preparation_inference_started=True)
                    atomic_json(audit_path, audit)
                    prepare_q(results["B"]["selected_checkpoint"], cfg)
                own_q = _own_q(q_path, cfg, results["B"]["selected_checkpoint_sha256"])
                audit["own_seed29_q"] = own_q
            q = own_q["q"] if stage == "D" else None
            audit.update(status="running", active=point["run_name"], training_started=True,
                         CSV_writes_started=True, training_diagnostic_CSV_writes_started=True,
                         updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(audit_path, audit)
            selected_path = train_stage(cfg, 29, stage, init=init, method=point["method"],
                                        weight=point["weight"], q=q, run_name=point["run_name"])
            result = _verify_seed29_stage(cfg, point, sources["source_bundle_hash"], checked["manifest_sha256"], init_hash=init_hash, q=q)
            if Path(selected_path).resolve() != Path(result["selected_checkpoint"]).resolve():
                raise RuntimeError("Original training stage returned an unexpected selected checkpoint")
            unchanged()
            if stage == "D" and _own_q(own_q["path"], cfg, results["B"]["selected_checkpoint_sha256"]) != own_q:
                raise RuntimeError("Own fixed canonical-train q changed")
            results[point["run_name"]] = result
            audit["completed"].append(result)
            atomic_json(audit_path, audit)
        control = checked["frozen_simple_control"]
        control_checkpoint = results["D0"] if control["method"] in {"shrink", "d0"} else results[control["run_id"]]
        audit["frozen_later_export_plan"] = dict(seed=29, own_B=results["B"], own_D0=results["D0"],
            RSI=dict(results[checked["frozen_RSI"]["run_id"]], alpha=1.0),
            strongest_simple_control=dict(control_checkpoint, method=control["method"], alpha=control["alpha"],
                                          weight=control["weight"], frozen_source_run_id=control["run_id"]),
            preserve_root_frozen_workpoints_even_if_seed29_noneligible=True, select_on_seed29=False,
            selection_subset="M", test_scoring_locked=True)
        audit.update(status="complete", active="frozen_seed29_training_complete_exports_pending")
        return audit
    except BaseException as error:
        audit.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        after = code_provenance()
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started,
                     frozen_sources_after=after, source_bundle_unchanged=after["source_sha256"] == sources["source_sha256"],
                     scientific_config_unchanged=json.dumps(cfg, sort_keys=True, allow_nan=False) == frozen_cfg)
        atomic_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--audit-file", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run_frozen(load_config(args.config), selection_path=args.selection_file,
                        audit_path=args.audit_file, execute=args.execute)
    print(json.dumps({key: result[key] for key in ("status", "active", "training_started", "q_preparation_inference_started", "CSV_writes_started")}))
    if result["status"] == "pending_decision":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
