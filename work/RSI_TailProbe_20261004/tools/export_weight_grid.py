"""Export the six registered selected weight-grid checkpoints after all DONE.

Delegate inference, canonical/original losses and M/H/T1 output reuse to the
frozen rsi.export.export_checkpoint. Use its previously verified content cache
and exact KD boundary context. Missing DONE means pending, never partial grid
inference or CSV publication. This tool performs no training or test scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rsi.export import export_checkpoint
from rsi.runtime import atomic_json, code_provenance, load_config, set_seed, sha256_file
from tools.cached_pilot_runner import ReferenceMetricCache, installed_metric_cache
from tools.train_remaining_pilot import completed_stage
from tools.train_weight_grid import PLAN, grid_preflight

FROZEN_TOOLS = {
    "tools/train_weight_grid.py": "67e1433c034d8c5efedb7d881fbb9c8e4d72fbf3b06a4d96b04e3ced6ded0e24",
    "tools/train_remaining_pilot.py": "9b8f348e412d7c339c567c40997a424c1f0b423415d0d60bd1594107f0a8912e",
    "tools/cached_pilot_runner.py": "e058aee110988bd85ee24d4a3b1c08f7b288276bfdf7f6f14087d2e5fed6bf02",
    "tools/fast_exact_metrics.py": "3f1a75687a1ca21d12144c4ddb9198cbf7ef018eb4f1e84d66d5cbde48db6296",
}


def export_preflight(cfg, sources):
    for name, expected in FROZEN_TOOLS.items():
        if sha256_file(PROJECT / name) != expected:
            raise ValueError(f"A borrowed frozen tool changed: {name}")
    checks = grid_preflight(cfg, sources)
    selected, pending = [], []
    for method, name, weight in PLAN:
        done_path = Path(cfg["run_root"]) / "seed17" / name / "DONE.json"
        if not done_path.is_file():
            pending.append(name)
            continue
        result = completed_stage(cfg, name, "D", method=method, weight=weight, q=checks["q"],
                                 manifest_hash=checks["manifest_hash"], source_hash=sources["source_bundle_hash"],
                                 init_hash=checks["B"]["selected_checkpoint_sha256"])
        selected.append(dict(result, method=method, weight=weight))
    proof_path = PROJECT / "outputs/exact_boundary_native_verification.json"
    proof = json.loads(proof_path.read_text())
    if (proof.get("status") != "complete" or not proof.get("frozen_sources_unchanged")
            or proof["frozen_sources_before"]["source_bundle_hash"] != sources["source_bundle_hash"]
            or proof.get("fast_tool_sha256") != FROZEN_TOOLS["tools/fast_exact_metrics.py"]
            or not all(case.get("every_field_exact_equal") for case in proof.get("cases", []))
            or not any(case.get("megapixels", 0.) >= 29 for case in proof.get("cases", []))
            or not any(5 <= case.get("megapixels", 0.) <= 7 for case in proof.get("cases", []))):
        raise ValueError("Borrowed exact boundary implementation lacks matching native-resolution proof")
    return dict(shared=checks, selected_grid_checkpoints=selected, pending_full_budget_DONE=pending,
                all_six_DONE_verified=not pending, native_proof_path=str(proof_path), native_proof_sha256=sha256_file(proof_path))


def expected_references(cfg):
    with Path(cfg["manifest"]).open(newline="") as stream:
        refs = {(row["subset"], row["image_id"], row["reference_id"]): row for row in csv.DictReader(stream)
                if row["split"] == "val" and row["subset"] in ("M", "H", "T1")}
    for subset in ("M", "H", "T1"):
        selected = [key for key in refs if key[0] == subset]
        expected = cfg["audited_counts"][subset]
        if len(selected) != expected["references"]["val"] or len({key[1] for key in selected}) != expected["images"]["val"]:
            raise ValueError("Configured manifest does not contain the complete expected validation cohort")
    return refs


def check_export_rows(rows, cfg, point, manifest_hash, expected):
    identities = [(row["subset"], row["image_id"], row["reference_id"]) for row in rows]
    if len(rows) != len(expected) or len(set(identities)) != len(rows) or set(identities) != set(expected):
        raise ValueError("Export is not exactly the complete M/H/T1 reference cohort")
    for row, key in zip(rows, identities):
        if (row["split"] != "val" or row["seed"] != 17 or row["protocol_id"] != cfg["protocol_id"]
                or row["manifest_hash"] != manifest_hash or row["run_id"] != point["run_name"]
                or row["checkpoint_hash"] != point["selected_checkpoint_sha256"]
                or row["method"] != point["method"] or row["lambda"] != point["weight"]
                or row["action"] != 1. or row["alpha"] != 1.
                or row["loss_view"] != "canonical_letterbox_valid"):
            raise ValueError("Export metadata or canonical-loss view differs from the selected grid checkpoint")
        for name in ("group_id", "seg_filename", "tool", "skill_level"):
            if row[name] != expected[key][name]:
                raise ValueError(f"Reference provenance differs: {name}")
        for name in ("dice", "dice_anchor", "dice_D0", "loss", "loss_anchor", "loss_D0",
                     "loss_orig", "loss_D0_orig", "gain_vs_D0"):
            if not math.isfinite(row[name]):
                raise ValueError(f"Missing or nonfinite required original/canonical D0 comparison: {name}")
    return dict(reference_rows=len(rows), rows_by_subset=dict(Counter(row["subset"] for row in rows)),
                images_by_subset={subset:len({row["image_id"] for row in rows if row["subset"]==subset})
                                  for subset in ("M", "H", "T1")}, includes_Anchor_duplicate=False,
                complete_reference_identity_and_provenance_match=True)


def run_exports(cfg, *, audit_path=None, validate_only=False):
    output_dir = (PROJECT / "outputs/per_reference/seed17").resolve()
    audit_path = Path(audit_path or PROJECT / "outputs/weight_grid_exports_seed17.json").resolve()
    protected = [PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected += [Path(cfg["run_root"]).resolve(), Path(cfg["manifest"]).resolve(), Path(cfg["audit"]).resolve()]
    core_path = PROJECT / "outputs/fixed_shrink_core_seed17.json"
    protected_inputs = [*protected, PROJECT / "outputs/per_reference", core_path,
                        PROJECT / "outputs/fixed_shrink_core_seed17.csv", PROJECT / "outputs/exact_boundary_native_verification.json"]
    if core_path.is_file():
        protected_inputs.append(Path(json.loads(core_path.read_text())["output_csv_path"]).resolve())
    if any(audit_path == path or audit_path.is_relative_to(path) or path.is_relative_to(audit_path) for path in protected_inputs):
        raise ValueError("Audit output overlaps a protected scientific input or export target")
    if any(output_dir.is_relative_to(path) or path.is_relative_to(output_dir) for path in protected):
        raise ValueError("CSV destinations overlap protected scientific input or source paths")
    sources = code_provenance()
    frozen_cfg = json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False)
    cache = ReferenceMetricCache()
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="preflight", active="preflight",
                 seed=17, tool_path=str(Path(__file__).resolve()), tool_sha256=sha256_file(__file__),
                 invocation=list(sys.argv), scientific_config=json.loads(frozen_cfg), frozen_sources_before=sources,
                 frozen_borrowed_tools=FROZEN_TOOLS, original_export_authority="unmodified rsi.export.export_checkpoint",
                 plan=[dict(method=method, run_name=name, weight=weight, alpha=1., include_anchor=False) for method,name,weight in PLAN],
                 completed=[], inference_started=False, CSV_writes_started=False, training_performed=False,
                 test_scoring_locked=True, cache_statistics=cache.statistics())
    atomic_json(audit_path, audit)
    tick = time.monotonic()
    def unchanged():
        if code_provenance()["source_sha256"] != sources["source_sha256"] or json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) != frozen_cfg:
            raise RuntimeError("Frozen source or scientific config changed")
    try:
        checks = export_preflight(cfg, sources)
        audit["preflight"] = checks
        if not checks["all_six_DONE_verified"]:
            audit.update(status="pending", active="waiting_for_all_six_full_budget_DONE")
            return audit
        if validate_only:
            audit.update(status="validated_only", active="validated_only")
            return audit
        expected = expected_references(cfg)
        with installed_metric_cache(cache, fast_exact=True):
            for point in checks["selected_grid_checkpoints"]:
                unchanged()
                # Match the original pilot's unconditional stage seed reset
                # and backend settings; pending/validate-only never reach it.
                set_seed(17)
                torch.set_num_threads(4)
                audit["inference_backend"] = dict(seed=17, cpu_threads=torch.get_num_threads(),
                                                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                                                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
                destination = output_dir / (point["run_name"] + ".csv")
                audit.update(status="running", active=point["run_name"], updated_utc=datetime.now(timezone.utc).isoformat(),
                             inference_started=True, CSV_writes_started=True)
                atomic_json(audit_path, audit)
                rows = export_checkpoint(point["selected_checkpoint"], cfg, run_id=point["run_name"],
                                         alpha=1., weight=point["weight"], method=point["method"],
                                         d0_checkpoint=checks["shared"]["D0"]["selected_checkpoint"],
                                         include_anchor=False, destination=destination, seed=17)
                result = check_export_rows(rows, cfg, point, checks["shared"]["manifest_hash"], expected)
                unchanged()
                audit["completed"].append(dict(point, **result, CSV_path=str(destination), CSV_sha256=sha256_file(destination),
                                                original_export_JSON_path=str(destination.with_suffix(".json")),
                                                original_export_JSON_sha256=sha256_file(destination.with_suffix(".json"))))
                audit["cache_statistics"] = cache.statistics()
                atomic_json(audit_path, audit)
        audit.update(status="complete", active="six_registered_grid_exports_complete")
        return audit
    except BaseException as error:
        audit.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        after = code_provenance()
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-tick,
                     cache_statistics=cache.statistics(), frozen_sources_after=after,
                     source_bundle_unchanged=after["source_sha256"] == sources["source_sha256"],
                     scientific_config_unchanged=json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) == frozen_cfg)
        atomic_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_exports(load_config(args.config), audit_path=args.audit_file, validate_only=args.validate_only)
    print(json.dumps(dict(status=result["status"], active=result["active"], completed=len(result["completed"]),
                          pending=result.get("preflight",{}).get("pending_full_budget_DONE",[]))))


if __name__ == "__main__":
    main()
