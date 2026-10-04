"""Wait for owned registered training, then finish only seed17 artifacts.

No training, signal, seed29, test scoring or scientific-success decision is
performed. All dependent commands are fixed argument lists and run serially.
Root remains responsible for interpreting exploratory selected-val results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rsi.runtime import atomic_json, code_provenance, load_config, sha256_file
from tools.export_weight_grid import FROZEN_TOOLS as EXPORT_DEPENDENCIES, export_preflight
from tools.train_weight_grid import PLAN

SOURCE_BUNDLE = "36825cbb7a90064f3db3cfd97eb16ec62b29aa1947e17e88c7ad4759024268fb"
FROZEN_TOOLS = dict(EXPORT_DEPENDENCIES, **{
    "tools/export_weight_grid.py": "f289d8b28b8c7c8a3216b084fa427aa81c87117a4eb0f8f8361998272d24f61a",
    "tools/finalize_pilot.py": "95de6989fc16e1de9ae477ab49c5252137907856a8f1f894c3c09d032bf766d8",
    "tools/source_threshold_sensitivity.py": "dfa049b96ef44dad15fda75627bad246128f46fcc07cde38ad98913307c4e99c",
    "tools/paired_control_comparison.py": "738f854195ee119fee387dc655c91a2a509387349dc0210d9169abaf02914ddb",
})
POLL_SECONDS = 45
IDENTITY_KEYS = ("pid", "started_utc", "command", "audit", "tool_sha256", "process_start_ticks", "uid")


def process_start_ticks(pid):
    """Linux kernel process identity, unaffected by wall-clock adjustment."""
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2:].split()[19])


def frozen_check(cfg, original_cfg, driver_hash=None):
    sources = code_provenance()
    if sources["source_bundle_hash"] != SOURCE_BUNDLE or json.dumps(cfg, sort_keys=True) != original_cfg:
        raise ValueError("Frozen training source or scientific configuration changed")
    if driver_hash is not None and sha256_file(__file__) != driver_hash:
        raise ValueError("Frozen completion supervisor changed")
    for name, digest in FROZEN_TOOLS.items():
        if sha256_file(PROJECT / name) != digest:
            raise ValueError(f"Frozen completion tool changed: {name}")
    return sources


def readiness(cfg, record_path, record_identity, record_sha256, completion_retry=False):
    record = json.loads(record_path.read_text())
    identity = {key: record[key] for key in IDENTITY_KEYS}
    if identity != record_identity or sha256_file(record_path) != record_sha256:
        raise ValueError("Owned training process registration changed")
    owned_pid = record_identity["pid"]
    audit_path = Path(record["audit"])
    training = json.loads(audit_path.read_text())
    expected_plan = [dict(method=method, run_name=name, weight=weight, stage="D", epochs=40) for method,name,weight in PLAN]
    if (training["tool_sha256"] != FROZEN_TOOLS["tools/train_weight_grid.py"] or training["seed"] != 17
            or training["scientific_config"] != cfg or training["plan"] != expected_plan
            or training["frozen_sources_before"]["source_bundle_hash"] != SOURCE_BUNDLE
            or training.get("test_scoring_locked") is not True):
        raise ValueError("Registered training audit differs from the frozen six-job scientific plan")
    if training["status"] == "failed":
        raise RuntimeError("Registered grid training failed: " + training.get("failure", "unspecified failure"))
    expected_names = [name for _, name, _ in PLAN]
    completed_names = [row["run_name"] for row in training["completed"]]
    if completed_names != expected_names[:len(completed_names)]:
        raise ValueError("Training completed list is not a unique ordered prefix of the registered grid")
    pending = [name for name in expected_names if not (Path(cfg["run_root"])/"seed17"/name/"DONE.json").is_file()]
    ready = training["status"] == "complete" and not pending and completed_names == expected_names
    if ready and (training.get("source_bundle_unchanged") is not True or training.get("scientific_config_unchanged") is not True):
        raise ValueError("Completed grid training did not preserve the frozen science")
    process = dict(pid=owned_pid, alive=False)
    if not ready:
        try:
            owned = psutil.Process(owned_pid)
            actual_command = owned.cmdline()
            actual_started = owned.create_time()
            actual_ticks = process_start_ticks(owned_pid)
            expected_started = datetime.fromisoformat(record["started_utc"]).timestamp()
            if (actual_command != record["command"] or abs(actual_started - expected_started) > 5
                    or owned.uids().real != os.getuid() or owned.uids().real != record["uid"]
                    or actual_ticks != record["process_start_ticks"]):
                raise ValueError("PID identity differs from the explicitly owned training process")
            process.update(alive=owned.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD),
                           command=actual_command, started_epoch_seconds=actual_started,
                           process_start_ticks=actual_ticks, uid=owned.uids().real)
        except (psutil.NoSuchProcess, FileNotFoundError):
            pass
        if not process["alive"]:
            # Re-read the atomic completion audit to avoid an exit/completion
            # observation race. An incomplete vanished job is a hard failure.
            final = json.loads(audit_path.read_text())
            if final["status"] == "complete" and not completion_retry:
                return readiness(cfg, record_path, record_identity, record_sha256, completion_retry=True)
            raise RuntimeError("Owned grid-training PID disappeared before registered completion")
    return dict(ready=ready, training_status=training["status"], training_active=training["active"],
                completed_training_jobs=len(completed_names), pending_full_budget_DONE=pending,
                owned_process=process, registered_training_audit_sha256=sha256_file(audit_path))


def command_plan(config_path):
    python = sys.executable
    return [
        ("export_weight_grid", [python, "-u", str(PROJECT/"tools/export_weight_grid.py"), "--config", str(config_path)]),
        ("finalize_full_T2", [python, "-u", str(PROJECT/"tools/finalize_pilot.py"), "--scope", "T2"]),
        ("source_threshold_sensitivity", [python, "-u", str(PROJECT/"tools/source_threshold_sensitivity.py"),
                                          "--inputs", str(PROJECT/"outputs/seed17/corrected_inputs/*.csv"),
                                          "--config", str(config_path)]),
        ("paired_control_comparison", [python, "-u", str(PROJECT/"tools/paired_control_comparison.py")]),
    ]


def finish(config_path, *, audit_path=None, log_path=None, validate_only=False):
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    original_cfg = json.dumps(cfg, sort_keys=True)
    record_path = PROJECT / "outputs/weight_grid_process_seed17.json"
    record = json.loads(record_path.read_text())
    if (type(record["pid"]) is not int or record["pid"] <= 0
            or type(record["process_start_ticks"]) is not int or record["process_start_ticks"] <= 0
            or type(record["uid"]) is not int or record["uid"] != os.getuid()
            or record["tool_sha256"] != FROZEN_TOOLS["tools/train_weight_grid.py"]):
        raise ValueError("Require the explicitly registered root-owned grid process")
    registered_command = record["command"]
    if (Path(registered_command[2]).resolve() != PROJECT/"tools/train_weight_grid.py"
            or "--config" not in registered_command
            or Path(registered_command[registered_command.index("--config")+1]).resolve() != config_path):
        raise ValueError("Registered training command does not use this frozen driver/config")
    identity = {key:record[key] for key in IDENTITY_KEYS}
    record_hash = sha256_file(record_path)
    audit_path = Path(audit_path or PROJECT/"outputs/weight_grid_completion_seed17.json").resolve()
    log_path = Path(log_path or PROJECT/"outputs/weight_grid_completion_console_seed17.log").resolve()
    for path, suffix in ((audit_path, ".json"), (log_path, ".log")):
        if (path.parent != PROJECT/"outputs" or not path.name.startswith("weight_grid_completion")
                or path.suffix != suffix):
            raise ValueError("Supervisor audit/log must use its own weight_grid_completion output namespace")
    protected = [PROJECT/name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected += [Path(cfg[name]).resolve() for name in ("run_root", "manifest", "audit", "data_root", "cache_dir")]
    protected += [
                  PROJECT/"outputs/per_reference", PROJECT/"outputs/seed17", record_path,
                  Path(record["audit"]).resolve(), Path(record["log"]).resolve(),
                  PROJECT/"outputs/fixed_shrink_core_seed17.json", PROJECT/"outputs/fixed_shrink_core_seed17.csv",
                  PROJECT/"outputs/exact_boundary_native_verification.json"]
    if audit_path == log_path or any(path == target or path.is_relative_to(target) or target.is_relative_to(path)
                                   for path in (audit_path,log_path) for target in protected):
        raise ValueError("Supervisor audit/log overlaps protected scientific inputs or dependent outputs")
    driver_hash = sha256_file(__file__)
    sources = frozen_check(cfg, original_cfg, driver_hash)
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="preflight", active="preflight",
                 tool_path=str(Path(__file__).resolve()), tool_sha256=driver_hash, invocation=list(sys.argv),
                 source_before=sources, frozen_tools=FROZEN_TOOLS, scientific_config=cfg,
                 config_path=str(config_path), config_sha256=sha256_file(config_path),
                 owned_training_registration=str(record_path), owned_training_registration_sha256=record_hash,
                 owned_training_identity=identity, poll_seconds=POLL_SECONDS,
                 commands=[dict(stage=name,args=args) for name,args in command_plan(config_path)], completed_commands=[],
                 subprocesses_started=False, test_scoring_locked=True, seed29_triggered=False,
                 root_scientific_decision_pending=True, scientific_success_declared=False)
    atomic_json(audit_path, audit)
    tick = time.monotonic()
    try:
        while True:
            frozen_check(cfg, original_cfg, driver_hash)
            state = readiness(cfg, record_path, identity, record_hash)
            audit.update(status="validated_only" if validate_only else "waiting", active="waiting_for_registered_grid", readiness=state)
            atomic_json(audit_path, audit)
            if validate_only:
                return audit
            if state["ready"]:
                break
            time.sleep(POLL_SECONDS)
        full = export_preflight(cfg, sources)
        if not full["all_six_DONE_verified"]:
            raise RuntimeError("Completed registration lacks the actual six selected full-budget checkpoints")
        audit["full_checkpoint_preflight"] = full
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", buffering=1) as log:
            for stage,args in command_plan(config_path):
                before = frozen_check(cfg, original_cfg, driver_hash)
                entry = dict(stage=stage,args=args,started_utc=datetime.now(timezone.utc).isoformat(),source_before=before)
                audit.update(status="running",active=stage,subprocesses_started=True,active_command=entry)
                atomic_json(audit_path, audit)
                print(json.dumps(dict(stage=stage,args=args)),file=log,flush=True)
                result = subprocess.run(args,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=False)
                after = frozen_check(cfg, original_cfg, driver_hash)
                entry.update(return_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),source_after=after)
                if result.returncode != 0:
                    raise RuntimeError(f"Dependent command failed: {stage} exit{result.returncode}")
                audit["completed_commands"].append(entry)
                atomic_json(audit_path, audit)
        audit.update(status="complete",active="seed17_artifacts_complete_root_decision_pending")
        return audit
    except BaseException as error:
        audit.update(status="failed",failure_type=type(error).__name__,failure=str(error))
        raise
    finally:
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick,
                     source_after=code_provenance(),root_scientific_decision_pending=True,
                     scientific_success_declared=False,seed29_triggered=False,test_scoring_locked=True)
        atomic_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT/"configs/p2_ima_m_v2.yaml")
    parser.add_argument("--audit-file",type=Path)
    parser.add_argument("--log-file",type=Path)
    parser.add_argument("--validate-only",action="store_true")
    args = parser.parse_args()
    result = finish(args.config,audit_path=args.audit_file,log_path=args.log_file,validate_only=args.validate_only)
    print(json.dumps(dict(status=result["status"],ready=result.get("readiness",{}).get("ready",False),
                          completed_commands=len(result["completed_commands"]))))


if __name__ == "__main__":
    main()
