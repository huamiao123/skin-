"""Train the three existing lambda1 pilot objectives after complete D0.

This thin scheduling driver delegates all scientific behavior to the frozen
rsi.train.train_stage. It performs no export, scoring, signal, weight migration
or additional epoch. Each original train_stage call resets seed17 before its
fresh optimizer, loaders and deterministic seed/epoch/image augmentation.
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

PLAN = (("rsi", "RSI-1"), ("mean_hinge", "MeanHinge-1"), ("abs_hard", "AbsHard-1"))


def completed_stage(cfg, name, stage, *, method, weight, q, manifest_hash, source_hash, init_hash=None):
    folder = Path(cfg["run_root"]) / "seed17" / name
    done_path, best = folder / "DONE.json", folder / "best.pth"
    done = json.loads(done_path.read_text())
    signature = done["signature"]
    expected = dict(config=cfg, seed=17, stage=stage, method=method, weight=weight, q=q,
                    init_checkpoint_hash=init_hash if init_hash is not None else signature["init_checkpoint_hash"],
                    manifest_hash=manifest_hash, source_bundle_hash=source_hash)
    if signature != expected or done["stage"] != stage or done["seed"] != 17:
        raise ValueError(f"Completed stage scientific signature differs: {name}")
    if done["epochs"] != cfg["epochs"][stage] or not done.get("test_scoring_locked"):
        raise ValueError(f"Stage has not completed its full locked budget: {name}")
    best_hash = sha256_file(best)
    if best_hash != done["best_sha256"]:
        raise ValueError(f"Selected checkpoint content differs from DONE: {name}")
    with (folder / "epoch_diagnostics.csv").open(newline="") as stream:
        epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
    if epochs != list(range(1, cfg["epochs"][stage] + 1)):
        raise ValueError(f"Stage epoch log is not exactly its fixed budget: {name}")
    return dict(run_name=name, stage=stage, epochs=done["epochs"], best_epoch=done["best_epoch"],
                best_val_dice=done["best_val_dice"], selected_checkpoint=str(best),
                selected_checkpoint_sha256=best_hash, DONE_path=str(done_path),
                DONE_sha256=sha256_file(done_path), signature_verified=True)


def preflight(cfg, source_hash):
    if not cfg["test_scoring_locked"] or cfg["epochs"]["B"] != 40 or cfg["epochs"]["D"] != 40:
        raise ValueError("Require the registered locked B40/D40 scientific configuration")
    manifest_hash = sha256_file(cfg["manifest"])
    base = completed_stage(cfg, "B", "B", method="d0", weight=0., q=None,
                           manifest_hash=manifest_hash, source_hash=source_hash)
    q_path = Path(base["selected_checkpoint"]).parent / "abs_hard_q.json"
    record = json.loads(q_path.read_text())
    q = record["q"]
    if (not isinstance(q, (float, int)) or not math.isfinite(q)
            or record.get("source") != "B_canonical_M_train" or record.get("quantile") != .75
            or cfg["abs_hard_quantile"] != .75 or record.get("fixed") is not True
            or record.get("gradient") is not False or record.get("manifest_hash") != manifest_hash
            or record.get("checkpoint_hash") != base["selected_checkpoint_sha256"]):
        raise ValueError("Stored q is not the fixed canonical M train quantile of the shared selected B")
    result = dict(B=base, q=q, q_file=str(q_path), q_file_sha256=sha256_file(q_path),
                  q_source=record, manifest_hash=manifest_hash)
    d0_done = Path(cfg["run_root"]) / "seed17/D0/DONE.json"
    if not d0_done.is_file():
        return dict(result, ready=False, reason="D0 has not yet published full-budget DONE.json")
    d0 = completed_stage(cfg, "D0", "D", method="d0", weight=0., q=q,
                         manifest_hash=manifest_hash, source_hash=source_hash,
                         init_hash=base["selected_checkpoint_sha256"])
    return dict(result, D0=d0, ready=True)


def run_remaining(cfg, *, audit_path=None, validate_only=False, invocation=None):
    audit_path = Path(audit_path or PROJECT / "outputs/train_remaining_pilot_seed17.json").resolve()
    protected = [PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected += [Path(cfg["run_root"]).resolve(), Path(cfg["manifest"]).resolve(), Path(cfg["audit"]).resolve()]
    if any(audit_path == path or audit_path.is_relative_to(path) or path.is_relative_to(audit_path) for path in protected):
        raise ValueError("Audit output overlaps a protected source, training or scientific input")
    sources = code_provenance()
    frozen_cfg = json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False)
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="preflight", active="preflight", seed=17,
                 tool_path=str(Path(__file__).resolve()), tool_sha256=sha256_file(__file__),
                 invocation=invocation if invocation is not None else list(sys.argv),
                 scientific_config=json.loads(frozen_cfg), frozen_sources_before=sources,
                 original_training_authority="unmodified rsi.train.train_stage",
                 original_rng_reset="train_stage calls set_seed(seed) unconditionally before DONE branch, loaders and optimizer; every call is seed17",
                 plan=[dict(method=method, run_name=name, stage="D", weight=1., epochs=cfg["epochs"]["D"])
                       for method, name in PLAN], completed=[], test_scoring_locked=True,
                 exports_performed=False, training_budget_or_weights_migrated=False)
    atomic_json(audit_path, audit)
    tick = time.monotonic()
    try:
        checks = preflight(cfg, sources["source_bundle_hash"])
        audit["preflight"] = checks
        if not checks["ready"]:
            audit.update(status="blocked_preflight", active="waiting_for_D0_DONE")
            if not validate_only:
                raise RuntimeError(checks["reason"])
            return audit
        if validate_only:
            audit.update(status="validated_only", active="validated_only")
            return audit
        for method, name in PLAN:
            if code_provenance()["source_sha256"] != sources["source_sha256"] or json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) != frozen_cfg:
                raise RuntimeError("Frozen source or scientific config changed")
            audit.update(status="running", active=name, updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(audit_path, audit)
            selected = train_stage(cfg, 17, "D", init=Path(checks["B"]["selected_checkpoint"]),
                                   method=method, weight=1., q=checks["q"], run_name=name)
            result = completed_stage(cfg, name, "D", method=method, weight=1., q=checks["q"],
                                     manifest_hash=checks["manifest_hash"], source_hash=sources["source_bundle_hash"],
                                     init_hash=checks["B"]["selected_checkpoint_sha256"])
            if Path(selected).resolve() != Path(result["selected_checkpoint"]).resolve():
                raise RuntimeError("Original train_stage returned an unexpected selected checkpoint")
            if code_provenance()["source_sha256"] != sources["source_sha256"] or json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) != frozen_cfg:
                raise RuntimeError("Frozen source or scientific config changed during the completed stage")
            audit["completed"].append(result)
            atomic_json(audit_path, audit)
        audit.update(status="complete", active="three_lambda1_objectives_complete")
        return audit
    except BaseException as error:
        audit.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        after = code_provenance()
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - tick,
                     frozen_sources_after=after, source_bundle_unchanged=after["source_sha256"] == sources["source_sha256"],
                     scientific_config_unchanged=json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False) == frozen_cfg)
        atomic_json(audit_path, audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run_remaining(load_config(args.config), audit_path=args.audit_file,
                           validate_only=args.validate_only)
    print(json.dumps(dict(status=result["status"], active=result["active"], completed=len(result["completed"]))))


if __name__ == "__main__":
    main()
