"""Resumable full controlled training queue; stops on the first failed stage."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from train_local import ROOT


def utc_now(): return datetime.now(timezone.utc).isoformat()


def main():
    output = ROOT / "models_phase3"
    output.mkdir(exist_ok=True)
    status_path = output / "run_status.json"
    status = dict(started_utc=utc_now(), state="RUNNING", completed=[], current=None)
    if status_path.exists():
        old = json.loads(status_path.read_text())
        status["completed"] = old.get("completed", [])
    jobs = [(family, seed) for seed in (17, 23, 42)
            for family in ("S64", "S96", "N0", "T8", "T32", "TG")]
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for family, seed in jobs:
        name = f"{family}_seed{seed}"
        done = output / name / "DONE.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "PASS":
            if name not in status["completed"]: status["completed"].append(name)
            continue
        status.update(current=name, current_started_utc=utc_now())
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("START", name, flush=True)
        log = output / f"{name}.log"
        with log.open("a") as handle:
            result = subprocess.run([sys.executable, "phase3_train.py", "--family", family,
                                     "--seed", str(seed)], cwd=ROOT, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state="FAILED", failure=name, exit_code=result.returncode, finished_utc=utc_now())
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            raise SystemExit(f"FAIL {name}; inspect {log}")
        status["completed"].append(name)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("DONE", name, flush=True)
    status.update(state="TRAINING_COMPLETE", current=None, finished_utc=utc_now())
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    for script in (("phase3_calibrate.py", "--layer", "controlled"),
                   ("phase3_readout.py", "--layer", "controlled"),
                   ("phase3_analyse.py",),
                   ("phase3_movement.py",),
                   ("phase3_ambiguity.py",),
                   ("phase3_oracle.py",),
                   ("phase3_alpha_curve.py",),
                   ("phase3_collect.py",)):
        print("START", script, flush=True)
        subprocess.run([sys.executable, *script], cwd=ROOT, env=environment, check=True)
    status.update(state="COMPLETE", finished_utc=utc_now())
    status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__": main()
