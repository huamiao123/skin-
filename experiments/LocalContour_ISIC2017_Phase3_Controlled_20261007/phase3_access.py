"""Append-only operational data access log; role checks remain the enforcement layer."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from train_local import ROOT


def log_access(stage: str, role: str, count: int, labels_used: bool):
    path = ROOT / "models_phase3/access_log.jsonl"
    path.parent.mkdir(exist_ok=True)
    row = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, role=role,
               image_count=count, labels_used=labels_used)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
