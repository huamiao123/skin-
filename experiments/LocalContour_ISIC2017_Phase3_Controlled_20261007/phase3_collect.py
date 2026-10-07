"""Collect reproducible public manifests and logs without copying weights or score caches."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import contourpy
import numba
import numpy as np
import scipy
import torch

from train_local import ROOT, CACHE


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    target = ROOT / "results_phase3/controlled"
    target.mkdir(parents=True, exist_ok=True)
    family_seed = [(family, seed) for seed in (17, 23, 42)
                   for family in ("S64", "S96", "N0", "T8", "T32", "TG")]
    entries = []
    log_dir = target / "training_logs"; log_dir.mkdir(exist_ok=True)
    for family, seed in family_seed:
        source = ROOT / "models_phase3" / f"{family}_seed{seed}"
        manifest = json.loads((source / "model_manifest.json").read_text())
        if manifest["status"] != "PASS" or sha(source / "best.pth") != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"Model provenance failure: {family} seed{seed}")
        if sha(source / "candidate_scores.npy") != manifest["score_sha256"]:
            raise RuntimeError(f"Score provenance failure: {family} seed{seed}")
        entries.append(manifest)
        shutil.copyfile(source / "training_log.csv", log_dir / f"{family}_seed{seed}.csv")
    (target / "model_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    environment = dict(python=sys.version, platform=platform.platform(),
                       numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                       contourpy=contourpy.__version__, numba=numba.__version__,
                       cuda_available=torch.cuda.is_available(),
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       source_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (target / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    keys = ["prepare_phase3.py", "candidate_layout.py", "local_model.py", "refiner_model.py",
            "phase3_train.py", "phase3_dp.py", "phase3_fast_raster.py", "phase3_calibrate.py",
            "phase3_readout.py", "evaluation.py", "build_feature_cache.py", "build_candidate_cache.py"]
    (target / "source_hashes.json").write_text(json.dumps({name: sha(ROOT / name) for name in keys}, indent=2) + "\n")
    files = [CACHE / f"{name}.npy" for name in ("features", "rgb", "probability", "boundary", "prediction", "gt", "node_context")]
    files += [CACHE / "candidates_R32" / f"{name}.npy" for name in ("points", "source_points", "normals", "valid", "gt_distance", "has_contour")]
    cache = {str(path.relative_to(ROOT)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files}
    (target / "cache_manifest.json").write_text(json.dumps(cache, indent=2) + "\n")
    profile = json.loads((ROOT / "results_phase3/full_dp_profile.json").read_text())
    profile.update(models=len(entries),
                   model_training_seconds={f"{e['family']}_seed{e['seed']}": e["seconds"] for e in entries},
                   model_peak_cuda_bytes={})
    for family, seed in family_seed:
        import csv
        with (log_dir / f"{family}_seed{seed}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        profile["model_peak_cuda_bytes"][f"{family}_seed{seed}"] = max(int(row["peak_cuda_bytes"]) for row in rows)
    profile["limitation"] = "Training GPU memory and full-DP time measured; uncached full-system CNN/candidate inference latency requires separate timing."
    (target / "resource_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print("COLLECTED", len(entries), "models", flush=True)


if __name__ == "__main__": main()
