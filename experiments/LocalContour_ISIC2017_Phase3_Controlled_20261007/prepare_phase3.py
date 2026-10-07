"""Freeze Phase 3 image roles before any new model is trained."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from project_paths import ISIC2017

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "splits"
VAL_ROLES = ROOT.parent / "contour_probe_20261005/assets/val_split_seed17.csv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ids(name: str, ids: list[str]) -> dict:
    path = SPLITS / f"{name}.txt"
    path.write_text("".join(f"{image_id}\n" for image_id in ids))
    return {"count": len(ids), "sha256": digest(path), "path": str(path.relative_to(ROOT))}


def main() -> None:
    train = (SPLITS / "train.txt").read_text().splitlines()
    val = (SPLITS / "val.txt").read_text().splitlines()
    test = (SPLITS / "test.txt").read_text().splitlines()
    assert len(train) == len(set(train)) == 2000
    assert len(val) == len(set(val)) == 150
    assert len(test) == len(set(test)) == 600
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    # The supplied public metadata has only image_id, age_approximate, and sex.
    metadata = ISIC2017 / "metadata/archive_metadata/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv"
    with metadata.open(newline="") as handle:
        columns = csv.DictReader(handle).fieldnames
    assert columns == ["image_id", "age_approximate", "sex"], columns
    ordered = sorted(train, key=lambda value: hashlib.sha256(f"LC-P3-v1|{value}".encode()).hexdigest())
    stop = ordered[:200]
    fit = ordered[200:]
    with VAL_ROLES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 150
    cal = [row["image_id"] for row in rows if row["role"] == "calibration50"]
    dev = [row["image_id"] for row in rows if row["role"] == "locked_verification100"]
    assert len(cal) == 50 and len(dev) == 100
    assert set(cal + dev) == set(val)
    roles = {"fit": fit, "stop": stop, "calibration": cal,
             "development_readout": dev, "test": test}
    assert sum(len(v) for v in roles.values()) == len(set().union(*(set(v) for v in roles.values())))
    manifest = {
        "protocol": "LC-P3-v1", "status": "LOCKED", "split_unit": "image_id",
        "patient_independence": "not established: public metadata has no patient/lesion group ID",
        "selection": "SHA256('LC-P3-v1|' + image_id), first 200 stop; remaining 1800 fit",
        "val_role_source": str(VAL_ROLES), "val_role_source_sha256": digest(VAL_ROLES),
        "metadata_columns": columns, "metadata_sha256": digest(metadata),
        "roles": {name: write_ids(name, ids) for name, ids in roles.items()},
        "role_intersections": "all empty", "official_test_access": "identity list only; no images, GT, or predictions",
    }
    (SPLITS / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
