from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = [
    "image_id", "image_path", "mask_path", "source_dataset", "source_original_split",
    "patient_id", "lesion_id", "group_id", "image_sha256", "mask_sha256",
    "original_h", "original_w", "parent_image_id", "assigned_split", "legacy_split",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_ids(mask_dir: Path) -> list[str]:
    return sorted(path.name.split("_")[1] for path in mask_dir.glob("ISIC_*_segmentation.png"))


def deterministic_roles(image_ids: list[str], seed: int) -> dict[str, str]:
    ordered = np.asarray(sorted(image_ids), dtype=object)
    np.random.default_rng(seed).shuffle(ordered)
    n = len(ordered)
    n_seg = round(n * 0.70)
    n_fit = round(n * 0.20)
    roles = {}
    for image_id in ordered[:n_seg]:
        roles[str(image_id)] = "S_seg"
    for image_id in ordered[n_seg:n_seg + n_fit]:
        roles[str(image_id)] = "R_fit"
    for image_id in ordered[n_seg + n_fit:]:
        roles[str(image_id)] = "R_val"
    return roles


def build_manifest(data_root: Path, official_mask_dir: Path, seed: int = 20260912) -> list[dict[str, str | int]]:
    ids = official_ids(official_mask_dir)
    train_images = sorted((data_root / "train/images").glob("*.png"), key=lambda p: int(p.stem))
    val_images = sorted((data_root / "val/images").glob("*.png"), key=lambda p: int(p.stem))
    if len(train_images) > len(ids):
        raise ValueError("legacy train split is larger than the available official ID list")

    train_ids = [f"ISIC_{ids[index]}" for index in range(len(train_images))]
    roles = deterministic_roles(train_ids, seed)
    rows = []
    seen_ids = set()
    for legacy_split, images in (("train", train_images), ("val", val_images)):
        for image_path in images:
            index = int(image_path.stem)
            mask_path = data_root / legacy_split / "masks" / image_path.name
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask for {image_path}")
            if legacy_split == "train":
                image_id = f"ISIC_{ids[index]}"
                assigned = roles[image_id]
                source_dataset = "ISIC2018_Task1"
                source_original_split = "challenge_training"
            else:
                official_index = len(train_images) + index
                if official_index < len(ids):
                    image_id = f"ISIC_{ids[official_index]}"
                    source_dataset = "ISIC2018_Task1"
                    source_original_split = "challenge_training"
                else:
                    image_id = f"legacy_extra_val_{index:04d}"
                    source_dataset = "legacy_ege_unknown"
                    source_original_split = "unknown"
                assigned = "V_dev"
            if image_id in seen_ids:
                raise ValueError(f"duplicate manifest image_id: {image_id}")
            seen_ids.add(image_id)
            with Image.open(image_path) as image:
                width, height = image.size
            with Image.open(mask_path) as mask:
                if mask.size != (width, height):
                    raise ValueError(f"image/mask size mismatch: {image_path} and {mask_path}")
            rows.append({
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "source_dataset": source_dataset,
                "source_original_split": source_original_split,
                "patient_id": "unknown",
                "lesion_id": "unknown",
                "group_id": image_id,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "original_h": height,
                "original_w": width,
                "parent_image_id": "",
                "assigned_split": assigned,
                "legacy_split": legacy_split,
            })
    return rows


def validate_manifest(rows: list[dict[str, str | int]]) -> dict[str, object]:
    if not rows:
        raise ValueError("manifest is empty")
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest image IDs are not unique")
    split_sets = defaultdict(set)
    for row in rows:
        split_sets[str(row["assigned_split"])].add(str(row["group_id"]))
    names = sorted(split_sets)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                overlaps[f"{left}:{right}"] = sorted(overlap)
    if overlaps:
        raise ValueError(f"group leakage across splits: {overlaps}")

    duplicate_images = [value for value, count in Counter(str(r["image_sha256"]) for r in rows).items() if count > 1]
    duplicate_masks = [value for value, count in Counter(str(r["mask_sha256"]) for r in rows).items() if count > 1]
    return {
        "n_rows": len(rows),
        "split_counts": dict(sorted(Counter(str(r["assigned_split"]) for r in rows).items())),
        "source_counts": dict(sorted(Counter(str(r["source_dataset"]) for r in rows).items())),
        "patient_id_status": "unavailable; group_id=image_id",
        "lesion_id_status": "unavailable",
        "exact_duplicate_image_hashes": len(duplicate_images),
        "exact_duplicate_mask_hashes": len(duplicate_masks),
        "near_duplicate_status": "pending manual review; not inferred from exact hashes",
        "mapping_note": (
            "Legacy numeric ordering mapped to sorted official Task1 IDs: train starts at 0; "
            "val continues after train. Remaining legacy validation rows are explicitly unknown."
        ),
    }


def write_manifest(rows: list[dict[str, str | int]], output: Path, audit_output: Path) -> tuple[str, dict[str, object]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256_file(output)
    audit = validate_manifest(rows)
    audit["manifest_sha256"] = digest
    audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return digest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    rows = build_manifest(args.data_root, args.official_mask_dir, args.seed)
    digest, audit = write_manifest(rows, args.output, args.audit_output)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"split_hash={digest}")


if __name__ == "__main__":
    main()

