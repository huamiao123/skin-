from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    with protocol_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_protocol(config, protocol_path.parent)
    return config


def validate_protocol(config: dict[str, Any], base_dir: str | Path = ".") -> None:
    if not isinstance(config, dict) or not isinstance(config.get("protocol"), dict):
        raise ValueError("protocol section is required")
    data = config.get("data", {})
    if not data.get("manifest"):
        raise ValueError("data.manifest must be set")
    manifest = Path(base_dir, data["manifest"])
    # The repository config stores a project-relative path; tolerate callers
    # passing the project root as base_dir as well.
    if not manifest.exists():
        manifest = Path(base_dir).parents[1] / data["manifest"]
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {data['manifest']}")
    expected_hash = data.get("split_hash")
    if not expected_hash or expected_hash == "pending_generation":
        raise ValueError("data.split_hash must be generated and frozen")
    actual_hash = sha256_file(manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"split_hash mismatch: expected {expected_hash}, got {actual_hash}")
    if config.get("model", {}).get("output_contract") != "probability":
        raise ValueError("research_v1 requires probability output contract")
    router = config.get("router", {})
    if router.get("enabled") and not router.get("frozen_segmentation", False):
        raise ValueError("router training requires frozen segmentation")
    if not 0 <= float(config.get("eval", {}).get("threshold", 0.5)) <= 1:
        raise ValueError("eval.threshold must be in [0,1]")
