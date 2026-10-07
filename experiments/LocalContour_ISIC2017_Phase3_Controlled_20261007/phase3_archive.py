"""Regenerate val150 logits from hash-verified Fixed checkpoints, without retraining."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from local_model import StrongLocal
from phase3_data import LocalInferenceCases
from refiner_model import NodeRefiner
from train_local import CACHE, ROOT, forward as local_forward, to_device

FIXED = ROOT.parent / "LocalContour_Phase2_Fixed_20261006"
FAMILIES = {"S64": ("local", 64), "S96": ("local_large", 96),
            "T8": ("local8", 8), "T32": ("medium32", 32), "TG": ("global", None)}
SEEDS = (17, 23, 42)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = list(csv.DictReader((FIXED / "results/training_records/training_manifest.csv").open()))
    lookup = {(row["family"], int(row["seed"])): row for row in rows}
    val = LocalInferenceCases("val")
    context = np.load(CACHE / "node_context.npy", mmap_mode="r")
    valid = np.load(CACHE / "candidates_R32/valid.npy", mmap_mode="r")
    output = ROOT / "archive_scores"; output.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = []
    for seed in SEEDS:
        for family, (old_name, parameter) in FAMILIES.items():
            weight = FIXED / "models" / f"{old_name}_seed{seed}" / "best.pth"
            old = lookup[(old_name, seed)]
            if digest(weight) != old["checkpoint_sha256_local_only"]:
                raise RuntimeError(f"Archive weight hash mismatch: {weight}")
            model = (StrongLocal(width=parameter, normalization="pixel_group") if family.startswith("S")
                     else NodeRefiner(parameter)).to(device)
            model.load_state_dict(torch.load(weight, map_location=device, weights_only=True)["state_dict"], strict=True)
            model.eval()
            path = output / f"{family}_seed{seed}.npy"
            scores = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(val), 256, 65))
            with torch.no_grad():
                if family.startswith("S"):
                    for start, raw in enumerate(DataLoader(val, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)):
                        batch = to_device(raw, device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            value = local_forward(model, batch)
                        scores[start*4:start*4+len(value)] = value.float().cpu().numpy().astype(np.float16)
                else:
                    upstream = np.load(output / f"S64_seed{seed}.npy", mmap_mode="r")
                    for start in range(0, len(val), 16):
                        rows_chunk = val.records[start:start+16]
                        indices = [row["index"] for row in rows_chunk]
                        c = torch.from_numpy(np.array(context[indices], copy=True)).to(device)
                        s = torch.from_numpy(np.array(upstream[start:start+16], copy=True)).to(device)
                        v = torch.from_numpy(np.array(valid[indices], copy=True)).to(device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            value = model(c, s, v)
                        scores[start:start+len(value)] = value.float().cpu().numpy().astype(np.float16)
            scores.flush()
            entry = dict(family=family, seed=seed, archive_checkpoint_sha256=digest(weight),
                         score_sha256=digest(path), score_dtype="float16", images=len(val),
                         checkpoint_selection="historical val150; exposed to current cal50/dev100")
            manifest.append(entry); print(json.dumps(entry), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
