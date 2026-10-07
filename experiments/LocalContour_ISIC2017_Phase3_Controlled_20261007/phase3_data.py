"""GT-free score-export datasets, distinct from fit/stop supervision views."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from train_local import CACHE, ROOT


class LocalInferenceCases(Dataset):
    def __init__(self, split):
        if split not in ("train", "val"): raise ValueError(split)
        done = json.loads((CACHE / "COMPLETE.json").read_text())
        self.records = [row for row in done["records"] if row["split"] == split]
        self.maps = {key: np.load(CACHE / f"{key}.npy", mmap_mode="r")
                     for key in ("features", "rgb", "probability", "boundary")}
        candidate = CACHE / "candidates_R32"
        self.geometry = {key: np.load(candidate / f"{key}.npy", mmap_mode="r")
                         for key in ("points", "source_points", "normals", "valid", "has_contour")}

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]; index = record["index"]
        return {"index": index, "image_id": record["image_id"],
                **{key: torch.from_numpy(np.array(value[index], copy=True))
                   for key, value in {**self.maps, **self.geometry}.items()}}


class NodeInferenceCases(Dataset):
    def __init__(self, split, seed):
        if split not in ("train", "val"): raise ValueError(split)
        done = json.loads((CACHE / "COMPLETE.json").read_text())
        self.records = [row for row in done["records"] if row["split"] == split]
        self.context = np.load(CACHE / "node_context.npy", mmap_mode="r")
        parent = ROOT / "models_phase3" / f"S64_seed{seed}"
        self.local = np.load(parent / "candidate_scores.npy", mmap_mode="r")
        self.valid = np.load(CACHE / "candidates_R32/valid.npy", mmap_mode="r")

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]; index = record["index"]
        return {"index": index, "image_id": record["image_id"],
                "context": torch.from_numpy(np.array(self.context[index], copy=True)),
                "local_scores": torch.from_numpy(np.array(self.local[index], copy=True)),
                "valid": torch.from_numpy(np.array(self.valid[index], copy=True))}
