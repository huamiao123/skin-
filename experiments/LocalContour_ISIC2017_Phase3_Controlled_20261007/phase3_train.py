"""Controlled Phase-3 fit/stop training; cal/dev/test are never Dataset members."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from local_model import StrongLocal, parameter_count
from phase3_access import log_access
from phase3_data import LocalInferenceCases, NodeInferenceCases
from refiner_model import NodeRefiner, PointwiseRefiner
from project_paths import PHASE1
from train_local import CachedCases, CACHE, ROOT, forward as local_forward, loss_and_accuracy, to_device

FAMILIES = ("S64", "S96", "N0", "T8", "T32", "TG")
SEEDS = (17, 23, 42)


class Phase3NodeCases(Dataset):
    def __init__(self, split: str, seed: int):
        done = json.loads((CACHE / "COMPLETE.json").read_text())
        self.records = [row for row in done["records"] if row["split"] == split]
        self.context = np.load(CACHE / "node_context.npy", mmap_mode="r")
        parent = ROOT / "models_phase3" / f"S64_seed{seed}"
        upstream = json.loads((parent / "DONE.json").read_text())
        if upstream["status"] != "PASS": raise RuntimeError("Phase3 S64 incomplete")
        self.local = np.load(parent / "candidate_scores.npy", mmap_mode="r")
        candidate = CACHE / "candidates_R32"
        self.labels = {key: np.load(candidate / f"{key}.npy", mmap_mode="r")
                       for key in ("valid", "gt_distance", "has_contour")}

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]; index = record["index"]
        return {"index": index, "image_id": record["image_id"],
                "context": torch.from_numpy(np.array(self.context[index], copy=True)),
                "local_scores": torch.from_numpy(np.array(self.local[index], copy=True)),
                **{key: torch.from_numpy(np.array(value[index], copy=True))
                   for key, value in self.labels.items()}}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def members(role: str) -> set[str]:
    assert role in ("fit", "stop")
    return set((ROOT / "splits" / f"{role}.txt").read_text().splitlines())


class RoleCases(Dataset):
    def __init__(self, role: str, family: str, seed: int):
        ids = members(role)
        if family.startswith("S"):
            self.source = CachedCases("train")
        else:
            self.source = Phase3NodeCases("train", seed)
        self.row_indices = [i for i, record in enumerate(self.source.records) if record["image_id"] in ids]
        if len(self.row_indices) != len(ids):
            raise RuntimeError(f"{role} membership mismatch: {len(self.row_indices)} vs {len(ids)}")
        log_access(f"training_dataset_{family}_seed{seed}", role, len(ids), True)

    def __len__(self): return len(self.row_indices)
    def __getitem__(self, i): return self.source[self.row_indices[i]]


def make_model(family: str) -> nn.Module:
    return (StrongLocal(width=64 if family == "S64" else 96, normalization="pixel_group")
            if family.startswith("S") else PointwiseRefiner() if family == "N0"
            else NodeRefiner({"T8": 8, "T32": 32, "TG": None}[family]))


def forward(model, batch, family):
    if family.startswith("S"):
        return local_forward(model, batch)
    # Explicitly pass inference fields only; GT and gt_distance are never model inputs.
    return model(batch["context"], batch["local_scores"], batch["valid"])


@torch.no_grad()
def evaluate(model, loader, device, family):
    model.eval(); total_loss = total_acc = total_weight = 0.0
    for raw in loader:
        batch = to_device(raw, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward(model, batch, family)
        loss, acc = loss_and_accuracy(logits, batch)
        weight = int(batch["has_contour"].sum().item())
        total_loss += float(loss) * weight; total_acc += float(acc) * weight; total_weight += weight
    return total_loss / max(1, total_weight), total_acc / max(1, total_weight)


@torch.no_grad()
def export_scores(model, family, seed, device, output):
    done = json.loads((CACHE / "COMPLETE.json").read_text())
    records = done["records"]
    score = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(records), 256, 65))
    model.eval()
    for split in ("train", "val"):
        source = LocalInferenceCases(split) if family.startswith("S") else NodeInferenceCases(split, seed)
        log_access(f"score_export_{family}_seed{seed}", split, len(source), False)
        loader = DataLoader(source, batch_size=4 if family.startswith("S") else 16,
                            shuffle=False, num_workers=0, pin_memory=True)
        for raw in loader:
            batch = to_device(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = forward(model, batch, family).float().cpu().numpy()
            score[batch["index"].cpu().numpy()] = result
    score.flush()


def train(family: str, seed: int, min_epochs=12, max_epochs=40, patience=8):
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit all Phase-3 source and protocol files before training")
    assert min_epochs >= 12 and max_epochs >= min_epochs and patience == 8
    manifest = json.loads((ROOT / "splits/phase3_manifest.json").read_text())
    assert manifest["status"] == "LOCKED"
    assert sha(ROOT / "splits/fit.txt") == manifest["roles"]["fit"]["sha256"]
    assert sha(ROOT / "splits/stop.txt") == manifest["roles"]["stop"]["sha256"]
    feature_meta = json.loads((CACHE / "COMPLETE.json").read_text())
    candidate_meta = json.loads((CACHE / "candidates_R32/COMPLETE.json").read_text())
    if feature_meta.get("gt_shape_verified_for_all") != len(feature_meta["records"]):
        raise RuntimeError("Feature/label cache audit failed")
    if candidate_meta.get("gt_distance_label_source") != "complete 2D official GT mask":
        raise RuntimeError("Candidate label provenance mismatch")
    cnn_weight = PHASE1 / "third_party/msgu_net/weights/best_model_isic2017.pth"
    boundary_weight = PHASE1 / "boundary_head/best.pth"
    if sha(cnn_weight) != feature_meta["cnn_sha256"]:
        raise RuntimeError("Frozen CNN weight differs from feature cache provenance")
    if not family.startswith("S"):
        parent_dir = ROOT / "models_phase3" / f"S64_seed{seed}"
        parent_meta = json.loads((parent_dir / "DONE.json").read_text())
        if sha(parent_dir / "candidate_scores.npy") != parent_meta["score_sha256"]:
            raise RuntimeError("Upstream S64 score hash mismatch")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(family).to(device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    # T8/T32/TG use identical seed initialization; only the attention mask differs.
    output = ROOT / "models_phase3" / f"{family}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    train_data = RoleCases("fit", family, seed)
    stop_data = RoleCases("stop", family, seed)
    batch_size = 4 if family.startswith("S") else 16
    training = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                          generator=torch.Generator().manual_seed(seed))
    stopping = DataLoader(stop_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if family.startswith("S") else 3e-4, weight_decay=1e-4)
    best = float("inf"); best_epoch = None; stale = 0; rows = []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train(); losses = []; accuracies = []; weights = []; gradients = []; seen = 0
        keep_targets = 0; labelled_nodes = 0; invalid_target_count = 0
        for step, raw in enumerate(training, 1):
            batch = to_device(raw, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward(model, batch, family)
                loss, acc = loss_and_accuracy(logits, batch)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach())); accuracies.append(float(acc.detach()))
            weights.append(int(batch["has_contour"].sum().item()))
            gradients.append(float(grad)); seen += len(batch["image_id"])
            eligible = batch["has_contour"].bool()
            if eligible.any():
                distance = batch["gt_distance"].float().masked_fill(~batch["valid"].bool(), 1e4)
                fixable = distance.min(dim=-1).values <= 4.0
                keep_targets += int((~fixable[eligible]).sum().item())
                labelled_nodes += int(fixable[eligible].numel())
            if step % 100 == 0: print(f"{family} seed={seed} epoch={epoch} step={step}/{len(training)}", flush=True)
        stop_loss, stop_acc = evaluate(model, stopping, device, family)
        row = dict(family=family, seed=seed, epoch=epoch,
                   fit_ce=float(np.average(losses, weights=weights)) if sum(weights) else 0.0,
                   fit_candidate_acc=float(np.average(accuracies, weights=weights)) if sum(weights) else 0.0,
                   stop_ce=stop_loss,
                   stop_candidate_acc=stop_acc, grad_norm_mean=float(np.mean(gradients)),
                   invalid_target_count=invalid_target_count,
                   keep_target_fraction=keep_targets/max(1,labelled_nodes),
                   updates=epoch * len(training), samples_seen=epoch * seen,
                   elapsed_seconds=round(time.monotonic()-started, 2),
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        rows.append(row); print(json.dumps(row), flush=True)
        if epoch == 8:
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, output / "epoch8.pth")
        if stop_loss < best - 1e-5:
            best = stop_loss; best_epoch = epoch; stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "stop_ce": best}, output / "best.pth")
        else:
            stale += 1
        with (output / "training_log.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        if epoch >= min_epochs and stale >= patience: break
    checkpoint = torch.load(output / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    export_scores(model, family, seed, device, output / "candidate_scores.npy")
    parent = None
    if not family.startswith("S"):
        parent = ROOT / "models_phase3" / f"S64_seed{seed}" / "best.pth"
    report = dict(status="PASS", family=family, seed=seed, source_git_commit=source_commit,
                  fit_ids_hash=manifest["roles"]["fit"]["sha256"], stop_ids_hash=manifest["roles"]["stop"]["sha256"],
                  upstream_checkpoint_sha256=sha(parent) if parent else None,
                  initial_parameter_sha256=initial_hash, parameters=parameter_count(model),
                  best_epoch=best_epoch, epochs_run=len(rows), stop_ce=best,
                  training_budget_edge=(len(rows) == max_epochs and best_epoch >= max_epochs-patience+1),
                  fit_images=len(train_data), stop_images=len(stop_data),
                  cal_or_dev_gradient_images=0, official_test_images_opened=0,
                  frozen_cnn_state_hash=feature_meta["cnn_state_hash"],
                  frozen_cnn_checkpoint_sha256=sha(cnn_weight),
                  frozen_boundary_head_checkpoint_sha256=sha(boundary_weight),
                  candidate_cache_layout=(candidate_meta["nodes"], candidate_meta["candidate_count"]),
                  score_dtype="float32", score_sha256=sha(output / "candidate_scores.npy"),
                  checkpoint_sha256=sha(output / "best.pth"), seconds=round(time.monotonic()-started, 2))
    (output / "DONE.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "model_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE " + str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    train(args.family, args.seed)
