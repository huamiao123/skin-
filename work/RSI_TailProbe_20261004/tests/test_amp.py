"""Real CUDA AMP overflow regressions using a one-weight toy model."""
import csv
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rsi import train as training
from rsi.datasets import collate_multi_reference


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="Real GradScaler overflow requires CUDA")


@CUDA_REQUIRED
def test_real_scaler_skips_overflow_then_recovers_without_changing_skip_parameters():
    model = nn.Linear(1, 1, bias=False).cuda()
    model.weight.data.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.cuda.amp.GradScaler()
    actual_calls = []
    observer = optimizer.register_step_post_hook(lambda *_: actual_calls.append(True))
    reports, scales = [], [scaler.get_scale()]
    try:
        for value in [100., 1.]:
            optimizer.zero_grad(set_to_none=True)
            before = model.weight.detach().clone()
            with torch.autocast("cuda"):
                logits = model(torch.tensor([[value]], device="cuda"))
                assert logits.dtype == torch.float16
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.ones_like(logits).float())
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            info = training.finish_optimizer_step(optimizer, scaler, list(model.parameters()), 1.)
            reports.append(info)
            scales.append(scaler.get_scale())
            if info["skipped"]:
                assert torch.equal(before, model.weight)
                assert not optimizer.state  # Adam moments and weight decay did not run.
            else:
                assert not torch.equal(before, model.weight)
                assert torch.isfinite(model.weight).all()
            assert len(optimizer._optimizer_step_post_hooks) == 1  # Only our observer remains.
    finally:
        observer.remove()
    assert math.isinf(reports[0]["norm"])
    assert reports[0]["skipped"] is True
    assert reports[1]["skipped"] is False
    assert math.isfinite(reports[1]["norm"])
    assert scales == [65536., 32768., 32768.]
    assert len(actual_calls) == 1
    summary = training.gradient_norm_summary([report["norm"] for report in reports])
    assert summary["gradient_norm_nonfinite_count"] == 1
    assert summary["gradient_norm_finite_count"] == 1
    assert summary["gradient_norm_before_clip"] == reports[1]["norm"]


class OverflowDataset(Dataset):
    def __init__(self, *, all_overflow=False, split="train"):
        self.all_overflow, self.split = all_overflow, split

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        value = 100. if index == 0 or self.all_overflow else 1.
        return dict(image=torch.full((3, 1, 1), value), masks=torch.ones(2, 1, 1, 1),
                    rater_present=torch.ones(2, dtype=torch.bool), pixel_valid=torch.ones(1, 1, 1),
                    image_id=str(index), group_id=f"group-{index}", subset="M",
                    references=[dict(reference_id=f"{index}-r{r}") for r in range(2)],
                    geometry=dict(original_h=1, original_w=1, resized_h=1, resized_w=1, top=0, left=0, size=1))


class OverflowModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cnn = nn.Module()
        self.cnn.encoder = nn.Linear(1, 1, bias=False)
        self.cnn.decoder = nn.Identity()
        self.cnn.encoder.weight.data.zero_()

    def set_stage(self, stage):
        assert stage == "A_CNN"

    def forward_cnn(self, images):
        return self.cnn.encoder(images[:, :1, 0, 0])[:, :, None, None]


def configure_overflow_training(monkeypatch, tmp_path, *, all_overflow=False, epochs=1):
    monkeypatch.setattr(training, "ControlledModel", OverflowModel)
    monkeypatch.setattr(training, "PROJECT", tmp_path / "mirror")
    monkeypatch.setattr(training, "code_provenance", lambda: {"source_bundle_hash": "real-CUDA-overflow-toy"})
    monkeypatch.setattr(training, "environment", lambda: {"device": "real CUDA AMP toy"})
    monkeypatch.setattr(training, "loaders", lambda *args, **kwargs:
                        (OverflowDataset(all_overflow=all_overflow), OverflowDataset(all_overflow=all_overflow),
                         OverflowDataset(split="val")))
    # Keep the deliberately overflowing sample before the finite sample; this
    # changes only toy sample ordering, not GradScaler/optimizer/loss behavior.
    monkeypatch.setattr(training, "loader", lambda data, batch_size, workers=0, **kwargs:
                        DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_multi_reference))

    def evaluate(*args, **kwargs):
        fields = ("loss", "seg", "risk", "weighted_risk", "rho", "J", "kappa_pm", "activation_rate", "mean_weight")
        return {**{name: 0. for name in fields}, "dice": .5, "iou": .3}, []

    monkeypatch.setattr(training, "evaluate", evaluate)
    actual_calls = []
    make_optimizer = training.make_optimizer

    def observed_optimizer(*args, **kwargs):
        optimizer, rates = make_optimizer(*args, **kwargs)
        optimizer.register_step_post_hook(lambda *_: actual_calls.append(True))
        return optimizer, rates

    monkeypatch.setattr(training, "make_optimizer", observed_optimizer)
    manifest, audit = tmp_path / "manifest.csv", tmp_path / "audit.json"
    manifest.write_text("real CUDA AMP overflow toy fixture\n")
    audit.write_text(json.dumps(dict(file_audit_complete=True, scope="full",
                                    final_references_sha256=training.sha256_file(manifest))))
    cfg = dict(audit=str(audit), manifest=str(manifest), run_root=str(tmp_path / "runs"), input_size=[1, 1],
               epochs={"A_CNN": epochs, "A_T": epochs, "B": epochs, "D": epochs},
               scheduler={"A_warmup_epochs": 2, "BD_warmup_epochs": 2},
               lr={"encoder": .001, "decoder": .003, "B": .0003, "D": .0001},
               weight_decay=.0001, amp=True, microbatch_start=1, effective_batch=1, workers=0,
               cache_dir=str(tmp_path / "cache"), gradient_clip=1., gradient_diagnostic_every_epochs=5)
    return cfg, actual_calls


@CUDA_REQUIRED
@pytest.mark.parametrize("all_overflow", [False, True], ids=["overflow-then-finite", "all-overflow"])
def test_train_epoch_logs_overflow_and_counts_actual_updates(monkeypatch, tmp_path, all_overflow):
    cfg, actual_calls = configure_overflow_training(monkeypatch, tmp_path, all_overflow=all_overflow)
    selected = training.train_stage(cfg, 17, "A_CNN", run_name="AMP-toy")
    checkpoint = torch.load(selected.parent / "latest.pth", map_location="cpu", weights_only=False)
    with (selected.parent / "epoch_diagnostics.csv").open(newline="") as handle:
        row, = csv.DictReader(handle)
    expected_actual, expected_skipped = (0, 2) if all_overflow else (1, 1)
    assert len(actual_calls) == expected_actual
    assert checkpoint["optimizer_steps"] == int(row["optimizer_steps"]) == expected_actual
    assert checkpoint["optimizer_step_attempts"] == int(row["optimizer_step_attempts"]) == 2
    assert int(row["amp_skipped_steps"]) == expected_skipped
    assert float(row["amp_skipped_step_fraction"]) == expected_skipped / 2
    assert int(row["gradient_norm_nonfinite_count"]) == expected_skipped
    assert int(row["gradient_norm_finite_count"]) == expected_actual
    assert math.isfinite(float(row["augmented_train_loss"]))
    assert checkpoint["scaler"]["scale"] == (16384. if all_overflow else 32768.)
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())
    if all_overflow:
        assert row["gradient_norm_before_clip"] == ""
        assert checkpoint["optimizer"]["state"] == {}
        assert checkpoint["model"]["cnn.encoder.weight"].item() == 0.
    else:
        assert math.isfinite(float(row["gradient_norm_before_clip"]))
        assert float(row["gradient_norm_before_clip"]) > 0.
        assert checkpoint["model"]["cnn.encoder.weight"].item() > 0.


@CUDA_REQUIRED
def test_amp_resume_restores_attempt_and_actual_counts(monkeypatch, tmp_path):
    cfg, actual_calls = configure_overflow_training(monkeypatch, tmp_path, epochs=2)
    save_checkpoint = training.save_checkpoint
    interruption = {"active": True}

    def stop_after_first_latest(path, state):
        save_checkpoint(path, state)
        if Path(path).name == "latest.pth" and state["epoch"] == 1 and interruption["active"]:
            raise RuntimeError("synthetic durable AMP epoch interruption")

    monkeypatch.setattr(training, "save_checkpoint", stop_after_first_latest)
    with pytest.raises(RuntimeError, match="durable AMP epoch interruption"):
        training.train_stage(cfg, 17, "A_CNN", run_name="AMP-resumed")
    interruption["active"] = False
    selected = training.train_stage(cfg, 17, "A_CNN", run_name="AMP-resumed")
    state = torch.load(selected.parent / "latest.pth", map_location="cpu", weights_only=False)
    with (selected.parent / "epoch_diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert [int(row["optimizer_step_attempts"]) for row in rows] == [2, 4]
    assert [int(row["optimizer_steps"]) for row in rows] == [1, 2]
    assert [int(row["amp_skipped_steps"]) for row in rows] == [1, 1]
    assert state["optimizer_step_attempts"] == 4
    assert state["optimizer_steps"] == len(actual_calls) == 2
    assert state["scaler"]["scale"] == 16384.
