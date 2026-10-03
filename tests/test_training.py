"""CPU-only checks of training control flow, not the GPU segmentation model."""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from rsi import train as training
from rsi.runtime import restore_rng, save_checkpoint


class ToyDataset(Dataset):
    def __init__(self, split="train", count=5):
        self.split, self.count, self.epoch = split, count, 0

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        r = 2 + index % 2
        image = torch.full((3, 1, 1), (index + 1) / 5)
        masks = torch.tensor([[(index + j) % 2] for j in range(r)], dtype=torch.float32).reshape(r, 1, 1, 1)
        return dict(image=image, masks=masks, rater_present=torch.ones(r, dtype=torch.bool),
                    pixel_valid=torch.ones(1, 1, 1), image_id=str(index), group_id=f"g{index}",
                    references=[{"reference_id": f"r{index}-{j}"} for j in range(r)],
                    geometry=dict(original_h=1, original_w=1, resized_h=1, resized_w=1, top=0, left=0, size=1),
                    subset="M")


class ToyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cnn = nn.Module()
        self.cnn.encoder = nn.Linear(1, 1, bias=False)
        self.cnn.decoder = nn.Linear(1, 1, bias=False)
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Linear(1, 1, bias=False)
        self.message = nn.Linear(1, 1, bias=False)
        self.set_stage("A_CNN")

    def cuda(self, *args, **kwargs):
        return self  # Deliberately never touches CUDA.

    def set_stage(self, stage):
        self.stage = stage
        for name, module in self.named_children():
            module.requires_grad_(name == {"A_CNN": "cnn", "A_T": "transformer", "B": "message", "D": "message"}[stage])
        return self

    def forward_cnn(self, images):
        x = images[:, :1, 0, 0]
        return self.cnn.decoder(self.cnn.encoder(x))[:, :, None, None]

    def forward_transformer(self, images):
        return self.transformer.encoder(images[:, :1, 0, 0])[:, :, None, None]

    def forward_pair(self, images, valid=None, alpha=1):
        z0 = self.forward_cnn(images).detach()
        message = self.message(images[:, :1, 0, 0])[:, :, None, None]
        return dict(z0=z0, z1=z0 + alpha * message, message=message)

    def frozen_state_hash(self):
        chunks = [p.detach().cpu().numpy().tobytes() for p in self.parameters() if not p.requires_grad]
        return hashlib.sha256(b"".join(chunks)).hexdigest()


def cpu_seed(seed, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=None)


def config(tmp_path, *, microbatch=2, epochs=1):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic CPU control-flow fixture\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(file_audit_complete=True, scope="full", final_references_sha256=training.sha256_file(manifest))))
    return dict(audit=str(audit), manifest=str(manifest), run_root=str(tmp_path / "runs"),
                input_size=[1, 1], epochs={"A_CNN": epochs, "A_T": epochs, "B": epochs, "D": epochs},
                scheduler={"A_warmup_epochs": 2, "BD_warmup_epochs": 2},
                lr={"encoder": 1e-3, "decoder": 3e-3, "B": 3e-4, "D": 1e-4},
                weight_decay=1e-4, amp=False, microbatch_start=microbatch, effective_batch=4,
                workers=0, cache_dir=str(tmp_path / "cache"), gradient_clip=1.0,
                gradient_diagnostic_every_epochs=5)


def install_cpu_training_fakes(monkeypatch, tmp_path, *, val_dice=0.5):
    monkeypatch.setattr(training, "ControlledModel", ToyModel)
    monkeypatch.setattr(training, "set_seed", cpu_seed)
    monkeypatch.setattr(training, "rng_state", cpu_rng)
    monkeypatch.setattr(training, "environment", lambda: {"device": "CPU toy"})
    monkeypatch.setattr(training, "code_provenance", lambda: {"source_bundle_hash": "cpu-control-fixture"})
    monkeypatch.setattr(training, "PROJECT", tmp_path / "mirror")
    monkeypatch.setattr(training, "loaders", lambda *args, **kwargs: (ToyDataset(), ToyDataset(), ToyDataset("val", 2)))
    original_device_batch = training.device_batch
    monkeypatch.setattr(training, "device_batch", lambda batch, device: original_device_batch(batch, torch.device("cpu")))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    def evaluate(*args, **kwargs):
        fields = ("loss", "seg", "risk", "weighted_risk", "rho", "J", "kappa_pm", "activation_rate", "mean_weight")
        return {**{k: 0.0 for k in fields}, "dice": val_dice, "iou": 0.3}, []
    monkeypatch.setattr(training, "evaluate", evaluate)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_training_actual_sample_accumulation_matches_larger_microbatch(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    first_cfg = config(tmp_path, microbatch=1)
    first = training.train_stage(first_cfg, 17, "A_CNN", run_name="micro1")
    first_last = read_checkpoint(first.parent / "latest.pth")
    second_cfg = dict(first_cfg, microbatch_start=2)
    second = training.train_stage(second_cfg, 17, "A_CNN", run_name="micro2")
    second_last = read_checkpoint(second.parent / "latest.pth")
    # Five images: full effective window four plus a flushed one-image tail.
    assert first_last["optimizer_steps"] == second_last["optimizer_steps"] == 2
    for key, value in first_last["model"].items():
        torch.testing.assert_close(value, second_last["model"][key], atol=2e-7, rtol=1e-6)
    assert first_last["selection_rule"] == "max_val_macro_mean_rater_hard_dice_earlier_tie"


def test_fixed_budget_warmup_cosine_and_earlier_epoch_tie(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path, epochs=4)
    best = training.train_stage(cfg, 17, "A_CNN", run_name="four_epochs")
    selected, latest = read_checkpoint(best), read_checkpoint(best.parent / "latest.pth")
    assert selected["epoch"] == selected["best_epoch"] == 1
    assert latest["epoch"] == 4
    assert latest["best_epoch"] == 1
    assert latest["optimizer_steps"] == 8
    with (best.parent / "epoch_diagnostics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3, 4]
    assert [float(row["lr_group0"]) for row in rows] == pytest.approx([0.0005, 0.001, 0.001, 0.0005])
    done = json.loads((best.parent / "DONE.json").read_text())
    assert done["test_scoring_locked"]
    assert done["best_sha256"] == training.sha256_file(best)


def test_epoch_checkpoint_resume_restores_optimizer_rng_and_selection(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path, epochs=4)
    original_save = training.save_checkpoint
    interrupted = {"active": True}
    def stop_after_second_epoch(path, payload):
        original_save(path, payload)
        if Path(path).name == "latest.pth" and payload["epoch"] == 2 and interrupted["active"]:
            raise RuntimeError("synthetic interrupt after durable epoch boundary")
    monkeypatch.setattr(training, "save_checkpoint", stop_after_second_epoch)
    with pytest.raises(RuntimeError, match="synthetic interrupt"):
        training.train_stage(cfg, 17, "A_CNN", run_name="resumed")
    interrupted["active"] = False
    resumed = training.train_stage(cfg, 17, "A_CNN", run_name="resumed")
    uninterrupted = training.train_stage(cfg, 17, "A_CNN", run_name="uninterrupted")
    resume_last, continuous_last = read_checkpoint(resumed.parent / "latest.pth"), read_checkpoint(uninterrupted.parent / "latest.pth")
    assert resume_last["epoch"] == continuous_last["epoch"] == 4
    assert resume_last["optimizer_steps"] == continuous_last["optimizer_steps"] == 8
    for key, value in continuous_last["model"].items():
        assert torch.equal(value, resume_last["model"][key])
    assert resume_last["scheduler"] == continuous_last["scheduler"]
    for index, states in continuous_last["optimizer"]["state"].items():
        for key, value in states.items():
            other = resume_last["optimizer"]["state"][index][key]
            if torch.is_tensor(value):
                assert torch.equal(value, other)
            else:
                assert value == other


def test_checkpoint_atomic_write_and_cpu_rng_round_trip(tmp_path):
    cpu_seed(31)
    state = cpu_rng()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, {"rng": state, "stage": "CPU toy"})
    restore_rng(read_checkpoint(path)["rng"])
    got = (random.random(), np.random.rand(), torch.rand(3))
    assert got[0] == expected[0]
    assert got[1] == expected[1]
    assert torch.equal(got[2], expected[2])
    assert not path.with_suffix(".tmp.pth").exists()


def test_evaluation_refuses_test_before_any_model_or_device_call():
    dataset = ToyDataset("test")
    with pytest.raises(PermissionError, match="test is locked"):
        training.evaluate(None, dataset, {}, "B")


def test_calculate_uses_reference_equal_within_image_and_anchor_no_gradient():
    model = ToyModel().set_stage("D")
    batch = training.collate_multi_reference([ToyDataset()[0], ToyDataset()[1]])
    out, objective, ell, ell0 = training.calculate(model, batch, "D", "rsi", 1)
    present = batch["rater_present"]
    expected = sum(ell[i, present[i]].mean() for i in range(2)) / 2
    torch.testing.assert_close(expected, objective["seg"])
    objective["loss"].backward()
    assert not out["z0"].requires_grad
    assert not ell0.requires_grad
    assert all(p.grad is None for p in model.cnn.parameters())
    assert model.message.weight.grad is not None


def test_completed_run_signature_and_manifest_binding_reject_reuse(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path)
    selected = training.train_stage(cfg, 17, "A_CNN", run_name="completed")
    assert training.train_stage(cfg, 17, "A_CNN", run_name="completed") == selected
    with pytest.raises(ValueError):
        training.train_stage(cfg, 17, "A_CNN", run_name="completed", method="rsi", weight=1)
    Path(cfg["manifest"]).write_text("changed after file audit")
    with pytest.raises(ValueError, match="manifest"):
        training.train_stage(cfg, 17, "A_CNN", run_name="another")


def test_resume_rejects_changed_objective_and_shared_init(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path, epochs=4)
    original_save = training.save_checkpoint
    def stop(path, payload):
        original_save(path, payload)
        if Path(path).name == "latest.pth":
            raise RuntimeError("checkpoint saved")
    monkeypatch.setattr(training, "save_checkpoint", stop)
    with pytest.raises(RuntimeError):
        training.train_stage(cfg, 17, "A_CNN", run_name="resume_guard")
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    with pytest.raises(ValueError):
        training.train_stage(cfg, 17, "A_CNN", run_name="resume_guard", method="rsi", weight=1)
    init = tmp_path / "different_init.pth"
    save_checkpoint(init, read_checkpoint(Path(cfg["run_root"]) / "seed17/resume_guard/latest.pth"))
    with pytest.raises(ValueError):
        training.train_stage(cfg, 17, "A_CNN", run_name="resume_guard", init=init)


def test_zero_risk_gradient_probe_has_na_cosine_and_no_parameter_update():
    model = ToyModel().set_stage("D")
    batch = training.collate_multi_reference([ToyDataset()[0], ToyDataset()[1]])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    diagnostics = training.gradient_diagnostic(model, batch, "D", "d0", 0, None)
    assert diagnostics["weighted_risk_gradient_norm"] == 0
    assert diagnostics["risk_seg_gradient_cosine"] is None
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_gradient_probe_restores_callers_eval_mode():
    model = ToyModel().set_stage("D").eval()
    batch = training.collate_multi_reference([ToyDataset()[0], ToyDataset()[1]])
    assert not model.training
    training.gradient_diagnostic(model, batch, "D", "rsi", 1, None)
    assert not model.training
    assert all(not module.training for module in model.modules())


def test_resume_recovers_selected_checkpoint_when_latest_written_before_best(monkeypatch, tmp_path):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path, epochs=2)
    original_save = training.save_checkpoint
    def stop_before_best(path, payload):
        original_save(path, payload)
        if Path(path).name == "latest.pth" and payload["epoch"] == 1:
            raise RuntimeError("latest durable, selected checkpoint not yet written")
    monkeypatch.setattr(training, "save_checkpoint", stop_before_best)
    with pytest.raises(RuntimeError):
        training.train_stage(cfg, 17, "A_CNN", run_name="durable_latest")
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    selected = training.train_stage(cfg, 17, "A_CNN", run_name="durable_latest")
    assert read_checkpoint(selected)["epoch"] == 1
    assert read_checkpoint(selected.parent / "latest.pth")["epoch"] == 2


@pytest.mark.parametrize("interrupted_epoch", [1, 2])
def test_resume_replays_uncommitted_epoch_without_duplicate_epoch_or_image_logs(
    monkeypatch, tmp_path, interrupted_epoch,
):
    install_cpu_training_fakes(monkeypatch, tmp_path)
    cfg = config(tmp_path, epochs=3)
    evaluate_without_image_logs = training.evaluate
    def evaluate_with_real_image_log(model, dataset, cfg, stage, **kwargs):
        result = evaluate_without_image_logs(model, dataset, cfg, stage, **kwargs)
        path = kwargs.get("per_image_path")
        if path is not None:
            for i in range(len(dataset)):
                sample = dataset[i]
                training.append_csv(path, dict(
                    epoch=kwargs["epoch"], split=kwargs.get("split_label", dataset.split),
                    image_id=sample["image_id"], group_id=sample["group_id"],
                    reference_count=len(sample["references"]), mean_loss=0.3,
                ))
        return result
    monkeypatch.setattr(training, "evaluate", evaluate_with_real_image_log)
    original_save = training.save_checkpoint
    def stop_before_latest_write(path, payload):
        if Path(path).name == "latest.pth" and payload["epoch"] == interrupted_epoch:
            raise RuntimeError("epoch logs appended but latest not written")
        original_save(path, payload)
    monkeypatch.setattr(training, "save_checkpoint", stop_before_latest_write)
    run_name = f"replay_epoch{interrupted_epoch}"
    with pytest.raises(RuntimeError, match="latest not written"):
        training.train_stage(cfg, 17, "A_CNN", run_name=run_name)
    run_dir = Path(cfg["run_root"]) / "seed17" / run_name
    with (run_dir / "epoch_diagnostics.csv").open() as f:
        pending = list(csv.DictReader(f))
    assert int(pending[-1]["epoch"]) == interrupted_epoch
    assert (run_dir / "latest.pth").exists() == (interrupted_epoch > 1)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    selected = training.train_stage(cfg, 17, "A_CNN", run_name=run_name)
    assert read_checkpoint(selected.parent / "latest.pth")["epoch"] == 3
    with (run_dir / "epoch_diagnostics.csv").open() as f:
        epoch_rows = list(csv.DictReader(f))
    assert [int(row["epoch"]) for row in epoch_rows] == [1, 2, 3]
    with (run_dir / "per_image_canonical_diagnostics.csv").open() as f:
        image_rows = list(csv.DictReader(f))
    image_keys = [(int(row["epoch"]), row["split"], row["image_id"]) for row in image_rows]
    assert len(image_keys) == len(set(image_keys)) == 3 * len(ToyDataset("val", 2))
    assert set(image_keys) == {(epoch, "val", str(i)) for epoch in (1, 2, 3) for i in range(2)}
