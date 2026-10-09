"""CPU export checks with a deliberately batch-dependent segmentation model."""
import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from rsi import export as exporting
from rsi.datasets import collate_multi_reference
from rsi.objectives import per_reference_loss


class SourceDataset(Dataset):
    """H/T1 contain M images with different batches and reference selections."""

    def __init__(self, manifest, *, split, subset, cache_dir, **kwargs):
        assert split == "val"
        self.subset = subset
        self.ids = {"M": ["a", "b", "c"], "H": ["b", "c"], "T1": ["a", "c"]}[subset]
        self.reference_indices = {"M": [0, 1, 2], "H": [1, 2], "T1": [2, 0]}[subset]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        number = ord(image_id) - ord("a") + 1
        width = 1 if image_id == "c" else 2
        valid = torch.zeros(1, 2, 2)
        valid[:, :, :width] = 1
        diagonal = torch.tensor([[[1., 0.], [0., 1.]]]) * valid
        masks = [valid.clone(), diagonal, torch.zeros_like(valid)]
        refs = [dict(reference_id=f"{image_id}-r{r}", annotator_id=f"annotator-{r}",
                     seg_filename=f"{image_id}-r{r}.png", tool="synthetic", skill_level="synthetic",
                     mask_path=f"{image_id}-r{r}") for r in self.reference_indices]
        return dict(image=torch.full((3, 2, 2), float(number)),
                    masks=torch.stack([masks[r] for r in self.reference_indices]),
                    rater_present=torch.ones(len(refs), dtype=torch.bool), pixel_valid=valid,
                    image_id=image_id, group_id=f"group-{image_id}", references=refs,
                    geometry=dict(original_h=2, original_w=width, resized_h=2,
                                  resized_w=width, top=0, left=0, size=2), subset=self.subset)


class BatchDependentModel(nn.Module):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("variant", torch.tensor(0.))
        self.calls = []
        self.instances.append(self)

    def cuda(self):
        return self

    def set_stage(self, stage):
        assert stage == "D"

    def anchor_state_hash(self):
        return "shared-CPU-toy-anchor"

    def forward_pair(self, images, valid=None, alpha=1.):
        ids = tuple(int(x) for x in images[:, 0, 0, 0])
        self.calls.append(ids)
        z0 = torch.full_like(images[:, :1], -0.2)
        # M batches (a,b)/(c) yield positive/negative logits, whereas H=(b,c)
        # and T1=(a,c) would change signs if either source ran another forward.
        delta = 1.75 - images[:, 0, 0, 0].mean() + self.variant
        message = torch.full_like(z0, float(delta))
        return dict(z0=z0, z1=z0 + alpha * message, message=message)


def test_source_exports_reuse_M_logits_and_recompute_selected_reference_losses(monkeypatch, tmp_path):
    BatchDependentModel.instances = []
    monkeypatch.setattr(exporting, "ControlledModel", BatchDependentModel)
    monkeypatch.setattr(exporting, "IMAMultiReferenceDataset", SourceDataset)
    monkeypatch.setattr(exporting, "loader", lambda data, batch_size, workers:
                        [collate_multi_reference([data[i] for i in range(start, min(start + batch_size, len(data)))])
                         for start in range(0, len(data), batch_size)])
    device_batch = exporting.device_batch
    monkeypatch.setattr(exporting, "device_batch", lambda batch, device:
                        device_batch(batch, torch.device("cpu")))

    def original_mask(ref, cache):
        image_id, reference_index = ref["reference_id"].split("-r")
        sample = SourceDataset(None, split="val", subset="M", cache_dir=None)
        item = sample[sample.ids.index(image_id)]
        width = item["geometry"]["original_w"]
        return item["masks"][int(reference_index), 0, :, :width].numpy().astype(bool)

    monkeypatch.setattr(exporting, "_mask_original", original_mask)
    checkpoint, d0_checkpoint = tmp_path / "method.pth", tmp_path / "d0.pth"
    torch.save({"model": {"variant": torch.tensor(.5)}}, checkpoint)
    torch.save({"model": {"variant": torch.tensor(.25)}}, d0_checkpoint)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic export fixture\n")
    destination = tmp_path / "per_reference.csv"
    cfg = dict(input_size=[2, 2], manifest=str(manifest), cache_dir=str(tmp_path / "cache"),
               eval_batch=2, workers=0, amp=False, protocol_id="CPU-source-export")
    rows = exporting.export_checkpoint(checkpoint, cfg, run_id="CPU-method",
                                       d0_checkpoint=d0_checkpoint, include_anchor=True,
                                       destination=destination)

    # Both the intervention and D0 see M exactly once. This also detects a
    # subset-specific recomputation whose rounded metrics happen to agree.
    assert len(BatchDependentModel.instances) == 2
    assert all(model.calls == [(1, 2), (3,)] for model in BatchDependentModel.instances)
    indexed = {(row["run_id"], row["subset"], row["image_id"], row["reference_id"]): row for row in rows}
    assert len(indexed) == len(rows) == 34
    compared = 0
    for row in rows:
        if row["subset"] == "M":
            continue
        corresponding = indexed[(row["run_id"], "M", row["image_id"], row["reference_id"])]
        # Reference metadata/cohort counts differ, but every image/reference
        # metric, canonical/original loss and update diagnostic must be exact.
        ignored = {"subset", "reference_count"}
        assert {k: v for k, v in row.items() if k not in ignored} == {
            k: v for k, v in corresponding.items() if k not in ignored}
        compared += 1
    assert compared == 16

    # The empty reference moves from M column 2 to T1 column 0. Check against
    # its own selected mask, rather than merely comparing two exported rows.
    empty_row = indexed[("CPU-method", "T1", "a", "a-r2")]
    selected = SourceDataset(None, split="val", subset="T1", cache_dir=None)[0]
    z = torch.full((1, 1, 2, 2), .55)
    expected = per_reference_loss(z, selected["masks"][None], selected["rater_present"][None],
                                  selected["pixel_valid"][None])
    assert empty_row["loss"] == float(expected[0, 0])
    assert empty_row["dice"] == 0.
    assert empty_row["loss"] != indexed[("CPU-method", "T1", "a", "a-r0")]["loss"]
    with destination.open(newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert np.isfinite([float(row["loss"]) for row in written]).all()
