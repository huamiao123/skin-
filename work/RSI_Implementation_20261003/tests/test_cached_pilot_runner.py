"""CPU numeric-equivalence checks outside the frozen training source bundle."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rsi import export as exporting, metrics
from rsi.datasets import collate_multi_reference
from tools import cached_pilot_runner as runner

SPEC = importlib.util.spec_from_file_location("cpu_export_fixture", Path(__file__).with_name("test_export.py"))
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("d0", [False, True])
@pytest.mark.parametrize("kind", ["empty", "full", "diagonal"])
def test_one_reference_every_scalar_matches_original_exactly(anchor, d0, kind):
    z = np.array([[0., .3, -2.], [1., -.5, 4.]], dtype=np.float64)
    z0, zd0 = z + .25, z - .4
    gt = {"empty": np.zeros_like(z), "full": np.ones_like(z),
          "diagonal": np.eye(2, 3)}[kind].astype(bool)
    arguments = (z, gt, z0 if anchor else None, zd0 if d0 else None)
    expected = metrics.evaluate_reference(*arguments)
    original = metrics.evaluate_reference
    cache = runner.ReferenceMetricCache()
    with runner.installed_metric_cache(cache):
        first = exporting.evaluate_reference(*arguments)
        assert first == expected
        first.pop("loss")
        second = metrics.evaluate_reference(*arguments)
        assert second == expected and first is not second
    assert metrics.evaluate_reference is original
    assert cache.hits >= 1 and all(isinstance(key, tuple) for key in cache.results)
    assert all(all(not isinstance(value, np.ndarray) for value in row.values()) for row in cache.results.values())


def test_recursive_global_lookups_and_dict_pop_cannot_pollute_the_cache():
    z = np.array([[-.3, .8], [1., -.4]])
    gt = np.array([[0, 1], [1, 0]], dtype=bool)
    z0 = z + .7
    cache = runner.ReferenceMetricCache()
    with runner.installed_metric_cache(cache):
        first = cache(z, gt, z0, z0)
        # Outer miss, anchor base miss, D0 identical base hit.
        assert (cache.calls, cache.misses, cache.hits) == (3, 2, 1)
        first.pop("loss_anchor")
        assert "loss_anchor" in cache(z, gt, z0, z0)
        base = cache(z0, gt)
        base.pop("loss")
        assert "loss" in cache(z0, gt)


def test_exact_bytes_and_layout_are_part_of_key_and_hits_still_validate():
    z = np.array([[.3, 1.], [-.5, .7]], dtype=np.float64)
    gt = np.eye(2, dtype=bool)
    changed = z.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    fortran = np.asfortranarray(z)
    expected = [metrics.evaluate_reference(value, gt) for value in (z, changed, fortran)]
    cache = runner.ReferenceMetricCache()
    with runner.installed_metric_cache(cache):
        assert [cache(value, gt) for value in (z, changed, fortran)] == expected
        assert cache.misses == 3 and cache.hits == 0
        assert cache(z.copy(), gt) == expected[0]
        assert cache.hits == 1
        bad = z.copy(); bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            cache(bad, gt)
        with pytest.raises(ValueError, match="binary"):
            cache(z, np.full_like(z, .5))
        with pytest.raises(ValueError, match="original mask size"):
            cache(z, gt, np.zeros((3, 2)))


@pytest.mark.parametrize("fast_exact", [False, True])
def test_full_export_all_rows_fields_and_written_bytes_match_original(monkeypatch, tmp_path, fast_exact):
    fixture.BatchDependentModel.instances = []
    monkeypatch.setattr(exporting, "ControlledModel", fixture.BatchDependentModel)
    monkeypatch.setattr(exporting, "IMAMultiReferenceDataset", fixture.SourceDataset)
    monkeypatch.setattr(exporting, "loader", lambda data, batch_size, workers:
                        [collate_multi_reference([data[i] for i in range(start, min(start + batch_size, len(data)))])
                         for start in range(0, len(data), batch_size)])
    original_device_batch = exporting.device_batch
    monkeypatch.setattr(exporting, "device_batch", lambda batch, device:
                        original_device_batch(batch, torch.device("cpu")))

    def original_mask(ref, cache):
        image_id, reference = ref["reference_id"].split("-r")
        data = fixture.SourceDataset(None, split="val", subset="M", cache_dir=None)
        item = data[data.ids.index(image_id)]
        return item["masks"][int(reference), 0, :, :item["geometry"]["original_w"]].numpy().astype(bool)

    monkeypatch.setattr(exporting, "_mask_original", original_mask)
    checkpoint, d0 = tmp_path / "method.pth", tmp_path / "d0.pth"
    torch.save({"model": {"variant": torch.tensor(.5)}}, checkpoint)
    torch.save({"model": {"variant": torch.tensor(.25)}}, d0)
    manifest = tmp_path / "manifest.csv"; manifest.write_text("CPU fixture\n")
    cfg = dict(input_size=[2, 2], manifest=str(manifest), cache_dir=str(tmp_path / "cache"),
               eval_batch=2, workers=0, amp=False, protocol_id="CPU-source-export")
    arguments = dict(run_id="CPU-method", method="rsi", d0_checkpoint=d0, include_anchor=True)
    baseline_path, cached_path = tmp_path / "baseline.csv", tmp_path / "cached.csv"
    expected = exporting.export_checkpoint(checkpoint, cfg, destination=baseline_path, **arguments)
    cache = runner.ReferenceMetricCache()
    original_metric, original_binary = metrics.evaluate_reference, metrics.binary_mask_metrics
    with runner.installed_metric_cache(cache, fast_exact=fast_exact):
        actual = exporting.export_checkpoint(checkpoint, cfg, destination=cached_path, **arguments)
    assert len(expected) == len(actual) == 34
    assert actual == expected  # Every returned field, including all floats, is exact.
    assert cached_path.read_bytes() == baseline_path.read_bytes()
    assert metrics.evaluate_reference is original_metric and metrics.binary_mask_metrics is original_binary
    assert cache.hits > cache.misses
    assert all(model.calls == [(1, 2), (3,)] for model in fixture.BatchDependentModel.instances)


def test_pilot_wrappers_only_change_B_metadata_and_reuse_done(monkeypatch, tmp_path):
    cfg = {"run_root": str(tmp_path / "runs"), "epochs": {"B": 40}}
    directory = tmp_path / "runs/seed17/B"; directory.mkdir(parents=True)
    done_path = directory / "DONE.json"
    done_path.write_text(json.dumps(dict(stage="B", seed=17, epochs=40, test_scoring_locked=True)))
    checkpoint = directory / "best.pth"; checkpoint.write_bytes(b"selected CPU fixture")
    before = done_path.read_bytes()
    called = []
    def fake_export(checkpoint, passed_cfg, **kwargs):
        called.append(kwargs)
        return [{"method": kwargs["method"], "loss": .123}]
    def fake_stage(passed_cfg, seed, stage, **kwargs):
        assert passed_cfg == cfg and seed == 17 and stage == "B" and kwargs == {"init": "prior.pth"}
        return checkpoint
    monkeypatch.setattr(runner.run_pilot, "export_checkpoint", fake_export)
    monkeypatch.setattr(runner.run_pilot, "train_stage", fake_stage)
    audit = {"stage_calls": [], "export_calls": []}
    cache = runner.ReferenceMetricCache()
    with runner.installed_pilot_patches(cache, cfg, audit):
        assert runner.run_pilot.train_stage(cfg, 17, "B", init="prior.pth") == checkpoint
        row = runner.run_pilot.export_checkpoint(checkpoint, cfg, run_id="B", include_anchor=True)[0]
        assert row["method"] == "B"
    assert called == [{"run_id": "B", "include_anchor": True, "method": "B"}]
    assert done_path.read_bytes() == before and audit["stage_calls"][0]["reused_DONE"]
    assert runner.run_pilot.export_checkpoint is fake_export and runner.run_pilot.train_stage is fake_stage
    done = json.loads(done_path.read_text()); done["epochs"] = 39; done_path.write_text(json.dumps(done))
    with runner.installed_pilot_patches(cache, cfg, audit):
        with pytest.raises(ValueError, match="full-budget"):
            runner.run_pilot.train_stage(cfg, 17, "B", init="prior.pth")


def test_patches_restore_on_failure_and_validate_only_never_runs_model(monkeypatch, tmp_path):
    original = (metrics.evaluate_reference, metrics.binary_mask_metrics, exporting.evaluate_reference,
                runner.run_pilot.export_checkpoint, runner.run_pilot.train_stage)
    with pytest.raises(RuntimeError, match="CPU injected"):
        with runner.installed_pilot_patches(runner.ReferenceMetricCache(), {}, {"stage_calls": [], "export_calls": []}):
            raise RuntimeError("CPU injected")
    assert original == (metrics.evaluate_reference, metrics.binary_mask_metrics, exporting.evaluate_reference,
                        runner.run_pilot.export_checkpoint, runner.run_pilot.train_stage)
    monkeypatch.setattr(runner.run_pilot, "run", lambda *args: pytest.fail("Validation must not start the pilot"))
    cfg = dict(run_root=str(tmp_path / "runs"), manifest=str(tmp_path / "manifest.csv"),
               audit=str(tmp_path / "file_audit.json"), test_scoring_locked=True)
    report = runner.run_cached(cfg, validate_only=True, audit_path=tmp_path / "validation.json")
    assert report["status"] == "validated_only" and report["cache_statistics"]["calls"] == 0
    assert report["scientific_config_unchanged"] and report["frozen_sources_before"] == report["frozen_sources_after"]
