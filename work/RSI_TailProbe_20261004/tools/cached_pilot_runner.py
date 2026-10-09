"""Run the frozen pilot with content-keyed numeric metric caching.

Only process-local metric/export bindings change. The original stage runner,
scientific configuration, checkpoint selection and numeric algorithms remain
the authority. This file is outside the frozen rsi/config source bundle.
Starting this command is an explicit orchestration action; --validate-only
checks configuration and provenance without running a model or a stage.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rsi import export as exporting, metrics, run_pilot
from rsi.runtime import atomic_json, code_provenance, load_config, sha256_file


def canonical_config(cfg):
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False, allow_nan=False)


class ReferenceMetricCache:
    """Only hashes and small numeric result dictionaries are retained.

    Raw dtype/shape/strides and logical C-order bytes form each content key.
    Including strides prevents a differently ordered floating reduction from
    sharing a cached result. Original validators run even on cache hits and
    the original evaluator receives the original inputs, without conversion.
    """

    def __init__(self, evaluator=None, max_entries=50000):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.original = evaluator or metrics.evaluate_reference
        self.max_entries = max_entries
        self.results = OrderedDict()
        self.calls = self.hits = self.misses = self.evictions = self.errors = 0
        self.hashing_seconds = self.calculation_seconds_including_recursion = 0.

    @staticmethod
    def _fingerprint(value):
        if value is None:
            return None
        array = metrics._array(value)
        # A temporary contiguous view/copy is released after hashing, never
        # retained by the cache. Array values are not stored as part of a key.
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape,
                                  "strides": array.strides}, sort_keys=True).encode())
        digest.update(memoryview(contiguous).cast("B"))
        return digest.hexdigest()

    @staticmethod
    def _validate(logits, mask, anchor, d0):
        z = metrics._image(logits, "logits")
        gt = metrics._mask(mask, "reference")
        if z.shape != gt.shape:
            raise ValueError("logits must already be restored to original mask size")
        for value in (anchor, d0):
            if value is not None and metrics._image(value, "logits").shape != gt.shape:
                raise ValueError("logits must already be restored to original mask size")

    def __call__(self, logits_orig, mask_orig, anchor_logits_orig=None, d0_logits_orig=None):
        self.calls += 1
        try:
            tick = time.perf_counter()
            self._validate(logits_orig, mask_orig, anchor_logits_orig, d0_logits_orig)
            key = tuple(self._fingerprint(value) for value in
                        (logits_orig, mask_orig, anchor_logits_orig, d0_logits_orig))
            self.hashing_seconds += time.perf_counter() - tick
            if key in self.results:
                self.hits += 1
                self.results.move_to_end(key)
                return self.results[key].copy()
            self.misses += 1
            tick = time.perf_counter()
            result = self.original(logits_orig, mask_orig, anchor_logits_orig, d0_logits_orig)
            self.calculation_seconds_including_recursion += time.perf_counter() - tick
            if not isinstance(result, dict) or any(not isinstance(v, (float, int, np.number)) for v in result.values()):
                raise TypeError("Metric cache only retains scalar numeric dictionaries")
            self.results[key] = result.copy()
            if len(self.results) > self.max_entries:
                self.results.popitem(last=False)
                self.evictions += 1
            return result.copy()
        except BaseException:
            self.errors += 1
            raise

    def statistics(self):
        return dict(calls=self.calls, hits=self.hits, misses=self.misses, errors=self.errors,
                    evictions=self.evictions, entries=len(self.results), max_entries=self.max_entries,
                    hashing_seconds=self.hashing_seconds,
                    calculation_seconds_including_recursion=self.calculation_seconds_including_recursion,
                    key="SHA256 of each raw dtype/shape/strides/C-order content; explicit None slots",
                    cache_payload="numeric scalar dictionaries only; no input arrays retained",
                    returns_independent_dictionary_copy=True)


@contextmanager
def installed_metric_cache(cache, *, fast_exact=False):
    """The metrics module's recursive global lookup uses this same cache."""
    original_metric = metrics.evaluate_reference
    original_export = exporting.evaluate_reference
    original_binary = metrics.binary_mask_metrics
    if fast_exact:
        from tools.fast_exact_metrics import binary_mask_metrics
        metrics.binary_mask_metrics = binary_mask_metrics
    metrics.evaluate_reference = cache
    exporting.evaluate_reference = cache
    try:
        yield cache
    finally:
        exporting.evaluate_reference = original_export
        metrics.evaluate_reference = original_metric
        metrics.binary_mask_metrics = original_binary


@contextmanager
def installed_pilot_patches(cache, cfg, audit, *, fast_exact=False, on_progress=None):
    original_export = run_pilot.export_checkpoint
    original_stage = run_pilot.train_stage
    expected_config = canonical_config(cfg)

    def export_checked(checkpoint, passed_cfg, **kwargs):
        if canonical_config(passed_cfg) != expected_config:
            raise ValueError("Export scientific config differs from the invoked pilot")
        actual = dict(kwargs)
        if actual.get("run_id") == "B":
            if actual.get("method", "d0") not in ("d0", "B"):
                raise ValueError("Unexpected B method metadata")
            actual["method"] = "B"
        record = dict(run_id=actual.get("run_id"), checkpoint=str(checkpoint),
                      method=actual.get("method", "d0"), alpha=actual.get("alpha", 1.),
                      weight=actual.get("weight", 0.), include_anchor=actual.get("include_anchor", False),
                      d0_checkpoint=str(actual["d0_checkpoint"]) if actual.get("d0_checkpoint") else None)
        audit["export_calls"].append(record)
        if on_progress:
            on_progress()
        result = original_export(checkpoint, passed_cfg, **actual)
        record.update(completed=True, reference_rows=len(result), cache_after=cache.statistics())
        if on_progress:
            on_progress()
        return result

    def stage_checked(passed_cfg, seed, stage, **kwargs):
        if canonical_config(passed_cfg) != expected_config:
            raise ValueError("Stage scientific config differs from the invoked pilot")
        name = kwargs.get("run_name") or (stage if stage != "D" else f"{kwargs.get('method', 'd0')}_{kwargs.get('weight', 0.):g}")
        done_path = Path(cfg["run_root"]) / f"seed{seed}" / name / "DONE.json"
        before = sha256_file(done_path) if done_path.is_file() else None
        if before:
            done = json.loads(done_path.read_text())
            if done["stage"] != stage or done["seed"] != seed or done["epochs"] != cfg["epochs"][stage] or not done.get("test_scoring_locked"):
                raise ValueError("Completed stage is not the full-budget locked stage")
        record = dict(run_name=name, stage=stage, seed=seed, existing_DONE_sha256=before,
                      authority="unmodified rsi.train.train_stage; signature, selected hash and resume verification")
        audit["stage_calls"].append(record)
        if on_progress:
            on_progress()
        selected = original_stage(passed_cfg, seed, stage, **kwargs)
        if before and sha256_file(done_path) != before:
            raise RuntimeError("A completed stage was rewritten instead of strictly reused")
        record.update(completed=True, reused_DONE=bool(before), selected_checkpoint=str(selected),
                      selected_checkpoint_sha256=sha256_file(selected), DONE_sha256=sha256_file(done_path))
        if on_progress:
            on_progress()
        return selected

    run_pilot.export_checkpoint = export_checked
    run_pilot.train_stage = stage_checked
    try:
        with installed_metric_cache(cache, fast_exact=fast_exact):
            yield
    finally:
        run_pilot.export_checkpoint = original_export
        run_pilot.train_stage = original_stage


def run_cached(cfg, *, seed=17, audit_path=None, fast_exact=False, validate_only=False, invocation=None):
    before_sources = code_provenance()
    before_config = canonical_config(cfg)
    cache = ReferenceMetricCache()
    audit_path = Path(audit_path or PROJECT / "outputs" / f"cached_runner_seed{seed}.json").resolve()
    protected = [PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git")]
    protected += [Path(cfg["run_root"]).resolve(), Path(cfg["manifest"]).resolve(), Path(cfg["audit"]).resolve()]
    if any(audit_path == path or audit_path.is_relative_to(path) or path.is_relative_to(audit_path) for path in protected):
        raise ValueError("Audit output overlaps protected sources, training or scientific inputs")
    audit = dict(started_utc=datetime.now(timezone.utc).isoformat(), status="validated_only" if validate_only else "running",
                 tool_path=str(Path(__file__).resolve()), tool_sha256=sha256_file(__file__),
                 invocation=invocation if invocation is not None else list(sys.argv), seed=seed,
                 scientific_config=json.loads(before_config), scientific_config_sha256=hashlib.sha256(before_config.encode()).hexdigest(),
                 frozen_sources_before=before_sources, test_scoring_locked=cfg["test_scoring_locked"],
                 original_run="rsi.run_pilot.run", original_config_loader="rsi.runtime.load_config",
                 patches=["metrics.evaluate_reference", "export.evaluate_reference", "run_pilot.export_checkpoint B metadata only", "run_pilot.train_stage delegate/reuse audit"],
                 stage_calls=[], export_calls=[], fast_exact_boundaries=fast_exact,
                 training_budget_or_weights_migrated=False, training_algorithm_changed=False,
                 cache_statistics=cache.statistics())
    if not cfg["test_scoring_locked"]:
        raise PermissionError("This runner requires locked test scoring")
    if fast_exact:
        fast_path = PROJECT / "tools/fast_exact_metrics.py"
        verification_path = PROJECT / "outputs/exact_boundary_native_verification.json"
        verification = json.loads(verification_path.read_text())
        if (verification.get("status") != "complete" or not verification.get("frozen_sources_unchanged")
                or verification.get("fast_tool_sha256") != sha256_file(fast_path)
                or len(verification.get("cases", [])) < 2
                or not any(case.get("megapixels", 0.) >= 29 for case in verification["cases"])
                or not any(5 <= case.get("megapixels", 0.) <= 7 for case in verification["cases"])
                or not all(case.get("every_field_exact_equal") for case in verification["cases"])):
            raise ValueError("Exact KD acceleration has not passed the recorded native-resolution numeric verification")
        audit["fast_exact_tool"] = dict(path=str(fast_path), sha256=sha256_file(fast_path),
                                        calculation="cKDTree Euclidean nearest boundary query; k=1, eps=0, workers=1",
                                        native_verification_path=str(verification_path),
                                        native_verification_sha256=sha256_file(verification_path))
    atomic_json(audit_path, audit)
    def progress():
        audit.update(updated_utc=datetime.now(timezone.utc).isoformat(), cache_statistics=cache.statistics())
        atomic_json(audit_path, audit)
    tick = time.monotonic()
    try:
        if not validate_only:
            with installed_pilot_patches(cache, cfg, audit, fast_exact=fast_exact, on_progress=progress):
                run_pilot.run(cfg, seed)
            audit["status"] = "completed"
        after_sources = code_provenance()
        if after_sources["source_sha256"] != before_sources["source_sha256"] or canonical_config(cfg) != before_config:
            raise RuntimeError("Frozen source or scientific configuration changed while running")
    except BaseException as error:
        audit.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        audit.update(finished_utc=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - tick,
                     cache_statistics=cache.statistics(), frozen_sources_after=code_provenance(),
                     scientific_config_unchanged=canonical_config(cfg) == before_config)
        atomic_json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fast-exact-boundaries", action="store_true")
    args = parser.parse_args()
    result = run_cached(load_config(args.config), seed=args.seed, audit_path=args.audit_file,
                        fast_exact=args.fast_exact_boundaries, validate_only=args.validate_only)
    print(json.dumps({"status": result["status"], "cache_statistics": result["cache_statistics"]}))


if __name__ == "__main__":
    main()
