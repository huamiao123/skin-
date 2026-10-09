"""Finalize frozen seed17 validation exports without changing training sources.

The original exporter labels ordinary B rows as method=d0. Correct only that
metadata in memory, retain byte-identical original CSVs, and reuse the frozen
summarizer. This command neither constructs a model nor scores test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rsi.summarize import _method, summarize_pilot


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_sources() -> dict:
    paths = sorted(PROJECT.glob("rsi/*.py")) + sorted(PROJECT.glob("configs/*.yaml"))
    hashes = {str(path.relative_to(PROJECT)): file_hash(path) for path in paths}
    return {"source_sha256": hashes,
            "source_bundle_hash": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_corrected_inputs(input_dir: Path, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rows, evidence, corrected = [], [], []
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No completed per-reference CSVs in {input_dir}")
    for path in files:
        before_hash = file_hash(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            current = list(reader)
        if not current or not fields or "method" not in fields:
            raise ValueError(f"Empty or unsupported export: {path}")
        originals = Counter()
        corrected_count = 0
        for row in current:
            if row.get("split") != "val":
                raise PermissionError(f"Only explicit val exports are allowed: {path}")
            if str(row.get("seed")) != str(seed):
                raise ValueError(f"Export seed differs: {path}")
            if row.get("run_id") == "B" and row["method"] != "B":
                originals[row["method"]] += 1
                row["method"] = "B"
                corrected_count += 1
            if row.get("run_id") == "Anchor" and row["method"] != "anchor":
                raise ValueError(f"Anchor metadata is unexpected; it will not be rewritten: {path}")
        evidence.append({"path": str(path.resolve()), "sha256": before_hash, "rows": len(current),
                         "correction": {"field": "method", "predicate": "run_id == B",
                                        "new_value": "B", "rows_corrected": corrected_count,
                                        "old_value_counts": dict(originals)}})
        corrected.append({"name": path.name, "fields": fields, "rows": current})
        rows.extend(current)
    return rows, evidence, corrected


def review_matrix(rows: list[dict], requested_scope: str) -> tuple[str, list[str], list[float]]:
    run_ids = {row["run_id"] for row in rows}
    if not {"B", "Anchor"}.issubset(run_ids):
        raise ValueError("Ordinary B and its exported Anchor are required")
    scope = "T1" if run_ids == {"B", "Anchor"} else "T2"
    if requested_scope != "auto" and requested_scope != scope:
        raise ValueError(f"Requested {requested_scope}, exports imply {scope}")
    trained = sorted(run_ids - {"Anchor"} - {r for r in run_ids if r.startswith("Shrink-D0-")})
    if scope == "T1":
        return scope, ["A_CNN", "A_T", "B"], []
    methods = defaultdict(set)
    shrink = set()
    d0_ids = set()
    for row in rows:
        if row["subset"] != "M":
            continue
        method = _method(row)
        if method == "d0":
            d0_ids.add(row["run_id"])
        elif method in {"rsi", "mean_hinge", "abs_hard"}:
            methods[method].add(float(row["lambda"]))
        elif method == "shrink":
            shrink.add(float(row["alpha"]))
        elif method not in {"anchor", "b"}:
            raise ValueError(f"Unexpected pilot method: {method}")
    if d0_ids != {"D0"}:
        raise ValueError("T2 requires exactly one true selected D0")
    weight_sets = [methods[name] for name in ("rsi", "mean_hinge", "abs_hard")]
    if not all(1.0 in weights for weights in weight_sets) or any(weights != weight_sets[0] for weights in weight_sets):
        raise ValueError("T2 weighted methods are incomplete or not searched symmetrically")
    if weight_sets[0] not in ({1.0}, {0.3, 1.0, 3.0}):
        raise ValueError("Require lambda1 only or both registered additional weights0.3/3")
    if len(shrink) != 4 or not all(any(math.isclose(a, b, abs_tol=1e-12) for a in shrink) for b in (0., 1/3, 2/3, 1.)):
        raise ValueError("All four registered Fixed-Shrink outputs are required")
    return scope, ["A_CNN", "A_T", *trained], sorted(weight_sets[0])


def review_completed_training(run_dir: Path, stages: list[str], rows: list[dict], sources: dict) -> tuple[dict, list[dict], list[dict]]:
    completions, epoch_rows, provenance = {}, [], []
    for stage in stages:
        done_path = run_dir / stage / "DONE.json"
        done = json.loads(done_path.read_text(encoding="utf-8"))
        signature = done["signature"]
        cfg = signature["config"]
        budget = cfg["epochs"][done["stage"]]
        if done["epochs"] != budget or not done.get("test_scoring_locked") or not cfg["test_scoring_locked"]:
            raise ValueError(f"Incomplete or unlocked training stage: {stage}")
        if signature["source_bundle_hash"] != sources["source_bundle_hash"]:
            raise ValueError(f"Frozen training source bundle differs: {stage}")
        completions[stage] = done
        epoch_path = run_dir / stage / "epoch_diagnostics.csv"
        with epoch_path.open(newline="", encoding="utf-8") as stream:
            current = list(csv.DictReader(stream))
        if sorted(int(row["epoch"]) for row in current) != list(range(1, budget + 1)):
            raise ValueError(f"Epoch logs do not contain each budget epoch exactly once: {stage}")
        epoch_rows.extend(current)
        provenance.append({"run_id": stage, "done_path": str(done_path.resolve()),
                           "done_sha256": file_hash(done_path), "epochs": done["epochs"],
                           "best_epoch": done["best_epoch"], "best_checkpoint_sha256": done["best_sha256"],
                           "source_bundle_hash": signature["source_bundle_hash"],
                           "epoch_csv": str(epoch_path.resolve()), "epoch_csv_sha256": file_hash(epoch_path)})
    cfg = completions["A_CNN"]["signature"]["config"]
    manifest_hash = completions["A_CNN"]["signature"]["manifest_hash"]
    for stage, done in completions.items():
        if done["signature"]["config"] != cfg or done["signature"]["manifest_hash"] != manifest_hash:
            raise ValueError(f"Training config or manifest differs: {stage}")
        if done["seed"] != 17 or done["signature"]["seed"] != 17:
            raise ValueError(f"Training completion is not seed17: {stage}")
        parent = {"A_T": "A_CNN", "B": "A_T", "D": "B"}.get(done["stage"])
        expected_init = completions[parent]["best_sha256"] if parent else None
        if done["signature"]["init_checkpoint_hash"] != expected_init:
            raise ValueError(f"Training does not share the selected initialization chain: {stage}")
        if done["stage"] in {"A_CNN", "A_T", "B"}:
            if (done["signature"]["method"], done["signature"]["weight"], done["signature"]["q"]) != ("d0", 0., None):
                raise ValueError(f"Ordinary training objective differs: {stage}")
        elif done["signature"]["q"] != completions["D0"]["signature"]["q"]:
            raise ValueError(f"D objectives do not use the same fixed q: {stage}")
    if "D0" in completions:
        baseline = completions["D0"]["signature"]
        if baseline["method"] != "d0" or baseline["weight"] != 0. or baseline["q"] is None or not math.isfinite(baseline["q"]):
            raise ValueError("D0 objective or fixed q is invalid")
    groups = defaultdict(list)
    for row in rows:
        if row.get("protocol_id") != cfg["protocol_id"] or row.get("manifest_hash") != manifest_hash:
            raise ValueError("Export protocol or manifest is not the completed training provenance")
        rid = row["run_id"]
        stage = "B" if rid == "Anchor" else "D0" if rid.startswith("Shrink-D0-") else rid
        if row["checkpoint_hash"] != completions[stage]["best_sha256"]:
            raise ValueError(f"Export does not identify the selected checkpoint: {rid}")
        alpha, action, weight = (float(row[key]) for key in ("alpha", "action", "lambda"))
        signature = completions[stage]["signature"]
        if action != alpha:
            raise ValueError(f"Export action differs from alpha: {rid}")
        if rid == "Anchor":
            valid = _method(row) == "anchor" and alpha == 0. and weight == 0.
        elif rid.startswith("Shrink-D0-"):
            valid = _method(row) == "shrink" and weight == 0.
        elif rid == "B":
            valid = _method(row) == "b" and alpha == 1. and weight == 0.
        else:
            valid = (_method(row) == signature["method"] and alpha == 1. and weight == signature["weight"])
        if not valid:
            raise ValueError(f"Export method, lambda or alpha differs from its training objective: {rid}")
        groups[(rid, row["subset"])].append(row)
    for rid in {row["run_id"] for row in rows}:
        for subset in ("M", "H", "T1"):
            refs = groups[(rid, subset)]
            expected = cfg["audited_counts"][subset]
            if len(refs) != expected["references"]["val"] or len({r["image_id"] for r in refs}) != expected["images"]["val"]:
                raise ValueError(f"Incomplete source cohort for {rid}/{subset}")
            identities = {(r["image_id"], r["reference_id"]) for r in refs}
            baseline = {(r["image_id"], r["reference_id"]) for r in groups[("B", subset)]}
            if identities != baseline:
                raise ValueError(f"Methods do not use the same source reference cohort: {rid}/{subset}")
    return completions, epoch_rows, provenance


def finalize(input_dir: Path, output_dir: Path, run_dir: Path, *, seed=17, scope="auto", bootstrap_repeats=2000) -> dict:
    input_dir, output_dir, run_dir = map(lambda p: Path(p).resolve(), (input_dir, output_dir, run_dir))
    if seed != 17:
        raise ValueError("This frozen pilot finalizer is for seed17; seed29 needs the separately frozen selected comparison")
    protected = [input_dir, run_dir, *(PROJECT / name for name in ("rsi", "configs", "tools", "tests", ".git"))]
    if any(path.is_relative_to(output_dir) or output_dir.is_relative_to(path) for path in protected):
        raise ValueError("Output replacement overlaps protected input, training or source directories")
    rows, input_evidence, corrected = read_corrected_inputs(input_dir, seed)
    scope, stages, weights = review_matrix(rows, scope)
    sources = frozen_sources()
    completions, epoch_rows, training_evidence = review_completed_training(run_dir, stages, rows, sources)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.finalize-", dir=output_dir.parent))
    backup = None
    try:
        for item in corrected:
            write_csv(staging / "corrected_inputs" / item["name"], item["rows"], item["fields"])
        table = summarize_pilot(rows, staging, split="val", bootstrap_repeats=bootstrap_repeats,
                                bootstrap_seed=seed, fixed_two_seed=17)
        write_csv(staging / "epoch_diagnostics.csv", epoch_rows)
        b = next(row for row in table if row["run_id"] == "B" and row["subset"] == "M")
        gate = b["mean_gain"] > 0 and b["G_plus"] > 0
        q = completions.get("D0", {}).get("signature", {}).get("q")
        decision = {"seed": seed, "scope": scope, "exploratory_validation_only": True,
                    "T1_average_gain_gate": gate, "T1_mean_gain": b["mean_gain"], "T1_G_plus": b["G_plus"],
                    "T1_full_mechanism_review_pending": True, "q": q,
                    "test_scoring_locked": True, "conditional_weight_grid": [.3, 1., 3.],
                    "exported_positive_weights": weights, "extra_grid_has_not_run": weights != [.3, 1., 3.],
                    "seed29_has_not_run": True, "formal_seeds_have_not_run": True,
                    "workpoint_comparison_available": scope == "T2",
                    "final_continue_or_stop_decision_pending": True}
        write_json(staging / "decision_inputs.json", decision)
        if scope == "T1":
            conclusion = ("普通消息正平均收益门槛未通过，优先检查基线及收敛；不据此宣称RSI无效。" if not gate else
                          "普通消息正平均收益门槛通过，但还须完成损害、来源、尾部及跨参考机制审阅。")
            text = f"已完成T1来源汇总。{conclusion} 当前没有D0或D阶段工作点，不能宣称工作点通过。"
        else:
            text = ("完整固定预算T2及四个实际Fixed-Shrink输出已汇总；工作点仅按M val的Dice非劣0.002和H_epsilon规则标记。"
                    "工作点被标记不等于方法胜出。须比较G+、负收益尾部、简单对照和配对区间后，记录是否对称补网格或复核seed29。")
        (staging / "decision_log.md").write_text(
            "# seed17 开发集决策输入\n\n" + text +
            "\n\n本工具仅在内存中把run_id=B行的method校正为B，Anchor及原始CSV保持不变。"
            "开发val同时用于checkpoint与工作点选择，配对区间仅为探索性证据。"
            "最终继续/停止决策尚待根代理记录；未评分test、未自动执行seed29或正式验证。\n", encoding="utf-8")
        for evidence in input_evidence:
            if file_hash(Path(evidence["path"])) != evidence["sha256"]:
                raise RuntimeError("Original CSV changed during finalization; refuse to publish")
        if frozen_sources() != sources:
            raise RuntimeError("Frozen training source changed during finalization")
        if output_dir.exists():
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = output_dir.with_name(output_dir.name + ".before_finalize." + suffix)
        evidence = {"reviewed_utc": datetime.now(timezone.utc).isoformat(), "scope": scope, "seed": seed,
                    "tool_path": str(Path(__file__).resolve()), "tool_sha256": file_hash(Path(__file__)),
                    "source_bundle_scope": "rsi/*.py and configs/*.yaml; this tools script is excluded",
                    "original_training_source": sources, "original_csvs": input_evidence,
                    "original_csv_bytes_unchanged": True, "training_completion_evidence": training_evidence,
                    "method_correction_only": True, "test_scoring_locked": True,
                    "models_constructed": False, "GPU_compute": False,
                    "prior_output_backup": str(backup) if backup else None,
                    "generated": sorted([str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()]
                                        + ["finalization_audit.json"])}
        write_json(staging / "finalization_audit.json", evidence)
        if backup:
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except Exception:
            if backup and not output_dir.exists():
                backup.rename(output_dir)
            raise
        return evidence
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[17], default=17)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "outputs/per_reference/seed17")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/seed17")
    parser.add_argument("--run-dir", type=Path, default=Path("/home/featurize/rsi_runs/P2-IMA-M-v2/seed17"))
    parser.add_argument("--scope", choices=["auto", "T1", "T2"], default="auto")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    result = finalize(args.input_dir, args.output_dir, args.run_dir, seed=args.seed,
                      scope=args.scope, bootstrap_repeats=args.bootstrap_repeats)
    print(json.dumps({"output_dir": str(args.output_dir), "scope": result["scope"],
                      "corrected_B_rows": sum(item["correction"]["rows_corrected"] for item in result["original_csvs"]),
                      "test_scoring_locked": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
