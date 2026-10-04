"""Summarize existing CPU lambda1 evidence without reselecting any experiment."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent


def record(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    review_path = OUT / "lambda1_scientific_review.json"
    partial_path = OUT.parent / "partial_RSI_D0_seed17/partial_scientific_review.json"
    review = json.loads(review_path.read_text())
    partial = json.loads(partial_path.read_text())
    inputs = [record(review_path), record(partial_path), record(Path(__file__))]
    for item in review["inputs"] + [review["source_comparison_csv"],
                                      review["applied_message_rms_provenance"]["input"],
                                      review["applied_message_rms_provenance"]["output"]]:
        assert record(Path(item["path"]))["sha256"] == item["sha256"], item["path"]
    assert review["test_scoring_locked"] and not review["test_inputs_read"]
    incidence = review["incidence_RSI_vs_alpha2thirds"]
    assert incidence["bootstrap_units"] == 222 and incidence["n_images"] == 223
    assert all(incidence[k + "_ci_low"] < 0 < incidence[k + "_ci_high"]
               for k in ("any_harm", "sign_flip"))
    near = next(r for r in review["main_contrast"] if r["control_run_id"] == "Shrink-D0-0.666667")
    assert all(near[k + "_ci_low"] < 0 < near[k + "_ci_high"]
               for k in ("dice", "G_plus", "H_epsilon", "worst_tail_10pct"))
    main_rows = {r["run_id"]: r for r in review["main_table"] if r["subset"] == "M"}
    assert main_rows["Shrink-D0-0.333333"]["workpoint_selected"] == "True"
    sources = partial["source_cohort_regression_description"]
    assert [r["subset"] for r in sources] == ["H", "T1"]
    rsi_sensitivity = next(r for r in review["native_source_threshold_rows"]
                           if r["run_id"] == "RSI-1" and r["subset"] == "M"
                           and float(r["epsilon_D"]) == .005)
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_intermediate_lambda1_scientific_readout",
        "scope": "seed17 selected-development lambda1; all registered four actual Fixed-Shrink points",
        "registered_symmetric_0p3_3_grid_results_pending": True,
        "interim_action": "complete the already registered symmetric weight grid and compare all controls",
        "unique_RSI_advantage_established": False,
        "RSI_idea_invalidated": False,
        "engineering_Dice_tolerance_eligibility_is_not_superiority": True,
        "seed29_choice_not_made": True,
        "formal_seeds_or_test_released": False,
        "H_T1_selection_used": False,
        "GPU_compute": False,
        "models_constructed": False,
        "raw_exports_modified": False,
        "inputs_sha256_verified": True,
        "inputs": inputs,
        "incidence_RSI_vs_alpha2thirds": incidence,
        "near_matching_control": near,
        "source_cohort_regression": sources,
        "canonical_soft_vs_hard": {
            k: rsi_sensitivity[k] for k in (
                "canonical_rho", "canonical_J_mean", "canonical_kappa_pm",
                "soft_hard_harm_disagreement_reference_count",
                "soft_hard_harm_disagreement_reference_fraction",
                "soft_hard_harm_disagreement_macro_rate")},
        "applied_message_provenance": review["applied_message_rms_provenance"],
        "statistical_limits": [
            "Checkpoint and workpoint selection reuse this validation set; all intervals exploratory and unadjusted.",
            "Paired bootstrap retains images, references and methods together, uses 222 known-group/image units for 223 M images, 2000 draws, seed17.",
            "An interval crossing zero does not establish equivalence; current evidence does not show a substantive unique advantage.",
            "Known groups and duplicate audit do not prove independence of unknown patient/lesion identities.",
            "H47 and strict-T1 20 are descriptive case/source sensitivity cohorts, never tuning inputs."],
    }
    text = """# seed17 λ=1 科学复核（中间结论）

工程可行性验证已完成，但 RSI 相对简单对照的独特优势仍未成立。该复核覆盖七项完整固定预算训练和四个实际 Fixed-Shrink 点的原图逐参考验证结果；λ=0.3/3 的对称网格结果尚未纳入。本记录不释放 test，不启动正式种子，不依据 H/T1 重选参数。

| M val 方法 | Dice | G+ | Hε (ε=0.005) | 最差10%逐图最小参考收益 |
|---|---:|---:|---:|---:|
"""
    for run, label in (("D0", "D0"), ("RSI-1", "RSI λ=1"),
                       ("Shrink-D0-0.333333", "Fixed-Shrink α=1/3（按规则选中）"),
                       ("Shrink-D0-0.666667", "Fixed-Shrink α=2/3")):
        row = main_rows[run]
        text += "| " + label + " | " + " | ".join(f"{float(row[k]):.6f}" for k in
                                                    ("dice", "G_plus", "H_epsilon", "worst_tail_10pct")) + " |\n"
    text += """
RSI 相对 D0、Mean-Hinge λ=1、Abs-Hard λ=1 的 Hε 与负收益尾部均改善，同时 G+ 减少。相对 D0，M Dice 差为 +0.000063，探索性配对区间跨0；BF1 的0.5%与1%容差有提高，0.25%与HD95区间跨0。以上证明风险收益分布发生变化，尚不能排除普通消息收缩解释。

实际 α=2/3 的 Dice、G+、Hε、最差10%尾部几乎复现 RSI，四项直接配对区间均跨0；BF1 三个容差和HD95的直接区间也跨0。末位小数上的优劣不构成实质优势，跨0也不能证明严格等价。

RSI 的 any-harm 为114/223，α=2/3为120/223，差 -2.69个百分点，95%配对区间[-7.21, +1.80]个百分点；sign-flip为79/223和86/223，差 -3.14个百分点，区间[-7.59, +1.35]个百分点。两项小幅下降尚不稳定，均是多个探索性开发集指标，不能增加为选点条件。

按预先规则选中的 α=1/3 保持 Dice 在 D0-0.002 范围内，Hε更小且最差尾部更好，也保留更少G+。RSI比它保留更多G+，同时Hε和尾部更差，因此当前是不同风险收益工作点。规则允许被选中不代表方法胜出。

H的47图/104参考、严格T1的20图/49参考中，RSI相对D0的Dice分别下降0.004200和0.002671。固定同批图像改用M参考（分别109和55参考）仍下降0.003976和0.002946，因此不能把下降仅归因于参考来源；这里同时包含病例队列敏感性。H的原生Dice差探索性区间在0以下，严格T1区间跨0；两者像素HD95均值升高，严格T1 BF1降低。它们只能作为已选M工作点的描述性风险，不能据此调参。

RSI M canonical软风险ρ约0.394、J均值0.003223、κ约0.543，惩罚信号没有消失；软损失伤害与硬Dice伤害仍有94/471参考不一致（19.96%，图宏权重19.43%），不能将软风险下降解释为所有参考的硬Dice安全。

导出中的message_rms是alpha缩放前的原始模块消息，alpha不同不能直接横比该列。新增的独立派生表保持原始CSV及所有分数不变，按applied_message_rms=abs(alpha)*raw_message_rms计算实际施加消息。M均值：RSI为0.497090，α=2/3为0.369147，α=1/3为0.184573；α=0严格为0。实际尾部已重算，logit变化和掩码变化仍使用真实导出。消息幅度不同而聚合分数接近，不足以单独证明独特机制。

所有95%区间来自已用于checkpoint/工作点选择的开发验证集，未进行多重比较校正。M的223图按222已知病例组/图像单位共同配对，2000次bootstrap、seed17；尾部在每次重采样后按各方法重新排序并取ceil(10%)。已知组隔离不等于缺失身份全部独立。

当前行动是完成已注册的三个加权方法λ=0.3/3对称网格，并与全部简单对照共同判断。不能仅因λ=1没有独特优势宣告RSI无效；也不能据工程可行性宣告成功。待完整网格后冻结M选出的λ/α及最强简单对照，再判断是否需要预定seed29复核。最终继续/停止决定仍待完整证据记录。

机器可读证据：`lambda1_decision_review.json`、`lambda1_scientific_review.json`、`paired_controls/paired_control_comparison.csv`、`source_fixed_cohort_comparisons.csv`和`update_diagnostics_applied_message_provenance.json`。
"""
    for item in inputs:
        assert record(Path(item["path"]))["sha256"] == item["sha256"]
    (OUT / "lambda1_decision_review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (OUT / "lambda1_scientific_decision.md").write_text(text)
    print(json.dumps({"status": result["status"], "inputs_sha256_verified": True,
                      "outputs": [str(OUT / "lambda1_decision_review.json"),
                                  str(OUT / "lambda1_scientific_decision.md")]}, indent=2))


if __name__ == "__main__":
    main()
