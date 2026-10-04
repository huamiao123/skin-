"""Scientific aggregation/control invariants, using tiny analytically known cases."""
import math
import numpy as np
import pytest

from tools.tail_probe_analysis import (ALPHAS, bootstrap_units, default_screening_rules,
    fixed_categories, paired_comparison, rms_control, screening_decision,
    summarize_point, tail_mean, validate_scope)


def reference(image, ref, gain, *, group=None):
    return dict(image_id=image,reference_id=ref,group_id=image if group is None else group,
                split="val",dice=.5+gain,dice_anchor=.5,iou=.4,loss=.2-gain,
                loss_anchor=.2,gain_dice=gain)


def test_image_equal_gain_harm_identity_and_distinct_incidence():
    # Unequal reference counts must not change image weights. Image a has
    # conflicting references but zero mean; b has one uniformly harmful ref.
    p=summarize_point([reference("a","1",.1),reference("a","2",-.1),reference("b","1",-.02)])
    assert p["values"]["G_plus"]==pytest.approx(.025)
    assert p["values"]["H_minus"]==pytest.approx(.035)
    assert p["values"]["mean_gain"]==pytest.approx(-.01)
    assert p["values"]["mean_gain"]==pytest.approx(p["values"]["G_plus"]-p["values"]["H_minus"])
    assert p["values"]["mean_harm"]==.5
    assert p["values"]["any_harm"]==1.
    assert p["values"]["all_harm"]==.5
    assert p["values"]["sign_conflict"]==.5
    assert p["values"]["H_epsilon"]==pytest.approx((.095/2+.015)/2)
    assert fixed_categories(p)=={"a":"conflict","b":"all-harm"}


def test_fixed_categories_use_each_reference_and_strict_epsilon():
    rows=[reference("a","1",.02),reference("a","2",.03),
          reference("b","1",-.02),reference("b","2",-.03),
          reference("c","1",.02),reference("c","2",-.03),
          reference("d","1",0),reference("d","2",.03)]
    assert fixed_categories(summarize_point(rows))=={"a":"all-benefit","b":"all-harm","c":"conflict","d":"other"}


def test_tail_ceil_counts_and_reordering_on_resampling():
    assert tail_mean(np.arange(223))==11.
    assert tail_mean(np.arange(20))==.5
    p=summarize_point([reference(str(i),"r",g) for i,g in enumerate([-.4,-.1,.1])])
    q=summarize_point([reference(str(i),"r",g) for i,g in enumerate([-.2,-.15,.1])])
    draws=np.asarray([[0,0,2],[1,1,2]])
    result=paired_comparison(p,q,image_only=True,draws=draws,repeats=2)
    # Each method's tail is independently reranked, not the tail of differences.
    assert result["worst_tail_10pct_difference"]==pytest.approx(-.2)
    assert result["worst_tail_10pct_ci_low"]==pytest.approx(np.percentile([-.2,.05],2.5))
    assert result["worst_tail_10pct_ci_high"]==pytest.approx(np.percentile([-.2,.05],97.5))
    assert result["tail_reranked_each_draw"]


def test_missing_groups_remain_distinct_and_known_groups_stay_paired():
    units=bootstrap_units(["a","b","c","d"],["case1","case1","unknown","unknown"])
    assert [x.tolist() for x in units]==[[0,1],[2],[3]]
    assert len(bootstrap_units(["a","b"],["case1","case1"],True))==2


def test_shared_draws_identical_methods_and_anchor_mismatch_rejected():
    p=summarize_point([reference("a","1",.02),reference("b","1",-.01)])
    result=paired_comparison(p,p,repeats=100)
    assert result["dice_difference"]==result["dice_ci_low"]==result["dice_ci_high"]==0
    q=summarize_point([dict(reference("a","1",.02),dice_anchor=.4,gain_dice=.12),reference("b","1",-.01)])
    with pytest.raises(ValueError,match="teacher identities"):
        paired_comparison(p,q)


def test_rms_control_uses_mean_image_mse_clips_and_marks_degeneracy():
    m=rms_control(4.,1.)
    assert m["alpha_RMS"]==.5
    assert m["RMS_absolute_mismatch"]==0
    assert m["strict_amplitude_match"]
    high=rms_control(1.,4.)
    assert high["alpha_RMS_raw"]==2.
    assert high["alpha_RMS"]==1.
    assert high["raw_exceeds_one"] and not high["strict_amplitude_match"]
    zero=rms_control(0.,0.)
    assert zero["alpha_RMS"]==0 and zero["denominator_zero"]
    with pytest.raises(ValueError):
        rms_control(float("nan"),1.)


def decision_fixture():
    mean=dict(dice=.85,G_plus=.02,H_epsilon=.01,H_minus=.011,worst_tail_10pct=-.1)
    rsi=dict(dice=.853,G_plus=.018,H_epsilon=.007,H_minus=.008,worst_tail_10pct=-.08)
    control=dict(dice=.85,G_plus=.017,H_epsilon=.009,H_minus=.01,worst_tail_10pct=-.09)
    main={}
    for subset in ("M","H","T1"):
        main[("U-Mean",subset)]=dict(mean)
        main[("U-RSI",subset)]=dict(rsi)
        main[("U-NoMessage",subset)]=dict(control)
        main[("U-Mean-logit-RMS",subset)]=dict(control)
        for alpha in ALPHAS:
            main[(f"U-Mean-logit-alpha-{alpha:.6f}",subset)]=dict(mean)
    return main


def screen(main,**kwargs):
    return screening_decision(main,rms_control(4.,1.),engineering_ok=kwargs.get("engineering_ok",True),
                              same_conditions=kwargs.get("same_conditions",True))


def test_preregistered_routes_and_engineering_failure_priority():
    main=decision_fixture()
    result=screen(main)
    assert result["route_A_pass"] and result["route_B_pass"]
    assert result["status"]=="CANDIDATE_FOR_CONFIRMATION"
    assert screen(main,engineering_ok=False)["status"]=="INVALID"
    assert screen(main,same_conditions=False)["status"]=="INCONCLUSIVE"


def test_predeclared_simple_control_practical_dominance_stops():
    main=decision_fixture()
    key=("U-Mean-logit-alpha-0.666667","M")
    main[key]=dict(main[("U-RSI","M")],G_plus=.020)
    result=screen(main)
    assert result["status"]=="STOP_CURRENT_RSI"
    assert result["simple_controls_dominating"]==[key[0]]


@pytest.mark.parametrize("cause",["source","total_harm","tail"])
def test_signal_with_source_total_harm_or_tail_regression_is_inconclusive(cause):
    main=decision_fixture()
    if cause=="source":
        main[("U-RSI","T1")]["dice"]-=.01
    elif cause=="total_harm":
        main[("U-RSI","M")]["H_minus"]=.012
    else:
        main[("U-RSI","M")]["worst_tail_10pct"]=-.12
    result=screen(main)
    assert result["status"]=="INCONCLUSIVE"


def test_low_risk_headroom_is_not_replaced_by_lower_thresholds():
    main=decision_fixture()
    for key in main:
        main[key]["H_epsilon"]=0.
    main[("U-RSI","M")]["dice"]=.85
    result=screen(main)
    assert not result["route_B_pass"]
    assert "LOW_RISK_HEADROOM" in result["flags"]
    assert result["status"]=="STOP_CURRENT_RSI"


def test_no_test_scope_or_changed_screening_rules():
    with pytest.raises(PermissionError):
        validate_scope({"seed":17,"test_scoring_locked":True},"test")
    rules=default_screening_rules()
    rules["route_A"]["dice_advantage_over_max_M_N_C"]=0.
    with pytest.raises(ValueError,match="thresholds changed"):
        screening_decision(decision_fixture(),rms_control(4.,1.),engineering_ok=True,same_conditions=True,rules=rules)
