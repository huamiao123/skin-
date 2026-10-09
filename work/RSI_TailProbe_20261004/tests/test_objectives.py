import math

import pytest
import torch

from rsi.objectives import (
    image_reference_mean,
    objective,
    objective_from_losses,
    per_reference_loss,
    weighted_reference_quantile,
)


def example():
    current = torch.tensor([[0.6, 0.2], [0.8, float("nan")]], requires_grad=True)
    anchor = torch.tensor([[0.4, 0.4], [0.7, float("nan")]], requires_grad=True)
    present = torch.tensor([[True, True], [True, False]])
    return current, anchor, present


@pytest.mark.parametrize("method,risk,total", [
    ("d0", 0, 0.6), ("rsi", 0.1, 0.8),
    ("mean_hinge", 0.05, 0.7), ("abs_hard", 0.175, 0.95),
])
def test_four_objectives_against_hand_calculation(method, risk, total):
    current, anchor, present = example()
    q = torch.tensor(0.5, requires_grad=True)
    out = objective_from_losses(current, anchor, present, method=method, weight=2, q=q)
    assert out["seg"].item() == pytest.approx(0.6)
    assert out["risk"].item() == pytest.approx(risk, abs=1e-7)
    assert out["loss"].item() == pytest.approx(total, abs=1e-7)
    assert out["rho"].item() == pytest.approx(0.75)
    assert out["J"].item() == pytest.approx(0.05, abs=1e-7)
    assert out["kappa_pm"].item() == pytest.approx(0.5)
    if method in ("rsi", "abs_hard"):
        assert out["activation_rate"].item() == pytest.approx(0.75)
        assert out["mean_weight"].item() == pytest.approx(2.5)
    out["loss"].backward()
    assert anchor.grad is None
    assert q.grad is None
    assert torch.isfinite(current.grad).all()
    assert current.grad[1, 1] == 0


@pytest.mark.parametrize("method", ["d0", "rsi", "mean_hinge", "abs_hard"])
def test_lambda_zero_is_exact_d0_value_and_gradient(method):
    current, anchor, present = example()
    out = objective_from_losses(current, anchor, present, method=method, weight=0, q=0.5)
    grad = torch.autograd.grad(out["loss"], current)[0]
    baseline = image_reference_mean(current, present)
    baseline_grad = torch.autograd.grad(baseline, current)[0]
    assert torch.equal(out["loss"], baseline)
    assert torch.equal(grad, baseline_grad)


def test_mean_hinge_is_per_image_before_hinge_not_cross_batch():
    current = torch.tensor([[0.2, 0.2], [0.5, 0.5]])
    anchor = torch.tensor([[0.4, 0.4], [0.4, 0.4]])
    present = torch.ones(2, 2, dtype=torch.bool)
    out = objective_from_losses(current, anchor, present, method="mean_hinge", weight=1)
    assert out["risk"].item() == pytest.approx(0.05)
    assert torch.relu((current - anchor).mean()).item() == 0


def test_pixel_loss_hand_calculation_padding_missing_and_input_gradient():
    logits = torch.tensor([[[[0.0, 0.0, float("nan")]]]], requires_grad=True)
    masks = torch.tensor([[[[[1.0, 0.0, float("nan")]]],
                           [[[1.0, 1.0, float("nan")]]],
                           [[[float("nan"), float("nan"), float("nan")]]]]])
    present = torch.tensor([[1, 1, 0]])
    valid = torch.tensor([[[[1, 1, 0]]]])
    out = per_reference_loss(logits, masks, present, valid, return_components=True)
    expected_dice = torch.tensor([[(1 + 1e-6) / (2 + 1e-6),
                                  (2 + 1e-6) / (3 + 1e-6), 0]])
    assert torch.allclose(out["soft_dice"], expected_dice, atol=1e-7)
    assert out["bce"][0, 0].item() == pytest.approx(math.log(2))
    assert out["loss"][0, 0].item() == pytest.approx(0.5 * math.log(2) + 0.5 * (1 - expected_dice[0, 0]))
    assert out["loss"][0, 2] == 0
    image_reference_mean(out["loss"], present).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 2] == 0


def test_per_reference_valid_is_supported_without_shared_pixel_assumption():
    logits = torch.zeros(1, 1, 1, 2)
    masks = torch.tensor([[[[[1.0, float("nan")]]], [[[float("nan"), 0.0]]]]])
    valid = torch.tensor([[[[[1, 0]]], [[[0, 1]]]]])
    result = per_reference_loss(logits, masks, torch.ones(1, 2), valid)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("method", ["d0", "rsi", "mean_hinge", "abs_hard"])
def test_reference_reordering_and_copying_entire_set_invariant(method):
    current, anchor, present = example()
    original = objective_from_losses(current, anchor, present, method=method, weight=1, q=0.5)
    reordered = objective_from_losses(current.flip(1), anchor.flip(1), present.flip(1), method=method, weight=1, q=0.5)
    copied = objective_from_losses(current.repeat(1, 2), anchor.repeat(1, 2), present.repeat(1, 2), method=method, weight=1, q=0.5)
    for name in ("loss", "risk", "rho", "J", "kappa_pm", "mean_weight"):
        assert torch.allclose(original[name], reordered[name], atol=1e-7)
        assert torch.allclose(original[name], copied[name], atol=1e-7)


def test_j_identity_single_reference_same_sign_and_numeric_kappa_tolerance():
    current = torch.tensor([[0.5, 0.7], [0.2, 0.3], [0.4, 0.4]])
    anchor = torch.tensor([[0.4, 0.4], [0.4, 0.4], [0.4000005, 0.3999995]])
    present = torch.ones(3, 2, dtype=torch.bool)
    out = objective_from_losses(current, anchor, present)
    assert torch.allclose(out["J_per_image"], out["J_identity_per_image"], atol=1e-7)
    assert out["J_per_image"][0].item() == pytest.approx(0, abs=1e-7)
    assert out["J_per_image"][1].item() == pytest.approx(0, abs=1e-7)
    assert out["kappa_pm"].item() == 0
    single = objective_from_losses(current[:, :1], anchor[:, :1], present[:, :1])
    assert torch.equal(single["J_per_image"], torch.zeros(3))


def test_weighted_quantile_is_image_equal_fixed_and_differentiation_free():
    losses = torch.tensor([[0.0, 1.0], [2.0, float("nan")]], requires_grad=True)
    present = torch.tensor([[1, 1], [1, 0]])
    q = weighted_reference_quantile(losses, present)
    assert q.item() == 2.0  # masses 0.25, 0.25, 0.5, not mask-equal
    assert not q.requires_grad
    assert weighted_reference_quantile(losses, present, 0.5).item() == 1.0
    assert weighted_reference_quantile(losses, present, 1.0).item() == 2.0
    assert weighted_reference_quantile(losses.repeat(1, 2), present.repeat(1, 2)).item() == 2.0
    with torch.no_grad():
        losses[1, 0] = 0
    assert q.item() == 2.0


def test_objective_wrapper_detaches_anchor_and_enforces_fp32():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    anchor = torch.ones(1, 1, 2, 2, requires_grad=True)
    masks = torch.ones(1, 2, 1, 2, 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = objective(logits, anchor, masks, torch.ones(1, 2), method="rsi", weight=1)
    assert out["loss"].dtype == torch.float32
    out["loss"].backward()
    assert logits.grad is not None
    assert anchor.grad is None


def test_invalid_empty_reference_and_nonbinary_data_fail_explicitly():
    z = torch.zeros(1, 1, 2, 2)
    masks = torch.zeros(1, 1, 1, 2, 2)
    with pytest.raises(ValueError, match="one present"):
        per_reference_loss(z, masks, torch.zeros(1, 1))
    with pytest.raises(ValueError, match="valid pixel"):
        per_reference_loss(z, masks, torch.ones(1, 1), torch.zeros_like(z))
    with pytest.raises(ValueError, match="binary"):
        per_reference_loss(z, masks + 0.5, torch.ones(1, 1))
    with pytest.raises(ValueError, match="requires"):
        objective_from_losses(torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1), method="abs_hard")
