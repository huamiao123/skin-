"""Mechanical checks of the new tail protocol, not experimental evidence.

The CPU architecture fixture has random weights and smaller spatial inputs to
test implementation invariants quickly. Formal preflight separately uses the
actual audited B checkpoint and actual train/validation images on the GPU.
"""
import copy

import pytest
import torch

from rsi.datasets import IMAMultiReferenceDataset, letterbox_geometry, restore_logits
from rsi.models import ControlledModel, state_hash
from rsi.objectives import objective, objective_from_losses, per_reference_loss
from rsi.tail_probe_model import EXPECTED_TRAINABLE, GROUPS, TailProbeModel, interpolate_update_logits


@pytest.fixture(scope="module")
def base():
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        net = ControlledModel(pretrained=False, input_size=32).set_stage("B").eval()
        # A learned B message is nonzero; copying must never reset Wo to zero.
        with torch.no_grad():
            net.message.Wo.weight.normal_(std=0.02)
            net.message.Wo.bias.normal_(std=0.02)
    return net


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(31)
    rgb = torch.rand(2, 3, 32, 32, generator=generator)
    masks = (torch.rand(2, 3, 1, 32, 32, generator=generator) > 0.6).float()
    valid = torch.ones(2, 1, 32, 32)
    valid[..., :3, :] = 0
    present = torch.tensor([[True, True, True], [True, True, False]])
    masks[1, 2] = float("nan")
    masks[..., :3, :] = float("nan")
    return rgb, masks, present, valid


def test_initial_f_u_outputs_equal_common_b_and_off_teacher(base, batch):
    rgb, _, _, valid = batch
    with torch.no_grad():
        original = base.forward_pair(rgb, valid)
    common_reference, common_on = None, None
    for group in GROUPS:
        model = TailProbeModel(base, group).eval()
        with torch.no_grad():
            reference, on = model.forward_pair(rgb, valid)
            off = model.forward_off(rgb, valid)
        torch.testing.assert_close(reference, original["z0"], atol=0, rtol=0)
        torch.testing.assert_close(off, reference, atol=0, rtol=0)
        if group != "U-NoMessage":
            torch.testing.assert_close(on, original["z1"], atol=0, rtol=0)
            if common_on is not None:
                torch.testing.assert_close(on, common_on, atol=0, rtol=0)
                torch.testing.assert_close(reference, common_reference, atol=0, rtol=0)
            common_reference, common_on = reference, on
        else:
            torch.testing.assert_close(on, reference, atol=0, rtol=0)


def test_copy_keeps_learned_message_rng_and_all_storage_independent(base):
    rng = torch.get_rng_state().clone()
    model = TailProbeModel(base, "U-Mean")
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.independent_tail_storage()
    for name, parameter in model.reference_tail.named_parameters():
        student = dict(model.student_tail.named_parameters())[name]
        assert torch.equal(parameter, student)
        assert parameter.untyped_storage().data_ptr() != student.untyped_storage().data_ptr()
    for name, tensor in base.state_dict().items():
        copied = model.state_dict()[name]
        assert torch.equal(tensor, copied)
        if tensor.numel():
            assert tensor.untyped_storage().data_ptr() != copied.untyped_storage().data_ptr()
    assert model.teacher_state_hash() == state_hash(base.cnn)
    assert torch.count_nonzero(model.message.Wo.weight) > 0
    with torch.no_grad():
        model.student_tail.output.bias.add_(1)
    assert model.teacher_state_hash() == state_hash(base.cnn)


@pytest.mark.parametrize("group", GROUPS)
def test_exact_gradient_optimizer_and_modes_keep_frozen_state(base, batch, group):
    rgb, masks, present, valid = batch
    model = TailProbeModel(base, group).train()
    prefixes = {"student_tail"} if group == "U-NoMessage" else {"message"}
    if group in {"U-Mean", "U-RSI"}:
        prefixes.add("student_tail")
    names = model.trainable_parameter_names()
    assert {name.split(".", 1)[0] for name in names} == prefixes
    assert sum(p.numel() for p in model.trainable_parameters()) == EXPECTED_TRAINABLE[group]
    for mode in (False, True, False, True):
        model.train(mode)
        assert model.trainable_parameter_names() == names
        for name, component in model._components().items():
            assert component.training == (mode and name in prefixes)
            assert all(p.requires_grad == (name in prefixes) for p in component.parameters())
    # Even a direct request cannot put the reference CNN in training mode.
    model.cnn.train(True)
    assert all(not module.training for module in model.cnn.modules())
    frozen_before = model.frozen_state_hash()
    teacher_before = model.teacher_state_hash()
    with torch.no_grad():
        teacher_logits = model.forward_reference(rgb).clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=1e-4)
    optimized = [p for g in optimizer.param_groups for p in g["params"]]
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters() if p.requires_grad}
    out = model.forward_details(rgb, valid)
    assert not out["z0"].requires_grad
    assert not out["f8"].requires_grad and not out["c4"].requires_grad
    if out["message"] is not None:
        out["message"].retain_grad()
    result = objective(out["z1"], out["z0"], masks, present, valid,
                       method="rsi" if group.endswith("RSI") else "d0",
                       weight=3 if group.endswith("RSI") else 0)
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    if out["message"] is not None:
        assert out["message"].grad.abs().sum() > 0
        assert model.message.q_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval().train()
    assert model.trainable_parameter_names() == names
    assert model.teacher_state_hash() == teacher_before
    assert model.frozen_state_hash() == frozen_before
    with torch.no_grad():
        torch.testing.assert_close(model.forward_reference(rgb), teacher_logits, atol=0, rtol=0)


def test_no_message_really_skips_transformer_message_and_teacher_in_deployment(base, batch):
    rgb, _, _, valid = batch
    model = TailProbeModel(base, "U-NoMessage").eval()
    calls = {name: 0 for name in ("transformer", "message", "reference", "student")}
    hooks = []
    for name, module in (("transformer", model.transformer), ("message", model.message),
                         ("reference", model.reference_tail), ("student", model.student_tail)):
        def count(_module, _args, _output, name=name):
            calls[name] += 1
        hooks.append(module.register_forward_hook(count))
    try:
        with torch.no_grad():
            initial = model(rgb, valid)
        assert calls == {"transformer": 0, "message": 0, "reference": 0, "student": 1}
        with torch.no_grad():
            # Output independence is stronger than a frozen/no-grad message.
            for parameter in model.transformer.parameters():
                parameter.fill_(float("nan"))
            for parameter in model.message.parameters():
                parameter.fill_(float("nan"))
            changed = model(rgb, valid)
        torch.testing.assert_close(initial, changed, atol=0, rtol=0)
        assert torch.isfinite(changed).all()
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("group", ("F-Mean", "F-RSI"))
def test_legacy_f_strict_forward_objective_and_gradients_identical(base, batch, group):
    rgb, masks, present, valid = batch
    legacy = copy.deepcopy(base).set_stage("D").train()
    with torch.no_grad():
        legacy.message.Wo.weight.mul_(1.07)
        legacy.message.Wo.bias.add_(0.01)
    model = TailProbeModel(base, group)
    result = model.load_legacy_state_dict(legacy.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    original = legacy.forward_pair(rgb, valid)
    current = model.forward_details(rgb, valid)
    for name in ("z0", "z1", "message"):
        torch.testing.assert_close(current[name], original[name], atol=0, rtol=0)
    method, weight = ("rsi", 3) if group == "F-RSI" else ("d0", 0)
    old_loss = objective(original["z1"], original["z0"], masks, present, valid,
                         method=method, weight=weight)
    new_loss = objective(current["z1"], current["z0"], masks, present, valid,
                         method=method, weight=weight)
    for name in ("loss", "seg", "risk", "rho", "J", "per_reference_loss"):
        torch.testing.assert_close(old_loss[name], new_loss[name], atol=0, rtol=0)
    old_loss["loss"].backward()
    new_loss["loss"].backward()
    for name, parameter in legacy.message.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(model.message.named_parameters())[name].grad,
                                   atol=0, rtol=0)
    assert torch.equal(original["z1"] >= 0, current["z1"] >= 0)


def test_legacy_reuse_rejects_missing_frozen_mismatch_and_u_initialization(base):
    model = TailProbeModel(base, "F-Mean")
    state = dict(base.state_dict())
    with pytest.raises(ValueError, match="strict"):
        model.load_legacy_state_dict(state, strict=False)
    incomplete = dict(state)
    incomplete.pop("message.Wo.bias")
    with pytest.raises(ValueError, match="keys"):
        model.load_legacy_state_dict(incomplete)
    drifted = dict(state)
    drifted["cnn.tail.output.bias"] = state["cnn.tail.output.bias"] + 1
    with pytest.raises(ValueError, match="frozen"):
        model.load_legacy_state_dict(drifted)
    with pytest.raises(ValueError, match="F groups"):
        TailProbeModel(base, "U-Mean").load_legacy_state_dict(state)
    with pytest.raises(ValueError, match="Unknown"):
        model.set_group("D0")


def test_rsi_is_per_reference_hinge_image_equal_fp32_and_risk_gradient_attached():
    current = torch.tensor([[0.7, 0.2, 0.9], [0.6, 0.3, float("nan")]], requires_grad=True)
    teacher = torch.tensor([[0.4, 0.4, 0.4], [0.4, 0.4, float("nan")]], requires_grad=True)
    present = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = objective_from_losses(current, teacher, present, method="rsi", weight=3)
    assert result["loss"].dtype == torch.float32
    expected_seg = (current[0].mean() + current[1, :2].mean()) / 2
    expected_risk = ((0.3 + 0.5) / 3 + 0.2 / 2) / 2
    assert result["seg"].item() == pytest.approx(expected_seg.item())
    assert result["risk"].item() == pytest.approx(expected_risk)
    result["loss"].backward()
    assert teacher.grad is None
    torch.testing.assert_close(current.grad,
                               torch.tensor([[4/6, 1/6, 4/6], [1, 1/4, 0]]),
                               atol=1e-7, rtol=1e-7)


def test_invalid_padding_and_absent_references_have_no_loss_or_gradient():
    z = torch.tensor([[[[0.4, -0.2, float("nan")]]]], requires_grad=True)
    reference = torch.tensor([[[[0.2, -0.5, float("nan")]]]], requires_grad=True)
    masks = torch.tensor([[[[[1., 0., float("nan")]]], [[[0., 1., float("nan")]]],
                           [[[float("nan"), float("nan"), float("nan")]]]]])
    present = torch.tensor([[1, 1, 0]])
    valid = torch.tensor([[[[1, 1, 0]]]])
    result = objective(z, reference, masks, present, valid, method="rsi", weight=3)
    independently_cropped = objective(z[..., :2], reference[..., :2], masks[..., :2],
                                      present, torch.ones_like(valid[..., :2]), method="rsi", weight=3)
    torch.testing.assert_close(result["loss"], independently_cropped["loss"], atol=0, rtol=0)
    result["loss"].backward()
    assert reference.grad is None
    assert torch.isfinite(z.grad).all()
    assert z.grad[..., 2].item() == 0
    assert per_reference_loss(z, masks, present, valid)[0, 2].item() == 0


def test_complete_update_interpolation_endpoints_and_coordinate_restore():
    reference = torch.tensor([[[[1e8, 0.1, -0.2, -1e8], [1., -2., 3., -4.],
                                 [2., -1., 4., -3.], [1e8, -0.1, 0.2, -1e8]]]])
    on = torch.tensor([[[[-1e8, -0.5, 0.6, 1e8], [-4., 3., -2., 1.],
                          [-3., 4., -1., 2.], [-1e8, 0.5, -0.6, 1e8]]]])
    assert torch.equal(interpolate_update_logits(reference, on, 0), reference)
    assert torch.equal(interpolate_update_logits(reference, on, 1), on)
    for alpha in (1/3, 2/3):
        torch.testing.assert_close(interpolate_update_logits(reference, on, alpha),
                                   reference + alpha*(on-reference), atol=0, rtol=0)
    geometry = letterbox_geometry(3, 7, size=4)
    recovered = restore_logits(interpolate_update_logits(reference, on, 1/3)[0], geometry)
    assert recovered.shape == (3, 7)
    assert recovered.dtype == torch.float32
    for alpha in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            interpolate_update_logits(reference, on, alpha)


def test_existing_dataset_test_lock_is_preserved(tmp_path):
    # It must refuse the scoring split even before trying a missing manifest.
    with pytest.raises(PermissionError, match="test"):
        IMAMultiReferenceDataset(tmp_path / "absent.csv", split="test")
