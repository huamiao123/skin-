"""T0 model contract checks; set RSI_TEST_DEVICE=cuda for GPU FP32 checks."""
import os
import json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from rsi.models import ControlledModel, CrossAttentionMessage, interpolate_deit_position


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = os.environ.get("RSI_TEST_DEVICE", "cpu")
    net = ControlledModel(pretrained=False).to(device).float()
    yield net
    del net
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture
def images(model):
    return torch.rand(1, 3, 256, 256, device=next(model.parameters()).device)


def reset_message(model):
    torch.nn.init.zeros_(model.message.Wo.weight)
    torch.nn.init.zeros_(model.message.Wo.bias)


def test_shapes_zero_message_and_off_restoration(model, images):
    reset_message(model)
    model.set_stage("B").eval()
    with torch.no_grad():
        f8, c4, z0 = model.cnn(images)
        t16 = model.transformer(images)
        assert f8.shape == (1, 128, 32, 32)
        assert c4.shape == (1, 64, 64, 64)
        assert t16.shape == (1, 384, 16, 16)
        assert z0.shape == (1, 1, 256, 256)
        assert model.aux_head(t16).shape == z0.shape
        assert torch.count_nonzero(model.message(f8, t16)) == 0
        torch.testing.assert_close(model(images), z0, atol=1e-6, rtol=1e-6)
        model.message.Wo.weight.normal_(std=0.02)
        model.message.Wo.bias.normal_(std=0.02)
        torch.testing.assert_close(model(images, alpha=0.0), z0, atol=1e-6, rtol=1e-6)
    assert not hasattr(model.cnn.encoder, "layer4")
    assert not hasattr(model.cnn.encoder, "fc")


def test_frozen_anchor_tail_input_gradient_and_later_internal_gradients(model, images):
    reset_message(model)
    model.set_stage("D").train()
    before_hash = model.frozen_state_hash()
    with torch.no_grad():
        baseline = model.forward_cnn(images).clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    first = model.forward_pair(images)
    assert not first["z0"].requires_grad
    assert first["z1"].requires_grad
    first["message"].retain_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(first["z1"], torch.ones_like(first["z1"]))
    loss.backward()
    assert first["message"].grad.abs().sum() > 0
    assert model.message.Wo.weight.grad.abs().sum() > 0
    assert model.message.Wo.bias.grad.abs().sum() > 0
    assert model.message.q_proj.weight.grad.abs().sum() == 0
    assert all(parameter.grad is None for parameter in model.tail.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model.forward_pair(images)
    torch.nn.functional.binary_cross_entropy_with_logits(second["z1"], torch.ones_like(second["z1"])).backward()
    assert model.message.q_proj.weight.grad.abs().sum() > 0
    assert model.message.k_proj.weight.grad.abs().sum() > 0
    assert model.message.v_proj.weight.grad.abs().sum() > 0
    assert not torch.equal(second["z1"].detach(), baseline)
    assert model.frozen_state_hash() == before_hash
    with torch.no_grad():
        torch.testing.assert_close(model.forward_cnn(images), baseline, atol=0, rtol=0)
        torch.testing.assert_close(model(images, alpha=0.0), baseline, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("stage,trainable", [("A_CNN", {"cnn"}), ("A_T", {"transformer", "aux_head"}),
                                            ("B", {"message"}), ("D", {"message"})])
def test_stage_train_override_keeps_frozen_modules_in_eval(model, stage, trainable):
    model.set_stage(stage).train()
    for name, component in model._components().items():
        assert component.training == (name in trainable)
        assert all(p.requires_grad == (name in trainable) for p in component.parameters())
    model.eval()
    assert all(not component.training for component in model._components().values())


def test_padding_context_token_mask_and_partial_token(model):
    device = next(model.parameters()).device
    valid = torch.zeros(1, 1, 256, 256, device=device)
    valid[:, :, 0, 0] = 1
    keep = model.context_token_valid(valid, 1)
    assert keep.shape == (1, 256)
    assert keep.sum() == 1 and keep[0, 0]
    message = CrossAttentionMessage().to(device).eval()
    with torch.no_grad():
        message.Wo.weight.copy_(torch.eye(128, device=device))
        f8 = torch.randn(1, 128, 2, 2, device=device)
        t16 = torch.randn(1, 384, 2, 2, device=device)
        keep = torch.tensor([[True, False, True, True]], device=device)
        original = message(f8, t16, keep)
        t16[:, :, 0, 1] += 1000
        torch.testing.assert_close(message(f8, t16, keep), original, atol=1e-6, rtol=1e-6)
        with pytest.raises(ValueError, match="at least one"):
            message(f8, t16, torch.zeros_like(keep))


def test_cls_position_interpolation():
    source = torch.arange(197 * 384, dtype=torch.float32).reshape(1, 197, 384)
    target = interpolate_deit_position(source, (16, 16))
    assert target.shape == (1, 257, 384)
    assert torch.equal(target[:, :1], source[:, :1])


def test_deployment_decodes_tail_once_and_transformer_has_no_bypass(model, images):
    model.set_stage("B").eval()
    calls = []
    hook = model.tail.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        first = model(images, alpha=0)
    hook.remove()
    assert len(calls) == 1
    replacement = model.transformer.register_forward_hook(lambda module, args, output: output * 100 + 100)
    with torch.no_grad():
        second = model(images, alpha=0)
    replacement.remove()
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_permanent_anchor_stays_eval_after_train(model, images):
    model.set_stage("B").eval()
    teacher = model.permanent_anchor()
    teacher.train()
    assert not teacher.training
    assert all(not module.training for module in teacher.modules())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    with torch.no_grad():
        torch.testing.assert_close(teacher(images)[2], model.forward_cnn(images), atol=0, rtol=0)


def test_actual_pretrained_load_and_cls_audit(tmp_path):
    model = ControlledModel(pretrained=True, audit_path=tmp_path / "pretrained.json")
    assert model.pretrained_audit["cnn"]["loaded"]
    assert model.pretrained_audit["transformer"]["loaded"]
    assert model.pretrained_audit["transformer"]["missing_keys"] == []
    assert model.pretrained_audit["transformer"]["unexpected_keys"] == []
    assert model.pretrained_audit["transformer"]["source_position_shape"] == [1, 197, 384]
    assert model.pretrained_audit["transformer"]["target_position_shape"] == [1, 257, 384]
    assert model.pretrained_audit["transformer"]["cls_position_preserved"]
    path = tmp_path / "pretrained.json"
    prior = json.loads(path.read_text())
    prior["gpu_acceptance_evidence"] = {"preserved": True}
    path.write_text(json.dumps(prior))
    repeated = ControlledModel(pretrained=True, audit_path=path)
    assert repeated.pretrained_audit["gpu_acceptance_evidence"] == {"preserved": True}
