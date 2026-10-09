import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.egeunet import EGEHRViTUNet


def test_full_model_forward():
    print("=== Test: Full EGEHRViTUNet forward ===")
    model = EGEHRViTUNet(
        num_classes=1, input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        bridge=True, gt_ds=True,
    )

    model.eval()
    model.halting_schedule["keep_all"] = True

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert "final_output" in output
    assert output["final_output"].shape == (2, 1, 256, 256)
    assert output["deep_supervision"] is not None
    assert len(output["deep_supervision"]) == 5
    assert output["aux_logits"] is not None
    print("  PASS: dict output shape correct")


def test_full_model_backward():
    print("\n=== Test: Full EGEHRViTUNet backward ===")
    model = EGEHRViTUNet(
        num_classes=1, input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        bridge=True, gt_ds=True,
    )

    model.train()
    model.set_epoch(0)

    x = torch.randn(2, 3, 256, 256)
    output = model(x)

    loss = output["final_output"].sum()
    loss.backward()
    print("  PASS: backward ok")


def test_warmup_and_transition():
    print("\n=== Test: Halting schedule ===")
    model = EGEHRViTUNet()

    model.set_epoch(0)
    hp = model.get_halting_params()
    assert hp["keep_all"] is True
    print(f"  Epoch 0: keep_all={hp['keep_all']}")

    model.set_epoch(30)
    hp = model.get_halting_params()
    assert hp["keep_all"] is False
    assert hp["min_keep_ratio"] == pytest.approx(0.40, abs=0.01)
    print(f"  Epoch 30: keep_all={hp['keep_all']}, min_keep={hp['min_keep_ratio']}")

    model.set_epoch(60)
    hp = model.get_halting_params()
    assert hp["keep_all"] is False
    assert hp["min_keep_ratio"] == pytest.approx(0.15, abs=0.01)
    print(f"  Epoch 60: keep_all={hp['keep_all']}, min_keep={hp['min_keep_ratio']}")
    print("  PASS: halting schedule correct")


if __name__ == "__main__":
    import pytest
    test_full_model_forward()
    test_full_model_backward()
    test_warmup_and_transition()
    print("\n=== All integration tests passed ===")
