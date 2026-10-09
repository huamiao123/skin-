import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.dynamic_adapter.ege_hrvit_adapter import EGEHRViTAdapter


def test_tokenization_and_size():
    print("=== Test 1: Tokenization and Size Recovery ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    t3 = torch.randn(2, 24, 32, 32)

    hp = {"keep_all": True, "threshold": 0.5, "min_keep_ratio": 1.0}
    output, stats = adapter(t3, **hp)

    assert output.shape == (2, 24, 32, 32), f"Expected (2,24,32,32), got {output.shape}"
    print(f"  PASS: Output shape = {output.shape}")


def test_full_token_path():
    print("\n=== Test 2: Full Token Path (all 1024 tokens, 12 blocks) ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    t3 = torch.randn(2, 24, 32, 32)

    hp = {"keep_all": True, "threshold": 0.5, "min_keep_ratio": 1.0}
    output, stats = adapter(t3, **hp)

    loss = output.sum()
    loss.backward()
    print("  PASS: Full 1024-token forward + backward ok")


def test_partial_keep():
    print("\n=== Test 3: Partial Keep (25% tokens) ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    t3 = torch.randn(2, 24, 32, 32)

    hp = {"keep_all": False, "threshold": 0.5, "min_keep_ratio": 0.15}
    output, stats = adapter(t3, **hp)

    assert output.shape == (2, 24, 32, 32)
    keep_mask = stats["keep_mask"]
    retention = keep_mask.float().mean()

    print(f"  Retention ratio: {retention:.4f}")
    print(f"  PASS: Partial keep output shape correct")

    loss = output.sum()
    loss.backward()
    print("  PASS: backward ok")


def test_token_position_reconstruction():
    print("\n=== Test 4: Token Position Reconstruction ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    B, H, W = 1, 4, 4
    t3 = torch.randn(B, 24, H, W)

    hp = {"keep_all": True, "threshold": 0.5, "min_keep_ratio": 1.0}
    output, stats = adapter(t3, **hp)

    assert output.shape == (B, 24, H, W), f"Expected {(B, 24, H, W)}, got {output.shape}"
    print(f"  PASS: Token position reconstruction on {H}x{W} grid")


def test_router_output():
    print("\n=== Test 5: Router outputs correct shapes ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    t3 = torch.randn(2, 24, 32, 32)

    hp = {"keep_all": False, "threshold": 0.5, "min_keep_ratio": 0.15}
    output, stats = adapter(t3, **hp)

    assert "router_logits" in stats
    assert "router_prob" in stats
    assert "keep_mask" in stats
    assert "retention_ratio" in stats
    assert "aux_logits" in stats

    assert stats["router_logits"].shape == (2, 1024)
    assert stats["router_prob"].shape == (2, 1024)
    assert stats["keep_mask"].shape == (2, 1024)

    print("  PASS: All router output shapes correct")


def test_residual_fusion():
    print("\n=== Test 6: Residual fusion (gamma) ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    assert adapter.gamma.item() == pytest.approx(0.1, abs=0.001)
    print(f"  gamma init = {adapter.gamma.item():.4f}")
    print("  PASS: gamma initialized correctly")


def test_context_refresh_present():
    print("\n=== Test 7: Context Refresh modules exist ===")
    adapter = EGEHRViTAdapter(
        in_channels=24, dim=48, num_heads=4, window_size=8,
        total_depth=12, halt_after=3,
    )

    assert adapter.context1 is not None
    assert adapter.context2 is not None
    print("  PASS: Context Refresh 1 and 2 are present")


if __name__ == "__main__":
    import pytest
    test_tokenization_and_size()
    test_full_token_path()
    test_partial_keep()
    test_token_position_reconstruction()
    test_router_output()
    test_context_refresh_present()
    test_residual_fusion()

    print("\n=== All tests passed ===")
