import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_window_partition_roundtrip():
    from models.window_cross_attention import window_partition, window_reverse

    x = torch.randn(2, 24, 32, 32)
    windows = window_partition(x, window_size=8)
    x_rebuild = window_reverse(windows, window_size=8, B=2, H=32, W=32, C=24)
    assert torch.allclose(x, x_rebuild), "Window partition roundtrip failed!"
    print("[PASS] window_partition roundtrip")


def test_window_cross_attention_shape():
    from models.window_cross_attention import WindowCrossAttention

    q = torch.randn(2, 24, 32, 32)
    kv = torch.randn(2, 24, 32, 32)

    module = WindowCrossAttention(dim=24, num_heads=4, window_size=8)
    out = module(q_map=q, kv_map=kv)

    assert out.shape == (2, 24, 32, 32), f"Expected (2,24,32,32), got {out.shape}"
    out.mean().backward()
    print("[PASS] WindowCrossAttention shape and backward")


def test_boundary_head_shape():
    from models.boundary_head import BoundaryHead

    t1 = torch.randn(2, 8, 128, 128)
    t2 = torch.randn(2, 16, 64, 64)
    t3 = torch.randn(2, 24, 32, 32)

    edge_head = BoundaryHead()
    edge_logits = edge_head(t1, t2, t3)

    assert edge_logits.shape == (2, 1, 32, 32), f"Expected (2,1,32,32), got {edge_logits.shape}"
    edge_logits.mean().backward()
    print("[PASS] BoundaryHead shape and backward")


def test_bgct_bridge_shape():
    from models.bgct_bridge import BoundaryGuidedCrossTransformerBridge

    t4 = torch.randn(2, 32, 16, 16)
    t3_enc = torch.randn(2, 24, 32, 32)
    mask_logits = torch.randn(2, 1, 32, 32)
    edge_logits = torch.randn(2, 1, 32, 32)

    bridge = BoundaryGuidedCrossTransformerBridge(
        dim_high=32, dim_low=24, window_size=8, num_heads=4
    )
    out = bridge(
        x_high=t4, x_low=t3_enc, mask_logits=mask_logits, edge_logits=edge_logits
    )

    assert out.shape == (2, 24, 32, 32), f"Expected (2,24,32,32), got {out.shape}"
    out.mean().backward()
    print("[PASS] BGCTBridge shape and backward")


def test_bgct_egeunet_forward():
    from models.egeunet import BGCTEGEUNet

    model = BGCTEGEUNet(
        num_classes=1, input_channels=3,
        c_list=[8,16,24,32,48,64],
        bridge=True, gt_ds=True,
        window_size=8, num_heads=4
    )
    x = torch.randn(2, 3, 256, 256)
    output = model(x)

    assert isinstance(output, dict), "Output should be a dict"
    assert "deep_supervision" in output
    assert "final_output" in output
    assert "edge_logits" in output
    assert output["final_output"].shape == (2, 1, 256, 256)
    assert output["edge_logits"].shape == (2, 1, 32, 32)
    assert len(output["deep_supervision"]) == 5

    print("[PASS] BGCTEGEUNet forward")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    from utils import GT_BceDiceLoss
    from losses.total_loss import compute_total_loss

    seg_loss_fn = GT_BceDiceLoss(wb=1, wd=1)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    loss, loss_dict = compute_total_loss(
        model_output=output, target=targets, epoch=10,
        original_seg_loss_fn=seg_loss_fn,
        lambda_edge=0.2, warmup_epochs=20
    )
    loss.backward()
    print(f"[PASS] Total loss backward, loss={loss.item():.4f}, seg_loss={loss_dict['seg']:.4f}, edge_loss={loss_dict['edge']:.4f}")


if __name__ == "__main__":
    test_window_partition_roundtrip()
    test_window_cross_attention_shape()
    test_boundary_head_shape()
    test_bgct_bridge_shape()
    test_bgct_egeunet_forward()
    print("\n[ALL TESTS PASSED]")
