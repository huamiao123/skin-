import json
from pathlib import Path

import torch

from refiner_model import NodeRefiner, PointwiseRefiner, parameter_count


ROOT = Path(__file__).resolve().parents[1]


def test_role_lock():
    manifest = json.loads((ROOT / "splits/phase3_manifest.json").read_text())
    roles = {name: set((ROOT / info["path"]).read_text().splitlines())
             for name, info in manifest["roles"].items()}
    assert {name: len(ids) for name, ids in roles.items()} == {
        "fit": 1800, "stop": 200, "calibration": 50,
        "development_readout": 100, "test": 600}
    assert len(set().union(*roles.values())) == 2750
    lock = json.loads((ROOT / "protocols/phase3_lock.json").read_text())
    assert lock["candidate_count"] == 65 and lock["zero_index"] == 32
    assert lock["models"] == ["S64", "S96", "N0", "T8", "T32", "TG"]


def test_n0_capacity_and_remote_invariance():
    torch.manual_seed(1)
    model = PointwiseRefiner().eval()
    assert parameter_count(model) == 286401
    assert parameter_count(NodeRefiner(None)) == 287425
    context = torch.randn(1, 256, 44)
    score = torch.randn(1, 256, 65)
    valid = torch.ones(1, 256, 65, dtype=torch.bool)
    with torch.no_grad():
        first = model(context, score, valid)[:, 0].clone()
        context[:, 1:] = torch.randn_like(context[:, 1:]) * 100
        score[:, 1:] = torch.randn_like(score[:, 1:]) * 100
        valid[:, 1:, 5:] = False
        second = model(context, score, valid)[:, 0]
    assert torch.equal(first, second)


def test_n0_all_parameters_participate_after_head_learns():
    torch.manual_seed(5)
    model = PointwiseRefiner().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context = torch.randn(2, 4, 44)
    score = torch.randn(2, 4, 65)
    valid = torch.ones(2, 4, 65, dtype=torch.bool)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = model(context, score, valid)
        loss = value.square().mean()
        loss.backward(); optimizer.step()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_attention_masks_and_circular_reach():
    local = NodeRefiner(8)
    medium = NodeRefiner(32)
    global_model = NodeRefiner(None)
    assert (~local.attention_mask[0]).sum().item() == 17
    assert (~medium.attention_mask[0]).sum().item() == 65
    assert not local.attention_mask[0, 255]
    assert global_model.attention_mask is None
    assert (~local.attention_mask[0]).nonzero().numel() == 17
    # Two circular layers can propagate no farther than twice the radius.
    for radius, expected in ((8, 33), (32, 129)):
        reachable = {0}
        for _ in range(2):
            reachable = {j for i in reachable for j in range(256)
                         if min(abs(j-i), 256-abs(j-i)) <= radius}
        assert len(reachable) == expected


def test_transformers_have_identical_seeded_initial_parameters():
    from phase3_train import make_model
    hashes = []
    for family in ("T8", "T32", "TG"):
        torch.manual_seed(17)
        model = make_model(family)
        hashes.append([value.detach().clone() for value in model.parameters()])
    for first, second, third in zip(*hashes):
        assert torch.equal(first, second) and torch.equal(second, third)


def test_inference_ignores_label_fields():
    from phase3_train import forward
    torch.manual_seed(4)
    model = PointwiseRefiner().eval()
    batch = {"context": torch.randn(1, 256, 44),
             "local_scores": torch.randn(1, 256, 65),
             "valid": torch.ones(1, 256, 65, dtype=torch.bool)}
    with torch.no_grad():
        first = forward(model, batch, "N0")
        batch["gt_distance"] = torch.randn(1, 256, 65)
        batch["gt"] = torch.randn(1, 256, 256)
        second = forward(model, batch, "N0")
    assert torch.equal(first, second)


def test_score_export_view_contains_no_gt():
    from phase3_data import LocalInferenceCases
    data = LocalInferenceCases("val")
    case = data[0]
    assert "gt" not in case and "gt_distance" not in case and "oracle_label" not in case
    assert tuple(case["points"].shape) == (256, 65, 2)


def test_reconstruction_metrics_edge_policies():
    import numpy as np
    from evaluation import full_metrics, selected_mask
    blank = np.zeros((32, 32), dtype=bool)
    assert full_metrics(blank, blank)["dice"] == 1.0
    single = blank.copy(); single[10, 10] = True
    result = full_metrics(single, blank)
    assert result["dice"] == 0.0 and np.isfinite(result["hd95"])
    complex_mask = blank.copy()
    complex_mask[3:24, 3:24] = True
    complex_mask[8:12, 8:12] = False  # retained hole
    complex_mask[27:30, 27:30] = True  # other component
    points = np.zeros((256, 65, 2), dtype=np.float32)
    # Zero-path reconstruction is deterministic and must retain pixel metrics.
    from evaluation import mask_contours, polygon_area, rasterize
    from contour.geometry import resample_closed
    exterior = max(mask_contours(complex_mask), key=lambda p: abs(polygon_area(p)))
    zero_polygon = resample_closed(exterior, 256)
    points[:, 32] = zero_polygon
    output = selected_mask(complex_mask, points, np.full(256, 32))
    assert output[9, 9] == 0 and output[28, 28] == 1
    assert np.isfinite(full_metrics(output, complex_mask)["bf1"])


def test_ambiguity_requires_separated_peaks():
    import numpy as np
    from phase3_ambiguity import separated_peak_margin
    value = np.zeros((2, 65), dtype=np.float32)
    value[0, 30] = 3; value[0, 32] = 2.9
    value[1, 20] = 3; value[1, 30] = 2.9
    valid = np.ones_like(value, dtype=bool)
    two, margin = separated_peak_margin(value, valid)
    assert not two[0]  # adjacent shoulders are one structural peak
    assert two[1] and margin[1] > 0
