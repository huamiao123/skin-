import random
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.dataset import NPY_datasets
from models.ege_dual import EGEDualUNet
from research.metrics_overlap import overlap_metrics
from research.metrics_surface import evaluate_surface
from research.model_output import unpack_segmentation_output
from research.transforms import ResearchTransform
from research.optimization import batch_weight, window_sample_counts
from research.protocol import load_protocol


def test_overlap_reports_macro_and_pooled_and_empty_cases():
    probabilities = np.zeros((2, 4, 4), dtype=np.float32)
    targets = np.zeros_like(probabilities)
    targets[1, 0, 0] = 1
    result = overlap_metrics(probabilities, targets)
    assert result["dice"].tolist() == [1.0, 0.0]
    assert result["macro_dice"] == 0.5
    assert result["pooled_foreground_dice"] == 0.0
    assert result["gt_empty"].tolist() == [True, False]


def test_surface_empty_and_translation_cases():
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:10, 4:10] = True
    same = evaluate_surface(gt, gt)
    assert same.surface_status == "finite"
    assert same.hd95_px == pytest.approx(0.0)
    assert same.assd_dirmean_px == pytest.approx(0.0)
    shifted = np.zeros_like(gt)
    shifted[4:10, 5:11] = True
    assert evaluate_surface(shifted, gt).hd95_px > 0
    empty = evaluate_surface(np.zeros_like(gt), gt)
    assert empty.surface_status == "one_empty"
    assert np.isinf(empty.hd95_px)


def test_research_transform_is_paired_and_binary():
    image = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4, 1), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    random.seed(42)
    output_image, output_mask = ResearchTransform(size=(4, 4), train=True)((image, mask))
    assert output_image.shape == (3, 4, 4)
    assert output_mask.shape == (1, 4, 4)
    assert float(output_image.min()) >= 0 and float(output_image.max()) <= 1
    assert set(torch.unique(output_mask).tolist()).issubset({0.0, 1.0})


def test_dataset_rejects_image_mask_id_mismatch(tmp_path):
    image_dir = tmp_path / "train/images"
    mask_dir = tmp_path / "train/masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    from PIL import Image
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_dir / "sample.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask_dir / "other.png")
    with pytest.raises(ValueError, match="id mismatch"):
        NPY_datasets(str(tmp_path) + "/", object(), train=True)


def test_dual_override_replay_and_output_contract():
    torch.manual_seed(1)
    model = EGEDualUNet(
        num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
        bridge=True, gt_ds=True, t_embed=48, t_depths=(2, 2, 2, 2),
        t_head_dim=16, t_sr_ratios=(4, 2, 1, 1), fusion_type="scalar",
    ).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        native = model(x, return_aux=True)
        final_native, deep_native = unpack_segmentation_output(native)
        replay = model(x, fusion_override=native["research_aux"]["gamma_native"],
                       fusion_mask=torch.ones(1, 4), return_aux=True)
        final_replay, _ = unpack_segmentation_output(replay)
    assert final_native.shape == (1, 1, 256, 256)
    assert len(deep_native) == 5
    assert torch.allclose(final_native, final_replay, atol=1e-6, rtol=1e-5)
    assert native["research_aux"]["gamma_native"].shape == (1, 4)
    assert native["research_aux"]["injection_ratio"].shape == (1, 4)


def test_gradient_accumulation_weights_tail_window_by_actual_samples():
    assert window_sample_counts(5, 2, 2) == [4, 1]
    assert batch_weight(5, 2, 2, 0) == pytest.approx(0.5)
    assert batch_weight(5, 2, 2, 1) == pytest.approx(0.5)
    assert batch_weight(5, 2, 2, 2) == pytest.approx(1.0)


def test_protocol_manifest_hash_is_frozen():
    config = load_protocol(Path(__file__).parents[1] / "research/configs/protocol_v1.yaml")
    assert config["protocol"]["id"] == "medseg_research_v1"
    assert config["data"]["split_hash"] != "pending_generation"
