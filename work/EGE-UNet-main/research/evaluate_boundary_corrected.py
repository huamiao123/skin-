"""Recompute historical boundary metrics with the frozen research_v1 definitions."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.dataset import NPY_datasets  # noqa: E402
from models.ege_dual import EGEDualUNet  # noqa: E402
from models.egeunet import EGEUNet, EGEWaveUNet  # noqa: E402
from research.metrics_surface import evaluate_surface  # noqa: E402
from utils import myNormalize, myResize, myToTensor  # noqa: E402


class EvalConfig:
    input_size_h = 256
    input_size_w = 256
    test_transformer = transforms.Compose([myNormalize("isic18", train=False), myToTensor(), myResize(256, 256)])


MODELS = {
    "EGE_baseline": ("results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth", lambda: EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True)),
    "EGE_Wave_v1": ("results/ege_wave_unet_isic18_Tuesday_11_August_2026_13h_32m_25s/checkpoints/best-epoch122-loss0.7867.pth", lambda: EGEWaveUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True, wave_mode="full", fusion_type="scalar")),
    "Wave_LL_only": ("results/ege_wave_ll_only_isic18_Thursday_20_August_2026_13h_42m_43s/checkpoints/best-epoch129-loss0.8160.pth", lambda: EGEWaveUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True, wave_mode="ll_only", fusion_type="scalar")),
    "EGE_Dual_difflr": ("results/ege_dual_difflr_isic18_Saturday_29_August_2026_19h_53m_18s/checkpoints/best-epoch97-loss0.7923.pth", lambda: EGEDualUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True)),
    "EGE_Dual_unified": ("results/ege_dual_isic18_Saturday_29_August_2026_16h_32m_55s/checkpoints/best-epoch75-loss0.8109.pth", lambda: EGEDualUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True)),
}


def main() -> None:
    data_root = Path("/root/isic2018_data")
    out_csv = ROOT / "results" / "corrected_boundary_metrics_v1.csv"
    loader = DataLoader(NPY_datasets(data_root, EvalConfig, train=False), batch_size=8, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, (rel_ckpt, builder) in MODELS.items():
        ckpt = ROOT / rel_ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        model = builder().to(device)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        model.eval()
        values = []
        with torch.no_grad():
            for images, masks in loader:
                output = model(images.to(device).float())
                if isinstance(output, tuple):
                    output = output[1]
                # Historical EGE checkpoints emit probabilities; protect against accidental logits.
                if output.detach().min() < 0 or output.detach().max() > 1:
                    output = torch.sigmoid(output)
                pred = (output.squeeze(1).cpu().numpy() >= 0.5)
                gt = (masks.squeeze(1).numpy() >= 0.5)
                for p, g in zip(pred, gt):
                    values.append(evaluate_surface(p, g).as_dict())
        finite_hd = [v["hd95_px"] for v in values if np.isfinite(v["hd95_px"])]
        finite_assd = [v["assd_dirmean_px"] for v in values if np.isfinite(v["assd_dirmean_px"])]
        rows.append({"model": name, "n_images": len(values), "bf1_2px_macro": np.mean([v["bf1_2px"] for v in values]), "hd95_px_macro_finite": np.mean(finite_hd) if finite_hd else np.nan, "assd_px_macro_finite": np.mean(finite_assd) if finite_assd else np.nan, "both_empty": sum(v["surface_status"] == "both_empty" for v in values), "one_empty": sum(v["surface_status"] == "one_empty" for v in values), "surface_impl": "surface-distance==0.1; 4n BF1; empty-aware"})
        print(rows[-1])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
