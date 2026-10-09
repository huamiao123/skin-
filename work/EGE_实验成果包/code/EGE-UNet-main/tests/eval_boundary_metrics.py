import sys
import os
import numpy as np
import torch
from skimage import measure

sys.path.insert(0, '/root/EGE-UNet-main')

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset import NPY_datasets
from utils import myNormalize, myToTensor, myResize


def boundary_f1(pred, gt, radius=2):
    p_b = pred ^ _dilate(pred, radius)
    g_b = gt ^ _dilate(gt, radius)
    tp = (p_b & g_b).sum()
    return 2 * tp / max(p_b.sum() + g_b.sum(), 1)


def _dilate(mask, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=r)


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')
    d1 = distance_transform_edt(1 - pred)
    d2 = distance_transform_edt(1 - gt)
    a = d1[gt > 0]
    b = d2[pred > 0]
    return max(np.percentile(a, 95), np.percentile(b, 95))


def assd(pred, gt):
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')
    d1 = distance_transform_edt(1 - pred)
    d2 = distance_transform_edt(1 - gt)
    return (d1[gt > 0].mean() + d2[pred > 0].mean()) / 2


class EvalConfig:
    data_path = '/root/EGE-UNet-main/data/isic2018/'
    input_size_h = 256
    input_size_w = 256
    test_transformer = transforms.Compose([
        myNormalize('isic18', train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])


def evaluate_boundary(model, val_loader, apply_sigmoid):
    model.eval()
    bf1s, hd95s, assds = [], [], []
    with torch.no_grad():
        for img, msk in val_loader:
            img = img.cuda().float()
            out = model(img)
            if isinstance(out, tuple):
                out = out[1]
            if apply_sigmoid:
                out = torch.sigmoid(out)
            p = (out > 0.5).float().squeeze().cpu().numpy().astype(bool)
            g = (msk.squeeze().cpu().numpy() > 0.5)
            if p.sum() == 0 or g.sum() == 0:
                continue
            bf1s.append(boundary_f1(p, g))
            h = hd95(p, g)
            a = assd(p, g)
            if not np.isnan(h):
                hd95s.append(h)
            if not np.isnan(a):
                assds.append(a)
    return np.mean(bf1s), np.mean(hd95s), np.mean(assds)


def run_model(name, ckpt_path, builder, apply_sigmoid):
    model = builder().cuda()
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('model_state_dict', sd.get('state_dict', sd))
    model.load_state_dict(sd)
    cfg = EvalConfig
    val_loader = DataLoader(NPY_datasets(cfg.data_path, cfg, train=False),
                            batch_size=1, shuffle=False, num_workers=4)
    bf1, hd, a = evaluate_boundary(model, val_loader, apply_sigmoid)
    print(f'{name}: BF1={bf1:.4f} HD95={hd:.3f} ASSD={a:.3f}')
    return bf1, hd, a


if __name__ == '__main__':
    from models.egeunet import EGEUNet, EGEWaveUNet
    from models.ege_dual import EGEDualUNet

    results = {}

    results['EGE baseline'] = run_model(
        'EGE baseline',
        '/root/EGE-UNet-main/results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth',
        lambda: EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True),
        False)

    results['EGE-Wave v1'] = run_model(
        'EGE-Wave v1',
        '/root/EGE-UNet-main/results/ege_wave_unet_isic18_Tuesday_11_August_2026_13h_32m_25s/checkpoints/best-epoch122-loss0.7867.pth',
        lambda: EGEWaveUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True,
                            wave_mode='full', fusion_type='scalar'),
        False)

    results['Wave LL-only'] = run_model(
        'Wave LL-only',
        '/root/EGE-UNet-main/results/ege_wave_ll_only_isic18_Thursday_20_August_2026_13h_42m_43s/checkpoints/best-epoch129-loss0.8160.pth',
        lambda: EGEWaveUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True,
                            wave_mode='ll_only', fusion_type='scalar'),
        False)

    results['EGE-Dual difflr'] = run_model(
        'EGE-Dual difflr',
        '/root/EGE-UNet-main/results/ege_dual_difflr_isic18_Saturday_29_August_2026_19h_53m_18s/checkpoints/best-epoch97-loss0.7923.pth',
        lambda: EGEDualUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True),
        False)

    results['EGE-Dual unified'] = run_model(
        'EGE-Dual unified',
        '/root/EGE-UNet-main/results/ege_dual_isic18_Saturday_29_August_2026_16h_32m_55s/checkpoints/best-epoch75-loss0.8109.pth',
        lambda: EGEDualUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True),
        False)

    print('\n=== SUMMARY ===')
    for k, v in results.items():
        print(f'{k}: BF1={v[0]:.4f} HD95={v[1]:.3f} ASSD={v[2]:.3f}')
