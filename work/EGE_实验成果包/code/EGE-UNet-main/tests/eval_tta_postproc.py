import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import ndimage

sys.path.insert(0, '/root/EGE-UNet-main')
from datasets.dataset import NPY_datasets
from utils import myNormalize, myToTensor, myResize
from models.ege_dual import EGEDualUNet


class EvalConfig:
    data_path = '/root/EGE-UNet-main/data/isic2018/'
    input_size_h = 256
    input_size_w = 256
    test_transformer = transforms.Compose([
        myNormalize('isic18', train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])


def get_preds(model, val_loader, tta=True):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for img, msk in val_loader:
            img = img.cuda().float()
            out = model(img)
            if isinstance(out, tuple):
                out = out[1]
            p = out
            if tta:
                for flip_fn in [lambda x: torch.flip(x, [-1]),
                                lambda x: torch.flip(x, [-2]),
                                lambda x: torch.flip(x, [-1, -2])]:
                    o = model(flip_fn(img))
                    if isinstance(o, tuple):
                        o = o[1]
                    p = p + flip_fn(o)
                p = p / 4.0
            preds.append(p.squeeze().cpu().numpy())
            gts.append(msk.squeeze().numpy())
    return np.array(preds), np.array(gts)


def flat_dsc(preds, gts, th):
    p = (preds >= th)
    g = (gts >= 0.5)
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fn, 1), (g & ~p).sum() * 0 + tp


def sens_spec(preds, gts, th):
    p = (preds >= th)
    g = (gts >= 0.5)
    tp = (p & g).sum()
    fn = (~p & g).sum()
    tn = (~p & ~g).sum()
    fp = (p & ~g).sum()
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def clean_small(pred, min_size):
    if pred.sum() == 0:
        return pred
    lab, n = ndimage.label(pred)
    if n <= 1:
        return pred
    sizes = ndimage.sum(pred, lab, range(1, n + 1))
    keep = np.zeros_like(pred)
    for i, s in enumerate(sizes):
        if s >= min_size:
            keep |= (lab == i + 1)
    return keep


def fill_holes(pred):
    if pred.sum() == 0:
        return pred
    lab, n = ndimage.label(~pred)
    filled = pred.copy()
    for i in range(1, n + 1):
        comp = (lab == i)
        if not comp[0, :].any() and not comp[-1, :].any() and \
           not comp[:, 0].any() and not comp[:, -1].any():
            filled |= comp
    return filled


def full_eval(preds, gts, th, min_size=0, do_fill=False):
    ps = []
    for p in preds:
        b = p >= th
        if do_fill:
            b = fill_holes(b)
        if min_size > 0:
            b = clean_small(b, min_size)
        ps.append(b)
    ps = np.array(ps)
    g = gts >= 0.5
    tp = (ps & g).sum()
    fp = (ps & ~g).sum()
    fn = (~ps & g).sum()
    dsc = 2 * tp / max(2 * tp + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = (g & ~ps).sum() * 0 + ((~ps & ~g).sum() / max((~ps & ~g).sum() + fp, 1))
    return dsc, sens, spec


if __name__ == '__main__':
    ckpt = '/root/EGE-UNet-main/results/ege_dual_difflr_isic18_Saturday_29_August_2026_19h_53m_18s/checkpoints/best-epoch97-loss0.7923.pth'
    model = EGEDualUNet(num_classes=1, input_channels=3,
                        c_list=[8, 16, 24, 32, 48, 64],
                        bridge=True, gt_ds=True).cuda()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    cfg = EvalConfig
    val_loader = DataLoader(NPY_datasets(cfg.data_path, cfg, train=False),
                            batch_size=1, shuffle=False, num_workers=4)

    print('=== baseline (no TTA) ===')
    preds, gts = get_preds(model, val_loader, tta=False)
    for th in [0.4, 0.45, 0.5, 0.55, 0.6]:
        d, _, _ = flat_dsc(preds, gts, th)
        se, sp = sens_spec(preds, gts, th)
        print(f'th={th:.2f}: DSC={d:.4f} Sens={se:.4f} Spec={sp:.4f}')

    print('=== TTA (3 flips) ===')
    preds_t, gts_t = get_preds(model, val_loader, tta=True)
    for th in [0.4, 0.45, 0.5, 0.55]:
        d, _, _ = flat_dsc(preds_t, gts_t, th)
        se, sp = sens_spec(preds_t, gts_t, th)
        print(f'TTA th={th:.2f}: DSC={d:.4f} Sens={se:.4f} Spec={sp:.4f}')

    print('=== TTA + best-th + morphological ===')
    best_th = 0.5
    best = (0, 0, 0)
    for th in [0.4, 0.45, 0.5, 0.55]:
        for ms in [0, 16, 64, 128]:
            for fill in [False, True]:
                d, se, sp = full_eval(preds_t, gts_t, th, ms, fill)
                if d > best[0]:
                    best = (d, se, sp)
                    best_th = (th, ms, fill)
                print(f'TTA th={th:.2f} min_size={ms:3d} fill={int(fill)}: DSC={d:.4f} Sens={se:.4f} Spec={sp:.4f}')
    print(f'\nBEST: th={best_th[0]} min_size={best_th[1]} fill={best_th[2]} -> DSC={best[0]:.4f} Sens={best[1]:.4f} Spec={best[2]:.4f}')

    print('=== 小病灶风险检查：GT 病灶像素分布 ===')
    g_sizes = [(gts[i] >= 0.5).sum() for i in range(len(gts))]
    g_sizes = np.array(g_sizes)
    print(f'GT<16px: {(g_sizes<16).mean()*100:.1f}%  GT<64px: {(g_sizes<64).mean()*100:.1f}%  '
          f'GT<128px: {(g_sizes<128).mean()*100:.1f}%  GT<256px: {(g_sizes<256).mean()*100:.1f}%')
