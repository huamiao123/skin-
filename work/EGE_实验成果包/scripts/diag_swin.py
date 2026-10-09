import sys, importlib.util, numpy as np, torch
sys.argv = ['diag']
import train_swin_isic as T
from sklearn.metrics import confusion_matrix

WD = '/root/Swin-Unet-main/results/swin_unet_isic18_Saturday_22_August_2026_15h_43m_56s/'
ckpt = torch.load(WD + 'checkpoints/best-epoch220-loss0.4458.pth', map_location='cpu')
model = T.SwinTransformerSys(img_size=256, patch_size=4, in_chans=3, num_classes=1,
    embed_dim=96, depths=[2,2,2,2], depths_decoder=[2,2,2,1], num_heads=[3,6,12,24],
    window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
    drop_path_rate=0.2, ape=False, patch_norm=True, final_upsample='expand_first').cuda()
model.load_state_dict(ckpt)
model.eval()

cfg = T.SwinConfig
from torch.utils.data import DataLoader
val_loader = DataLoader(T.NPY_datasets(cfg.data_path, cfg, train=False), batch_size=1,
                        shuffle=False, num_workers=4)

per_dsc = []
pred_stats = []
with torch.no_grad():
    for i, (img, msk) in enumerate(val_loader):
        img = img.cuda().float(); msk = msk.cuda().float()
        out = torch.sigmoid(model(img))
        p = (out > 0.5).float(); g = (msk > 0.5).float()
        inter = (p*g).sum(); union = (p+g).sum() + 1e-8
        dsc = (2*inter/union).item()
        per_dsc.append(dsc)
        pred_stats.append(out.mean().item())
        if i < 3:
            print(f'img {i}: pred mean={out.mean().item():.3f}, mask mean={msk.mean().item():.3f}, '
                  f'pred min={out.min().item():.3f}, max={out.max().item():.3f}')

per_dsc = np.array(per_dsc)
pred_stats = np.array(pred_stats)
print(f'per-image DSC: mean={per_dsc.mean():.4f}, median={np.median(per_dsc):.4f}, '
      f'min={per_dsc.min():.4f}, frac<0.5={(per_dsc<0.5).mean():.3f}')
print(f'pred mean over val: overall={pred_stats.mean():.4f}, '
      f'frac images with pred mean<0.01: {(pred_stats<0.01).mean():.3f}, '
      f'frac>0.99: {(pred_stats>0.99).mean():.3f}')
