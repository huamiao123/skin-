import torch
import numpy as np
import os, sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/EGE-UNet-main')
from models.egeunet import EGEUNet, EGEHRViTUNet, EGEWaveUNet
from utils import set_seed

set_seed(42)
device = torch.device('cuda')

ege_ckpt = '/root/EGE-UNet-main/results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth'
hrvit_ckpt = '/root/EGE-UNet-main/results/ege_hrvit_unet_isic18_Friday_07_August_2026_19h_56m_04s/checkpoints/latest.pth'
wave_ckpt = '/root/EGE-UNet-main/results/ege_wave_unet_isic18_Tuesday_11_August_2026_13h_32m_25s/checkpoints/best-epoch122-loss0.7867.pth'

ege = EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
ege.load_state_dict(torch.load(ege_ckpt, map_location=device))
ege.eval()

hrvit = EGEHRViTUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
hckpt = torch.load(hrvit_ckpt, map_location=device)
hrvit.load_state_dict(hckpt['model_state_dict'])
hrvit.eval()
hrvit.set_epoch(300)

wave = EGEWaveUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
wave.load_state_dict(torch.load(wave_ckpt, map_location=device))
wave.eval()

print(f"EGE-UNet    params: {sum(p.numel() for p in ege.parameters()):,}")
print(f"EGE-HRViT   params: {sum(p.numel() for p in hrvit.parameters()):,}")
print(f"EGE-Wave    params: {sum(p.numel() for p in wave.parameters()):,}")

val_dir = '/root/isic2018_data/val/images/'
mask_dir = '/root/isic2018_data/val/masks/'
img_files = sorted(os.listdir(val_dir))[:10]

is_mean, is_std = 149.034, 32.022

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np - is_mean) / is_std
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0
    img_np = img_np / 255.0
    t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
    return np.array(img), t

def load_mask(path):
    m = Image.open(path).convert('L').resize((256, 256))
    return np.array(m) / 255.0

def dice(p, t):
    p_b, t_b = (p>0.5).astype(np.float32), (t>0.5).astype(np.float32)
    return 2*(p_b*t_b).sum()/(p_b.sum()+t_b.sum()+1e-8)

# --- 1. 5-column mask comparison ---
fig, axes = plt.subplots(10, 5, figsize=(10, 24))
plt.subplots_adjust(wspace=0.01, hspace=0.15)

results = []
for idx in range(10):
    img_np, img_t = preprocess(os.path.join(val_dir, img_files[idx]))
    mask = load_mask(os.path.join(mask_dir, img_files[idx]))
    with torch.no_grad():
        _, ege_out = ege(img_t)
        ege_pred = ege_out.squeeze().cpu().numpy()
        hrvit_out = hrvit(img_t)
        hrvit_pred = hrvit_out['final_output'].squeeze().cpu().numpy()
        _, wave_out = wave(img_t)
        wave_pred = wave_out.squeeze().cpu().numpy()
    ege_d = dice(ege_pred, mask)
    hrvit_d = dice(hrvit_pred, mask)
    wave_d = dice(wave_pred, mask)
    results.append((ege_d, hrvit_d, wave_d))

    for j, (title, data) in enumerate([
        ('Input', img_np),
        ('GT', mask),
        ('EGE-UNet', (ege_pred > 0.5).astype(np.float32)),
        ('EGE-HRViT', (hrvit_pred > 0.5).astype(np.float32)),
        ('EGE-Wave', (wave_pred > 0.5).astype(np.float32)),
    ]):
        ax = axes[idx][j]
        if j == 0: ax.imshow(data)
        else: ax.imshow(data, cmap='gray', vmin=0, vmax=1)
        if idx == 0: ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

os.makedirs('/root/EGE-UNet-main/results/comparison', exist_ok=True)
plt.savefig('/root/EGE-UNet-main/results/comparison/three_model_mask_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 2. Boundary overlay ---
from skimage import measure
def get_contours(mask_bin, level=0.5):
    contours = measure.find_contours(mask_bin.astype(np.float64), level)
    return contours

fig, axes = plt.subplots(10, 4, figsize=(8, 24))
plt.subplots_adjust(wspace=0.01, hspace=0.15)

for idx in range(10):
    img_np, img_t = preprocess(os.path.join(val_dir, img_files[idx]))
    mask = load_mask(os.path.join(mask_dir, img_files[idx]))
    with torch.no_grad():
        _, ege_out = ege(img_t)
        ege_pred = ege_out.squeeze().cpu().numpy()
        hrvit_out = hrvit(img_t)
        hrvit_pred = hrvit_out['final_output'].squeeze().cpu().numpy()
        _, wave_out = wave(img_t)
        wave_pred = wave_out.squeeze().cpu().numpy()

    gt_ct = get_contours((mask > 0.5).astype(np.float32))
    ege_ct = get_contours((ege_pred > 0.5).astype(np.float32))
    hrvit_ct = get_contours((hrvit_pred > 0.5).astype(np.float32))
    wave_ct = get_contours((wave_pred > 0.5).astype(np.float32))

    for j, (title, pred_ct, color) in enumerate([
        ('EGE-UNet', ege_ct, 'red'),
        ('EGE-HRViT', hrvit_ct, 'blue'),
        ('EGE-Wave', wave_ct, 'lime'),
        ('GT+Input', gt_ct, 'yellow'),
    ]):
        ax = axes[idx][j]
        ax.imshow(img_np)
        for contour in pred_ct:
            ax.plot(contour[:, 1], contour[:, 0], color, linewidth=1.0)
        if idx == 0: ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 255); ax.set_ylim(255, 0)

plt.savefig('/root/EGE-UNet-main/results/comparison/three_model_boundary_overlay.png', dpi=150, bbox_inches='tight')
plt.close()

ege_mean = np.mean([r[0] for r in results])
hrvit_mean = np.mean([r[1] for r in results])
wave_mean = np.mean([r[2] for r in results])
print(f"Avg Dice 10 samples - EGE: {ege_mean:.4f}, HRViT: {hrvit_mean:.4f}, Wave: {wave_mean:.4f}")
print("Saved: three_model_mask_comparison.png, three_model_boundary_overlay.png")
