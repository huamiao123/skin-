import torch
import numpy as np
import os, sys
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/EGE-UNet-main')
from models.egeunet import EGEUNet, EGEHRViTUNet
from utils import set_seed

set_seed(42)
device = torch.device('cuda')

base_ckpt = '/root/EGE-UNet-main/results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth'
hrvit_ckpt = '/root/EGE-UNet-main/results/ege_hrvit_unet_isic18_Friday_07_August_2026_19h_56m_04s/checkpoints/latest.pth'

ege = EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
ege.load_state_dict(torch.load(base_ckpt, map_location=device))
ege.eval()

hrvit = EGEHRViTUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
ckpt = torch.load(hrvit_ckpt, map_location=device)
hrvit.load_state_dict(ckpt['model_state_dict'])
hrvit.eval()
hrvit.set_epoch(300)

print(f"EGE-UNet params: {sum(p.numel() for p in ege.parameters()):,}")
print(f"EGE-HRViT params: {sum(p.numel() for p in hrvit.parameters()):,}")

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

ege_preds, hrvit_preds, images, masks = [], [], [], []
for idx in range(10):
    img_np, img_t = preprocess(os.path.join(val_dir, img_files[idx]))
    mask = load_mask(os.path.join(mask_dir, img_files[idx]))
    with torch.no_grad():
        _, ege_out = ege(img_t)
        ege_pred = (ege_out.squeeze().cpu().numpy() > 0.5).astype(np.float32)
        hrvit_out = hrvit(img_t)
        hrvit_pred = (hrvit_out['final_output'].squeeze().cpu().numpy() > 0.5).astype(np.float32)
    images.append(img_np)
    masks.append(mask)
    ege_preds.append(ege_pred)
    hrvit_preds.append(hrvit_pred)

fig, axes = plt.subplots(10, 4, figsize=(8, 22))
plt.subplots_adjust(wspace=0.01, hspace=0.15)

for idx in range(10):
    for j, (title, data) in enumerate([
        ('Input', images[idx]),
        ('GT', masks[idx]),
        ('EGE-UNet', ege_preds[idx]),
        ('EGE-HRViT', hrvit_preds[idx]),
    ]):
        ax = axes[idx][j]
        if j == 0:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
        if idx == 0:
            ax.set_title(title, fontsize=10)
        ax.axis('off')

os.makedirs('/root/EGE-UNet-main/results/comparison', exist_ok=True)
plt.savefig('/root/EGE-UNet-main/results/comparison/ege_vs_hrvit.png', dpi=150, bbox_inches='tight')
plt.close()

dice = lambda p, t: 2*(p*t).sum()/(p.sum()+t.sum()+1e-8)
ege_dice = sum(dice(p, m) for p, m in zip(ege_preds, masks)) / 10
hrvit_dice = sum(dice(p, m) for p, m in zip(hrvit_preds, masks)) / 10
print(f"Avg Dice on 10 samples -- EGE-UNet: {ege_dice:.4f}, EGE-HRViT: {hrvit_dice:.4f}")
print("Saved to /root/EGE-UNet-main/results/comparison/ege_vs_hrvit.png")
