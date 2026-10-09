import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/EGE-UNet-main')
from models.egeunet import EGEUNet, BGCTEGEUNet
from utils import set_seed

set_seed(42)

device = torch.device('cuda')

ege_best = '/root/EGE-UNet-main/results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth'
bgct_best = '/root/EGE-UNet-main/results/bgct_egeunet_isic18_Friday_07_August_2026_11h_19m_00s/checkpoints/best-epoch43-loss0.9772.pth'

ege = EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
ege.load_state_dict(torch.load(ege_best, map_location=device))
ege.eval()

bgct = BGCTEGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
bgct.load_state_dict(torch.load(bgct_best, map_location=device))
bgct.eval()

print(f"EGE-UNet params: {sum(p.numel() for p in ege.parameters()):,}")
print(f"BGCT-EGE-UNet params: {sum(p.numel() for p in bgct.parameters()):,}")

val_dir = '/root/isic2018_data/val/images/'
mask_dir = '/root/isic2018_data/val/masks/'
img_files = sorted(os.listdir(val_dir))[:20]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

isege_mean, isege_std = 148.429, 25.748

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np - isege_mean) / isege_std
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0
    img_np = img_np / 255.0
    t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
    return np.array(img), t

def load_mask(path):
    m = Image.open(path).convert('L').resize((256, 256))
    return np.array(m) / 255.0

fig, axes = plt.subplots(8, 4, figsize=(12, 24))
plt.subplots_adjust(wspace=0.02, hspace=0.15)

for idx in range(8):
    img_np, img_t = preprocess(os.path.join(val_dir, img_files[idx]))
    mask = load_mask(os.path.join(mask_dir, img_files[idx]))

    with torch.no_grad():
        _, ege_out = ege(img_t)
        ege_pred = ege_out.squeeze().cpu().numpy()
    with torch.no_grad():
        bgct_out = bgct(img_t)
        bgct_pred = bgct_out['final_output'].squeeze().cpu().numpy()

    for j, (title, data) in enumerate([
        ('Input', img_np),
        ('Ground Truth', mask),
        ('EGE-UNet', (ege_pred > 0.5).astype(np.float32)),
        ('BGCT-EGE-UNet', (bgct_pred > 0.5).astype(np.float32)),
    ]):
        ax = axes[idx][j]
        if j == 0:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

os.makedirs('/root/EGE-UNet-main/results/comparison', exist_ok=True)
plt.savefig('/root/EGE-UNet-main/results/comparison/comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved comparison.png")
