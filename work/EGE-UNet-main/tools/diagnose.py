import torch
import numpy as np
import os, sys, csv
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.insert(0, '/root/EGE-UNet-main')
from models.egeunet import EGEUNet, BGCTEGEUNet
from utils import set_seed
set_seed(42)

device = torch.device('cuda')

# ============================================================
# CHECKPOINTS
# ============================================================
EGE_CKPT = '/root/EGE-UNet-main/results/egeunet_isic18_Friday_07_August_2026_13h_04m_56s/checkpoints/best-epoch120-loss0.8128.pth'
BGCT_CKPT = '/root/EGE-UNet-main/results/bgct_egeunet_isic18_Friday_07_August_2026_11h_19m_00s/checkpoints/best-epoch43-loss0.9772.pth'

IMAGE_DIR = '/root/isic2018_data/val/images/'
MASK_DIR = '/root/isic2018_data/val/masks/'

OUT_DIR = '/root/EGE-UNet-main/tools/outputs/'
os.makedirs(OUT_DIR + 'ege/val_probs', exist_ok=True)
os.makedirs(OUT_DIR + 'ege/test_probs', exist_ok=True)
os.makedirs(OUT_DIR + 'bgct/val_probs', exist_ok=True)
os.makedirs(OUT_DIR + 'bgct/test_probs', exist_ok=True)
os.makedirs(OUT_DIR + 'vis', exist_ok=True)

ISEGE_MEAN, ISEGE_STD = 148.429, 25.748
IMAGE_SIZE = 256

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading models...")
ege = EGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
ege.load_state_dict(torch.load(EGE_CKPT, map_location=device))
ege.eval()
print(f"EGE-UNet params: {sum(p.numel() for p in ege.parameters()):,}")

bgct = BGCTEGEUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True).to(device)
bgct.load_state_dict(torch.load(BGCT_CKPT, map_location=device))
bgct.eval()
print(f"BGCT-EGE-UNet params: {sum(p.numel() for p in bgct.parameters()):,}")


# ============================================================
# HELPERS
# ============================================================
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np - ISEGE_MEAN) / ISEGE_STD
    mn, mx = img_np.min(), img_np.max()
    img_np = (img_np - mn) / (mx - mn) * 255.0
    img_np = img_np / 255.0
    t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
    return np.array(img), t

def load_mask(path):
    m = Image.open(path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(m) / 255.0

def dice_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)

def iou_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)

def calc_metrics(pred_bin, gt):
    TP = (pred_bin * gt).sum()
    TN = ((1 - pred_bin) * (1 - gt)).sum()
    FP = (pred_bin * (1 - gt)).sum()
    FN = ((1 - pred_bin) * gt).sum()
    dsc = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sen = TP / (TP + FN) if (TP + FN) > 0 else 0
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    return {'dsc': dsc, 'iou': iou, 'acc': acc, 'sen': sen, 'spe': spe, 'pre': pre}

def mask_to_boundary(mask, kernel_size=3):
    """Dilate - Erode to get boundary band of width kernel_size"""
    k = kernel_size
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    dilated = torch.nn.functional.max_pool2d(mask, k, stride=1, padding=k//2)
    eroded = -torch.nn.functional.max_pool2d(-mask, k, stride=1, padding=k//2)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary.squeeze().numpy()


# ============================================================
# TASK 1: SAVE PROBABILITIES + STATISTICS
# ============================================================
print("\n" + "="*60)
print("TASK 2: Saving probabilities for all images...")
print("="*60)

image_files = sorted(os.listdir(IMAGE_DIR))
n_val = int(len(image_files) * 0.5)
val_files = image_files[:n_val]
test_files = image_files[n_val:]

all_ege_val_probs = []
all_bgct_val_probs = []
all_ege_test_probs = []
all_bgct_test_probs = []
all_val_gts = []
all_test_gts = []

for split_name, file_list, ege_probs_list, bgct_probs_list, gt_list, out_sub in [
    ('val', val_files, all_ege_val_probs, all_bgct_val_probs, all_val_gts, 'val_probs'),
    ('test', test_files, all_ege_test_probs, all_bgct_test_probs, all_test_gts, 'test_probs'),
]:
    print(f"\nProcessing {split_name} set ({len(file_list)} images)...")
    for fname in tqdm(file_list):
        sid = fname.replace('.png', '').replace('.jpg', '')
        img_np, img_t = preprocess(os.path.join(IMAGE_DIR, fname))
        mask_np = load_mask(os.path.join(MASK_DIR, fname))
        gt_list.append(mask_np)

        with torch.no_grad():
            _, ege_out = ege(img_t)
            ege_prob = ege_out.squeeze().cpu().numpy()
            ege_probs_list.append(ege_prob)

            bgct_out = bgct(img_t)
            bgct_prob = bgct_out['final_output'].squeeze().cpu().numpy()
            bgct_probs_list.append(bgct_prob)

        np.save(f'{OUT_DIR}ege/{out_sub}/{sid}.npy', ege_prob)
        np.save(f'{OUT_DIR}bgct/{out_sub}/{sid}.npy', bgct_prob)


# ============================================================
# PROBABILITY STATISTICS (GT interior / boundary / bg)
# ============================================================
print("\n" + "="*60)
print("Probability Statistics (Test Set)")
print("="*60)

def compute_prob_stats(probs_list, gt_list, boundary_width=3):
    interior_probs = []
    boundary_probs = []
    bg_probs = []
    for prob, gt in zip(probs_list, gt_list):
        gt_b = (gt > 0.5).astype(np.float32)
        boundary = mask_to_boundary(gt_b, kernel_size=boundary_width*2+1)
        interior = gt_b - boundary
        bg = 1.0 - gt_b
        interior_probs.append(prob[interior > 0.5].mean() if interior.sum() > 0 else 0)
        boundary_probs.append(prob[boundary > 0.5].mean() if boundary.sum() > 0 else 0)
        bg_probs.append(prob[bg > 0.5].mean() if bg.sum() > 0 else 0)
    return {
        'interior': (np.mean(interior_probs), np.std(interior_probs)),
        'boundary': (np.mean(boundary_probs), np.std(boundary_probs)),
        'background': (np.mean(bg_probs), np.std(bg_probs)),
    }

ege_stats = compute_prob_stats(all_ege_test_probs, all_test_gts)
bgct_stats = compute_prob_stats(all_bgct_test_probs, all_test_gts)

print(f"{'Metric':<25} {'EGE':>10} {'BGCT':>10}")
print("-"*45)
for k in ['interior', 'boundary', 'background']:
    print(f"GT {k} prob:        {ege_stats[k][0]:>8.4f}±{ege_stats[k][1]:.4f}  {bgct_stats[k][0]:>8.4f}±{bgct_stats[k][1]:.4f}")


# ============================================================
# TASK 3: THRESHOLD SWEEP ON VALIDATION SET
# ============================================================
print("\n" + "="*60)
print("TASK 3: Threshold Sweep on Validation Set")
print("="*60)

thresholds = [round(x, 2) for x in np.arange(0.30, 0.72, 0.02)]

def sweep_thresholds(probs_list, gt_list, thresholds):
    results = []
    for T in thresholds:
        all_metrics = []
        for prob, gt in zip(probs_list, gt_list):
            pred = (prob >= T).astype(np.float32)
            gt_b = (gt > 0.5).astype(np.float32)
            m = calc_metrics(pred, gt_b)
            all_metrics.append(m)
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        avg['threshold'] = T
        results.append(avg)
    return results

ege_sweep = sweep_thresholds(all_ege_val_probs, all_val_gts, thresholds)
bgct_sweep = sweep_thresholds(all_bgct_val_probs, all_val_gts, thresholds)

ege_best = max(ege_sweep, key=lambda x: x['dsc'])
bgct_best = max(bgct_sweep, key=lambda x: x['dsc'])

T_EGE_BEST = ege_best['threshold']
T_BGCT_BEST = bgct_best['threshold']

print(f"\nEGE best threshold: {T_EGE_BEST:.2f}  (Val DSC={ege_best['dsc']:.4f})")
print(f"BGCT best threshold: {T_BGCT_BEST:.2f}  (Val DSC={bgct_best['dsc']:.4f})")

# Save CSV
for name, sweep in [('ege', ege_sweep), ('bgct', bgct_sweep)]:
    with open(f'{OUT_DIR}threshold_scan_{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['threshold','dsc','iou','acc','sen','spe','pre'])
        w.writeheader()
        for r in sweep:
            w.writerow({k: f'{r[k]:.6f}' for k in w.fieldnames})

# Threshold curves plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, title, metric in [(axes[0], 'DSC', 'dsc'), (axes[1], 'Sensitivity', 'sen')]:
    ax.plot([r['threshold'] for r in ege_sweep], [r[metric] for r in ege_sweep], 'b-o', ms=4, label='EGE-UNet')
    ax.plot([r['threshold'] for r in bgct_sweep], [r[metric] for r in bgct_sweep], 'r-s', ms=4, label='BGCT-EGE-UNet')
    ax.set_xlabel('Threshold'); ax.set_ylabel(title); ax.legend(); ax.grid(True, alpha=0.3)
axes[1].plot([r['threshold'] for r in ege_sweep], [r['spe'] for r in ege_sweep], 'b--o', ms=4, label='EGE Specificity')
axes[1].plot([r['threshold'] for r in bgct_sweep], [r['spe'] for r in bgct_sweep], 'r--s', ms=4, label='BGCT Specificity')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}threshold_curves.png', dpi=150)
plt.close()
print("Threshold curves saved.")


# ============================================================
# TASK 4: BOUNDARY METRICS ON TEST SET
# ============================================================
print("\n" + "="*60)
print("TASK 4: Boundary Metrics on Test Set")
print("="*60)

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    import subprocess; subprocess.run(['pip', 'install', 'scipy', '-q'])
    from scipy.ndimage import distance_transform_edt

def compute_boundary_metrics(prob, gt, threshold=0.5, bw=3):
    """Compute per-image boundary metrics"""
    pred_bin = (prob >= threshold).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)

    # Standard metrics
    std_m = calc_metrics(pred_bin, gt_bin)

    # Boundary IoU
    gt_b = mask_to_boundary(gt_bin, kernel_size=bw*2+1)
    pd_b = mask_to_boundary(pred_bin, kernel_size=bw*2+1)

    inter = (gt_b * pd_b).sum()
    union = (gt_b + pd_b).clamp(0, 1).sum()
    biou = inter / union if union > 0 else (1.0 if gt_b.sum() == 0 and pd_b.sum() == 0 else 0.0)

    # Boundary F1 with tolerance
    gt_pts = np.argwhere(gt_b > 0.5)
    pd_pts = np.argwhere(pd_b > 0.5)

    if len(gt_pts) == 0 and len(pd_pts) == 0:
        bf1 = 1.0; bp = 1.0; br = 1.0
    elif len(gt_pts) == 0 or len(pd_pts) == 0:
        bf1 = 0.0; bp = 0.0; br = 0.0
    else:
        gt_dist = np.full((IMAGE_SIZE, IMAGE_SIZE), IMAGE_SIZE*2)
        for y, x in gt_pts: gt_dist[y, x] = 0
        distance_transform_edt(gt_dist, output=gt_dist)
        matched_pd = sum(1 for y, x in pd_pts if gt_dist[y, x] <= bw)
        bp = matched_pd / len(pd_pts)
        br = matched_pd / len(gt_pts)
        bf1 = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0

    # HD95 / ASSD
    if len(gt_pts) == 0 and len(pd_pts) == 0:
        hd95 = 0.0; assd = 0.0
    elif len(gt_pts) == 0 or len(pd_pts) == 0:
        hd95 = float(IMAGE_SIZE); assd = float(IMAGE_SIZE)
    else:
        gt_dist = np.full((IMAGE_SIZE, IMAGE_SIZE), IMAGE_SIZE*2, dtype=np.float64)
        pd_dist = np.full((IMAGE_SIZE, IMAGE_SIZE), IMAGE_SIZE*2, dtype=np.float64)
        for y, x in gt_pts: gt_dist[y, x] = 0
        for y, x in pd_pts: pd_dist[y, x] = 0
        distance_transform_edt(gt_dist, output=gt_dist)
        distance_transform_edt(pd_dist, output=pd_dist)
        gt_to_pd = np.array([pd_dist[y, x] for y, x in gt_pts])
        pd_to_gt = np.array([gt_dist[y, x] for y, x in pd_pts])
        all_dists = np.concatenate([gt_to_pd, pd_to_gt])
        hd95 = np.percentile(all_dists, 95)
        assd = (gt_to_pd.mean() + pd_to_gt.mean()) / 2.0

    empty_pred = 1 if pred_bin.sum() == 0 and gt_bin.sum() > 0 else 0

    return {
        **std_m, 'biou': biou, 'bf1': bf1, 'bp': bp, 'br': br,
        'hd95': hd95, 'assd': assd, 'empty_pred': empty_pred
    }

# Compute boundary metrics for both models @0.5
ege_bm_05 = [compute_boundary_metrics(p, g, 0.5) for p, g in zip(all_ege_test_probs, all_test_gts)]
bgct_bm_05 = [compute_boundary_metrics(p, g, 0.5) for p, g in zip(all_bgct_test_probs, all_test_gts)]

# Save per-image CSV
for name, bm_list, files in [('ege', ege_bm_05, test_files), ('bgct', bgct_bm_05, test_files)]:
    with open(f'{OUT_DIR}boundary_metrics_threshold_05_{name}.csv', 'w', newline='') as f:
        fields = ['sample_id','dsc','iou','acc','sen','spe','pre','biou','bp','br','bf1','hd95','assd','empty_pred']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fid, m in zip(files, bm_list):
            row = {'sample_id': fid.replace('.png','').replace('.jpg','')}
            for k in fields[1:]: row[k] = f'{m[k]:.6f}'
            w.writerow(row)

# Summary table
def summarize(bm_list):
    keys = ['dsc','iou','acc','sen','spe','biou','bf1','hd95','assd']
    return {k: (np.mean([m[k] for m in bm_list]), np.std([m[k] for m in bm_list])) for k in keys}

ege_sum = summarize(ege_bm_05)
bgct_sum = summarize(bgct_bm_05)
ege_empty = sum(m['empty_pred'] for m in ege_bm_05)
bgct_empty = sum(m['empty_pred'] for m in bgct_bm_05)

print(f"\nEmpty predictions: EGE={ege_empty}, BGCT={bgct_empty}")
print(f"\n{'Metric':<22} {'EGE@0.5':>18} {'BGCT@0.5':>18} {'Delta':>12}")
print("-"*70)
for k in ['dsc','iou','acc','sen','spe','biou','bf1','hd95','assd']:
    d = bgct_sum[k][0] - ege_sum[k][0]
    print(f"{k:<22} {ege_sum[k][0]:>9.4f}±{ege_sum[k][1]:.4f}  {bgct_sum[k][0]:>9.4f}±{bgct_sum[k][1]:.4f}  {d:>+10.4f}")


# ============================================================
# TASK 5: RE-EVALUATE WITH VAL BEST THRESHOLDS
# ============================================================
print("\n" + "="*60)
print(f"TASK 5: Test Set with Val Best Thresholds (EGE={T_EGE_BEST:.2f}, BGCT={T_BGCT_BEST:.2f})")
print("="*60)

ege_bm_best = [compute_boundary_metrics(p, g, T_EGE_BEST) for p, g in zip(all_ege_test_probs, all_test_gts)]
bgct_bm_best = [compute_boundary_metrics(p, g, T_BGCT_BEST) for p, g in zip(all_bgct_test_probs, all_test_gts)]

ege_best_sum = summarize(ege_bm_best)
bgct_best_sum = summarize(bgct_bm_best)

print(f"\nTable 1: threshold=0.5")
print(f"{'Metric':<22} {'EGE@0.5':>18} {'BGCT@0.5':>18}")
print("-"*60)
for k in ['dsc','iou','sen','spe','biou','bf1','hd95','assd']:
    print(f"{k:<22} {ege_sum[k][0]:>9.4f}±{ege_sum[k][1]:.4f}  {bgct_sum[k][0]:>9.4f}±{bgct_sum[k][1]:.4f}")

print(f"\nTable 2: own val-best threshold")
print(f"{'Metric':<22} {'EGE@'+str(T_EGE_BEST):>18} {'BGCT@'+str(T_BGCT_BEST):>18}")
print("-"*60)
for k in ['dsc','iou','sen','spe','biou','bf1','hd95','assd']:
    print(f"{k:<22} {ege_best_sum[k][0]:>9.4f}±{ege_best_sum[k][1]:.4f}  {bgct_best_sum[k][0]:>9.4f}±{bgct_best_sum[k][1]:.4f}")


# ============================================================
# TASK 2: VISUALIZATION (PROBABILITY MAPS)
# ============================================================
print("\n" + "="*60)
print("Generating probability visualizations...")
print("="*60)

# Select interesting samples from test set
test_indices = []
for i in range(len(all_test_gts)):
    ege_dsc = dice_score((all_ege_test_probs[i] >= 0.5).astype(np.float32), (all_test_gts[i] > 0.5).astype(np.float32))
    bgct_dsc = dice_score((all_bgct_test_probs[i] >= 0.5).astype(np.float32), (all_test_gts[i] > 0.5).astype(np.float32))
    dsc_diff = abs(ege_dsc - bgct_dsc)
    gt_area = (all_test_gts[i] > 0.5).sum()
    test_indices.append((i, dsc_diff, gt_area, ege_dsc, bgct_dsc))

# Pick diverse samples
selected = []
# Top DSC differences
by_diff = sorted(test_indices, key=lambda x: -x[1])
selected.extend([x[0] for x in by_diff[:4]])
# Small lesions
by_small = sorted(test_indices, key=lambda x: x[2])
selected.extend([x[0] for x in by_small if x[0] not in selected][:3])
# Large lesions
by_large = sorted(test_indices, key=lambda x: -x[2])
selected.extend([x[0] for x in by_large if x[0] not in selected][:3])
# Random
np.random.seed(42)
remaining = [i for i in range(len(all_test_gts)) if i not in selected]
selected.extend(np.random.choice(remaining, min(6, len(remaining)), replace=False).tolist())

n_samples = len(selected)
cols = 6
rows = n_samples
fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
plt.subplots_adjust(wspace=0.02, hspace=0.25)

for row_idx, sample_idx in enumerate(selected):
    img_np, img_t = preprocess(os.path.join(IMAGE_DIR, test_files[sample_idx]))
    gt = (all_test_gts[sample_idx] > 0.5).astype(np.float32)
    ege_p = all_ege_test_probs[sample_idx]
    bgct_p = all_bgct_test_probs[sample_idx]
    diff = bgct_p - ege_p

    for col_idx, (title, data, cmap, vmin, vmax) in enumerate([
        ('Input', img_np, None, None, None),
        ('GT', gt, 'gray', 0, 1),
        ('EGE Prob', ege_p, 'hot', 0, 1),
        ('BGCT Prob', bgct_p, 'hot', 0, 1),
        ('Diff (B-E)', diff, 'RdBu_r', -0.5, 0.5),
        ('BGCT Bin', (bgct_p >= 0.5).astype(np.float32), 'gray', 0, 1),
    ]):
        ax = axes[row_idx][col_idx] if rows > 1 else axes[col_idx]
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

plt.savefig(f'{OUT_DIR}vis/probability_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Probability map visualization saved ({n_samples} samples).")


# ============================================================
# FINAL CONCLUSIONS
# ============================================================
print("\n" + "="*60)
print("FINAL DIAGNOSIS")
print("="*60)

dsc_delta = bgct_sum['dsc'][0] - ege_sum['dsc'][0]
biou_delta = bgct_sum['biou'][0] - ege_sum['biou'][0]
hd95_delta = bgct_sum['hd95'][0] - ege_sum['hd95'][0]
gt_interior_diff = bgct_stats['interior'][0] - ege_stats['interior'][0]
gt_boundary_diff = bgct_stats['boundary'][0] - ege_stats['boundary'][0]
gt_bg_diff = bgct_stats['background'][0] - ege_stats['background'][0]

print(f"\nEGE val-best threshold: {T_EGE_BEST:.2f}")
print(f"BGCT val-best threshold: {T_BGCT_BEST:.2f}")
print(f"\n@0.5: DSC Δ={dsc_delta:+.4f}, BIoU Δ={biou_delta:+.4f}, HD95 Δ={hd95_delta:+.4f}")
print(f"GT interior prob: EGE={ege_stats['interior'][0]:.4f}, BGCT={bgct_stats['interior'][0]:.4f}  Δ={gt_interior_diff:+.4f}")
print(f"GT boundary prob: EGE={ege_stats['boundary'][0]:.4f}, BGCT={bgct_stats['boundary'][0]:.4f}  Δ={gt_boundary_diff:+.4f}")
print(f"GT background prob: EGE={ege_stats['background'][0]:.4f}, BGCT={bgct_stats['background'][0]:.4f}  Δ={gt_bg_diff:+.4f}")

if abs(dsc_delta) < 0.005 and abs(biou_delta) < 0.01 and abs(hd95_delta) < 1.0:
    print("\nFINAL CONCLUSION: D - Current BGCT design shows NO significant improvement on boundary metrics or DSC at threshold=0.5.")
elif biou_delta > 0.01 or hd95_delta < -1.0:
    print("\nFINAL CONCLUSION: C - BGCT shows boundary improvement but limited DSC gain.")
else:
    print("\nFINAL CONCLUSION: B - BGCT mainly shifts the precision-recall tradeoff.")
