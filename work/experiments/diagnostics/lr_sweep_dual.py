import sys; sys.path.insert(0, '/root/EGE-UNet-main')
import sys, numpy as np, torch
sys.argv = ['sweep']
import train as T
from models.ege_dual import EGEDualUNet
from torch.utils.data import DataLoader, Subset

cfg = T.setting_config
train_ds = T.NPY_datasets(cfg.data_path, cfg, train=True)
loader = DataLoader(Subset(train_ds, list(range(100))), batch_size=16,
                    shuffle=True, num_workers=4, pin_memory=True)

def make_model():
    return EGEDualUNet(num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                       bridge=True, gt_ds=True)

def run(lr):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model().cuda()
    criterion = T.BceDiceLoss(wb=1, wd=1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    for ep in range(1, 41):
        model.train(); losses = []
        for img, msk in loader:
            img = img.cuda().float(); msk = msk.cuda().float()
            ds, out = model(img)
            loss = T.GT_BceDiceLoss(wb=1, wd=1)(ds, out, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    model.eval(); dscs = []
    with torch.no_grad():
        for img, msk in loader:
            _, out = model(img.cuda().float())
            p = (out.cpu() > 0.5).float()
            g = (msk > 0.5).float()
            inter = (p*g).sum(dim=(1,2,3)); union = (p+g).sum(dim=(1,2,3)) + 1e-8
            dscs.append((2*inter/union).numpy())
    g = {k: round(v.item(), 3) for k, v in model._gamma_stats.items()}
    print(f'lr={lr:.1e}: loss={np.mean(losses):.4f}, DSC={np.concatenate(dscs).mean():.4f}, gamma={g}')
    del model; torch.cuda.empty_cache()

for lr in [1e-3, 3e-4, 1e-4]:
    run(lr)
