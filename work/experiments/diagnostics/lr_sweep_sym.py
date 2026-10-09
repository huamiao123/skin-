import sys, numpy as np, torch
sys.path.insert(0, '/root/pure_transformer_unet')
from torch.utils.data import DataLoader, Subset
import train as T

cfg = T.PTUConfig
train_ds = T.NPY_datasets(cfg.data_path, cfg, train=True)
loader = DataLoader(Subset(train_ds, list(range(100))), batch_size=16,
                    shuffle=True, num_workers=4, pin_memory=True)

def make_model():
    return T.build_model(cfg, [2,2,12,2], base_embed=96, decoder_depths=[12,2,2])

def run(lr):
    torch.manual_seed(42); np.random.seed(42)
    model = make_model().cuda()
    criterion = T.BceDiceLoss(wb=1, wd=1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    for ep in range(1, 41):
        model.train(); losses = []
        for img, msk in loader:
            img = img.cuda().float(); msk = msk.cuda().float()
            out = torch.sigmoid(model(img))
            loss = criterion(out, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    model.eval(); dscs = []
    with torch.no_grad():
        for img, msk in loader:
            p = (torch.sigmoid(model(img.cuda().float())).cpu() > 0.5).float()
            g = (msk > 0.5).float()
            inter = (p*g).sum(dim=(1,2,3)); union = (p+g).sum(dim=(1,2,3)) + 1e-8
            dscs.append((2*inter/union).numpy())
    print(f'lr={lr:.1e}: loss={np.mean(losses):.4f}, DSC={np.concatenate(dscs).mean():.4f}')
    del model; torch.cuda.empty_cache()

for lr in [5e-5, 1e-4, 3e-4]:
    run(lr)
