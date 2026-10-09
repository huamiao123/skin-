import sys, importlib.util, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
sys.argv = ['overfit4']
import train_swin_isic as T

cfg = T.SwinConfig
cfg.epochs = 40
cfg.train_transformer = transforms.Compose([
    T.myNormalize('isic18', train=True), T.myToTensor(), T.myResize(256, 256)])
torch.manual_seed(42); np.random.seed(42)

train_ds = T.NPY_datasets(cfg.data_path, cfg, train=True)
loader = DataLoader(Subset(train_ds, list(range(100))), batch_size=16,
                    shuffle=True, num_workers=4, pin_memory=True)

def make_model():
    return T.SwinTransformerSys(img_size=256, patch_size=4, in_chans=3, num_classes=1,
        embed_dim=96, depths=[2,2,2,2], depths_decoder=[2,2,2,1], num_heads=[3,6,12,24],
        window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, ape=False, patch_norm=True, final_upsample='expand_first')

def run(name, opt_fn, lr, epochs=40, wd=0.0):
    def _make_opt(params):
        return opt_fn(params, lr=lr, weight_decay=wd)
    model = make_model().cuda()
    criterion = T.BceDiceLoss(wb=1, wd=1)
    opt = _make_opt(model.parameters())
    gn_first = None
    for ep in range(1, epochs+1):
        model.train(); losses = []
        for img, msk in loader:
            img = img.cuda().float(); msk = msk.cuda().float()
            out = torch.sigmoid(model(img))
            loss = criterion(out, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if gn_first is None:
                gn_first = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        if ep % 10 == 0:
            model.eval(); dscs = []
            with torch.no_grad():
                for img, msk in loader:
                    p = (torch.sigmoid(model(img.cuda().float())).cpu() > 0.5).float()
                    g = (msk > 0.5).float()
                    inter = (p*g).sum(dim=(1,2,3)); union = (p+g).sum(dim=(1,2,3)) + 1e-8
                    dscs.append((2*inter/union).numpy())
            print(f'[{name}] ep {ep}: loss={np.mean(losses):.4f} DSC={np.concatenate(dscs).mean():.4f}')
    print(f'[{name}] first-step grad norm: {gn_first:.4f}')

run('AdamW 3e-4', torch.optim.AdamW, 3e-4)
run('AdamW 1e-2', torch.optim.AdamW, 1e-2)
run('SGD 0.05 mom0.9 wd1e-4', lambda p, lr, weight_decay: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=weight_decay), 0.05, wd=1e-4)
