#!/bin/bash
cd /root/pure_transformer_unet

echo "=== PTU-SYM prep: memory/speed benchmark ==="
python3 -c "
import sys, torch, time
sys.argv = ['x']
import train as T
model = T.build_model(T.PTUConfig, [2,2,12,2], base_embed=96, decoder_depths=[12,2,2]).cuda()
n = sum(p.numel() for p in model.parameters())
print(f'params: {n:,}')
try:
    torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(32,3,256,256).cuda(); tb = torch.rand(32,1,256,256).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(xb)), tb); l.backward(); opt.step()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(4):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(xb)), tb); l.backward(); opt.step()
    torch.cuda.synchronize()
    print(f'bs=32: {(time.time()-t0)/4*1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated()/1024**3:.1f} GB')
except RuntimeError as e:
    print('bs=32 OOM:', str(e)[:80])
" 2>&1 | grep -vE "FutureWarning|warnings.warn"

echo "=== PTU-SYM lr screening (100-img proxy) ==="
cat > /tmp/lr_sweep_sym.py << 'EOF'
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
EOF
python /tmp/lr_sweep_sym.py 2>&1 | grep -E "^lr="

echo "=== PTU-SYM full training (lr=1e-4 default) ==="
python3 train.py --depth3 12 --ddec3 12 --embed 96 --lr 1e-4 --tag sym12 > /root/ptu_sym12_run.log 2>&1
echo "=== PTU-SYM12 COMPLETE ===" >> /root/ptu_sym12_run.log
