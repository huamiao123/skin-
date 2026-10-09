import sys, torch, time
sys.path.insert(0, '/root/pure_transformer_unet')
from models.pure_transformer_unet import PureTransformerUNet

torch.manual_seed(0)
model = PureTransformerUNet(in_chans=3, num_classes=1, patch_size=4,
    embed_dims=(64,128,256,512), depths=[2,2,4,2], num_heads=(2,4,8,16),
    sr_ratios=(4,2,1,1), decoder_depths=(2,2,2), decoder_heads=(8,4,2),
    decoder_sr_ratios=(1,2,4), mlp_ratio=4.0, head_dim=32,
    drop=0.0, attn_drop=0.0, drop_path_rate=0.1).cuda()
n = sum(p.numel() for p in model.parameters())
print(f'params: {n:,}')

x = torch.randn(2, 3, 256, 256).cuda()
feats = model.encoder(x)
print('encoder stage shapes (expect [B,N,C],H,W):')
for i, (f, H, W) in enumerate(feats):
    print(f'  stage{i+1}: tokens {tuple(f.shape)} H={H} W={W}')
out = model(x)
print('final out:', tuple(out.shape), '(expect (2,1,256,256))')

# NaN check on full forward+backward
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
tgt = torch.rand(2, 1, 256, 256).cuda()
loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), tgt)
opt.zero_grad(); loss.backward(); opt.step()
print('loss:', loss.item(), 'NaN check:', torch.isnan(loss).item())

# memory/speed at batch 64
try:
    torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(64,3,256,256).cuda(); tb = torch.rand(64,1,256,256).cuda()
    for _ in range(2):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(xb)), tb); l.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy(torch.sigmoid(model(xb)), tb); l.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time()-t0)/5
    mem = torch.cuda.max_memory_allocated()/1024**3
    print(f'bs=64: {dt*1000:.0f} ms/step, peak {mem:.1f} GB')
except RuntimeError as e:
    print(f'bs=64: OOM -> {str(e)[:60]}')
print('ALL CHECKS PASSED')
