import sys
import time
import torch

sys.path.insert(0, '/root/EGE-UNet-main')
sys.path.insert(0, '/root')
sys.path.insert(1, '/root/Swin-Unet-main')
sys.path.insert(2, '/root/Pytorch-UNet-master')

from thop import profile

from models.egeunet import EGEUNet, EGEWaveUNet
from models.ege_dual import EGEDualUNet
from pure_transformer_unet.models.pure_transformer_unet import PureTransformerUNet
from pure_transformer_unet.models.ptu_dual import PTUDualUNet
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from unet.unet_model import UNet


def bench(name, model, bs, is_ege_tuple=False):
    model = model.cuda().eval()
    x = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(x,), verbose=False)
    torch.cuda.reset_peak_memory_stats()
    xb = torch.randn(bs, 3, 256, 256).cuda()
    tgt = torch.rand(bs, 1, 256, 256).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        opt.zero_grad()
        out = model(xb)
        if is_ege_tuple:
            out = out[1]
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), tgt)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad()
        out = model(xb)
        if is_ege_tuple:
            out = out[1]
        loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(out), tgt)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f'{name}: params={params/1e6:.2f}M FLOPs={flops/1e9:.2f}G '
          f'bs{bs} {dt*1000:.0f}ms/step peak={mem:.1f}GB')
    del model
    torch.cuda.empty_cache()


bench('EGE baseline', EGEUNet(num_classes=1, input_channels=3,
      c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True), 64, True)
bench('EGE-Wave v1', EGEWaveUNet(num_classes=1, input_channels=3,
      c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True,
      wave_mode='full', fusion_type='scalar'), 64, True)
bench('EGE-Dual difflr', EGEDualUNet(num_classes=1, input_channels=3,
      c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True), 64, True)
bench('PTU [2,2,4,2]', PureTransformerUNet(in_chans=3, num_classes=1,
      patch_size=4, embed_dims=(64,128,256,512), depths=[2,2,4,2],
      num_heads=(2,4,8,16), sr_ratios=(4,2,1,1), decoder_depths=(2,2,2),
      decoder_heads=(8,4,2), decoder_sr_ratios=(1,2,4), mlp_ratio=4.0,
      head_dim=32, drop=0.0, attn_drop=0.0, drop_path_rate=0.1), 64)
bench('PTU-Dual', PTUDualUNet(in_chans=3, num_classes=1, patch_size=4,
      embed_dims=(64,128,256,512), depths=[2,2,4,2], num_heads=(2,4,8,16),
      sr_ratios=(4,2,1,1), decoder_depths=(2,2,2), decoder_heads=(8,4,2),
      decoder_sr_ratios=(1,2,4), mlp_ratio=4.0, head_dim=32,
      drop=0.0, attn_drop=0.0, drop_path_rate=0.1,
      cnn_channels=(32,64,128,256)), 64)
bench('Swin [2,2,6,2] pretrain-cfg', SwinTransformerSys(img_size=256,
      patch_size=4, in_chans=3, num_classes=1, embed_dim=96,
      depths=[2,2,6,2], depths_decoder=[2,2,2,1], num_heads=[3,6,12,24],
      window_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.0,
      attn_drop_rate=0.0, drop_path_rate=0.2, ape=False, patch_norm=True,
      final_upsample='expand_first'), 32)
bench('U-Net', UNet(n_channels=3, n_classes=1, bilinear=False), 32)
