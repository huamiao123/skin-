"""Inspect pinned HGM decoder blocks on 256-input scales, without training."""
from common import *
import ast
import copy
from torch import nn
from torch.cuda.amp import autocast
from mamba_ssm import Mamba

SRC=ROOT/'references/hgm_decoder_source.py'
EXPECTED='8c783b14dcedd13d98c433213d2a8f33c8ea0aec5bb4fa8281914b2d73c7a8da'


def main():
    setup()
    assert sha(SRC)==EXPECTED
    tree=ast.parse(SRC.read_text())
    names={'BiPixelMambaLayer','BiWindowMambaLayer'}
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names]
    assert {n.name for n in classes}==names
    module=ast.Module(body=classes,type_ignores=[])
    ast.fix_missing_locations(module)
    namespace=globals().copy()
    exec(compile(module,str(SRC),'exec'),namespace)
    # Reproduce the two p values constructed by the upstream BMD, rather than
    # guessing a window configuration from the paper diagram.
    records=[]
    for i,side in enumerate((64,128)):
        pixel_arg=2**((2-i+1)//2-1)
        window_arg=2**((2-i+1)//2)//2
        pixel=namespace['BiPixelMambaLayer'](32,pixel_arg)
        window=namespace['BiWindowMambaLayer'](32,window_arg)
        records.append(dict(stage=i,side=side,pixel_partition=pixel.p,
                            pixel_partition_applied=(side%pixel.p==0),
                            window_pool_factor=window.p,window_reduces_resolution=window.p>1 and side%window.p==0))
    del pixel,window
    # Test the original blocks with a small 32x32 feature, using the installed
    # Mamba implementation. Capture BF16 compatibility separately from FP32.
    results={}
    for name in ('BiPixelMambaLayer','BiWindowMambaLayer'):
        layer=namespace[name](32,1).cuda().train()
        bf16_error=None
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                y=layer(torch.randn(1,32,32,32,device='cuda',dtype=torch.bfloat16))
            torch.cuda.synchronize()
            bf16_ok=bool(torch.isfinite(y).all())
            del y
        except Exception as exc:
            bf16_ok=False
            bf16_error=f'{type(exc).__name__}: {str(exc)[:500]}'
        layer.zero_grad(set_to_none=True)
        x=torch.randn(1,32,32,32,device='cuda',requires_grad=True)
        try:
            y=layer(x)
            assert torch.isfinite(y).all()
            y.square().mean().backward()
            missing=[n for n,p in layer.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not missing,missing
            fp32_ok=True
            fp32_error=None
        except Exception as exc:
            fp32_ok=False
            fp32_error=f'{type(exc).__name__}: {str(exc)[:500]}'
        results[name]=dict(bf16_direct_passed=bf16_ok,bf16_error=bf16_error,
                           fp32_forward_backward_passed=fp32_ok,fp32_error=fp32_error)
        del layer,x
        torch.cuda.empty_cache()
    report=dict(source_sha256=EXPECTED,revision='49ae05afc9785dbbc5a303d39e63588b068e5328',
                gpu=torch.cuda.get_device_name(),mamba_version='1.2.2 (HGM requirements: 2.2.4)',
                bmd_256_resolution=records,isolated_32x32_blocks=results,
                note='Isolated upstream blocks only; not full HGM or a trained experiment')
    dump(ROOT/'analysis/hgm_diagnostic.json',report)
    print('HGM_DIAGNOSTIC',json.dumps(report),flush=True)


if __name__=='__main__':main()
