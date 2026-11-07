"""Validate the adapted SegMAN numerical kernels and actual feature contract."""
from segman_model import *
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder
from stage3_confirmation.models import segmentation_loss, tensor_hash
from stage3_confirmation.data import ImageData


def maxdiff(a,b):return float((a.detach().float()-b.detach().float()).abs().max())


def main():
    setup()
    assert torch.cuda.is_bf16_supported()
    result=dict(gpu=torch.cuda.get_device_name(),seed=42,segman_revision='9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8',
                source_sha256=sha(SRC),cross_scan_source_sha256=sha(CSM),
                method='SegMAN-small decoder adapted with standalone framework shims and standard selective scan',
                official_reproduction=False,precision='BF16 autocast with FP32 loss')
    # Four-direction official Triton scan, compared to a direct tensor formula.
    x=torch.randn(2,3,8,8,device='cuda',requires_grad=True)
    x2=x.detach().clone().requires_grad_()
    direct=torch.stack([x2.flatten(2),x2.transpose(2,3).flatten(2),
                        x2.flatten(2).flip(-1),x2.transpose(2,3).flatten(2).flip(-1)],dim=1)
    actual=CrossScanTriton.apply(x)
    result['cross_scan_max_abs']=maxdiff(actual,direct)
    weight=torch.randn_like(actual)
    (actual*weight).sum().backward()
    (direct*weight).sum().backward()
    result['cross_scan_grad_max_abs']=maxdiff(x.grad,x2.grad)
    assert result['cross_scan_max_abs']<1e-6 and result['cross_scan_grad_max_abs']<1e-6
    del x,x2,direct,actual,weight

    # Standard CUDA scan vs independent PyTorch reference, grouped B/C as in
    # SegMAN. This does not compare against the unavailable oflex extension.
    shapes=[(2,12,16),(2,12,16),(12,1),(2,4,1,16),(2,4,1,16),(12,),(12,)]
    values=[torch.randn(*s,device='cuda') for s in shapes]
    values[2]=-values[2].exp()
    values[5]=values[5].abs()
    left=[v.detach().clone().requires_grad_() for v in values]
    right=[v.detach().clone().requires_grad_() for v in values]
    out=selective_scan_fn(*left[:6],delta_bias=left[6],delta_softplus=True)
    ref=selective_scan_ref(*right[:6],delta_bias=right[6],delta_softplus=True)
    result['standard_scan_reference_max_abs']=maxdiff(out,ref)
    w=torch.randn_like(out)
    (out*w).sum().backward()
    (ref*w).sum().backward()
    result['standard_scan_reference_gradient_max_abs']=max(maxdiff(a.grad,b.grad) for a,b in zip(left,right))
    assert result['standard_scan_reference_max_abs']<2e-4
    assert result['standard_scan_reference_gradient_max_abs']<1e-3
    del values,left,right,out,ref,w

    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    original=tensor_hash(encoder)
    assert original==json.loads((ROOT/'preflight.json').read_text())['encoder_sha256']
    images,target=data.batch('train',torch.arange(4,device='cuda'))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        features=encoder(images)
    model=MambaVisionSegMAN().cuda().train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(features)
    assert list(logits.shape)==[4,1,256,256] and torch.isfinite(logits).all()
    loss=segmentation_loss(logits,target)
    assert torch.isfinite(loss)
    loss.backward()
    failed=[n for n,p in model.named_parameters() if p.requires_grad and
            (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not failed,failed
    assert tensor_hash(encoder)==original and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    result.update(real_feature_shapes=[list(x.shape) for x in features],sample_loss=float(loss),
                  logits_shape=list(logits.shape),encoder_unchanged=True,
                  encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in model.parameters()))
    result['passed']=True
    dump(Path(__file__).parent/'preflight.json',result)
    print('SEGMAN PREFLIGHT PASSED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
