"""Test true batch64 with real images and a resident full validation cache."""
from common import *
from models import RegionProbe,segmentation_loss,region_targets,region_loss_numerators
from stage4_fusion.models import EncoderPair
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode


def main():
    setup()
    data=ImageData(42)
    encoder=EncoderPair().cuda().eval()
    probe=RegionProbe('hiram').cuda().train()
    val_features,val_targets=encode(encoder,data,'val',64)
    order,matrices=data.geometry(1)
    images,gt=data.batch('train',order[:64],matrices)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(images)
    encoder_peak=torch.cuda.max_memory_allocated()/1024**3
    targets=region_targets(gt)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):logits,regions=probe(features,return_regions=True)
    num=region_loss_numerators(regions,targets)
    loss=segmentation_loss(logits,gt)+num[0]/(targets[0].sum()+1e-8)+num[1]/(targets[1].sum()+1e-8)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters() if p.requires_grad)
    torch.cuda.synchronize()
    record=dict(passed=True,gpu=torch.cuda.get_device_name(),batch=64,validation_cache_samples=len(val_targets),
        encoder_peak_allocated_gib=encoder_peak,training_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        hiram_forward_backward_seconds=time.perf_counter()-start,finite_loss=float(loss.item()))
    dump(ROOT/'capacity_preflight.json',record)
    print('CAPACITY PREFLIGHT PASSED',json.dumps(record),flush=True)


if __name__=='__main__':main()
