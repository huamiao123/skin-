"""Measure real GPU capacity, keeping validation features resident."""
from common import *
import gc
from models import DecoderProbe,FrozenEncoder,segmentation_loss,region_targets,region_loss_numerators
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode


def main():
    setup()
    data=ImageData(42)
    encoder=FrozenEncoder().cuda().eval()
    probe=DecoderProbe('hiram').cuda().train()
    # Test whole encoder batch with validation cache already resident.
    # The encoder works on no-grad features and can be split independently.
    val_features,val_targets=encode(encoder,data,'val',32)
    order,matrices=data.geometry(1)
    images,gt=data.batch('train',order[:64],matrices)
    torch.cuda.empty_cache()
    encoder_batch=None
    encoder_attempts=[]
    for batch in (64,32,16,8):
        try:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=encoder(images[:batch])
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()/1024**3
            del y
            encoder_attempts.append(dict(batch=batch,peak_gib=peak,passed=True))
            if peak<8.8:
                encoder_batch=batch
                break
        except torch.cuda.OutOfMemoryError:
            encoder_attempts.append(dict(batch=batch,passed=False,error='CUDA out of memory'))
            gc.collect();torch.cuda.empty_cache()
    assert encoder_batch
    parts=None
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,64,encoder_batch):
            z=encoder(images[start:start+encoder_batch])
            if parts is None:parts=[[] for _ in z]
            for dst,v in zip(parts,z):dst.append(v.detach())
    features=[torch.cat(v,0) for v in parts]
    targets=region_targets(gt)
    attempts=[]
    selected=None
    for batch in (64,32,16,8):
        try:
            probe.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,regions=probe([x[:batch] for x in features],return_regions=True)
            num=region_loss_numerators(regions,[x[:batch] for x in targets])
            loss=segmentation_loss(logits,gt[:batch])+num[0]/(targets[0][:batch].sum()+1e-8)+num[1]/(targets[1][:batch].sum()+1e-8)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters() if p.requires_grad)
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated()/1024**3
            elapsed=time.perf_counter()-start
            attempts.append(dict(batch=batch,peak_gib=peak,seconds=elapsed,passed=True))
            del logits,regions,num,loss
            probe.zero_grad(set_to_none=True)
            if peak<8.8:
                selected=batch
                break
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(batch=batch,passed=False,error='CUDA out of memory'))
            probe.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
    assert selected is not None,attempts
    record=dict(passed=True,gpu=torch.cuda.get_device_name(),effective_batch=64,
        train_micro_batch=selected,encoder_batch=encoder_batch,evaluation_batch=selected,
        validation_cache_samples=len(val_targets),encoder_attempts=encoder_attempts,training_attempts=attempts,
        max_accepted_peak_gib=8.8,partial_batch_semantics='weighted segmentation and globally normalized region KL; optimizer step once per 64')
    dump(ROOT/'capacity_preflight.json',record)
    print('CAPACITY PREFLIGHT PASSED',json.dumps(record),flush=True)


if __name__=='__main__':main()
