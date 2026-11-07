"""Test exact batch 64: SyncBN precludes claiming smaller microbatches equivalent."""
from segman_model import *
from stage3_confirmation.models import segmentation_loss


def main():
    setup()
    pre=json.loads((Path(__file__).parent/'preflight.json').read_text())
    assert pre['passed']
    reserve=[torch.empty((808,c,s,s),device='cuda',dtype=torch.bfloat16) for c,s in zip(CHANNELS,(64,32,16,8))]
    reserve.extend([torch.empty((808,1,256,256),device='cuda'),
                    torch.empty((2694,4,256,256),device='cuda',dtype=torch.uint8)])
    model=MambaVisionSegMAN().cuda().train()
    feats=[torch.randn(64,c,s,s,device='cuda',dtype=torch.bfloat16) for c,s in zip(CHANNELS,(64,32,16,8))]
    targets=torch.rand(64,1,256,256,device='cuda')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(feats)
    loss=segmentation_loss(logits,targets)
    assert torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize()
    result=dict(passed=True,train_batch=64,effective_batch=64,encoder_batch=64,evaluation_batch=64,
                forward_backward_seconds=time.perf_counter()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                note='BF16, includes reserved validation features and resident image data; no microbatching')
    dump(Path(__file__).parent/'capacity_preflight.json',result)
    print('SEGMAN CAPACITY',json.dumps(result),flush=True)


if __name__=='__main__':main()
