"""Probe real CUDA peak with the validation cache and trainable models resident."""
from common import *
from models import DecoderProbe, segmentation_loss


def main():
    setup()
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    # Reserve the same dtype and shapes as the actual BF16 validation cache and
    # resident uint8 image/mask data. Synthetic tensors are for capacity only.
    reserve=[torch.empty((808,c,s,s),device='cuda',dtype=torch.bfloat16) for c,s in zip(CHANNELS,(64,32,16,8))]
    reserve += [torch.empty((808,1,256,256),device='cuda'),
                torch.empty((2694,4,256,256),device='cuda',dtype=torch.uint8)]
    models=[DecoderProbe(n).cuda().train() for n in ('paired_baseline','gate_only','axis_bridge')]
    checks=[]
    selected=None
    for n in (64,32,16,8):
        feats=target=logits=loss=None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            feats=[torch.randn(n,c,s,s,device='cuda',dtype=torch.bfloat16) for c,s in zip(CHANNELS,(64,32,16,8))]
            target=torch.rand(n,1,256,256,device='cuda')
            start=time.perf_counter()
            for m in models:
                m.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits=m(feats)
                loss=segmentation_loss(logits,target)
                loss.backward()
                assert torch.isfinite(loss)
                m.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            checks.append(dict(micro_batch=n,passed=True,peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                               seconds_for_three_forward_backward=time.perf_counter()-start))
            selected=n
            break
        except torch.cuda.OutOfMemoryError:
            checks.append(dict(micro_batch=n,passed=False,reason='CUDA out of memory'))
            for m in models:m.zero_grad(set_to_none=True)
        finally:
            del feats,target,logits,loss
            torch.cuda.empty_cache()
    assert selected is not None
    result=dict(passed=True,effective_batch=64,train_micro_batch=selected,encoder_batch=64,
                evaluation_batch=64,checks=checks,
                accumulation='size-weighted per-image BCE and Dice; one optimizer update per effective batch',
                note='Peak includes reserved validation features and uint8 images; not a formal result')
    dump(ROOT/'capacity_preflight.json',result)
    print('CAPACITY',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
