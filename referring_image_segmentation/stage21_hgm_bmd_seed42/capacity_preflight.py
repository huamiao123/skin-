"""Real data, cached validation features and exact batch 64 training step."""
from common import *
from hgm_model import MambaVisionHGMProbe, FrozenEncoder, segmentation_loss
from stage3_confirmation.data import ImageData
from engine import encode_val, extract


def main():
    setup()
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    data=ImageData(42)
    encoder=FrozenEncoder().cuda().eval()
    val_features,val_targets=encode_val(encoder,data,64)
    model=MambaVisionHGMProbe().cuda().train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001,weight_decay=.01)
    order,matrices=data.geometry(1)
    images,target=data.batch('train',order[:64],matrices)
    features=extract(encoder,images,64)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(features)
    loss=segmentation_loss(logits,target)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    torch.cuda.synchronize()
    result=dict(passed=True,train_batch=64,effective_batch=64,encoder_batch=64,evaluation_batch=64,
        forward_backward_update_seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        note='real features, validation cache, resident dataset, encoder, optimizer; no gradient accumulation')
    assert len(val_targets)==808 and len(val_features)==4
    dump(ROOT/'capacity_preflight.json',result)
    print('CAPACITY',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
