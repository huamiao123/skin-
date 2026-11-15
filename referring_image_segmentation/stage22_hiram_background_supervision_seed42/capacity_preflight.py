"""Actual batch64 plus resident validation cache and all three models."""
from common import *
from models import NAMES, DecoderProbe, FrozenEncoder, region_targets, auxiliary_numerators, auxiliary_loss, segmentation_loss
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode


def main():
    setup()
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    data=ImageData(42)
    encoder=FrozenEncoder().cuda().eval()
    val_features,val_targets=encode(encoder,data,'val',64)
    models={n:DecoderProbe('hiram').cuda().train() for n in NAMES}
    opts={n:torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.001,weight_decay=.01)
          for n,m in models.items()}
    order,matrices=data.geometry(1)
    images,gt=data.batch('train',order[:64],matrices)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(images)
    targets=region_targets(gt);denoms=[x.sum()+1e-8 for x in targets]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for name,model in models.items():
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,regions=model(features,return_regions=True)
        loss=segmentation_loss(logits,gt)+auxiliary_loss(name,auxiliary_numerators(regions,targets),denoms)
        assert torch.isfinite(loss);loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
        opts[name].step();opts[name].zero_grad(set_to_none=True)
        del logits,regions,loss
    torch.cuda.synchronize()
    assert len(val_targets)==808 and len(val_features)==4
    result=dict(passed=True,gpu=torch.cuda.get_device_name(),effective_batch=64,train_micro_batch=64,
        encoder_batch=64,evaluation_batch=64,model_count=3,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        three_model_batch_seconds=time.perf_counter()-start,
        note='real input, all three models, resident data and validation feature cache, full batch64')
    dump(ROOT/'capacity_preflight.json',result)
    print('CAPACITY',json.dumps(result),flush=True)


if __name__=='__main__':main()
