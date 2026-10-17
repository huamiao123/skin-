"""Check paired initialization, both branches' gradients and full-image inference."""
import json
import time
import torch
import timm
from models import ROOT, EncoderPair, build_probes, select_features, tensor_hash, segmentation_loss, NAMES


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=EncoderPair().cuda()
    hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():assert h==json.loads((ROOT.parent/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    probes=build_probes(42)
    initial={n:tensor_hash(p) for n,p in probes.items()}
    all_params=[]
    for p in probes.values():all_params.extend(map(id,p.parameters()))
    assert len(all_params)==len(set(all_params))
    for d in ('light','swin'):
        assert tensor_hash(probes[f'fusion_{d}'].dino_adapter)==tensor_hash(probes[f'dinov3_{d}'].adapter)
        assert tensor_hash(probes[f'fusion_{d}'].mamba_adapter)==tensor_hash(probes[f'vmamba_{d}'].adapter)
        assert len({tensor_hash(probes[f'{n}_{d}'].decoder) for n in ('dinov3','vmamba','fusion')})==1
    assert tensor_hash(probes['fusion_light'].fusion)==tensor_hash(probes['fusion_swin'].fusion)
    params={}
    for n,p in probes.items():
        names=('dinov3','vmamba') if n.startswith('fusion_') else (n.split('_')[0],)
        enc_count=sum(getattr(encoder,k).audit['params'] for k in names)
        train_count=sum(t.numel() for t in p.parameters())
        params[n]=dict(encoder=enc_count,trainable=train_count,total=enc_count+train_count,
            decoder=sum(t.numel() for t in p.decoder.parameters()))
    x=torch.randn(64,3,256,256,device='cuda');target=torch.rand(2,1,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(x[:2])
    for n,p in probes.items():
        p.cuda().train()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(select_features(n,features))
        loss=segmentation_loss(logits,target);loss.backward()
        assert logits.shape==target.shape and all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in p.parameters()),n
        p.zero_grad(set_to_none=True)
        if n.startswith('fusion_'):
            p.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                full=p(features)
                for branch in (0,1):
                    ablated=[f if i//4!=branch else torch.zeros_like(f) for i,f in enumerate(features)]
                    assert not torch.equal(full,p(ablated)),(n,'Branch has no effect',branch)
        p.cpu()
    del features,logits,loss,full,ablated,target
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
    assert all(p.grad is None for p in encoder.parameters())
    timings={}
    for n,p in probes.items():
        p.cuda().eval();timings[n]={}
        enc=encoder if n.startswith('fusion_') else getattr(encoder,n.split('_')[0])
        for batch in (1,64):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            resident=torch.cuda.memory_allocated()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(3):result=p(enc(x[:batch]))
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(10):result=p(enc(x[:batch]))
                torch.cuda.synchronize()
            seconds=(time.perf_counter()-t)/10
            timings[n][str(batch)]=dict(seconds_per_batch=seconds,images_per_second=batch/seconds,
                incremental_peak_gib=(torch.cuda.max_memory_allocated()-resident)/1024**3,
                process_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        p.cpu();print('BENCHMARK',n,timings[n],flush=True)
    report=dict(passed=True,initial_probe_hashes=initial,parameters=params,inference=timings,
        inference_note='Full encoder+adapter+decoder, no cache, synthetic normalized images, BF16, 3 warmups/10 repeats. Incremental peak excludes resident models/input; process peak includes both encoders.',
        initialization_matched=True,independent_parameters=True,both_branches_affect_predictions=True,
        finite_nonzero_gradients=True,frozen_encoder_hashes=hashes,
        torch_version=torch.__version__,timm_version=timm.__version__,gpu=torch.cuda.get_device_name(),
        total_gpu_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFLIGHT PASSED',json.dumps(params),flush=True)


if __name__=='__main__':main()
