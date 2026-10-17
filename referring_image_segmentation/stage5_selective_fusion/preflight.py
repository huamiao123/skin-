"""Validate controlled initialization, branch use, updates and end-to-end costs."""
import json
import time
import torch
import timm
from models import ROOT,EncoderPair,build_probes,select_features,NAMES,SPEC,tensor_hash,segmentation_loss
from stage4_fusion.models import FusionProbe,CHANNELS
from stage3_confirmation.models import Probe as SingleProbe


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    encoder=EncoderPair().cuda();hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():assert h==json.loads((ROOT.parent/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    probes=build_probes(42);initial={n:tensor_hash(p) for n,p in probes.items()}
    saved={n:{k:v.clone() for k,v in p.state_dict().items()} for n,p in probes.items()}
    unique=[id(t) for p in probes.values() for t in p.parameters()];assert len(unique)==len(set(unique))
    a,b=probes['static_residual'],probes['dynamic_gate']
    for field in ('dino_adapter','mamba_adapter','phi','decoder'):assert tensor_hash(getattr(a,field))==tensor_hash(getattr(b,field))
    old=FusionProbe('swin',42)
    for field in ('dino_adapter','mamba_adapter','decoder'):assert tensor_hash(getattr(a,field))==tensor_hash(getattr(old,field))
    assert tensor_hash(probes['capacity_concat'].fusion)==tensor_hash(old.fusion)
    assert tensor_hash(probes['capacity_concat'].decoder)==tensor_hash(old.decoder)
    params={}
    for n,p in probes.items():
        ec=encoder.dinov3.audit['params']+(0 if n=='capacity_concat' else encoder.vmamba.audit['params'])
        tc=sum(t.numel() for t in p.parameters())
        params[n]=dict(encoder=ec,trainable=tc,total=ec+tc,decoder=sum(t.numel() for t in p.decoder.parameters()))
    assert params['capacity_concat']['trainable']==638305
    assert params['dynamic_gate']['trainable']-params['static_residual']['trainable']==512
    x=torch.randn(64,3,256,256,device='cuda');target=torch.rand(2,1,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(x[:2])
    a.cuda().eval();b.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        la,ga=a(features,return_gates=True);lb,gb=b(features,return_gates=True)
    assert torch.equal(la,lb),'Static/dynamic do not start with the same function'
    assert all(torch.equal(u,v) and torch.allclose(u,torch.full_like(u,.1)) for u,v in zip(ga,gb))
    a.cpu();b.cpu();del la,lb,ga,gb
    for n,p in probes.items():
        p.cuda().train();batch=select_features(n,features)
        optimizer=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.01)
        before=tensor_hash(p)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(batch)
        loss=segmentation_loss(logits,target);loss.backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert tensor_hash(p)!=before
        if n=='capacity_concat':assert tensor_hash(p.dino_adapter)!=tensor_hash(p.mamba_adapter),'Capacity branches remained identical'
        if n=='dynamic_gate':assert all(layer.weight.abs().sum()>0 for layer in p.gate)
        p.load_state_dict(saved[n]);assert tensor_hash(p)==initial[n]
        p.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            full=p(batch)
            for branch in (0,1):
                ablated=[f if j//4!=branch else torch.zeros_like(f) for j,f in enumerate(batch)]
                assert not torch.equal(full,p(ablated)),(n,branch)
        p.cpu();del optimizer,logits,loss,full,ablated
    del features,batch,target
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
    assert all(t.grad is None for t in encoder.parameters())
    comparators=dict(dinov3_swin=SingleProbe('swin',42,CHANNELS['dinov3']),concat_swin=old,**probes)
    timings={}
    for n,p in comparators.items():
        p.cuda().eval();timings[n]={}
        enc=encoder.dinov3 if n in ('capacity_concat','dinov3_swin') else encoder
        def predict(image):
            fs=enc(image)
            return p(fs+fs if n=='capacity_concat' else fs)
        for batch_size in (1,64):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(3):result=predict(x[:batch_size])
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(10):result=predict(x[:batch_size])
                torch.cuda.synchronize()
            secs=(time.perf_counter()-t)/10
            timings[n][str(batch_size)]=dict(seconds_per_batch=secs,images_per_second=batch_size/secs,
                incremental_peak_gib=(torch.cuda.max_memory_allocated()-resident)/1024**3,
                process_peak_gib=torch.cuda.max_memory_allocated()/1024**3)
        p.cpu();print('BENCHMARK',n,timings[n],flush=True)
    report=dict(passed=True,initial_probe_hashes=initial,parameters=params,inference=timings,model_spec=SPEC,
        static_dynamic_initial_logits_identical=True,capacity_adapters_independent_and_diverge=True,
        all_gradients_finite_nonzero=True,both_branches_affect_output=True,encoders_unchanged=True,
        frozen_encoder_hashes=hashes,torch_version=torch.__version__,timm_version=timm.__version__,
        gpu=torch.cuda.get_device_name(),total_gpu_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
        inference_note='Full forward without feature cache; synthetic BF16; 3 warmups/10 repeats. Capacity uses one DINO forward. Process memory includes both encoders; incremental peak excludes resident tensors.')
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print('PREFLIGHT PASSED',params,flush=True)


if __name__=='__main__':main()
