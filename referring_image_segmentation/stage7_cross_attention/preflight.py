"""Attention math/window tests, reference reuse, gradient checks and efficiency."""
import copy
import json
import time
import numpy as np
import torch
import timm
from models import ROOT,PROJECT,NAMES,SPEC,EncoderPair,build_probes,WindowCrossAttention,partition,reverse,tensor_hash,segmentation_loss
from references import REFERENCE_PATHS,build_reference_probes,reference_features
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode,evaluate
from stage5_selective_fusion.diagnostics import sha,dump


def mathematical_checks():
    torch.manual_seed(42)
    a=WindowCrossAttention().double();b=copy.deepcopy(a)
    d=torch.randn(2,64,64,dtype=torch.double,requires_grad=True);m=torch.randn_like(d,requires_grad=True)
    dd=d.detach().clone().requires_grad_();mm=m.detach().clone().requires_grad_()
    y=a.attention(d,m);z=b.attention(dd,mm,manual=True)
    torch.testing.assert_close(y,z,atol=1e-10,rtol=1e-9)
    y.square().mean().backward();z.square().mean().backward()
    torch.testing.assert_close(d.grad,dd.grad,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(m.grad,mm.grad,atol=1e-10,rtol=1e-8)
    for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
        assert n==nn
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-8)
    for side in (8,16,32,64):
        x=torch.arange(2*side*side*64).reshape(2,side,side,64)
        assert torch.equal(reverse(partition(x),2,side,side),x)
    a=a.float().cuda().eval()
    d=torch.randn(2,64,16,16,device='cuda');m=torch.randn_like(d)
    with torch.no_grad():
        y=a(d,m);changed=m.clone();changed[:,0,0,0]+=5;z=a(d,changed)
        assert not torch.equal(y[:,:,:8,:8],z[:,:,:8,:8])
        assert torch.equal(y[:,:,8:,:],z[:,:,8:,:]) and torch.equal(y[:,:,:8,8:],z[:,:,:8,8:])
        changed=d.clone();changed[:,0,0,0]+=5;z=a(changed,m)
        assert not torch.equal(y[:,:,0,0],z[:,:,0,0])
        delta=(y-z).abs();delta[:,:,0,0]=0;assert delta.max()==0
    return dict(manual_sdpa_forward_and_backward_match=True,window_roundtrip=True,
        vmamba_values_affect_within_window=True,no_cross_window_leakage=True,dino_query_changes_only_its_retrieval=True)


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checks=mathematical_checks();print('ATTENTION_MATH_PASSED',flush=True)
    for stage in ('stage5_selective_fusion','stage6_fusion_confirmation'):
        assert json.loads((PROJECT/stage/'validation.json').read_text())['passed']
        for path,digest in json.loads((PROJECT/stage/'launch_provenance.json').read_text())['source_sha256'].items():assert sha(path)==digest,path
    enc=EncoderPair().cuda();hashes={n:tensor_hash(getattr(enc,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():assert h==json.loads((PROJECT/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    probes=build_probes(42);initial={n:tensor_hash(p) for n,p in probes.items()}
    saved={n:{k:v.clone() for k,v in p.state_dict().items()} for n,p in probes.items()}
    ref=build_reference_probes();params={}
    ids=[id(p) for model in probes.values() for p in model.parameters()];assert len(ids)==len(set(ids))
    for p in probes.values():
        for attr in ('dino_adapter','mamba_adapter','decoder'):assert tensor_hash(getattr(p,attr))==tensor_hash(getattr(ref['fusion'],attr))
    for n,p in dict(**ref,**probes).items():
        ec=enc.dinov3.audit['params'] if n in ('dinov3','capacity_concat') else enc.vmamba.audit['params'] if n=='vmamba' else sum(getattr(enc,k).audit['params'] for k in hashes)
        tc=sum(t.numel() for t in p.parameters());params[n]=dict(encoder=ec,trainable=tc,total=ec+tc,decoder=sum(t.numel() for t in p.decoder.parameters()))
    assert params['window_cross_attention']['trainable']==670321
    assert params['local_conv']['trainable']==670401
    x=torch.randn(64,3,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):features=enc(x[:2])
    target=torch.rand(2,1,256,256,device='cuda')
    for n,p in probes.items():
        p.cuda().train();opt=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.01)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(features)
        loss=segmentation_loss(logits,target);loss.backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
        assert tensor_hash(p)!=initial[n]
        p.load_state_dict(saved[n]);assert tensor_hash(p)==initial[n];p.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            full=p(features)
            for branch in (0,1):
                ablated=[f if i//4!=branch else torch.zeros_like(f) for i,f in enumerate(features)]
                assert not torch.equal(full,p(ablated)),(n,branch)
        p.cpu();del opt,loss,logits,full,ablated
    del features,target,saved
    print('INITIALIZATION_AND_GRADIENTS_PASSED',flush=True)
    data=ImageData(42);assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    cache,targets=encode(enc,data,'val',64);reuse={}
    fixed=dict(seed=42,epochs=300,input_size=256,batch=64,encoder_batch=64,evaluation_batch=64,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,T_max=300,eta_min=1e-5,clip=1.)
    for n,p in ref.items():
        base=REFERENCE_PATHS[n];cfg=json.loads((base.parent/'config.json').read_text())
        for k,v in fixed.items():assert cfg[k]==v,(n,k)
        history=json.loads((base.parent/'history.json').read_text())
        history_name=n+'_swin' if n in ('dinov3','vmamba','fusion') else n
        p.cuda();reuse[n]={}
        for criterion in ('best_dice','best_loss','last'):
            path=base/criterion;assert json.loads((path/'ids.json').read_text())==data.ids['val']
            record=json.loads((path/'metrics.json').read_text())
            chosen=max(history,key=lambda r:r['models'][history_name]['val']['macro']['dice']) if criterion=='best_dice' else min(history,key=lambda r:r['models'][history_name]['val']['loss']) if criterion=='best_loss' else history[-1]
            assert chosen['epoch']==record['selected_epoch'] and chosen['models'][history_name]['val']==record['final']
            state=torch.load(base/f'{criterion}.pth',map_location='cpu');p.load_state_dict(state['probe'])
            current,prob,_=evaluate(p,reference_features(n,cache),targets,export=True)
            assert current['counts']==record['final']['counts'],(n,criterion,'counts')
            for mode in ('macro','pooled'):
                for k in ('dice','iou','precision','recall'):assert abs(current[mode][k]-record['final'][mode][k])<1e-12
            delta=float(np.abs(prob-np.load(path/'probabilities.npy',mmap_mode='r')).max())
            assert delta<=1e-6,(n,criterion,delta)
            reuse[n][criterion]=dict(max_probability_abs_delta=delta,checkpoint_sha256=sha(base/f'{criterion}.pth'),probabilities_sha256=sha(path/'probabilities.npy'))
        p.cpu();del state,prob;print('REFERENCE_PASSED',n,flush=True)
    del cache,targets,data
    timings={}
    for n,p in dict(**ref,**probes).items():
        p.cuda().eval();timings[n]={}
        encoder=enc.dinov3 if n in ('dinov3','capacity_concat') else enc.vmamba if n=='vmamba' else enc
        def predict(images):
            fs=encoder(images)
            return p(fs+fs if n=='capacity_concat' else fs)
        for bs in (1,64):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(3):out=predict(x[:bs])
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(10):out=predict(x[:bs])
                torch.cuda.synchronize()
            sec=(time.perf_counter()-t)/10
            timings[n][str(bs)]=dict(seconds_per_batch=sec,images_per_second=bs/sec,incremental_peak_gib=(torch.cuda.max_memory_allocated()-resident)/1024**3)
        p.cpu();print('BENCHMARK',n,flush=True)
    assert all(tensor_hash(getattr(enc,n))==h for n,h in hashes.items()) and all(p.grad is None for p in enc.parameters())
    dump(ROOT/'preflight.json',dict(passed=True,initial_probe_hashes=initial,parameters=params,inference=timings,model_spec=SPEC,
        inference_note='Current 48GB RTX4090; full BF16 forward, no feature caching/disk/postprocessing, 3 warmups/10 repeats. Incremental memory excludes resident tensors.',
        attention_checks=checks,all_gradients_finite_nonzero=True,both_branches_affect_output=True,common_initial_tensors_match=True,
        data_manifest_match=True,source_hashes_match=True,reused_references=reuse,frozen_encoder_hashes=hashes,
        torch_version=torch.__version__,timm_version=timm.__version__,gpu=torch.cuda.get_device_name(),
        total_gpu_gib=torch.cuda.get_device_properties(0).total_memory/1024**3))
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
