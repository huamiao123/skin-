"""Audit baseline reuse, seeded initialization, branch training and full inference."""
import json
import time
import numpy as np
import torch
import timm
from models import ROOT, PROJECT, SEEDS, NAMES, EncoderPair, build_probes, select_features, SingleProbe, tensor_hash, segmentation_loss, SPEC
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode, evaluate
from stage5_selective_fusion.diagnostics import sha, dump


def main():
    torch.set_num_threads(4);torch.manual_seed(43)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old_prov=json.loads((PROJECT/'stage3_confirmation/launch_provenance.json').read_text())
    for path,digest in old_prov['source_sha256'].items():assert sha(path)==digest,path
    assert json.loads((PROJECT/'stage3_confirmation/validation.json').read_text())['passed']
    encoder=EncoderPair().cuda();hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():assert h==json.loads((PROJECT/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    data=ImageData(43)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    cache,targets=encode(encoder,data,'val',64)
    initial={};reuse={};params={}
    old_preflight=json.loads((PROJECT/'stage3_confirmation/preflight.json').read_text())
    common=dict(epochs=300,input_size=256,batch=64,encoder_batch=64,evaluation_batch=64,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,T_max=300,eta_min=1e-5,
        clip=1.,precision='BF16 autocast; FP32 loss',loss='BCEWithLogits + per-image soft Dice; smooth=1',
        augmentation_seed='seed+epoch*100003',frozen_encoder=True)
    for seed in SEEDS:
        baseline=SingleProbe('swin',seed)
        assert tensor_hash(baseline)==old_preflight['initial_probe_hashes'][str(seed)]['swin']
        folder=PROJECT/f'stage3_confirmation/results/seed{seed}'
        config=json.loads((folder/'config.json').read_text())
        for key,value in common.items():assert config[key]==value,(seed,key)
        assert config['seed']==seed and config['encoder_initial_hash']==hashes['dinov3']
        assert config['initial']['swin']['adapter_hash']==tensor_hash(baseline.adapter)
        assert config['initial']['swin']['decoder_hash']==tensor_hash(baseline.decoder)
        done=json.loads((folder/'DONE.json').read_text());assert done['epochs']==300
        history=json.loads((folder/'history.json').read_text());assert len(history)==300
        probes=build_probes(seed);initial[str(seed)]={n:tensor_hash(p) for n,p in probes.items()}
        all_parameters=[id(p) for m in probes.values() for p in m.parameters()]
        assert len(set(all_parameters))==len(all_parameters)
        for p in probes.values():
            assert tensor_hash(p.dino_adapter)==tensor_hash(baseline.adapter)
            assert tensor_hash(p.decoder)==tensor_hash(baseline.decoder)
        assert tensor_hash(probes['capacity_concat'].fusion)==tensor_hash(probes['fusion_concat'].fusion)
        cap=probes['capacity_concat']
        assert not (set(map(id,cap.dino_adapter.parameters())) & set(map(id,cap.mamba_adapter.parameters())))
        for n,p in probes.items():
            ec=encoder.dinov3.audit['params']+(encoder.vmamba.audit['params'] if n=='fusion_concat' else 0)
            tc=sum(t.numel() for t in p.parameters());params[n]=dict(encoder=ec,trainable=tc,total=ec+tc,decoder=sum(t.numel() for t in p.decoder.parameters()))
            p.cuda().train();batch=select_features(n,[f[:2] for f in cache]);before=tensor_hash(p)
            opt=torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.01)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(batch)
            loss=segmentation_loss(logits,targets[:2]);loss.backward()
            assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in p.parameters()),(seed,n)
            torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
            assert tensor_hash(p)!=before
            if n=='capacity_concat':assert tensor_hash(p.dino_adapter)!=tensor_hash(p.mamba_adapter)
            p.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                full=p(batch)
                for branch in (0,1):
                    ablated=[f if i//4!=branch else torch.zeros_like(f) for i,f in enumerate(batch)]
                    assert not torch.equal(full,p(ablated)),(seed,n,branch)
            p.cpu();del opt,loss,logits,full,ablated,batch
        params['dinov3']=dict(encoder=encoder.dinov3.audit['params'],trainable=sum(p.numel() for p in baseline.parameters()),
            total=encoder.dinov3.audit['params']+sum(p.numel() for p in baseline.parameters()),decoder=sum(p.numel() for p in baseline.decoder.parameters()))
        baseline.cuda();reuse[str(seed)]=dict(source=str(folder/'swin'),config_sha256=sha(folder/'config.json'),checkpoints={})
        for criterion in ('best_dice','best_loss','last'):
            path=folder/'swin'/criterion
            assert json.loads((path/'ids.json').read_text())==data.ids['val']
            record=json.loads((path/'metrics.json').read_text())
            selected=max(history,key=lambda r:r['models']['swin']['val']['macro']['dice']) if criterion=='best_dice' else min(history,key=lambda r:r['models']['swin']['val']['loss']) if criterion=='best_loss' else history[-1]
            assert record['selected_epoch']==selected['epoch']
            state=torch.load(folder/'swin'/f'{criterion}.pth',map_location='cpu')
            baseline.load_state_dict(state['probe'])
            current,prob,_=evaluate(baseline,cache[:4],targets,export=True)
            assert current['counts']==record['final']['counts'],(seed,criterion,'counts')
            for mode in ('macro','pooled'):
                for key in ('dice','iou','precision','recall'):assert abs(current[mode][key]-record['final'][mode][key])<1e-12
            delta=float(np.abs(prob-np.load(path/'probabilities.npy',mmap_mode='r')).max())
            assert delta<=1e-6,(seed,criterion,delta)
            reuse[str(seed)]['checkpoints'][criterion]=dict(selected_epoch=record['selected_epoch'],max_probability_abs_delta=delta,
                checkpoint_sha256=sha(folder/'swin'/f'{criterion}.pth'),probabilities_sha256=sha(path/'probabilities.npy'))
        baseline.cpu();del baseline,probes,prob,state
        print('REUSE_AND_GRADIENTS_PASSED',seed,flush=True)
    # Seed42 classes must exactly reproduce the previous definitions/initial tensors.
    check42=build_probes(42)
    assert tensor_hash(check42['capacity_concat'])==json.loads((PROJECT/'stage5_selective_fusion/preflight.json').read_text())['initial_probe_hashes']['capacity_concat']
    assert tensor_hash(check42['fusion_concat'])==json.loads((PROJECT/'stage4_fusion/preflight.json').read_text())['initial_probe_hashes']['fusion_swin']
    del cache,targets,data,check42
    timings={};x=torch.randn(64,3,256,256,device='cuda')
    for n,p in dict(dinov3=SingleProbe('swin',43),**build_probes(43)).items():
        p.cuda().eval();timings[n]={};enc=encoder if n=='fusion_concat' else encoder.dinov3
        def predict(x):
            fs=enc(x)
            return p(fs+fs if n=='capacity_concat' else fs)
        for bs in (1,64):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for _ in range(3):result=predict(x[:bs])
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(10):result=predict(x[:bs])
                torch.cuda.synchronize()
            secs=(time.perf_counter()-t)/10
            timings[n][str(bs)]=dict(seconds_per_batch=secs,images_per_second=bs/secs,
                incremental_peak_gib=(torch.cuda.max_memory_allocated()-resident)/1024**3)
        p.cpu()
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
    assert all(p.grad is None for p in encoder.parameters())
    report=dict(passed=True,initial_probe_hashes=initial,parameters=params,reused_baselines=reuse,
        source_audit_passed=True,data_manifest_match=True,capacity_adapters_diverge=True,branch_ablation_passed=True,
        all_gradients_finite_nonzero=True,seed42_initial_hashes_match=True,frozen_encoder_hashes=hashes,
        inference=timings,inference_note='Full forward BF16; 3 warmups/10 repeats; no feature caching, disk or postprocessing; same device. Incremental memory excludes resident tensors.',
        gpu=torch.cuda.get_device_name(),total_gpu_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
        torch_version=torch.__version__,timm_version=timm.__version__,model_spec=SPEC)
    dump(ROOT/'preflight.json',report);print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
