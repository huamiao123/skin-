"""Real-data preflight before the single formal run; never train historical models."""
from common import *
from models import FrozenEncoder,build_probe,tensor_hash,segmentation_loss
from data import ImageData
import importlib.metadata as metadata
import platform

def main():
    setup();encoder=FrozenEncoder().cuda();probe=build_probe().cuda();data=ImageData()
    initial_encoder=tensor_hash(encoder);initial_probe=tensor_hash(probe)
    expected=json.loads((PROJECT/'stage4_fusion/preflight.json').read_text())['initial_probe_hashes']['vmamba_swin']
    assert initial_probe==expected,'Historical same-channel Adapter/Decoder initial tensors differ'
    x,y=data.batch('train',torch.arange(64,device='cuda'))
    baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(x)
    torch.cuda.synchronize();encoder_time=time.perf_counter()-start
    encoder_peak=torch.cuda.max_memory_allocated()
    assert all(torch.isfinite(f).all() for f in features)
    shapes=[list(f.shape) for f in features]
    pyramid=[a(f) for a,f in zip(probe.adapter,[f.float() for f in features])]
    assert [list(p.shape) for p in pyramid]==[[64,64,s,s] for s in (64,32,16,8)]
    del pyramid
    encoder.train(True);assert not any(m.training for m in encoder.modules())
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    probe.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=probe(features)
    assert list(logits.shape)==[64,1,256,256]
    loss=segmentation_loss(logits,y);assert loss.dtype==torch.float32 and torch.isfinite(loss)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
    torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);optimizer.step();torch.cuda.synchronize()
    decoder_time=time.perf_counter()-start;decoder_peak=torch.cuda.max_memory_allocated()
    assert tensor_hash(probe)!=initial_probe and tensor_hash(encoder)==initial_encoder
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    del logits,loss,features,optimizer;probe.zero_grad(set_to_none=True);torch.cuda.empty_cache();probe.eval()
    # Complete, uncached single-image path; same warmup/repetitions as stage4.
    sample=x[:1].contiguous();torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=probe(encoder(sample))
        del out;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before=torch.cuda.memory_allocated();start=time.perf_counter()
        for _ in range(10):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=probe(encoder(sample))
        torch.cuda.synchronize();seconds=(time.perf_counter()-start)/10
    inference_peak=torch.cuda.max_memory_allocated()
    assert tensor_hash(encoder)==initial_encoder
    params=dict(encoder=encoder.audit['params'],adapter=sum(p.numel() for p in probe.adapter.parameters()),
        decoder_including_head=sum(p.numel() for p in probe.decoder.parameters()),
        segmentation_head=sum(p.numel() for p in probe.decoder.readout.parameters()),
        trainable=sum(p.numel() for p in probe.parameters()))
    params['decoder_without_head']=params['decoder_including_head']-params['segmentation_head']
    params['total']=params['encoder']+params['trainable']
    versions=dict(python=platform.python_version(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3)
    for name in ('timm','transformers','mamba-ssm','einops','numpy','scipy','Pillow','safetensors'):
        versions[name]=metadata.version(name)
    import selective_scan_cuda,mamba_ssm
    versions['selective_scan_cuda_path']=selective_scan_cuda.__file__
    versions['selective_scan_cuda_sha256']=sha(selective_scan_cuda.__file__)
    try:versions['causal-conv1d']=metadata.version('causal-conv1d')
    except metadata.PackageNotFoundError:versions['causal-conv1d']='not installed; official MambaVision uses torch conv1d and CUDA selective_scan_fn'
    config=dict(seed=42,epochs=300,encoder='NVIDIA MambaVision-S-1K',channels=list(CHANNELS),input_size=256,
        train_samples=1886,validation_samples=808,batch=64,encoder_batch=64,evaluation_batch=64,
        updates_per_epoch=30,total_updates=9000,drop_last=False,gradient_accumulation=1,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR',T_max=300,eta_min=1e-5,warmup=False,
        scheduler_step='after all training updates, before validation; log LR actually used for training',
        loss='historical BCEWithLogits + per-image Soft Dice, smooth=1',clip=1.,
        precision='BF16 autocast, FP32 loss',frozen_encoder=True,
        augmentation_seed='42+epoch*100003',normalization='historical ImageNet mean/std',
        data_processing='unmodified stage3_confirmation.data.ImageData / stage1_visual.data',
        evaluation='256x256 development validation; probability>=0.5; original common evaluation.py',
        primary_selection='maximum validation macro Dice; earliest exact tie',
        secondary_selection='minimum sample-weighted validation loss; earliest exact tie',
        final_selection='epoch300',encoder_initial_hash=initial_encoder,probe_initial_hash=initial_probe,
        parameters=params,pretrained=encoder.audit,environment=versions)
    dump(ROOT/'config.json',config)
    dump(ROOT/'preflight.json',dict(passed=True,checks=dict(data_manifest_exact=True,n_train=1886,n_val=808,
        real_batch64_forward=True,finite_stage_features=True,stage_shapes=shapes,output_shape=[64,1,256,256],
        matches_historical_vmamba_swin_initial_probe=True,finite_trainable_gradients=True,
        trainable_weights_updated=True,encoder_no_grad=True,encoder_eval_recursive=True,
        encoder_parameters_and_buffers_unchanged=True,cuda_selective_scan_executed=True),
        encoder_hash=initial_encoder,initial_probe_hash=initial_probe,pretrained=encoder.audit,parameters=params,environment=versions,
        memory=dict(encoder_batch64_baseline_gib=baseline/1024**3,encoder_batch64_peak_gib=encoder_peak/1024**3,
            decoder_train_batch64_peak_gib=decoder_peak/1024**3),
        throughput=dict(encoder_batch64_seconds=encoder_time,decoder_train_batch64_seconds=decoder_time),
        inference=dict(batch=1,includes='Encoder + Adapter + Swin Pixel Decoder + segmentation head; no cache',
            warmup=3,repetitions=10,precision='BF16 autocast',cuda_synchronized=True,
            seconds_per_image=seconds,images_per_second=1/seconds,process_peak_gib=inference_peak/1024**3,
            incremental_peak_gib=(inference_peak-before)/1024**3,baseline_gib=before/1024**3,
            note='Preflight process also holds historical GPU-resident data; separate standalone model memory measured at final evaluation')))
    print('PREFLIGHT_PASSED',json.dumps(dict(shapes=shapes,params=params,encoder_peak_gib=encoder_peak/1024**3,full_latency_ms=seconds*1000)),flush=True)

if __name__=='__main__':main()
