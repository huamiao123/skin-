"""Standalone uncached full model timing without any training/data feature cache."""
from common import *
from models import FrozenEncoder,build_probe,tensor_hash
from PIL import Image
import argparse

def main():
    setup();verify_frozen_sources()
    done=json.loads((ROOT/'results/DONE.json').read_text());assert done['epochs']==300
    encoder=FrozenEncoder().cuda();probe=build_probe().cuda().eval()
    checkpoint=torch.load(ROOT/'results/best_dice.pth',map_location='cpu');probe.load_state_dict(checkpoint['probe'])
    # One real validation RGB with identical resize/normalization; preprocessing outside timing.
    data=PROJECT/'data/isic2018/val/images';path=sorted(p for p in data.iterdir() if p.is_file())[0]
    image=np.asarray(Image.open(path).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
    x=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda().float()/255
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1);x=(x-mean)/std
    with torch.no_grad():
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16):y=probe(encoder(x))
        del y;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        before=torch.cuda.memory_allocated();times=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):y=probe(encoder(x))
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
    assert tensor_hash(encoder)==done['final_encoder_hash']
    record=dict(batch=1,warmup=3,repetitions=10,device=torch.cuda.get_device_name(),
        precision='BF16 autocast, FP32 parameters',includes='Encoder + Adapter + Swin Decoder + segmentation head',
        excludes='disk I/O, CPU resize/normalization and postprocessing; no cached features',cuda_synchronized=True,
        mean_ms=float(np.mean(times)*1000),median_ms=float(np.median(times)*1000),std_ms=float(np.std(times)*1000),
        times_seconds=times,images_per_second=1/float(np.mean(times)),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        baseline_allocated_gib=before/1024**3,incremental_peak_gib=(torch.cuda.max_memory_allocated()-before)/1024**3,
        flops=None,flops_note='No complete operator accounting for selective_scan; no partial FLOPs presented as full-model FLOPs',
        historical_latency_note='Stage4 timings recorded on its original device/session; only descriptive, not a new same-session benchmark')
    dump(ROOT/'analysis/inference.json',record);print('FULL_MODEL_BENCHMARK',json.dumps(record),flush=True)

if __name__=='__main__':main()
