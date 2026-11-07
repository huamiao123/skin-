"""Full batch-one latency of the adapted model on the current GPU."""
from segman_model import *
import gc
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder
from stage3_confirmation.data import ImageData


def main():
    setup()
    local=Path(__file__).resolve().parent
    out=local/'results'
    assert json.loads((out/'DONE.json').read_text())['epochs']==300
    data=ImageData(42)
    image,_=data.batch('val',torch.tensor([0],device='cuda'))
    del data
    gc.collect()
    torch.cuda.empty_cache()
    encoder=FrozenEncoder().cuda().eval()
    model=MambaVisionSegMAN().cuda().eval()
    model.load_state_dict(torch.load(out/'best_dice.pth',map_location='cpu')['model'],strict=True)
    elapsed=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(10):model(encoder(image))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(50):
            torch.cuda.synchronize()
            start=time.perf_counter()
            model(encoder(image))
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
    result=dict(gpu=torch.cuda.get_device_name(),precision='BF16 autocast',batch=1,warmup=10,repeats=50,
        scope='full MambaVision-S + adapted SegMAN decoder + mask head, no data loading',
        mean_ms=float(np.mean(elapsed)),median_ms=float(np.median(elapsed)),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    dump(local/'analysis/benchmark.json',result)
    print('BENCHMARK',json.dumps(result),flush=True)


if __name__=='__main__':main()
