"""Measure complete uncached batch-one segmentation inference on real data."""
from common import *
from models import DecoderProbe,FrozenEncoder
from stage3_confirmation.data import ImageData


def main():
    setup()
    out=ROOT/'results'
    assert json.loads((out/'DONE.json').read_text())['epochs']==300
    data=ImageData(42)
    encoder=FrozenEncoder().cuda().eval()
    img,_=data.batch('val',torch.tensor([0],device='cuda'))
    records={}
    for name in ('region_only','hiram'):
        model=DecoderProbe(name).cuda().eval()
        state=torch.load(out/name/'best_dice.pth',map_location='cpu')
        model.load_state_dict(state['model'],strict=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(5):model(encoder(img))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            for _ in range(30):
                torch.cuda.synchronize()
                start=time.perf_counter()
                model(encoder(img))
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
        records[name]=dict(mean_ms=float(np.mean(elapsed)),median_ms=float(np.median(elapsed)),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        del model
    dump(ROOT/'analysis/benchmark.json',dict(gpu=torch.cuda.get_device_name(),warmup=5,repeats=30,batch=1,
        precision='BF16',scope='MambaVision-S + adapter + decoder (including HiRAM where applicable) + head, no data loading',models=records))
    print('BENCHMARK',json.dumps(records),flush=True)


if __name__=='__main__':main()
