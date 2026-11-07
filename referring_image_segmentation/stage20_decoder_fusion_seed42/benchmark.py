"""Real full-model BF16 batch-one GPU inference on one validation image."""
from common import *
import gc
from models import DecoderProbe, FrozenEncoder
from stage3_confirmation.data import ImageData
from train import NAMES


def main():
    setup()
    out=ROOT/'results'
    assert json.loads((out/'DONE.json').read_text())['epochs']==300
    data=ImageData(42)
    img,_=data.batch('val',torch.tensor([0],device='cuda'))
    del data
    gc.collect()
    torch.cuda.empty_cache()
    encoder=FrozenEncoder().cuda().eval()
    records={}
    for name in NAMES:
        model=DecoderProbe(name).cuda().eval()
        state=torch.load(out/name/'best_dice.pth',map_location='cpu')
        model.load_state_dict(state['model'],strict=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(10):model(encoder(img))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            for _ in range(50):
                torch.cuda.synchronize()
                start=time.perf_counter()
                model(encoder(img))
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
        records[name]=dict(mean_ms=float(np.mean(elapsed)),median_ms=float(np.median(elapsed)),
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        del model
        torch.cuda.empty_cache()
    dump(ROOT/'analysis/benchmark.json',dict(gpu=torch.cuda.get_device_name(),warmup=10,repeats=50,batch=1,
        precision='BF16 autocast',scope='MambaVision-S + adapter + full decoder + head; no data loading',models=records))
    print('BENCHMARK',json.dumps(records),flush=True)


if __name__=='__main__':main()
