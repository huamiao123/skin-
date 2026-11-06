"""Measure complete uncached segmentation inference, real batch one."""
from common import *
from models import Segmentor
from train import load_trainable,NAMES
from stage3_confirmation.data import ImageData


def main():
    setup()
    out=ROOT/'results'
    assert json.loads((out/'DONE.json').read_text())['epochs']==300
    data=ImageData(42)
    img,_=data.batch('val',torch.tensor([0],device='cuda'))
    records={}
    for name in NAMES:
        model=Segmentor(name).cuda().eval()
        state=torch.load(out/name/'best_dice.pth',map_location='cpu')
        load_trainable(model,state['model'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(5):model(img)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[]
            for _ in range(30):
                torch.cuda.synchronize();start=time.perf_counter();model(img);torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
        records[name]=dict(mean_ms=float(np.mean(elapsed)),median_ms=float(np.median(elapsed)),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        del model
        torch.cuda.empty_cache()
    dump(ROOT/'analysis/benchmark.json',dict(gpu=torch.cuda.get_device_name(),warmup=5,repeats=30,batch=1,
        precision='BF16',scope='entire uncached encoder/router + adapters + Swin + mask head, excluding I/O',models=records))
    print('BENCHMARK',json.dumps(records),flush=True)


if __name__=='__main__':main()
