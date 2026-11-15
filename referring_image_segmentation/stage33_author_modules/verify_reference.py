"""Replay archived validation only; test data are never loaded here."""
import gc,json
from pathlib import Path
import torch
from train import ROOT,PROJECT,base,dump,digest,load_data,cache_new,fill_cache
from stage29_context_fusion import train as prior
def main():
    base.setup()
    records=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    data,_=load_data();cache=cache_new(len(data.ids['val']));checks=[]
    for record in records:
        assert digest(Path(record['checkpoint']))==record['sha256']
        m=prior.System(record['arm']).cuda()
        state=torch.load(record['checkpoint'],map_location='cpu');m.load_state_dict(state['model'],strict=True);del state
        fill_cache(m,data,'val',cache)
        metric,_=base.evaluate(m,cache,64)
        assert metric['counts']==record['validation']['counts'],record['arm']
        assert abs(metric['macro']['dice']-record['validation']['macro']['dice'])<1e-12
        checks.append(dict(arm=record['arm'],passed=True,n=762,macro_dice=metric['macro']['dice']))
        print('REFERENCE_REPLAY_OK',checks[-1],flush=True)
        del m;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',dict(passed=True,checks=checks,test_read=False))
if __name__=='__main__':main()

