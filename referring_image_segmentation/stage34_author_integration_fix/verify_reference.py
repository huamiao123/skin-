"""Replay archived validation of matching original arms and baseline; never load test images."""
import gc,json
from pathlib import Path
import torch
from train import ROOT,PROJECT,base,dump,digest,load_data,cache_new,fill_cache
from stage29_context_fusion import train as prior
from author_modules import _import
def main():
    base.setup()
    records=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    original=_import(PROJECT/'stage33_author_modules/author_modules.py','stage33_original_for_replay')
    data,_=load_data()
    cache=cache_new(len(data.ids['val']))
    checks=[]
    for record in records:
        assert digest(Path(record['checkpoint']))==record['sha256']
        if record['source_stage']=='stage29':
            m=prior.System(record['source_arm'])
        else:
            m=base.System(finetune=True,stochastic_depth=False)
            m.probe.decoder=original.AuthorModuleDecoder(record['source_arm'],m.probe.decoder,seed=42)
        m=m.cuda()
        state=torch.load(record['checkpoint'],map_location='cpu')
        m.load_state_dict(state['model'],strict=True)
        del state
        m.eval()
        fill_cache(m,data,'val',cache)
        metric,_=base.evaluate(m,cache,64)
        assert metric['counts']==record['validation']['counts'],record['arm']
        assert abs(metric['macro']['dice']-record['validation']['macro']['dice'])<1e-12
        checks.append(dict(arm=record['arm'],passed=True,n=762,macro_dice=metric['macro']['dice']))
        print('REFERENCE_REPLAY_OK',checks[-1],flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',dict(passed=True,checks=checks,test_images_read=False))
if __name__=='__main__':main()
