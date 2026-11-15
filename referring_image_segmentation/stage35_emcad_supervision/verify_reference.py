"""Historical asset verification plus validation-only hardware drift measurement."""
from pathlib import Path
import gc,json,csv
import torch
from train import ROOT,PROJECT,base,dump,digest,load_data,cache_new,fill_cache
from author_modules import _import
def main():
    base.setup()
    records=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    for record in records:
        assert digest(Path(record['checkpoint']))==record['sha256']
        for split,key in [('val','val_rows'),('test','test_rows')]:
            with Path(record[key]).open() as f:ids=[row['id'] for row in csv.DictReader(f)]
            assert ids==partition['splits'][split]
    record=next(v for v in records if v['arm']=='stage34_emcad64_exp2')
    old=_import(PROJECT/'stage34_author_integration_fix/author_modules.py','old_exp2_for_hardware_replay')
    m=base.System(finetune=True,stochastic_depth=False)
    m.probe.decoder=old.AuthorModuleDecoder('emcad64_exp2',m.probe.decoder,seed=42)
    state=torch.load(record['checkpoint'],map_location='cpu')
    m.load_state_dict(state['model'],strict=True);del state
    m=m.cuda().eval()
    data,_=load_data();cache=cache_new(len(data.ids['val']))
    fill_cache(m,data,'val',cache)
    metric,_=base.evaluate(m,cache,64)
    delta=metric['macro']['dice']-record['validation']['macro']['dice']
    # A large replay discrepancy indicates environment/input drift and stops training.
    assert abs(delta)<.005,('large unexplained validation replay change',delta)
    result=dict(passed=True,historical_checkpoint_hashes_verified=len(records),
        current_hardware=torch.cuda.get_device_name(0),test_images_read=False,
        validation_n=762,historical_macro_dice=record['validation']['macro']['dice'],
        current_macro_dice=metric['macro']['dice'],macro_delta_pp=100*delta,
        confusion_counts_identical=metric['counts']==record['validation']['counts'],
        primary_control='fresh emcad_single trained on same4090, not the historical A4000 scores')
    dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',result)
    print('REFERENCE_OK',json.dumps(result),flush=True)
if __name__=='__main__':main()
