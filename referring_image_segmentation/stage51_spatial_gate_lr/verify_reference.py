"""Replay baseline and both matched Stage50 best checkpoints on all762 validation images."""
from pathlib import Path
import json,gc,torch,numpy as np
from train import ROOT,PROJECT,System,base,digest,dump
from evaluate import helpers
def main():
    base.setup();refs=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    ids=json.loads((ROOT/'identity_partition.json').read_text())['splits']['val']
    data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
    results=[]
    for ref in refs:
        assert digest(ref['checkpoint'])==ref['sha256']
        state=torch.load(ref['checkpoint'],map_location='cpu')
        for rel,sha in state['config']['sources'].items():assert digest(PROJECT/rel)==sha,rel
        model=System(ref.get('model_arm','original')).cuda()
        model.load_state_dict(state['model'],strict=True);model.requires_grad_(False).eval()
        target=ROOT/(ref['arm']+'_reference_predictions.npz')
        metric,_=helpers.evaluate(model,data,ids,base,target)
        assert metric['counts']==ref['validation']['counts']
        assert abs(metric['macro']['dice']-ref['validation']['macro']['dice'])<1e-12
        prior=Path(ref.get('val_predictions',str(Path(ref['checkpoint']).parent/'uper64_val_predictions.npz')))
        old=np.load(prior);new=np.load(target)
        assert old.files==new.files
        for k in old.files:assert np.array_equal(old[k],new[k]),(ref['arm'],k)
        results.append(dict(arm=ref['arm'],n=762,binary_predictions_exact=True,sha256=ref['sha256']))
        del model,state;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',dict(passed=True,references=results,test_read=False))
    print('ALL_REFERENCES_VERIFIED',flush=True)
if __name__=='__main__':main()
