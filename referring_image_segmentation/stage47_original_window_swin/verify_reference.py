"""Replay all762 validation predictions of Stage45 UPer64 with the original encoder path."""
from pathlib import Path
import json,sys,gc
import numpy as np
import torch
from train import ROOT,PROJECT,System,base,digest,dump
from evaluate import helpers
def main():
    base.setup()
    ref=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]
    assert digest(Path(ref['checkpoint']))==ref['sha256']
    state=torch.load(ref['checkpoint'],map_location='cpu')
    for rel,sha in state['config']['sources'].items():assert digest(PROJECT/rel)==sha,rel
    m=System('original').cuda()
    m.load_state_dict(state['model'],strict=True);m.requires_grad_(False).eval()
    ids=json.loads((ROOT/'identity_partition.json').read_text())['splits']['val']
    data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
    metric,_=helpers.evaluate(m,data,ids,base,ROOT/'reference_replay_predictions.npz')
    assert metric['counts']==ref['validation']['counts']
    assert abs(metric['macro']['dice']-ref['validation']['macro']['dice'])<1e-12
    prior=np.load(Path(ref['checkpoint']).parent/'uper64_val_predictions.npz')
    actual=np.load(ROOT/'reference_replay_predictions.npz')
    assert prior.files==actual.files
    for key in prior.files:assert np.array_equal(prior[key],actual[key])
    dump(ROOT/'REFERENCE_REPLAY_VERIFIED.json',dict(passed=True,n=762,binary_predictions_exact=True,
        macro_dice=metric['macro']['dice'],checkpoint_sha256=ref['sha256'],test_read=False))
    print('REFERENCE_REPLAY_VERIFIED',flush=True)
if __name__=='__main__':main()
