"""Frozen seed42 reference definitions; probabilities and checkpoints stay read-only."""
import json
import numpy as np
from models import PROJECT, SingleProbe, CHANNELS, FusionProbe
from stage5_selective_fusion.models import CapacityProbe, ResidualProbe

REFERENCE_PATHS={
    **{n:PROJECT/f'stage4_fusion/results/seed42/{n}_swin' for n in ('dinov3','vmamba','fusion')},
    **{n:PROJECT/f'stage5_selective_fusion/results/seed42/{n}' for n in ('capacity_concat','static_residual','dynamic_gate')}}


def build_reference_probes():
    return dict(dinov3=SingleProbe('swin',42,CHANNELS['dinov3']),vmamba=SingleProbe('swin',42,CHANNELS['vmamba']),
        fusion=FusionProbe('swin',42),capacity_concat=CapacityProbe(42),static_residual=ResidualProbe(42,False),dynamic_gate=ResidualProbe(42,True))


def reference_features(name,features):
    if name=='dinov3':return features[:4]
    if name=='vmamba':return features[4:]
    if name=='capacity_concat':return features[:4]+features[:4]
    assert name in ('fusion','static_residual','dynamic_gate')
    return features


def references(criterion,ids):
    probs={}
    for n,base in REFERENCE_PATHS.items():
        assert json.loads((base/criterion/'ids.json').read_text())==ids
        probs[n]=np.load(base/criterion/'probabilities.npy',mmap_mode='r')
    probs['mean']=(probs['dinov3']+probs['vmamba'])*.5
    return probs
