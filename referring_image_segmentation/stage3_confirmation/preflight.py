"""Check seed control and equivalence to archived model and geometry definitions."""
import json
from pathlib import Path
import torch
from models import Probe,tensor_hash
from data import ImageData,BaseData

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from stage1_visual.models import Probe as OldLight
    from stage2_decoder.models import Probe as OldSwin
    torch.manual_seed(42);old_light=OldLight([384]*4)
    torch.manual_seed(42);old_swin=OldSwin([384]*4)
    assert tensor_hash(old_light)==tensor_hash(Probe('light',42))
    assert tensor_hash(old_swin)==tensor_hash(Probe('swin',42))
    dummy=ImageData.__new__(ImageData);dummy.seed=42;dummy.ids={'train':list(range(1886))}
    for epoch in (1,2,300):
        old=BaseData.geometry(dummy,epoch);new=ImageData.geometry(dummy,epoch)
        assert all(torch.equal(a,b) for a,b in zip(old,new))
    hashes={};geometries={}
    for seed in (43,44,45):
        light,swin=Probe('light',seed),Probe('swin',seed)
        assert tensor_hash(light.adapter)==tensor_hash(swin.adapter)
        assert tensor_hash(light.decoder.readout)==tensor_hash(swin.decoder.readout)
        hashes[str(seed)]={d:tensor_hash(p) for d,p in [('light',light),('swin',swin)]}
        dummy.seed=seed;geometries[seed]=dummy.geometry(1)
    for d in ('light','swin'):assert len({v[d] for v in hashes.values()})==3
    assert not torch.equal(geometries[43][0],geometries[44][0])
    report=dict(passed=True,seed42_matches_archived_initializations=True,seed42_geometry_matches=True,
        distinct_seeds=True,paired_adapter_readout_equal=True,initial_probe_hashes=hashes)
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
