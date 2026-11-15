"""Shared initialization, objective and inference audit before real-data training."""
import json,gc
from pathlib import Path
import torch
from train import ROOT,PROJECT,System,base,dump,digest,setup,objective
from author_modules import _import
def main():
    setup()
    for p,sha in json.loads((ROOT/'ARCHIVED_SOURCE_HASHES.json').read_text()).items():
        assert digest(Path(p))==sha
    single=System('emcad_single')
    deep=System('emcad_deep')
    a=single.state_dict();b=deep.state_dict()
    assert all(torch.equal(v,b[k]) for k,v in a.items())
    added=set(b)-set(a)
    assert added and all(k.startswith('probe.decoder.aux_heads.') for k in added)
    prior=_import(PROJECT/'stage34_author_integration_fix/author_modules.py','original_exp2_for_init')
    original=prior.AuthorModuleDecoder('emcad64_exp2',base.System(finetune=True,stochastic_depth=False).probe.decoder,seed=42)
    base.assert_tree_exact(single.probe.decoder.state_dict(),original.state_dict())
    for m in [single,deep]:
        opt=base.optimizer(m)
        assert {id(q) for g in opt.param_groups for q in g['params']}=={id(q) for q in m.parameters() if q.requires_grad}
    features=[torch.randn(2,64,h,h) for h in [64,32,16,8]]
    single.eval();deep.eval()
    with torch.no_grad():
        x=single.probe.decoder(features);y=deep.probe.decoder(features)
        assert torch.equal(x,y)
    deep.probe.decoder.train()
    outputs=deep.probe.decoder(features)
    assert isinstance(outputs,list) and len(outputs)==4
    assert all(q.shape==(2,1,256,256) and torch.isfinite(q).all() for q in outputs)
    target=torch.randint(0,2,(2,1,256,256)).float()
    actual=objective(outputs,target)
    expected=(sum(base.segmentation_loss(q.float(),target) for q in outputs)
        +base.segmentation_loss(torch.stack([q.float() for q in outputs]).sum(0),target))/5
    assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-7)
    deep.probe.decoder.eval()
    assert isinstance(deep.probe.decoder(features),torch.Tensor)
    result=dict(passed=True,shared_initial_state_exact=True,original_exp2_initial_state_exact=True,
       extra_parameters=sum(q.numel() for q in deep.probe.decoder.aux_heads.parameters()),
       identical_initial_inference=True,four_training_outputs=True,five_term_loss_mean_verified=True,
       one_output_inference=True,optimizer_owns_all_trainable=True,archived_sources_unchanged=True,test_read=False)
    assert result['extra_parameters']==195
    dump(ROOT/'INTEGRATION_VERIFIED.json',result)
    print('INTEGRATION_OK',json.dumps(result),flush=True)
if __name__=='__main__':main()
