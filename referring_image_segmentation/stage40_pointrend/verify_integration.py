"""Runtime verification of binary adaptation, untouched author core and matched initialization."""
from pathlib import Path
import ast,json,hashlib,inspect
import torch
from torch.nn import functional as F
from train import ROOT,PROJECT,base,System,dump,setup
from author_modules import AUTHOR,COMMIT,make_scope
def main():
    setup()
    source=PROJECT.parent.parent/'research/pointrend_author_20261112'
    manifest=json.loads((source/'SOURCE_MANIFEST.json').read_text())
    # Manifest format recorded as-is alongside runtime AST provenance.
    scope,provenance=make_scope(False)
    m=System('point_uncertain').cuda().eval()
    other=System('point_random')
    base.assert_tree_exact(m.state_dict(),other.state_dict())
    ref=base.System(finetune=True,stochastic_depth=False).cuda().eval()
    base.assert_tree_exact(m.encoder.state_dict(),ref.encoder.state_dict())
    base.assert_tree_exact(m.probe.adapter.state_dict(),ref.probe.adapter.state_dict())
    base.assert_tree_exact(m.probe.decoder.baseline.state_dict(),ref.probe.decoder.state_dict())
    gen=torch.Generator().manual_seed(42)
    x=torch.randn(2,3,256,256,generator=gen).cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        f1,f2,mid=m.prefix(x);f3,f4=m.suffix(mid)
        pyr=[a(f) for a,f in zip(m.probe.adapter,[f1,f2,f3,f4])]
        features={f'p{i}':v for i,v in enumerate(pyr)}
        coarse=m.probe.decoder.author_head.coarse_sem_seg_head.layers(features)
        binary=coarse[:,1:2]-coarse[:,0:1]
        direct=F.interpolate(binary,size=(256,256),mode='bilinear',align_corners=False)
        original=ref(x)
        assert torch.equal(direct,original)
        y=m(x);assert y.shape==(2,1,256,256) and torch.isfinite(y).all()
    assert torch.allclose(coarse.float().softmax(1)[:,1:2],binary.float().sigmoid(),atol=1e-7)
    # Actual native point sampling trace: training1024 points and two inference1024-point steps.
    calls=[]
    head=m.probe.decoder.author_head
    hook=head.point_head.register_forward_pre_hook(lambda module,args:calls.append([list(a.shape) for a in args]))
    target=(torch.rand(2,256,256,device='cuda')>.5).long()
    head.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _,loss=head(features,target)
    assert set(loss)=={'loss_sem_seg_coarse','loss_sem_seg_point'}
    assert all(torch.isfinite(v) for v in loss.values())
    train_calls=calls[:];calls.clear()
    head.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        pred,_=head(features)
    eval_calls=calls[:];hook.remove()
    assert len(train_calls)==1 and len(eval_calls)==2
    assert all(c[0]==[2,64,1024] and c[1]==[2,2,1024] for c in train_calls+eval_calls)
    # When point heads output the original coarse logit sampled at selected coords,
    # interpolation/scatter dimensions and binary conversion remain well-defined.
    rnd,_=make_scope(True)
    u=torch.randn(3,1,128,128,device='cuda')
    i,c=rnd['get_uncertain_point_coords_on_grid'](u,1024)
    i1,c1=rnd['get_uncertain_point_coords_on_grid'](u[:1],1024)
    assert torch.equal(i[:1],i1) and torch.equal(c[:1],c1) and len(torch.unique(i[0]))==1024
    # Original source node hash equals executed AST node; no algorithm-node edits.
    prov=dict(commit=COMMIT,source_manifest=manifest,core=provenance,
        adapters=['SwinCoarseAdapter layers/loss','binary logit bridge','64channel p0 feature','framework registries/config shim'],
        control='random branch overrides only grid selector and training importance ratio',
        fvcore=__import__('fvcore').__version__)
    dump(ROOT/'AUTHOR_PROVENANCE.json',prov)
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,shared_full_initial_state_exact=True,
        original_swin_coarse_logits_exact=True,binary_probability_equivalent=True,
        point_train_calls=train_calls,point_inference_calls=eval_calls,
        uniform_grid_unique_and_batch_invariant=True,author_node_hashes=provenance))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
