import json,torch
from train import ROOT,System,base,dump,setup,optimizer
def main():
    setup()
    early=System('early_adapt').cuda()
    late=System('swin_control').cuda()
    base.assert_tree_exact(early.state_dict(),late.state_dict())
    ep={n for n,p in early.named_parameters() if p.requires_grad}
    lp={n for n,p in late.named_parameters() if p.requires_grad}
    extra=ep-lp
    assert extra and lp<=ep
    assert all(n.startswith(('encoder.model.patch_embed.','encoder.model.levels.0.','encoder.model.levels.1.')) for n in extra)
    x=torch.randn(4,3,256,256,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        f=early.prefix(x);g=late.prefix(x)
        assert all(torch.equal(a,b) for a,b in zip(f,g))
        assert all(a.requires_grad for a in f)
        assert all(not a.requires_grad for a in g)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=early(x);b=late(x)
        assert torch.equal(a,b) and a.shape==(4,1,256,256)
    assert all(not m.training for m in early.encoder.modules())
    opt=optimizer(early)
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,
        identical_initial_parameters_and_buffers=True,initial_forward_bitwise_exact=True,
        gradient_enabled_early_prefix=True,no_feature_cache=True,encoder_buffers_eval=True,
        extra_trainable_names=sorted(extra),optimizer_groups=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in opt.param_groups]))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
