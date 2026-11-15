from pathlib import Path
import json,torch
from train import ROOT,PROJECT,System,base,dump,setup,digest
from author_modules import PointRendDecoder
def main():
    setup()
    ref=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]
    assert digest(Path(ref['checkpoint']))==ref['sha256']
    state=torch.load(ref['checkpoint'],map_location='cpu')
    for rel,sha in state['config']['sources'].items():assert digest(PROJECT/rel)==sha,rel
    m=System().cuda().eval()
    b=base.System(finetune=False,stochastic_depth=False).cuda().eval()
    b.load_state_dict({k.replace('probe.decoder.baseline.','probe.decoder.'):v for k,v in state['model'].items()},strict=True)
    base.assert_tree_exact(m.encoder.state_dict(),b.encoder.state_dict())
    base.assert_tree_exact(m.probe.adapter.state_dict(),b.probe.adapter.state_dict())
    base.assert_tree_exact(m.probe.decoder.baseline.state_dict(),b.probe.decoder.state_dict())
    native=PointRendDecoder('point_uncertain',b.probe.decoder).cuda().eval()
    base.assert_tree_exact(m.point_head.state_dict(),native.author_head.point_head.state_dict())
    x=torch.randn(4,3,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        fine,coarse=m.frozen_features(x)
        got=m._head(fine,coarse)
        f1,f2,mid=b.prefix(x);f3,f4=b.suffix(mid)
        pyr=[a(f) for a,f in zip(b.probe.adapter,[f1,f2,f3,f4])]
        expected=native(pyr)
        assert torch.equal(got,expected)
        m.inference_policy='coarse_direct'
        assert torch.equal(m(x),b(x))
        m.inference_policy='uncertain'
    trainable=[n for n,p in m.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith('probe.decoder.author_head.point_head.') for n in trainable)
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,baseline_state_exact=True,native_point_head_init_exact=True,
        cached_native_head_logits_exact=True,coarse_logits_exact=True,only_point_parameters_trainable=True,
        trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
        reference_sha256=ref['sha256'],frozen_digest=m.frozen_digest()))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
