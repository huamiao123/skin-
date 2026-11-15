"""Check pinned author LoRA and actual late-block call sites before training."""
import json,torch
from torch.nn import functional as F
from train import ROOT,PROJECT,System,base,setup,dump,digest
from author_modules import LoRALinear,AUTHOR
def main():
    setup()
    for row in json.loads((PROJECT.parent.parent/'research/peft_followup_20261112/SOURCE_MANIFEST.json').read_text()):
        assert digest(row['local'])==row['sha256']
    original=System('original').cuda().eval()
    shared=base.cpu_tree(original.state_dict())
    x=torch.randn(4,3,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):expected=original(x)
    checks={}
    for arm in ['frozen','lora8']:
        model=System(arm).cuda().eval()
        base.assert_tree_exact({k:v for k,v in model.state_dict().items() if '.lora_' not in k},shared)
        calls={};handles=[]
        for name,module in model.encoder.named_modules():
            if isinstance(module,LoRALinear):
                def hit(m,a,b,n=name):calls[n]=calls.get(n,0)+1
                handles.append(module.register_forward_hook(hit))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(x)
        assert torch.equal(actual,expected)
        for h in handles:h.remove()
        if arm=='lora8':
            assert len(calls)==48 and all(v==1 for v in calls.values())
            assert {r['path'] for r in model.adapter_targets}==set(calls)
            module=next(m for m in model.encoder.modules() if isinstance(m,LoRALinear))
            v=torch.randn(2,5,module.in_features,device='cuda')
            with torch.no_grad():
                module.lora_B.normal_(std=.01)
                y=module(v)
                manual=F.linear(v,module.weight,module.bias)+(v@module.lora_A.T@module.lora_B.T)*module.scaling
                assert torch.equal(y,manual)
                w=module.weight.clone()
                model.train();model.eval();model.train()
                assert torch.equal(w,module.weight) and not module.merged and not module.merge_weights
        checks[arm]=dict(shared_weights_exact=True,initial_output_exact=True,actual_adapter_calls=calls,
            encoder_trainable=sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
            decoder_trainable=sum(p.numel() for p in model.probe.parameters() if p.requires_grad),
            total=sum(p.numel() for p in model.parameters()),targets=model.adapter_targets)
        del model
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,checks=checks,
        native_lora_file=str(AUTHOR),native_lora_sha256=digest(AUTHOR),manual_forward_exact=True,
        automatic_merge_disabled=True,no_ssm_or_window_change=True))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
