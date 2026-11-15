"""Audit native CAB, pixel/token mapping, padding isolation and actual residual call path."""
import json,copy,gc,torch
from torch.nn import functional as F
from train import ROOT,PROJECT,System,base,setup,dump,digest,source_hashes
from author_modules import native_classes,SOURCE,AUTHOR
def main():
    setup()
    pin=json.loads((SOURCE/'PIN.json').read_text())
    assert pin['commit']=='33d7b3460c4665229334cc8de38c7f4c766ed3be'
    original=System('original').cuda().eval();shared=base.cpu_tree(original.state_dict())
    x=torch.randn(4,3,256,256,device='cuda');original_mixer=[]
    hook=original.encoder.model.levels[2].blocks[3].mixer.register_forward_hook(lambda m,a,b:original_mixer.append(b.detach().clone()))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):expected=original(x)
    hook.remove();checks={};conv_states={}
    for arm in ['cab','conv_control']:
        model=System(arm).cuda().eval()
        base.assert_tree_exact({k:v for k,v in model.state_dict().items() if '.spatial_adapter.' not in k},shared)
        a=model.encoder.model.levels[2].blocks[3].mlp.spatial_adapter
        conv_states[arm]=base.cpu_tree({k:v for k,v in a.state_dict().items() if k.startswith(('cab.cab.0.','cab.cab.2.'))})
        seen=[];mix=[]
        h=a.register_forward_hook(lambda m,i,o:seen.append((i[0].detach().clone(),o.detach().clone())))
        h2=model.encoder.model.levels[2].blocks[3].mixer.register_forward_hook(lambda m,i,o:mix.append(o.detach().clone()))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(x)
        h.remove();h2.remove()
        assert len(seen)==1 and len(mix)==1 and torch.equal(mix[0],original_mixer[0])
        assert torch.equal(actual,expected) and torch.count_nonzero(seen[0][1])==0
        # Exact coordinate roundtrip, not a reshape that mixes channels and pixels.
        coords=torch.arange(2*384*16*16,device='cuda').reshape(2,384,16,16).float()
        t=a.partition(F.pad(coords,(0,12,0,12)),14)
        back=a.reverse(t,14,28,28)[:,:,:16,:16]
        assert torch.equal(coords,back)
        mask=a.partition(F.pad(torch.ones(2,1,16,16,device='cuda'),(0,12,0,12)),14).bool()
        # Nonzero adapter: compare true full16x16 CAB with returned window tokens.
        with torch.no_grad():
            a.scale.fill_(.1)
            local=a.cab(coords/10000)*a.scale[None,:,None,None]
            direct=a.partition(F.pad(local,(0,12,0,12)),14)
            delta=a(t/10000)
            assert torch.equal(delta,direct)
            perturbed=t/10000+torch.where(mask,0.,123.)
            assert torch.equal(a(perturbed),delta)
            assert torch.count_nonzero(delta.masked_select(~mask.expand_as(delta)))==0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):changed=model(x)
        assert not torch.equal(changed,expected)
        max_change=float((changed.float()-expected.float()).abs().max())
        checks[arm]=dict(shared_original_weights_exact=True,zero_scale_initial_logits_exact=True,
            stage3_last_mamba_mixer_output_exact=True,adapter_called_once=True,
            pixel_token_roundtrip_exact=True,full_spatial_manual_forward_exact=True,
            padding_cannot_affect_adapter=True,adapter_padded_output_zero=True,
            nonzero_scale_changes_logits=True,max_logit_change=max_change,
            native_ast_hashes=a.ast_hashes,adapter_parameters=sum(p.numel() for p in a.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            target=model.adapter_targets)
        del model,a,seen,mix,coords,t,back,mask,delta,local,direct,perturbed,changed
        gc.collect();torch.cuda.empty_cache()
    base.assert_tree_exact(conv_states['cab'],conv_states['conv_control'])
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,checks=checks,
        ordinary_convolution_weights_matched=True,author_file=str(AUTHOR),author_sha256=digest(AUTHOR),
        sources=source_hashes()))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
