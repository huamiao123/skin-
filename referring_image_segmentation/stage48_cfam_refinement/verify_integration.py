"""Stage48 integration checks, before any formal segmentation training."""
from pathlib import Path
import copy,gc,inspect,json
import torch
from torch.nn import functional as F
from train import ROOT,PROJECT,System,base,dump,digest,setup,objective
from author_modules import RefinedUPerProbe,NativeUPerProbe,SOURCE,load_cfam

def manual_forward(head,features,refiner):
    inputs=head._transform_inputs(features)
    laterals=[f(inputs[i]) for i,f in enumerate(head.lateral_convs)]
    laterals.append(head.psp_forward(inputs))
    for i in range(len(laterals)-1,0,-1):
        laterals[i-1]=laterals[i-1]+F.interpolate(laterals[i],size=laterals[i-1].shape[2:],mode='bilinear',align_corners=head.align_corners)
    outs=[f(laterals[i]) for i,f in enumerate(head.fpn_convs)]
    before=outs[1]
    outs[1]=refiner(before).to(before.dtype)
    outs.append(laterals[-1])
    for i in range(len(outs)-1,0,-1):
        outs[i]=F.interpolate(outs[i],size=outs[0].shape[2:],mode='bilinear',align_corners=head.align_corners)
    y=head.cls_seg(head.fpn_bottleneck(torch.cat(outs,dim=1)))
    return F.interpolate(y,size=(256,256),mode='bilinear',align_corners=False)

def main():
    setup()
    manifest=json.loads((SOURCE.parent/'SOURCE_MANIFEST.json').read_text())
    for x in manifest:
        if x['repo']=='xmindflow/cenet':assert digest(SOURCE/x['path'])==x['sha256']
    original=System('original').cuda().eval()
    original_state=base.cpu_tree(original.state_dict())
    x=torch.randn(4,3,256,256,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        f1,f2,mid=original.prefix(x);f3,f4=original.suffix(mid);features=[f1,f2,f3,f4]
        reference=original.probe(features)
    checks={};provenance={}
    for arm in ['cfam','conv_control']:
        model=System(arm).cuda().eval()
        shared={k:v for k,v in model.state_dict().items() if not k.startswith('probe.refiner.')}
        base.assert_tree_exact(shared,original_state)
        for mode in ['bypass','zero_scale']:
            model.probe.enabled=mode!='bypass'
            saved={}
            if mode=='zero_scale':
                for n,p in model.probe.refiner.named_parameters():
                    if 'layer_scale' in n:
                        saved[n]=p.detach().clone()
                        with torch.no_grad():p.zero_()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=model.probe(features)
            assert y.dtype==reference.dtype and torch.equal(y,reference),(arm,mode,float((y-reference).abs().max()))
            for n,p in model.probe.refiner.named_parameters():
                if n in saved:
                    with torch.no_grad():p.copy_(saved[n])
        model.probe.enabled=True
        seen={}
        def record(m,args,out):seen.update(shape=list(args[0].shape),input_dtype=str(args[0].dtype),output_dtype=str(out.dtype))
        hook=model.probe.refiner.register_forward_hook(record)
        twin=copy.deepcopy(model.probe.refiner).eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            actual=model.probe(features)
            expected=manual_forward(original.probe.decoder,features,twin)
        assert torch.equal(actual,expected),(arm,float((actual-expected).abs().max()))
        hook.remove()
        # Reset to train only after forward/identity checks; encoder remains eval.
        model.train();assert model.probe.refiner.training
        assert all(not m.training for m in model.encoder.modules())
        assert all(p.requires_grad for level in model.encoder.model.levels[2:] for p in level.parameters())
        assert all(not p.requires_grad for level in model.encoder.model.levels[:2] for p in level.parameters())
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        soft=torch.full_like(logits,.501,dtype=torch.float32)
        loss=objective(logits,soft);loss.backward()
        grads={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in model.probe.refiner.named_parameters()}
        assert all(v is not None and v>=0 for v in grads.values())
        assert all(torch.isfinite(p.grad).all() for p in model.probe.refiner.parameters())
        assert sum(grads.values())>0
        checks[arm]=dict(shared_encoder_and_decoder_initial_state_exact=True,neutral_bypass_logits_exact=True,
            zero_layer_scale_logits_exact=True,independent_manual_insertion_logits_exact=True,interface=seen,
            author_math_preserved_with_explicit_output_dtype_cast=True,gradient_l1=grads,
            encoder_late_only=True,encoder_all_eval=True,refiner_train_mode=True,
            refiner_parameters=sum(p.numel() for p in model.probe.refiner.parameters()),
            no_native_loss_call=True,loss=float(loss))
        provenance[arm]=model.probe.provenance
        del model,shared,twin,actual,expected,logits,loss,soft,grads
        gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'AUTHOR_PROVENANCE.json',dict(author_repository='xmindflow/cenet',
        commit=json.loads((SOURCE/'PIN.json').read_text())['commit'],
        class_definitions=provenance,source_manifest_sha256=digest(SOURCE.parent/'SOURCE_MANIFEST.json'),
        integration='Unmodified author CFAM AST, fpn_convs[1] forward-output hook, cast back to FPN dtype. Full CENet not reproduced.',
        control='Conventional parameter-matched 2conv residual; no attention; not FLOP-matched.'))
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,checks=checks,test_images_read=False,
        existing_environment_unchanged=True,baseline_architecture='stage45_uper64'))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
