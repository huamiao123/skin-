"""Check pinned author forward, exact initialization and loss/data interface before formal runs."""
from pathlib import Path
import inspect,json,gc,importlib.metadata
import torch
from torch.nn import functional as F
from train import ROOT,PROJECT,System,base,dump,digest,setup,objective
from author_modules import NativeUPerProbe,RESEARCH,ConvModule,BaseModule
def main():
    setup()
    manifest=json.loads((RESEARCH/'SOURCE_MANIFEST.json').read_text())
    # The fetched-source manifest is checked by path independently of the module import.
    for repo in manifest['repositories']:
        root=RESEARCH/'official_code'/repo['repository'].replace('/','__')
        for entry in repo['files']:
            q=root/entry['path']
            assert digest(q)==entry['sha256'],str(q)
    audit=json.loads((RESEARCH/'ENVIRONMENT_AUDIT.json').read_text())
    assert audit['existing_package_changes']=={} and audit['numpy_bridge_passed']
    checks={}
    provenance={}
    for arm,width in [('uper64',64),('uper128',128)]:
        model=System(arm).cuda()
        twin=NativeUPerProbe(width)
        base.assert_tree_exact(model.probe.state_dict(),twin.state_dict())
        assert model.probe.decoder._is_init
        head=model.probe.decoder
        assert list(head.in_channels)==[96,192,384,768] and head.channels==width
        assert head.input_transform=='multiple_select' and head.out_channels==1
        assert tuple(head.psp_modules.pool_scales)==(1,2,3,6)
        assert isinstance(head.psp_modules[0][1].norm,torch.nn.SyncBatchNorm)
        assert abs(float(head.conv_seg.weight.std())-.01)<.003
        # Native MMSeg loss is intentionally not part of this controlled adaptation.
        def forbid_loss(*_):raise AssertionError('Native CE loss must not replace project soft BCE+Dice')
        h=head.loss_decode.register_forward_pre_hook(forbid_loss)
        x=torch.randn(4,3,256,256,device='cuda')
        shapes={}
        def capture(name):
            def hook(module,args,output):shapes[name]=list(output.shape)
            return hook
        hooks=[head.psp_modules[i].register_forward_hook(capture('ppm_'+str(i))) for i in range(4)]
        hooks += [m.register_forward_hook(capture('lateral_'+str(i))) for i,m in enumerate(head.lateral_convs)]
        hooks += [m.register_forward_hook(capture('fpn_'+str(i))) for i,m in enumerate(head.fpn_convs)]
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            f1,f2,mid=model.prefix(x);f3,f4=model.suffix(mid);features=[f1,f2,f3,f4]
            output=model.probe(features)
            expected=F.interpolate(head(features),size=(256,256),mode='bilinear',align_corners=False)
            assert torch.equal(output,expected) and output.shape==(4,1,256,256)
        for hook in hooks:hook.remove()
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        soft=torch.full_like(logits,.501,dtype=torch.float32)
        loss=objective(logits,soft)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(head.conv_seg.weight.grad).all() and head.conv_seg.weight.grad.abs().sum()>0
        assert all(not m.training for m in model.encoder.modules())
        assert all(not z.requires_grad for z in model.encoder.model.patch_embed.parameters())
        assert all(not z.requires_grad for stage in model.encoder.model.levels[:2] for z in stage.parameters())
        assert all(z.requires_grad for stage in model.encoder.model.levels[2:] for z in stage.parameters())
        checks[arm]=dict(native_head_initialized=True,deterministic_initial_state=True,
            native_forward_exact=True,intermediate_shapes=shapes,soft_target_path_preserved=True,
            native_ce_not_called=True,encoder_late_only=True,all_encoder_modules_eval=True,
            decoder_parameters=sum(z.numel() for z in model.probe.parameters()),
            total_parameters=sum(z.numel() for z in model.parameters()))
        provenance[arm]=model.probe.provenance
        h.remove()
        del model,twin,head,x,f1,f2,mid,f3,f4,features,output,expected,logits,loss,soft
        gc.collect();torch.cuda.empty_cache()
    deps={name:importlib.metadata.version(name) for name in ['mmcv-lite','mmengine','numpy','torch']}
    dependency_sources={inspect.getsourcefile(obj):digest(inspect.getsourcefile(obj)) for obj in [ConvModule,BaseModule]}
    dump(ROOT/'AUTHOR_PROVENANCE.json',dict(untouched_extracted_ast_nodes=provenance,
        dependency_versions=deps,dependency_sources=dependency_sources,manifest_sha256=digest(RESEARCH/'SOURCE_MANIFEST.json'),
        used_native_forward=True,used_native_init=True,
        adaptation='Forward-only main head integration; native loss/predict_by_feat framework APIs are not used; common project soft BCE+Dice and fixed evaluation retained.'))
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,checks=checks,source_manifest_checked=True,
        existing_numerical_environment_preserved=True,test_images_read=False))
    print('INTEGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
