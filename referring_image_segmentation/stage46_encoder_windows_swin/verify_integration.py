"""Verify pretrained transfer, author kernels, real window communication and init."""
from pathlib import Path
import json,hashlib,inspect,copy,gc
import torch
from train import ROOT,PROJECT,System,base,dump,digest,setup
from encoder_adaptation import RESEARCH,AUTHOR,native_scope,assert_initial_transfer,DropPath,to_2tuple,trunc_normal_
def main():
    setup()
    manifest=json.loads((RESEARCH/'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:assert digest(RESEARCH/item['path'])==item['sha256']
    ref=System('original').cuda().eval()
    models={a:System(a).cuda().eval() for a in ['window8','swin8']}
    scope,nodes=native_scope()
    shape_audit={}
    x=torch.randn(2,3,256,256,device='cuda')
    for arm,model in models.items():
        assert_initial_transfer(model,ref)
        base.assert_tree_exact(model.probe.state_dict(),ref.probe.state_dict())
        assert all(not module.training for module in model.encoder.modules())
        hooks=[];record={}
        def capture(label):
            def hook(module,args):record[label]=list(args[0].shape)
            return hook
        for i in [2,3]:
            level=model.encoder.model.levels[i]
            assert level.window_size==8
            hooks.append(level.register_forward_pre_hook(capture('stage'+str(i+1)+'_map')))
            hooks.append(level.blocks[0].mixer.register_forward_pre_hook(capture('stage'+str(i+1)+'_mamba')))
            if arm=='swin8':
                for j,b in enumerate(level.blocks[level.mamba_count:]):
                    hooks.append(b.register_forward_pre_hook(capture('stage'+str(i+1)+'_swin'+str(j))))
                    assert b.shift_size==(4 if i==2 and j==1 else 0)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(x)
            assert output.shape==(2,1,256,256) and torch.isfinite(output).all()
        for h in hooks:h.remove()
        assert record['stage3_mamba']==[8,64,384] and record['stage4_mamba']==[2,64,768]
        if arm=='swin8':
            assert record['stage3_swin1']==[2,256,384] and record['stage4_swin0']==[2,64,768]
        shape_audit[arm]=record
    # The no-shift zero-bias native block must reproduce the original Attention+MLP block.
    fixed=models['window8'].encoder.model.levels[2].blocks[4]
    native=models['swin8'].encoder.model.levels[2].blocks[4]
    tokens=torch.randn(2,256,384,device='cuda')
    with torch.no_grad():
        windows=scope['window_partition'](tokens.view(2,16,16,384),8).view(-1,64,384)
        expected=scope['window_reverse'](fixed(windows).view(-1,8,8,384),8,16,16).view(2,256,384)
        actual=native(tokens)
        max_error=float((actual-expected).abs().max())
        assert torch.allclose(actual,expected,atol=3e-5,rtol=1e-5),max_error
    # Use actual transferred native shifted block; alter another ORIGINAL window only.
    shifted=models['swin8'].encoder.model.levels[2].blocks[5]
    unshifted=copy.deepcopy(shifted);unshifted.shift_size=0;unshifted.attn_mask=None
    z=torch.randn(1,256,384,device='cuda');changed=z.clone()
    changed[0,8*16+6,0]+=4
    with torch.no_grad():
        d_fixed=float((unshifted(z)[0,7*16+6]-unshifted(changed)[0,7*16+6]).abs().max())
        d_shift=float((shifted(z)[0,7*16+6]-shifted(changed)[0,7*16+6]).abs().max())
        assert d_fixed==0 and d_shift>1e-6,(d_fixed,d_shift)
        wrapped=z.clone();wrapped[0,15*16+6,0]+=4
        d_wrap=float((shifted(z)[0,0*16+6]-shifted(wrapped)[0,0*16+6]).abs().max())
        assert d_wrap<1e-6,d_wrap
    transfer=models['swin8'].transfer_records
    assert len([p for n,p in models['swin8'].named_parameters() if 'relative_position_bias_table' in n])==5
    helpers={inspect.getsourcefile(obj):digest(inspect.getsourcefile(obj)) for obj in [DropPath,to_2tuple,trunc_normal_]}
    dump(ROOT/'AUTHOR_PROVENANCE.json',dict(repository=manifest['repository'],commit=manifest['commit'],
        source_sha256=digest(AUTHOR),untouched_ast_nodes=nodes,timm_helper_sources=helpers,
        transferred_blocks=transfer,relative_bias_init='zero after native constructor; other old pretrained tensors exact',
        unchanged_uper_author_module_sha256=digest(PROJECT/'stage45_upernet_decoder/author_modules.py')))
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,all_old_pretrained_tensors_exact=True,
        identical_uper64_initialization=True,native_swin_source_hashes_verified=True,
        no_shift_zero_bias_fp32_max_error=max_error,cross_window_perturbation=dict(fixed=d_fixed,shifted=d_shift,wrapped=d_wrap),
        runtime_shapes=shape_audit,zero_padding_with8_windows=True,encoder_eval=True,test_images_read=False))
    print('INTEGRATION_VERIFIED',max_error,d_fixed,d_shift,d_wrap,flush=True)
if __name__=='__main__':main()
