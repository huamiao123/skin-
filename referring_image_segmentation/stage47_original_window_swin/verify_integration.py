"""Original-window Swin: preserve Mamba exactly and isolate attention interventions."""
from pathlib import Path
import json,inspect,copy
import torch
from train import ROOT,PROJECT,System,ARMS,base,dump,digest,setup
from encoder_adaptation import RESEARCH,AUTHOR,native_scope,assert_initial_transfer,DropPath,to_2tuple,trunc_normal_

def stage_mamba_output(level,x,count):
    saved={}
    def hook(module,args,output):saved['tokens']=output.detach().clone()
    h=level.blocks[count-1].register_forward_hook(hook)
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):level(x)
    finally:h.remove()
    return saved['tokens']

def main():
    setup()
    manifest=json.loads((RESEARCH/'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:assert digest(RESEARCH/item['path'])==item['sha256']
    ref=System('original').cuda().eval()
    models={a:System(a).cuda().eval() for a in ARMS}
    scope,nodes=native_scope()
    checks={}
    for arm,model in models.items():
        assert_initial_transfer(model,ref)
        base.assert_tree_exact(model.probe.state_dict(),ref.probe.state_dict())
        assert all(not m.training for m in model.encoder.modules())
        for i,ws,padded in [(2,14,28),(3,7,14)]:
            lev=model.encoder.model.levels[i]
            assert lev.window_size==ws and lev.padded_hw==padded
            for j,b in enumerate(lev.blocks[lev.mamba_count:]):
                assert b.shift_size==(ws//2 if arm=='shifted_windows' and j%2 else 0)
                assert b.attn.relative_position_bias_table.requires_grad==(arm!='native_fixed')
    # All parameter values/buffers match; only bias trainability differs.
    base.assert_tree_exact(models['native_fixed'].state_dict(),models['position_bias'].state_dict())
    runtime={};mamba={}
    x=torch.randn(2,3,256,256,device='cuda')
    outputs={}
    for arm,model in models.items():
        shapes={};hooks=[]
        def capture(key):
            def hook(module,args):shapes[key]=list(args[0].shape)
            return hook
        for i in (2,3):
            level=model.encoder.model.levels[i]
            hooks.append(level.blocks[0].mixer.register_forward_pre_hook(capture('stage'+str(i+1)+'_mamba')))
            hooks.append(level.blocks[level.mamba_count].register_forward_pre_hook(capture('stage'+str(i+1)+'_attention')))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            outputs[arm]=model(x)
            assert outputs[arm].shape==(2,1,256,256) and torch.isfinite(outputs[arm]).all()
        for h in hooks:h.remove()
        assert shapes['stage3_mamba']==[8,196,384] and shapes['stage4_mamba']==[8,49,768]
        assert shapes['stage3_attention']==[2,784,384] and shapes['stage4_attention']==[2,196,768]
        runtime[arm]=shapes
    assert torch.equal(outputs['native_fixed'],outputs['position_bias'])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):original_output=ref(x)
    implementation_delta=float((outputs['native_fixed']-original_output).abs().max())
    for i,hw,ws,dim,count in [(2,16,14,384,4),(3,8,7,768,3)]:
        z=torch.randn(2,dim,hw,hw,device='cuda')
        expected=stage_mamba_output(ref.encoder.model.levels[i],z,count)
        for arm,model in models.items():
            actual=stage_mamba_output(model.encoder.model.levels[i],z,count)
            assert torch.equal(actual,expected),(arm,i,'Mamba path')
        padded=((hw+ws-1)//ws)*ws
        valid=torch.zeros(2,padded,padded,1,device='cuda')
        valid[:,:hw,:hw]=1
        valid=scope['window_partition'](valid,ws).view(-1,ws*ws).bool()
        nonzero_fraction=float((expected[~valid].abs().sum(-1)>0).float().mean())
        mamba[str(i+1)]=dict(shape=list(expected.shape),bitwise_equal_all_arms=True,padded_tokens_nonzero_fraction=nonzero_fraction)
    equivalence={};interaction={}
    for i,dim,ws,padded,mc in [(2,384,14,28,4),(3,768,7,14,3)]:
        tokens=torch.randn(2,padded*padded,dim,device='cuda')
        old=ref.encoder.model.levels[i].blocks[mc]
        new=models['native_fixed'].encoder.model.levels[i].blocks[mc]
        with torch.no_grad():
            win=scope['window_partition'](tokens.view(2,padded,padded,dim),ws).view(-1,ws*ws,dim)
            expected=scope['window_reverse'](old(win).view(-1,ws,ws,dim),ws,padded,padded).view(2,padded*padded,dim)
            actual=new(tokens)
            err=float((actual-expected).abs().max())
            old64=copy.deepcopy(old).double();old64.mixer.fused_attn=False
            new64=copy.deepcopy(new).double()
            win64=scope['window_partition'](tokens.double().view(2,padded,padded,dim),ws).view(-1,ws*ws,dim)
            expected64=scope['window_reverse'](old64(win64).view(-1,ws,ws,dim),ws,padded,padded).view(2,padded*padded,dim)
            actual64=new64(tokens.double())
            diag=dict(stage=i+1,max_abs_fp32=err,max_abs_fp64=float((actual64-expected64).abs().max()),
                relative_l2_fp32=float(torch.linalg.vector_norm((actual-expected).double())/torch.linalg.vector_norm(expected.double())),
                original_fp32_vs_fp64=float(torch.linalg.vector_norm(expected.double()-expected64)/torch.linalg.vector_norm(expected64)),
                native_fp32_vs_fp64=float(torch.linalg.vector_norm(actual.double()-expected64)/torch.linalg.vector_norm(expected64)),
                output_max_abs=float(expected.abs().max()),tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
            dump(ROOT/f'PRECISION_DIAGNOSTIC_STAGE{i+1}.json',diag)
            print('PRECISION_DIAGNOSTIC',diag,flush=True)
            assert torch.equal(actual64,expected64),diag
            # FP32 fused SDPA vs explicit native matmul/softmax: compare scale-aware error.
            assert diag['relative_l2_fp32']<=2e-6,diag
            assert err/max(1.,diag['output_max_abs'])<=5e-6,diag
            assert diag['original_fp32_vs_fp64']<=2e-6 and diag['native_fp32_vs_fp64']<=2e-6,diag
        equivalence[str(i+1)]=diag
        shifted=models['shifted_windows'].encoder.model.levels[i].blocks[mc+1]
        fixed=copy.deepcopy(shifted);fixed.shift_size=0;fixed.attn_mask=None
        z=torch.randn(2,padded*padded,dim,device='cuda');altered=z.clone()
        col=ws//2-1;target=(ws-1)*padded+col
        altered[0,ws*padded+col,0]+=4
        with torch.no_grad():
            a=fixed(z);b=fixed(altered);c=shifted(z);d=shifted(altered)
            d_fixed=float((a[0,target]-b[0,target]).abs().max())
            d_shift=float((c[0,target]-d[0,target]).abs().max())
            d_batch=float((c[1]-d[1]).abs().max())
            assert d_fixed==0 and d_shift>1e-6 and d_batch==0,(i,d_fixed,d_shift,d_batch)
            wrap=z.clone();wrap[0,(padded-1)*padded+col,0]+=4
            d_wrap=float((shifted(z)[0,col]-shifted(wrap)[0,col]).abs().max())
            assert d_wrap<1e-6,(i,d_wrap)
        interaction[str(i+1)]=dict(fixed=d_fixed,shifted=d_shift,cyclic_wrap=d_wrap,other_sample=d_batch)
    helpers={inspect.getsourcefile(obj):digest(inspect.getsourcefile(obj)) for obj in [DropPath,to_2tuple,trunc_normal_]}
    dump(ROOT/'AUTHOR_PROVENANCE.json',dict(repository=manifest['repository'],commit=manifest['commit'],
        source_sha256=digest(AUTHOR),untouched_ast_nodes=nodes,timm_helper_sources=helpers,
        transferred_blocks={arm:m.transfer_records for arm,m in models.items()},
        bias_init='Zero in all arms; frozen only in native_fixed',original_window_and_padding=True,
        unchanged_uper_author_module_sha256=digest(PROJECT/'stage45_upernet_decoder/author_modules.py')))
    dump(ROOT/'INTEGRATION_VERIFIED.json',dict(passed=True,pretrained_tensors_exact=True,decoder_init_exact=True,
        fixed_vs_position_initial_model_values_and_logits_exact=True,original_mamba_outputs=mamba,
        native_vs_original_full_bf16_max_difference=implementation_delta,native_vs_original_attention_fp32_max_error=equivalence,
        cross_window_interaction=interaction,runtime_shapes=runtime,test_read=False))
    print('INTEGRATION_VERIFIED',equivalence,interaction,flush=True)
if __name__=='__main__':main()
