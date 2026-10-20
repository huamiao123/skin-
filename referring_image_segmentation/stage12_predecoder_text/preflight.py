"""Check that language acts before Swin, masked tokens are ignored, and gradients flow."""
from common import *
import inspect,time,numpy as np,torch
from models import build_probes,FrozenEncoder,VisualProbe,tensor_hash,masked_loss
from data import MedicalData,FeatureCache
from engine import predict,score,restore

def main():
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old=json.loads((CONTROLS/'preflight.json').read_text())
    reference=VisualProbe('swin',SEED,(384,)*4)
    assert tensor_hash(reference)==old['initial_hashes']['E0_visual']
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();frozen=tensor_hash(enc);assert frozen==old['frozen_visual_hash']
    cache=FeatureCache(enc,data);cache.encode('train',epoch=1,limit=32);cache.encode('val',limit=16)
    ps={n:p.cuda() for n,p in build_probes().items()};initial={n:tensor_hash(p) for n,p in ps.items()};assert len(set(initial.values()))==1
    for p in ps.values():
        assert tensor_hash(p.adapter)==tensor_hash(reference.adapter)
        assert tensor_hash(p.decoder)==tensor_hash(reference.decoder)
        assert tuple(inspect.signature(p.decode).parameters)==('pyramid',)
        assert not hasattr(p,'text_projection') and not hasattr(p,'pixel_projection') and not hasattr(p,'log_tau')
    ids=[id(v) for p in ps.values() for v in p.parameters()];assert len(ids)==len(set(ids))
    positions=torch.arange(BATCH,device='cuda')
    fs,y,valid,correct=cache.batch('train',positions)
    fixed=cache.batch('train',positions,fixed=True)[3]
    p=ps[NAMES[1]];p.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=p.predecoder_features(fs,correct);b=p.predecoder_features(fs,fixed)
        changes=[float((x-y).abs().max()) for x,y in zip(a,b)]
        assert all(d>1e-6 for d in changes),changes
        za=p.decode(a);zb=p.decode(b);assert not torch.allclose(za,zb,atol=1e-6)
        altered=dict(tokens=correct['tokens'].clone(),valid=correct['valid'])
        altered['tokens'][~altered['valid']]=17.
        zp=p(fs,altered);assert torch.allclose(za,zp,atol=1e-5,rtol=0),float((za-zp).abs().max())
    torch.cuda.reset_peak_memory_stats();parameters={};gradients={}
    text_count=json.loads((ROOT/'token_provenance.json').read_text())['parameters']
    for n,p in ps.items():
        p.train();text=fixed if n==NAMES[0] else correct;opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,text)
        loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()),n
        norms=[float(m.in_proj_weight.grad.norm()) for m in p.cross_attention]
        assert all(g>0 for g in norms) and float(p.word_projection.weight.grad.norm())>0
        gradients[n]=dict(attention_input_projection_norms=norms,word_projection_norm=float(p.word_projection.weight.grad.norm()))
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();assert tensor_hash(p)!=initial[n]
        count=sum(v.numel() for v in p.parameters());parameters[n]=dict(trainable=count,total=enc.audit['params']+text_count+count,
            added_vs_visual=count-506465,batch=BATCH,training_step_seconds=time.perf_counter()-start)
    changed=y*valid+(1-y)*(1-valid)
    assert torch.allclose(masked_loss(z.detach(),y,valid),masked_loss(z.detach(),changed,valid),atol=1e-6)
    for row in data.rows['val'][:16]:
        x=restore(np.ones((64,64),np.float32),row);assert x.shape==(row['height'],row['width']) and np.allclose(x,1)
    low,_=predict(ps[NAMES[1]],cache,'val',limit=16);m,_=score(low,data,'val',full=True)
    assert m['n']==16 and all(np.isfinite(v) for v in m['macro'].values())
    assert all(v.grad is None and not v.requires_grad for v in enc.parameters()) and tensor_hash(enc)==frozen
    dump(ROOT/'preflight.json',dict(passed=True,initial_hashes=initial,frozen_visual_hash=frozen,parameters=parameters,
        matching_visual_adapter_decoder_head_initialization=True,independent_models=True,
        predecoder_feature_changes=changes,text_changes_decoder_input_and_output=True,masked_token_padding_invariant=True,
        decoder_has_no_text_input=True,no_postdecoder_text_matching_head=True,gradient_checks=gradients,
        padding_loss_invariant=True,native_geometry_and_metrics_passed=True,encoder_gradients_absent=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('PREFLIGHT_PASSED',json.dumps(parameters),'predecoder_changes',changes,flush=True)

if __name__=='__main__':main()
