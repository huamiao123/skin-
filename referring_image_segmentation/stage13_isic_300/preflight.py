"""Verify all five fresh initializations and sentence/word routing before 300 epochs."""
from common import *
import time,numpy as np,torch
from models import build_probes,FrozenEncoder,tensor_hash,masked_loss
from data import MedicalData,FeatureCache
from engine import predict,score,restore

def main():
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    lineage=json.loads((ROOT/'lineage.json').read_text());assert lineage['passed']
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();frozen=tensor_hash(enc)
    assert frozen==lineage['expected_frozen_visual_hash']
    cache=FeatureCache(enc,data);cache.encode('train',epoch=1,limit=32);cache.encode('val',limit=16)
    ps={n:p.cuda() for n,p in build_probes().items()};initial={n:tensor_hash(p) for n,p in ps.items()}
    assert initial==lineage['expected_initial_hashes']
    assert initial[NAMES[1]]==initial[NAMES[2]] and initial[NAMES[3]]==initial[NAMES[4]]
    assert len({tensor_hash(p.adapter) for p in ps.values()})==1
    assert len({tensor_hash(p.decoder.blocks) for p in ps.values()})==1
    assert tensor_hash(ps[NAMES[0]].decoder.readout)==tensor_hash(ps[NAMES[3]].decoder.readout)==tensor_hash(ps[NAMES[4]].decoder.readout)
    ids=[id(v) for p in ps.values() for v in p.parameters()];assert len(ids)==len(set(ids))
    pos=torch.arange(BATCH,device='cuda');indices=data.indices['train'][pos];fs,y,valid,_=cache.batch('train',pos)
    text_count=json.loads((ROOT/'token_provenance.json').read_text())['parameters'];params={};routes={}
    torch.cuda.reset_peak_memory_stats()
    for n,p in ps.items():
        text=data.text_for_model(n,indices)
        if n==NAMES[0]:assert text is None;routes[n]='none'
        elif n in PRE_NAMES:
            assert text['tokens'].shape==(BATCH,77,512) and text['valid'].any(1).all();routes[n]='word tokens'
            if n in FIXED_NAMES:assert torch.equal(text['tokens'][0],text['tokens'][-1])
        else:
            assert text.shape==(BATCH,512);routes[n]='sentence vector'
            if n in FIXED_NAMES:assert torch.equal(text[0],text[-1])
        p.train();opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,text)
        loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();assert tensor_hash(p)!=initial[n]
        count=sum(v.numel() for v in p.parameters())
        params[n]=dict(trainable=count,total=count+enc.audit['params']+(text_count if n!=NAMES[0] else 0),
            batch=BATCH,training_step_seconds=time.perf_counter()-start)
        del opt
    changed=y*valid+(1-y)*(1-valid)
    assert torch.allclose(masked_loss(z.detach(),y,valid),masked_loss(z.detach(),changed,valid),atol=1e-6)
    for r in data.rows['val'][:16]:
        a=restore(np.ones((64,64),np.float32),r);assert a.shape==(r['height'],r['width']) and np.allclose(a,1)
    low,_=predict(ps[NAMES[0]],cache,'val',limit=16);m,_=score(low,data,'val',full=True)
    assert m['n']==16 and all(np.isfinite(v) for v in m['macro'].values())
    assert all(v.grad is None and not v.requires_grad for v in enc.parameters()) and tensor_hash(enc)==frozen
    updates=EPOCHS*51;warmup=5*51
    assert updates==15300 and warmup==255
    # The endpoint is the new full-budget cosine endpoint, not the old epoch100 endpoint.
    def lr(step):
        import math
        return 1e-4*(step+1)/warmup if step<warmup else 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*(step-warmup)/(updates-warmup-1)))
    assert abs(lr(updates-1)-1e-6)<1e-12 and lr(5100-1)>1e-6
    dump(ROOT/'preflight.json',dict(passed=True,initial_hashes=initial,parameters=params,frozen_visual_hash=frozen,
        exact_previous_initialization=True,all_models_independent=True,shared_visual_initialization=True,model_input_routes=routes,
        fresh_300_epoch_schedule=dict(epochs=EPOCHS,updates=updates,warmup_updates=warmup,epoch100_end_lr=lr(5099),final_lr=lr(15299)),
        encoder_gradients_absent=True,padding_loss_invariant=True,native_geometry_and_metrics_passed=True,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('PREFLIGHT_PASSED',json.dumps(params),'updates',updates,'lr_at_epoch100',lr(5099),flush=True)

if __name__=='__main__':main()
