"""Check frozen weights, matched initialization, gradients, masks and text use."""
from common import *
import time, numpy as np, torch
from models import build_probes,EncoderPair,tensor_hash,masked_loss
from data import MedicalData,FeatureCache
from engine import restore,score,predict

def main():
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=MedicalData();enc=EncoderPair().cuda()
    hashes={n:tensor_hash(getattr(enc,n)) for n in ('dinov3','vmamba')}
    for n,digest in hashes.items():assert digest==json.loads((PROJECT/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    cache=FeatureCache(enc,data);cache.encode('train',1,limit=32);cache.encode('val',limit=16)
    probes={n:p.cuda() for n,p in build_probes().items()}
    initial={n:tensor_hash(p) for n,p in probes.items()}
    assert initial['dinov3_text']==initial['dinov3_fixed_text']
    assert len({tensor_hash(p.decoder) for p in probes.values()})==1
    assert len({tensor_hash(p.text_projection) for p in probes.values()})==1
    params=[id(x) for p in probes.values() for x in p.parameters()];assert len(params)==len(set(params))
    fs,y,valid,text=cache.batch('train',torch.arange(BATCH,device='cuda'))
    report={};torch.cuda.reset_peak_memory_stats()
    for n,p in probes.items():
        p.train();optimizer=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
        before=tensor_hash(p);torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,text)
        loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1);optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;assert tensor_hash(p)!=before
        report[n]=dict(trainable=sum(x.numel() for x in p.parameters()),batch16_step_seconds=seconds)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=p(fs,text);b=p(fs,data.text[data.fixed_index].expand(BATCH,-1))
        assert not torch.allclose(a,b,atol=1e-6),n
    # Loss is independent of padding labels.
    changed=y*(valid)+(1-y)*(1-valid)
    assert torch.allclose(masked_loss(z.detach(),y,valid),masked_loss(z.detach(),changed,valid),atol=1e-6)
    # Restore covers exactly the original image; constant logits remain constant.
    for r in data.rows['val'][:16]:
        restored=restore(np.ones((64,64),np.float32),r)
        assert restored.shape==(r['height'],r['width']) and np.allclose(restored,1)
    low,_=predict(probes['dinov3_text'],cache,'val',limit=16)
    start=time.perf_counter();metrics,rows=score(low,data,'val',full=True);full_seconds=time.perf_counter()-start
    assert metrics['n']==16 and all(np.isfinite(v) for v in metrics['macro'].values())
    assert all(x.grad is None and not x.requires_grad for x in enc.parameters())
    assert hashes=={n:tensor_hash(getattr(enc,n)) for n in hashes}
    text_count=json.loads((ROOT/'text_provenance.json').read_text())['parameters']
    for n in report:
        visual=sum(getattr(enc,k).audit['params'] for k in ('dinov3','vmamba')) if n=='fusion_text' else getattr(enc,'vmamba' if n=='vmamba_text' else 'dinov3').audit['params']
        report[n]['total']=visual+text_count+report[n]['trainable']
    dump(ROOT/'preflight.json',dict(passed=True,initial_hashes=initial,frozen_visual_hashes=hashes,
        parameters=report,encoder_gradients_absent=True,independent_models=True,matched_decoder_and_head_initialization=True,
        text_changes_logits=True,padding_loss_invariant=True,native_geometry_and_full_metrics_passed=True,
        full_metrics_16_seconds=full_seconds,cache_gib=cache.feature_bytes/1024**3,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,torch_version=torch.__version__))
    print('PREFLIGHT_PASSED',json.dumps(report),flush=True)

if __name__=='__main__':main()
