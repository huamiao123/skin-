"""Check paired initializations, independent gradients, geometry and batch64 memory."""
from common import *
import time, numpy as np, torch
from models import build_probes, FrozenEncoder, tensor_hash, masked_loss
from data import MedicalData, FeatureCache
from engine import restore, score, predict

def shared_hash(probe):
    h=hashlib.sha256()
    for key,value in probe.state_dict().items():
        if key.startswith('adapter.') or key.startswith('decoder.blocks.'):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    initial={}; shared={}
    for seed in SEEDS:
        ps=build_probes(seed); initial[str(seed)]={n:tensor_hash(p) for n,p in ps.items()}
        assert initial[str(seed)][NAMES[1]]==initial[str(seed)][NAMES[2]]
        sh={n:shared_hash(p) for n,p in ps.items()}; assert len(set(sh.values()))==1;shared[str(seed)]=sh
        ids=[id(x) for p in ps.values() for x in p.parameters()];assert len(ids)==len(set(ids))
        del ps
    assert len({initial[str(s)][NAMES[0]] for s in SEEDS})==len(SEEDS)
    data=MedicalData(); enc=FrozenEncoder('dinov3').cuda(); frozen=tensor_hash(enc)
    old=json.loads((PROJECT/'stage1_visual/benchmark_dinov3.json').read_text())
    assert frozen==old['frozen_hash']
    limits={'train':BATCH,'val':16};cache=FeatureCache(enc,data,limits=limits)
    cache.encode('train',limit=BATCH);cache.encode('val',limit=16)
    ps={n:p.cuda() for n,p in build_probes(42).items()}
    fs,y,valid,text=cache.batch('train',torch.arange(BATCH,device='cuda'))
    report={};torch.cuda.reset_peak_memory_stats()
    text_count=json.loads((ROOT/'text_provenance.json').read_text())['parameters']
    for n,p in ps.items():
        txt=data.text[data.fixed_index].expand(BATCH,-1) if n==NAMES[1] else text
        before=tensor_hash(p);opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,txt)
        loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step();opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;assert tensor_hash(p)!=before
        trainable=sum(x.numel() for x in p.parameters())
        report[n]=dict(trainable=trainable,total=enc.audit['params']+trainable+(text_count if n!=NAMES[0] else 0),
                       batch=BATCH,training_step_seconds=elapsed)
        if n==NAMES[2]:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                a=p(fs,text);b=p(fs,data.text[data.fixed_index].expand(BATCH,-1))
            assert not torch.allclose(a,b,atol=1e-6)
        del opt
    changed=y*valid+(1-y)*(1-valid)
    assert torch.allclose(masked_loss(z.detach(),y,valid),masked_loss(z.detach(),changed,valid),atol=1e-6)
    for r in data.rows['val'][:16]:
        a=restore(np.ones((64,64),np.float32),r)
        assert a.shape==(r['height'],r['width']) and np.allclose(a,1)
    low,_=predict(ps[NAMES[0]],cache,'val',limit=16);m,_=score(low,data,'val',full=True)
    assert m['n']==16 and all(np.isfinite(v) for v in m['macro'].values())
    assert all(x.grad is None and not x.requires_grad for x in enc.parameters());assert tensor_hash(enc)==frozen
    peak=torch.cuda.max_memory_allocated()/1024**3;available=torch.cuda.get_device_properties(0).total_memory/1024**3
    assert peak < available*.85,(peak,available)
    dump(ROOT/'preflight.json',dict(passed=True,initial_hashes=initial,shared_initial_hashes=shared,frozen_visual_hash=frozen,
        parameters=report,batch=BATCH,independent_models=True,text_input_changes_logits=True,
        padding_loss_invariant=True,native_geometry_and_full_metrics_passed=True,encoder_gradients_absent=True,
        subset_peak_gpu_gib=peak,estimated_full_train_val_cache_gib=(len(data.rows['train'])+len(data.rows['val']))*4*384*16*16*2/1024**3,
        torch_version=torch.__version__,encoder_source=enc.audit))
    print('PREFLIGHT_PASSED',json.dumps(report),'peak_GiB',peak,flush=True)

if __name__=='__main__':main()
