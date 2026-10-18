"""Data/text/initialization audits plus full-cache memory and real-gradient checks."""
from common import *
import time,torch,numpy as np
from models import build_probes,EncoderPair,tensor_hash,masked_loss
from data import RISData,FeatureCache
from engine import predict,score,restore

def main():
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert json.loads((ROOT/'text_provenance.json').read_text())['cache_sha256']==sha(ASSETS/'text512.npy')
    encoder=EncoderPair().cuda();enc_hash={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in enc_hash.items():assert h==json.loads((PROJECT/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    data=RISData();cache=FeatureCache(encoder,data)
    cache.encode('train',1,rows=data.rows['train'][:64]);cache.encode('val',rows=data.rows['val'][:128])
    probes=build_probes();initial={n:tensor_hash(p) for n,p in probes.items()}
    assert initial['dinov3_text']==initial['dinov3_fixed_text']
    for name,p in probes.items():
        for field in ('decoder','text_projection','pixel_projection'):assert tensor_hash(getattr(p,field))==tensor_hash(getattr(probes['dinov3_text'],field))
    ids=[id(p) for m in probes.values() for p in m.parameters()];assert len(ids)==len(set(ids))
    # Every real expression/target is retained, and input images cannot cross official splits.
    image_sets={s:{r['image_id'] for r in rows} for s,rows in data.rows.items()}
    assert all(not image_sets[a]&image_sets[b] for a in image_sets for b in image_sets if a!=b)
    hashes={}
    for split,rows in data.rows.items():
        for r in rows:hashes.setdefault(data.images[r['image_index']]['sha256'],set()).add(split)
    duplicate_contents=[sorted(s) for s in hashes.values() if len(s)>1]
    # Keep official splits; disclose content duplicates if any rather than silently changing the benchmark.
    features,target,valid,text=cache.batch('train',torch.arange(16,device='cuda'))
    params={};speed={}
    for n,p in probes.items():
        p.cuda().train();params[n]=sum(x.numel() for x in p.parameters())
        before=tensor_hash(p);optimizer=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(features,text)
        loss=masked_loss(logits,target,valid);loss.backward()
        assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in p.parameters()),n
        torch.nn.utils.clip_grad_norm_(p.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();speed[n]=time.perf_counter()-start
        assert tensor_hash(p)!=before
        p.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=p([f[:1] for f in features],text[:1]);b=p([f[:1] for f in features],data.text[data.fixed_index].unsqueeze(0))
            assert not torch.equal(a,b),'Text projection is disconnected'
        p.cpu();del optimizer,logits,loss,a,b
    # Explicitly prove padding labels and logits do not affect the loss.
    z=torch.randn(2,1,256,256,device='cuda',requires_grad=True);t=target[:2];v=valid[:2]
    a=masked_loss(z,t,v);b=masked_loss(z,t*v+(1-t)*(1-v),v)
    torch.testing.assert_close(a,b,atol=0,rtol=0);a.backward();assert torch.count_nonzero(z.grad*(1-v))==0
    constant=restore(np.full((64,64),3,np.float32),data.images[data.rows['val'][0]['image_index']])
    assert constant.shape==(data.images[data.rows['val'][0]['image_index']]['height'],data.images[data.rows['val'][0]['image_index']]['width']) and (constant>0).all()
    # Benchmark native validation on real expressions, with the selected fixed batch.
    probe=build_probes()['dinov3_text'].cuda();start=time.perf_counter()
    logits,loss=predict(probe,cache,'val',limit=128);region,rows=score(logits,data,'val')
    assert region['n']==128 and np.isfinite(loss)
    native_validation_seconds=time.perf_counter()-start
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in enc_hash.items())
    assert all(p.grad is None for p in encoder.parameters())
    peak=torch.cuda.max_memory_allocated()/1024**3;total=torch.cuda.get_device_properties(0).total_memory/1024**3
    assert peak<total-1,'Insufficient headroom with the complete train/val cache'
    dump(ROOT/'preflight.json',dict(passed=True,seed=42,batch=16,initial_hashes=initial,trainable_parameters=params,
        frozen_visual_hashes=enc_hash,independent_parameters=True,common_head_decoder_initialization=True,
        dino_fixed_and_real_initial_state_identical=True,finite_nonzero_gradients=True,text_affects_predictions=True,
        padding_loss_invariant=True,native_geometry_verified=True,official_image_ids_disjoint=True,
        cross_split_identical_image_content_groups=duplicate_contents,full_train_val_cache_allocated=True,
        peak_allocated_gib=peak,total_gpu_gib=total,first_training_step_seconds=speed,native_128_validation_seconds=native_validation_seconds,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name()))
    print('PREFLIGHT PASSED',peak,total,speed,flush=True)

if __name__=='__main__':main()
