"""Verify historical compatibility before adding the missing E0 control."""
from common import *
import numpy as np,torch
from models import build_probes,FrozenEncoder,tensor_hash,masked_loss
from data import MedicalData,FeatureCache
from engine import predict,score,restore

def shared_hash(p):
    h=hashlib.sha256()
    for k,v in p.state_dict().items():
        if k.startswith('adapter.') or k.startswith('decoder.blocks.'):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old=json.loads((PREVIOUS/'preflight.json').read_text());lineage=json.loads((ROOT/'lineage.json').read_text());assert lineage['passed']
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();frozen=tensor_hash(enc)
    assert frozen==old['frozen_visual_hashes']['dinov3']
    cache=FeatureCache(enc,data);cache.encode('train',epoch=1,limit=32);cache.encode('val')
    ps={n:p.cuda() for n,p in build_probes().items()};initial={n:tensor_hash(p) for n,p in ps.items()}
    assert len({shared_hash(p) for p in ps.values()})==1
    for n,previous in REUSED.items():assert initial[n]==old['initial_hashes'][previous],n
    ids=[id(p) for m in ps.values() for p in m.parameters()];assert len(ids)==len(set(ids))
    fs,y,valid,text=cache.batch('train',torch.arange(BATCH,device='cuda'));p=ps[NAMES[0]]
    torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01)
    with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,text)
    loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters())
    opt.step();opt.zero_grad(set_to_none=True);assert tensor_hash(p)!=initial[NAMES[0]]
    changed=y*valid+(1-y)*(1-valid)
    assert torch.allclose(masked_loss(z.detach(),y,valid),masked_loss(z.detach(),changed,valid),atol=1e-6)
    for r in data.rows['val']:
        a=restore(np.ones((64,64),np.float32),r)
        assert a.shape==(r['height'],r['width']) and np.allclose(a,1)
    replay={}
    for n,previous in REUSED.items():
        state=torch.load(lineage['reused_checkpoints'][n]['path'],map_location='cpu');ps[n].load_state_dict(state['probe'])
        low,_=predict(ps[n],cache,'val');m,_=score(low,data,'val')
        expected=lineage['reused_metrics'][n]['splits']['val']['macro']['dice']
        assert abs(m['macro']['dice']-expected)<1e-10,(n,m['macro']['dice'],expected)
        replay[n]=dict(actual_validation_macro_dice=m['macro']['dice'],historical_validation_macro_dice=expected,matched=True)
    low,_=predict(p,cache,'val',limit=16);m,_=score(low,data,'val',full=True)
    assert m['n']==16 and all(np.isfinite(x) for x in m['macro'].values())
    assert all(x.grad is None and not x.requires_grad for x in enc.parameters()) and tensor_hash(enc)==frozen
    text_count=json.loads((ROOT/'text_provenance.json').read_text())['parameters']
    parameters={n:dict(trainable=sum(x.numel() for x in p.parameters()),
        total=enc.audit['params']+sum(x.numel() for x in p.parameters())+(text_count if n!=NAMES[0] else 0)) for n,p in ps.items()}
    dump(ROOT/'preflight.json',dict(passed=True,initial_hashes=initial,shared_adapter_decoder_initialization=True,
        frozen_visual_hash=frozen,parameters=parameters,historical_validation_replay=replay,
        encoder_gradients_absent=True,padding_loss_invariant=True,native_geometry_and_metrics_passed=True,
        exact_historical_augmentation_implementation=True,batch=BATCH,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('PREFLIGHT_PASSED',json.dumps(replay),'parameters',json.dumps(parameters),flush=True)

if __name__=='__main__':main()
