"""Verify exact initial equivalence, independent branch gates and actual optimizer updates."""
from branch_model import OUT, attach, official, method_config
import fit
import torch

def check():
    net=official.load_model()
    data=official.dataset('ISIC2018',['train'])
    order,geom=data.geometry(1)
    x,_=data.batch('train',order[:4],geom)
    base_count=sum(p.numel() for p in net.parameters())
    def capture(mode):
        net.train(mode)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            return {k:v.detach().clone() for k,v in net(x).items()}
    base_eval=capture(False)
    rng=official.rng_state()
    # Save BN buffers because this comparison intentionally exercises training mode.
    buffers={n:t.clone() for n,t in net.named_buffers()}
    base_train=capture(True)
    for n,t in net.named_buffers():t.copy_(buffers[n])
    attach(net)
    test_eval=capture(False)
    official.restore_rng(rng)
    test_train=capture(True)
    assert set(base_train)=={'main','aux'}
    for old,new in ((base_eval,test_eval),(base_train,test_train)):
        assert all(torch.equal(a,new[k]) for k,a in old.items())
    del buffers,base_eval,test_eval,base_train,test_train
    net.train();opt=fit.optimizer(net)
    params=list(net.probe.native_branches.named_parameters())
    lookup={id(p) for p in opt.param_groups[1]['params']}
    assert all(id(p) in lookup for n,p in params)
    trace=[]
    for step in range(2):
        before={n:p.detach().clone() for n,p in params}
        _,base_audit=fit.update(net,opt,data,order[64*step:64*(step+1)],geom,64,check=(step==0))
        delta={n:(p-before[n]).detach().abs().max().item() for n,p in params}
        grad={n:p.grad.detach().float().norm().item() for n,p in params}
        assert all(torch.isfinite(p).all() for _,p in params)
        if step==0:
            assert all(delta[f'paths.{i}.strength']>0 for i in (0,1))
            assert all(v==0 for k,v in grad.items() if not k.endswith('strength'))
        else:
            assert all(v>0 for v in grad.values()) and all(v>0 for v in delta.values())
        trace.append(dict(grad=grad,update=delta))
    official.write(OUT/'audit.json',dict(protocol=method_config(),baseline_parameters=base_count,
        added_parameters=sum(p.numel() for n,p in params),equal_eval_main=True,equal_train_main_and_aux=True,
        microbatch=64,first_two_updates=trace,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
    print('VERIFIED initialization, training/eval equivalence and both branches',flush=True)
if __name__=='__main__':check()

