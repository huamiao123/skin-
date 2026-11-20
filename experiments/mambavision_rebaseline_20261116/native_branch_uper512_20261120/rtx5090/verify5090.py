import gc,json,statistics,time,sys
import numpy as np
import torch
from resident_images import HERE, Base,ResidentImages,loader,official
import train_branch,fit
from branch_model import load_branch_model
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref

def scan_check():
    torch.manual_seed(23)
    results={}
    for dtype in (torch.float32,torch.bfloat16):
        def rand(*shape):return (torch.randn(*shape,device='cuda')*.2).to(dtype)
        vals=[rand(2,16,32),rand(2,16,32),-torch.rand(16,8,device='cuda'),rand(2,8,32),rand(2,8,32),torch.randn(16,device='cuda')*.1]
        args=[v.detach().requires_grad_(True) for v in vals]
        grad=rand(2,16,32)
        def go(fn):
            out=fn(*args,delta_softplus=True)
            grads=torch.autograd.grad(out,args,grad)
            return [out,*grads]
        native=go(selective_scan_fn);ref=go(selective_scan_ref)
        errors=[(v.float()-r.float()).abs().max().item() for v,r in zip(native,ref)]
        tol=.012 if dtype==torch.bfloat16 else 2e-4
        for v,r in zip(native,ref):torch.testing.assert_close(v.float(),r.float(),rtol=tol,atol=tol)
        results[str(dtype)]=errors
    return results

def bench():
    dev=torch.cuda.get_device_properties(0)
    assert torch.cuda.get_device_capability()==(12,0)
    out=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=dev.name,scan_max_abs_errors=scan_check())
    net=load_branch_model(); opt=fit.optimizer(net);net.train()
    with loader(Base):host=official.dataset('ISIC2018',['train','val'])
    with loader(ResidentImages):gpu=official.dataset('ISIC2018',['train','val'])
    order,geom=host.geometry(4)
    reforder,refgeom=gpu.geometry(4)
    assert torch.equal(order,reforder) and torch.equal(geom,refgeom)
    for part,ix,transform in [('train',order[:64],geom),('val',torch.arange(64,device='cuda'),None)]:
        a=host.batch(part,ix,transform);b=gpu.batch(part,ix,transform)
        assert all(torch.equal(v,w) for v,w in zip(a,b))
    out['input_bitwise_equal']=True
    times={'stream':[],'resident':[]}
    for step,name in enumerate(['stream','resident']+['resident','stream','stream','resident']*3):
        torch.cuda.synchronize();t=time.monotonic()
        fit.update(net,opt,host if name=='stream' else gpu,order[(step*64):((step+1)*64)],geom,64)
        torch.cuda.synchronize()
        if step>=2:times[name].append(time.monotonic()-t)
    out['median_step_s']={k:statistics.median(v) for k,v in times.items()};out['step_times']=times
    out['images_s']={k:64/statistics.median(v) for k,v in times.items()}
    out['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    out['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
    out['driver_used_gib']=(dev.total_memory-torch.cuda.mem_get_info()[0])/1024**3
    del opt,net,gpu,host;gc.collect();torch.cuda.empty_cache()
    # Cross-device replay of the original validation checkpoint, before resuming optimizer state.
    folder=HERE.parent/'ISIC2018'
    ck=torch.load(str(folder/'resume.pth'),weights_only=False,mmap=True,map_location='cpu')
    best=ck['best'];out['resume_epoch']=len(ck['history']);del ck
    net=load_branch_model();net.load_state_dict(torch.load(folder/best['file'],weights_only=True,map_location='cpu'))
    with loader(ResidentImages):val=official.dataset('ISIC2018',['val'])
    score,_=fit.measure(net,val,'val')
    out['old_best_epoch']=best['epoch'];out['old_best_validation']=best['score']
    out['same_weights_new_gpu_validation']=score;out['replay_delta_pp']=100*(score-best['score'])
    official.write(HERE/'check5090.json',out)
    if abs(out['replay_delta_pp'])>.1:raise RuntimeError('Inspect unexpected migration replay difference before resume')
    print('5090 VERIFIED',out,flush=True)
if __name__=='__main__':bench()

