"""Explicitly launched, synthetic-data GPU preflight. Never runs on import.

This is not a trainer or experiment. Run one arm and one micro-batch per process.
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from model import ARMS, SegmentationSystem, load_pretrained_encoder, objective

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',required=True,choices=tuple(ARMS))
    ap.add_argument('--micro-batch',required=True,type=int)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    dest=Path(__file__).resolve().parents[1]/'reports'/f'gpu_preflight_{args.arm}_mb{args.micro_batch}.json'
    if dest.exists():raise FileExistsError(f'refusing to overwrite {dest}')
    if args.micro_batch < 2 or args.micro_batch > 64:
        raise ValueError('micro-batch must be 2..64: training SyncBN includes a 1x1 PPM branch')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; check driver and scheduling first')
    torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device);start=time.perf_counter()
    m=SegmentationSystem(load_pretrained_encoder(),args.arm,42).to(args.device)
    opt=torch.optim.AdamW([{'params':list(m.probe.parameters()),'lr':1e-3},
                           {'params':[p for p in m.encoder.parameters() if p.requires_grad],'lr':1e-4}],weight_decay=.01)
    images=torch.rand(args.micro_batch,3,256,256,device=args.device)
    target=torch.rand(args.micro_batch,1,256,256,device=args.device)
    m.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        direct=m(images)['main'];f1,f2,mid=m.prefix(images);cached=m.from_cache(f1,f2,mid,(256,256))['main']
    diff=float((direct.float()-cached.float()).abs().max());
    if not torch.allclose(direct.float(),cached.float(),rtol=1e-3,atol=1e-3):raise AssertionError(f'cache mismatch {diff}')
    frozen={k:v.detach().cpu().clone() for k,v in m.encoder.state_dict().items() if not k.startswith(('model.levels.2.','model.levels.3.'))}
    buffers={k:v.detach().cpu().clone() for k,v in m.encoder.named_buffers()}
    m.train();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):output=m.from_cache(f1,f2,mid,(256,256))
    loss,parts=objective(output,target)
    if not torch.isfinite(loss):raise AssertionError('nonfinite loss')
    loss.backward()
    if any(p.grad is not None for p in m.encoder.parameters() if not p.requires_grad):raise AssertionError('frozen gradient')
    for i in (2,3):
        gradients=[p.grad for p in m.encoder.model.levels[i].parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum()>0 for g in gradients):raise AssertionError(f'late stage gradient {i}')
    for label,head in [('main',m.probe.decode_head),('aux',m.probe.auxiliary_head)]:
        if head is not None:
            grads=[p.grad for p in head.parameters() if p.requires_grad]
            if any(g is None or not torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):raise AssertionError(f'{label} gradient')
    opt.step();torch.cuda.synchronize(args.device)
    if any(not torch.equal(v,m.encoder.state_dict()[k].detach().cpu()) for k,v in frozen.items()):raise AssertionError('frozen encoder changed')
    if any(not torch.equal(v,m.encoder.get_buffer(k).detach().cpu()) for k,v in buffers.items()):raise AssertionError('encoder buffer changed')
    report={'arm':args.arm,'micro_batch':args.micro_batch,'device':torch.cuda.get_device_name(args.device),
            'device_total_bytes':torch.cuda.get_device_properties(args.device).total_memory,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),'elapsed_seconds':time.perf_counter()-start,
            'cache_direct_max_abs_diff':diff,'loss':float(loss),'synthetic_only':True,'steps':1,'passed':True}
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
