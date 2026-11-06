"""Isolated real-GPU capacity checks with both models resident."""
from common import *
import argparse
import subprocess
from models import Segmentor,region_targets,region_numerators,segmentation_loss
from stage3_confirmation.data import ImageData


def candidate(batch):
    setup()
    data=ImageData(42)
    models={n:Segmentor(n).cuda() for n in ('ordered_prior','region_scan')}
    optimizers={n:torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=.001,weight_decay=.01)
                for n,m in models.items()}
    order,matrices=data.geometry(1)
    images,targets=data.batch('train',order[:64],matrices)
    soft=region_targets(targets)
    model=models['region_scan'].train();opt=optimizers['region_scan']
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):logits,route=model(images[:batch])
    mass=soft[:batch].sum((0,2,3))+1e-8
    loss=segmentation_loss(logits,targets[:batch])+(region_numerators(route,soft[:batch])/mass).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    opt.step();torch.cuda.synchronize()
    training_peak=torch.cuda.max_memory_allocated()/1024**3
    seconds=time.perf_counter()-start
    del logits,route,loss
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for m in models.values():
            m.eval();v,_=m(images[:batch]);assert torch.isfinite(v).all()
    torch.cuda.synchronize()
    eval_peak=torch.cuda.max_memory_allocated()/1024**3
    result=dict(batch=batch,passed=True,train_peak_gib=training_peak,eval_peak_gib=eval_peak,
        region_scan_forward_backward_step_seconds=seconds)
    print('CANDIDATE '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=int)
    args=parser.parse_args()
    if args.candidate:
        candidate(args.candidate);return
    attempts=[]
    selected=None
    for batch in (64,32,16,8,4,2,1):
        p=subprocess.run([sys.executable,'-u',str(ROOT/'capacity_preflight.py'),'--candidate',str(batch)],
                         cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (ROOT/f'capacity_attempt_{batch}.log').write_text(p.stdout)
        found=[line.removeprefix('CANDIDATE ') for line in p.stdout.splitlines() if line.startswith('CANDIDATE ')]
        if p.returncode==0 and len(found)==1:
            row=json.loads(found[0]);attempts.append(row)
            if max(row['train_peak_gib'],row['eval_peak_gib'])<8.6:
                selected=batch;break
        else:
            # Fail visibly on implementation errors; only an OOM permits fallback.
            is_oom='out of memory' in p.stdout.lower()
            attempts.append(dict(batch=batch,passed=False,returncode=p.returncode,oom=is_oom))
            if not is_oom:raise RuntimeError(f'non-memory failure for batch {batch}: see capacity_attempt_{batch}.log')
    assert selected,attempts
    record=dict(passed=True,gpu='NVIDIA GeForce RTX 3080',effective_batch=64,
        train_micro_batch=selected,evaluation_batch=selected,attempts=attempts,
        max_accepted_peak_gib=8.6,
        semantics='segmentation loss sample-weighted; class CE denominators over full effective batch; 30 optimizer updates per epoch')
    dump(ROOT/'capacity_preflight.json',record)
    print('CAPACITY PREFLIGHT PASSED',json.dumps(record),flush=True)


if __name__=='__main__':main()
