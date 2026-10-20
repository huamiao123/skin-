"""Matched three-model runs; resumable pilot, smoke and frozen-budget formal training."""
from common import *
import argparse, fcntl, math, random, time, signal
import numpy as np, torch
from models import build_probes, FrozenEncoder, tensor_hash, masked_loss
from data import MedicalData, FeatureCache
from engine import predict, score

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pth.tmp');torch.save(obj,tmp);tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','pilot','formal'),required=True)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();assert args.seed in SEEDS;assert args.phase=='smoke' or args.stop_after_epoch is None
    assert args.phase=='formal' or args.seed==42
    out=ROOT/('results' if args.phase=='formal' else args.phase)/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():return
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    pre=json.loads((ROOT/'preflight.json').read_text());assert pre['passed'] and pre['batch']==BATCH
    if args.phase=='formal':
        budget=json.loads((ROOT/'budget_decision.json').read_text());assert budget['passed'];epochs=budget['formal_epochs']
        frozen=json.loads((ROOT/'formal_provenance.json').read_text())
        for section in ('source_sha256','record_sha256'):
            for path,digest in frozen[section].items():assert sha(path)==digest,path
    else:epochs=2 if args.phase=='smoke' else PILOT_EPOCHS
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();frozen_hash=tensor_hash(enc);assert frozen_hash==pre['frozen_visual_hash']
    count=BATCH if args.phase=='smoke' else len(data.rows['train']);val_count=16 if args.phase=='smoke' else len(data.rows['val'])
    cache=FeatureCache(enc,data,limits={'train':count,'val':val_count})
    cache.encode('train',limit=count);cache.encode('val',limit=val_count)
    ps={n:p.cuda() for n,p in build_probes(args.seed).items()}
    assert {n:tensor_hash(p) for n,p in ps.items()}==pre['initial_hashes'][str(args.seed)]
    opts={n:torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n,p in ps.items()}
    per_epoch=math.ceil(count/BATCH);warmup=5*per_epoch;total=epochs*per_epoch
    config=dict(phase=args.phase,seed=args.seed,epochs=epochs,batch=BATCH,train_samples=count,validation_samples=val_count,
        optimizer='AdamW',lr=1e-4,weight_decay=.01,eta_min=1e-6,warmup_steps=warmup,total_schedule_steps=total,
        augmentation='none; exactly cached frozen DINO features',loss='valid BCE + per-image SoftDice; 1:1; smooth1',
        selection='maximum original-resolution validation macro Dice; earliest exact tie',
        frozen_encoders=True,previous_segmentation_weights_loaded=False,initial_hashes=pre['initial_hashes'][str(args.seed)],
        data_sha256=sha(ROOT/'data_manifest.json'),text_sha256=sha(ROOT/'text_provenance.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    history=[];best={n:dict(score=-1.,epoch=0) for n in NAMES};epoch=1;cursor=step=0;running={n:0. for n in NAMES};resumed=False
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu');assert state['config']==config
        for n in NAMES:ps[n].load_state_dict(state['probes'][n]);opts[n].load_state_dict(state['optimizers'][n])
        history=state['history'];best=state['best'];epoch=state['epoch'];cursor=state['cursor'];step=state['step'];running=state['running'];resumed=True
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
    def checkpoint():
        save(out/'latest.pth',dict(config=config,probes={n:p.state_dict() for n,p in ps.items()},optimizers={n:o.state_dict() for n,o in opts.items()},
            history=history,best=best,epoch=epoch,cursor=cursor,step=step,running=running,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print('START',json.dumps(dict(phase=args.phase,seed=args.seed,epochs=epochs,epoch=epoch,resumed=resumed,cache_gib=cache.feature_bytes/1024**3)),flush=True)
    checked={n:False for n in NAMES}
    while epoch<=epochs:
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        order=torch.randperm(count,generator=torch.Generator().manual_seed(args.seed+epoch*100003)).cuda()
        for p in ps.values():p.train()
        for offset in range(cursor,count,BATCH):
            pos=order[offset:offset+BATCH];fs,y,valid,text=cache.batch('train',pos)
            lr=1e-4*(step+1)/warmup if step<warmup else 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup-1)))
            for n,p in ps.items():
                opt=opts[n]
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad(set_to_none=True);txt=data.text[data.fixed_index].expand(len(pos),-1) if n==NAMES[1] else text
                with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,txt)
                loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
                if not checked[n]:assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters());checked[n]=True
                torch.nn.utils.clip_grad_norm_(p.parameters(),1.);opt.step();running[n]+=float(loss)*len(pos)
            step+=1;cursor=offset+len(pos)
            if step%25==0 or stopping[0]:
                checkpoint();dump(out/'status.json',dict(phase='training',epoch=epoch,epochs=epochs,samples_done=cursor,samples_per_epoch=count,updates=step,lr=lr))
            if stopping[0]:print('STOPPED_WITH_CHECKPOINT',flush=True);return
        validation={}
        for n,p in ps.items():
            low,loss=predict(p,cache,'val',limit=val_count);m,_=score(low,data,'val');m['loss']=loss;validation[n]=m
            if m['macro']['dice']>best[n]['score']:
                best[n]=dict(score=m['macro']['dice'],epoch=epoch)
                save(out/n/'best_dice.pth',dict(probe=p.state_dict(),epoch=epoch,metrics=m,config=config))
        assert all(x.grad is None for x in enc.parameters())
        row=dict(epoch=epoch,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=running[n]/count,val=validation[n],best=best[n]) for n in NAMES})
        history.append(row);dump(out/'history.json',history);dump(out/'status.json',dict(phase='epoch_complete',epochs=epochs,**row))
        print('EPOCH',json.dumps(dict(seed=args.seed,epoch=epoch,seconds=row['seconds'],peak_gpu_gib=row['peak_gpu_gib'],macro_dice={n:m['macro']['dice'] for n,m in validation.items()})),flush=True)
        epoch+=1;cursor=0;running={n:0. for n in NAMES};checkpoint()
        if args.stop_after_epoch==epoch-1 or stopping[0]:return
    assert step==epochs*per_epoch;assert tensor_hash(enc)==frozen_hash
    dump(out/'DONE.json',dict(passed=True,seed=args.seed,phase=args.phase,epochs=epochs,resumed=resumed,updates_per_model=step,
        best=best,all_three_models_complete=True,encoders_unchanged=True,checkpoint_selection_locked=True))
    print('TRAINING_DONE',args.phase,args.seed,flush=True)

if __name__=='__main__':main()
