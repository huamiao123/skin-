"""Five independent seed42 models; identical inputs and a fresh 300-epoch schedule."""
from common import *
import argparse,fcntl,math,random,signal,time
import numpy as np,torch
from models import build_probes,FrozenEncoder,tensor_hash,masked_loss
from data import MedicalData,FeatureCache
from engine import predict,score

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pth.tmp');torch.save(obj,tmp);tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();assert args.stop_after_epoch is None or args.smoke
    out=ROOT/('smoke' if args.smoke else 'results')/'seed42';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():return
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    pre=json.loads((ROOT/'preflight.json').read_text());assert pre['passed']
    if not args.smoke:
        provenance=json.loads((ROOT/'launch_provenance.json').read_text())
        for part in ('source_sha256','record_sha256'):
            for p,h in provenance[part].items():assert sha(p)==h,p
    data=MedicalData();enc=FrozenEncoder('dinov3').cuda();assert tensor_hash(enc)==pre['frozen_visual_hash']
    cache=FeatureCache(enc,data);ps={n:p.cuda() for n,p in build_probes().items()}
    assert {n:tensor_hash(p) for n,p in ps.items()}==pre['initial_hashes']
    opts={n:torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n,p in ps.items()}
    count=32 if args.smoke else len(data.rows['train']);val_count=16 if args.smoke else len(data.rows['val'])
    epochs=2 if args.smoke else EPOCHS;per_epoch=math.ceil(count/BATCH)
    full_steps=math.ceil(len(data.rows['train'])/BATCH);warmup=5*full_steps;total=EPOCHS*full_steps
    config=dict(seed=SEED,epochs=epochs,train_samples=count,validation_samples=val_count,batch=BATCH,input_size=256,
        optimizer='AdamW',lr=1e-4,weight_decay=.01,eta_min=1e-6,warmup_steps=warmup,total_schedule_steps=total,
        models=list(NAMES),loss='valid-pixel BCE + per-image soft Dice; equal weights; smooth1',
        selection='maximum original-resolution validation macro Dice; earliest exact tie',frozen_encoders=True,
        previous_segmentation_weights_loaded=False,fresh_start=True,augmentation='exact stage9 brightness/contrast .9-1.1',
        initial_hashes=pre['initial_hashes'],data_sha256=sha(ROOT/'data_manifest.json'),token_sha256=sha(ROOT/'token_provenance.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'))
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
    cache.encode('val',limit=val_count);checked={n:False for n in NAMES}
    print('START',json.dumps(dict(seed=SEED,epochs=epochs,epoch=epoch,resumed=resumed,models=list(NAMES),full_schedule_updates=total)),flush=True)
    while epoch<=epochs:
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats();cache.encode('train',epoch,limit=count)
        order=torch.randperm(count,generator=torch.Generator().manual_seed(SEED+epoch*100003)).cuda()
        for p in ps.values():p.train()
        for offset in range(cursor,count,BATCH):
            pos=order[offset:offset+BATCH];fs,y,valid,_=cache.batch('train',pos);indices=data.indices['train'][pos]
            lr=1e-4*(step+1)/warmup if step<warmup else 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup-1)))
            for n,p in ps.items():
                opt=opts[n]
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);text=data.text_for_model(n,indices)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=p(fs,text)
                loss=masked_loss(z,y,valid);assert torch.isfinite(loss);loss.backward()
                if not checked[n]:assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.parameters());checked[n]=True
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
        assert all(v.grad is None for v in enc.parameters())
        row=dict(epoch=epoch,seconds=time.perf_counter()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=running[n]/count,val=validation[n],best=best[n]) for n in NAMES})
        history.append(row);dump(out/'history.json',history);dump(out/'status.json',dict(phase='epoch_complete',epochs=epochs,**row))
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],val_macro_dice={n:m['macro']['dice'] for n,m in validation.items()},best=best)),flush=True)
        epoch+=1;cursor=0;running={n:0. for n in NAMES};checkpoint()
        if args.stop_after_epoch==epoch-1 or stopping[0]:return
    assert step==epochs*per_epoch and tensor_hash(enc)==pre['frozen_visual_hash']
    dump(out/'DONE.json',dict(passed=True,seed=SEED,epochs=epochs,updates_per_model=step,resumed=resumed,smoke_only=args.smoke,
        best=best,all_five_models_complete=True,checkpoint_selection_locked=True,encoders_unchanged=True))
    print('TRAINING_DONE',flush=True)

if __name__=='__main__':main()
