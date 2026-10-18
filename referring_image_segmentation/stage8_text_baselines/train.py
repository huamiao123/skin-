"""Four independent probes, one frozen feature cache, one seed, native-IoU selection."""
from common import *
import argparse,fcntl,math,random,time
import numpy as np,torch
from models import build_probes,EncoderPair,tensor_hash,masked_loss
from data import RISData,FeatureCache
from engine import predict,score

def save(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after-epoch',type=int);args=p.parse_args()
    assert args.stop_after_epoch is None or args.smoke
    out=ROOT/('smoke' if args.smoke else 'results')/'seed42';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():return
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    pre=json.loads((ROOT/'preflight.json').read_text());assert pre['passed']
    encoder=EncoderPair().cuda();enc_hash={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    assert enc_hash==pre['frozen_visual_hashes']
    probes={n:p.cuda() for n,p in build_probes().items()}
    assert {n:tensor_hash(p) for n,p in probes.items()}==pre['initial_hashes']
    data=RISData();cache=FeatureCache(encoder,data)
    optimizers={n:torch.optim.AdamW(p.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n,p in probes.items()}
    count=128 if args.smoke else len(data.rows['train']);val_count=128 if args.smoke else len(data.rows['val']);epochs=2 if args.smoke else 30
    per_epoch=math.ceil(count/16);warmup=2*7539;total=30*7539
    config=dict(seed=42,epochs=epochs,train_expressions=count,validation_expressions=val_count,batch=16,input_size=256,
        loss='masked BCE + masked per-image soft Dice smooth1',optimizer='AdamW',lr=1e-4,eta_min=1e-6,weight_decay=.01,
        warmup_steps=warmup,total_schedule_steps=total,primary_selection='maximum native per-expression validation mean IoU; earliest exact tie',
        frozen_encoders=True,text='CLIP ViT-B/32 EOS hidden512; frozen; common learned projection',
        initialization=pre['initial_hashes'],data_sha256=sha(ROOT/'data_manifest.json'),text_sha256=sha(ROOT/'text_provenance.json'))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    history=[];best={n:dict(score=-1.,epoch=0) for n in NAMES};epoch=1;cursor=0;step=0;running={n:0. for n in NAMES};resumed=False
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu');assert state['config']==config
        for n in NAMES:probes[n].load_state_dict(state['probes'][n]);optimizers[n].load_state_dict(state['optimizers'][n])
        history=state['history'];best=state['best'];epoch=state['epoch'];cursor=state['cursor'];step=state['step'];running=state['running'];resumed=True
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
    def checkpoint():
        save(out/'latest.pth',dict(probes={n:p.state_dict() for n,p in probes.items()},optimizers={n:o.state_dict() for n,o in optimizers.items()},
            config=config,history=history,best=best,epoch=epoch,cursor=cursor,step=step,running=running,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    print('START',json.dumps(dict(epoch=epoch,cursor=cursor,step=step,resumed=resumed,smoke=args.smoke)),flush=True)
    cache.encode('val',rows=data.rows['val'][:val_count])
    checked={n:False for n in NAMES}
    while epoch<=epochs:
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats();cache.encode('train',epoch,rows=data.rows['train'][:count])
        order=torch.randperm(count,generator=torch.Generator().manual_seed(42+epoch*100003)).cuda()
        for p in probes.values():p.train()
        for offset in range(cursor,count,16):
            positions=order[offset:offset+16];features,target,valid,text=cache.batch('train',positions)
            lr=1e-4*(step+1)/warmup if step<warmup else 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup-1)))
            for n,p in probes.items():
                opt=optimizers[n]
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True)
                txt=data.text[data.fixed_index].expand(len(positions),-1) if n=='dinov3_fixed_text' else text
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(features,txt)
                loss=masked_loss(logits,target,valid);assert torch.isfinite(loss);loss.backward()
                if not checked[n]:assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in p.parameters());checked[n]=True
                norm=torch.nn.utils.clip_grad_norm_(p.parameters(),1.);assert torch.isfinite(norm)
                opt.step();running[n]+=float(loss)*len(positions)
            step+=1;cursor=offset+len(positions)
            if step%500==0:
                checkpoint();dump(out/'status.json',dict(phase='training',epoch=epoch,epochs=epochs,expressions_done=cursor,
                    expressions_per_epoch=count,updates=step,lr=lr,elapsed_epoch_seconds=time.perf_counter()-started))
                print('STEP',epoch,cursor,count,step,flush=True)
        validation={}
        for n,p in probes.items():
            low,loss=predict(p,cache,'val',limit=val_count);region,_=score(low,data,'val');region['loss']=loss;validation[n]=region
            if region['macro']['iou']>best[n]['score']:
                best[n]=dict(score=region['macro']['iou'],epoch=epoch)
                folder=out/n;folder.mkdir(exist_ok=True)
                save(folder/'best_miou.pth',dict(probe=p.state_dict(),epoch=epoch,metrics=region))
                np.save(folder/'best_miou_logits64.npy',low)
            print('VALIDATION',epoch,n,region['macro']['iou'],flush=True)
        assert all(tensor_hash(getattr(encoder,n))==h for n,h in enc_hash.items())
        assert all(p.grad is None for p in encoder.parameters())
        row=dict(epoch=epoch,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=running[n]/count,val=validation[n],best=best[n]) for n in NAMES})
        history.append(row);dump(out/'history.json',history);dump(out/'status.json',dict(phase='epoch_complete',**row))
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],macro_iou={n:r['macro']['iou'] for n,r in validation.items()})),flush=True)
        epoch+=1;cursor=0;running={n:0. for n in NAMES};checkpoint()
        if args.stop_after_epoch==epoch-1:return
    for n,p in probes.items():save(out/n/'last.pth',dict(probe=p.state_dict(),epoch=epochs,metrics=history[-1]['models'][n]['val']))
    assert step==epochs*per_epoch
    dump(out/'DONE.json',dict(passed=True,seed=42,epochs=epochs,smoke_only=args.smoke,resumed=resumed,updates_per_model=step,
        best=best,encoders_unchanged=True,all_four_models_complete=True,checkpoint_selection_locked=True))
    print('TRAINING DONE',flush=True)

if __name__=='__main__':main()
