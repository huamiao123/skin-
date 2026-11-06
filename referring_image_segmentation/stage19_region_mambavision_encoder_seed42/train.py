"""Train paired window-local region routers with effective batch 64."""
from common import *
import argparse
import fcntl
import signal
from models import Segmentor, tensor_hash, region_targets, region_numerators, segmentation_loss
from stage3_confirmation.data import ImageData
from engine import evaluate

NAMES=('ordered_prior','region_scan')
EFFECTIVE_BATCH=64


def trainable_state(model):
    return dict(probe=model.probe.state_dict(),router=model.encoder.mixer.router.state_dict(),
                region_prior=model.encoder.mixer.region_prior.detach())


def load_trainable(model,state):
    model.probe.load_state_dict(state['probe'],strict=True)
    model.encoder.mixer.router.load_state_dict(state['router'],strict=True)
    with torch.no_grad():model.encoder.mixer.region_prior.copy_(state['region_prior'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    assert args.smoke or args.stop_after_epoch is None
    setup()
    preflight=json.loads((ROOT/'preflight.json').read_text())
    capacity=json.loads((ROOT/'capacity_preflight.json').read_text())
    assert preflight['passed'] and capacity['passed'] and capacity['effective_batch']==64
    micro=capacity['train_micro_batch']; val_batch=capacity['evaluation_batch']
    out=ROOT/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=True)
    lock=(out/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():
        print('ALREADY DONE',flush=True);return
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    started=time.perf_counter()
    budget=2 if args.smoke else 300
    config=dict(smoke_only=args.smoke,seed=42,epochs=budget,models=list(NAMES),
        encoder='NVIDIA MambaVision-S-1K, region router inside levels[2].blocks[0].mixer',
        original_encoder_frozen=True,encoder_eval=True,new_encoder_router_trainable=True,
        dataset='ISIC2018 legacy development split 1886/808',effective_batch=64,
        micro_batch=micro,evaluation_batch=val_batch,updates_per_epoch=30,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR(T_max=300, eta_min=1e-5), epoch end',
        segmentation_loss='historical BCEWithLogits + per-image Soft Dice smooth=1',
        region_loss='mean of three class-balanced soft CE; full-effective-batch target mass; weight=1',
        region_targets='width2 5x5 morphology, area-pool to 16x16; background/boundary/interior',
        augmented_seed='42+epoch*100003',precision='BF16 autocast / FP32 loss',clip=1.,
        selection='max val macro Dice; earliest exact tie',secondary='min val loss; epoch300',
        val_encoder_cache=False,route_train='Gumbel hard for region_scan',route_eval='argmax',
        pretrain_weight_sha256=WEIGHT_SHA256,gpu=torch.cuda.get_device_name())
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    models={n:Segmentor(n).cuda() for n in NAMES}
    assert tensor_hash(models[NAMES[0]])==tensor_hash(models[NAMES[1]])
    assert all(m.encoder.frozen_state_hash()==preflight['frozen_hash'] for m in models.values())
    optimizers={n:torch.optim.AdamW((p for p in models[n].parameters() if p.requires_grad),
         lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n in NAMES}
    schedulers={n:torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[n],T_max=300,eta_min=1e-5) for n in NAMES}
    start=1;updates={n:0 for n in NAMES};history=[];resumed=False;prev_seconds=0.
    selectors={n:{'best_dice':{'epoch':0,'score':-1.},'best_loss':{'epoch':0,'score':1e30}} for n in NAMES}
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu')
        assert state['config']==config and state['frozen_hash']==preflight['frozen_hash']
        for n in NAMES:
            load_trainable(models[n],state['models'][n])
            optimizers[n].load_state_dict(state['optimizers'][n])
            schedulers[n].load_state_dict(state['schedulers'][n])
            assert tree_hash(trainable_state(models[n]))==tree_hash(state['models'][n])
            assert tree_hash(optimizers[n].state_dict())==tree_hash(state['optimizers'][n])
            assert tree_hash(schedulers[n].state_dict())==tree_hash(state['schedulers'][n])
        restore_rng(state['rng'])
        assert tree_hash(rng_state())==tree_hash(state['rng'])
        start=state['epoch']+1;updates=state['updates'];selectors=state['selectors']
        history=state['history'];prev_seconds=state['active_seconds'];resumed=True
        dump(out/'resume_audit.json',dict(resumed=True,checkpoint_epoch=state['epoch'],
            start_epoch=start,updates=updates,optimizer_scheduler_and_rng_restored=True))
        del state
    assert all(updates[n]==(start-1)*30 for n in NAMES)
    print('START',json.dumps(dict(smoke=args.smoke,start_epoch=start,budget=budget,resumed=resumed,micro_batch=micro)),flush=True)
    for epoch in range(start,budget+1):
        torch.cuda.synchronize();epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        order,matrices=data.geometry(epoch)
        for model in models.values():model.train()
        train_total={n:0. for n in NAMES};train_seg={n:0. for n in NAMES};train_aux={n:0. for n in NAMES}
        lrs={n:optimizers[n].param_groups[0]['lr'] for n in NAMES}
        for offset in range(0,len(order),EFFECTIVE_BATCH):
            ids=order[offset:offset+EFFECTIVE_BATCH]
            images,target=data.batch('train',ids,matrices)
            soft=region_targets(target);masses=soft.sum((0,2,3))+1e-8;size=len(target)
            for name in NAMES:
                model=models[name];opt=optimizers[name];opt.zero_grad(set_to_none=True)
                step_loss=step_seg=step_aux=0.
                for j in range(0,size,micro):
                    end=min(j+micro,size)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits,route=model(images[j:end])
                    seg=segmentation_loss(logits,target[j:end])
                    aux=(region_numerators(route,soft[j:end])/masses).mean()
                    loss=seg*((end-j)/size)+aux
                    assert torch.isfinite(loss),(epoch,name,offset,j)
                    loss.backward()
                    step_loss+=float(loss.item());step_seg+=float(seg.item())*(end-j)/size;step_aux+=float(aux.item())
                norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                assert torch.isfinite(norm),(epoch,name,offset)
                opt.step();updates[name]+=1
                train_total[name]+=step_loss*size;train_seg[name]+=step_seg*size;train_aux[name]+=step_aux*size
            del images,target,soft,logits,route,loss
        assert all(updates[n]==epoch*30 for n in NAMES)
        for scheduler in schedulers.values():scheduler.step()
        vals={n:evaluate(models[n],data,val_batch)[0] for n in NAMES}
        for name in NAMES:
            dest=out/name;dest.mkdir(exist_ok=True)
            for criterion,score in [('best_dice',vals[name]['macro']['dice']),('best_loss',vals[name]['loss'])]:
                old=selectors[name][criterion]['score']
                if (score>old if criterion=='best_dice' else score<old):
                    selectors[name][criterion]={'epoch':epoch,'score':score}
                    checkpoint(dest/(criterion+'.pth'),dict(model=trainable_state(models[name]),epoch=epoch,
                        metrics=vals[name],criterion=criterion,seed=42,frozen_hash=preflight['frozen_hash']))
        if epoch==1 or epoch%25==0 or epoch==budget:
            for model in models.values():
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                assert not any(m.training for m in model.encoder.net.modules())
                assert model.encoder.frozen_state_hash()==preflight['frozen_hash']
        torch.cuda.synchronize()
        row=dict(epoch=epoch,updates=updates.copy(),seconds=time.perf_counter()-epoch_start,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=train_total[n]/1886,train_seg=train_seg[n]/1886,
                train_aux=train_aux[n]/1886,val=vals[n],lr=lrs[n],selectors=selectors[n].copy()) for n in NAMES})
        history.append(row)
        checkpoint(out/'latest.pth',dict(epoch=epoch,models={n:trainable_state(models[n]) for n in NAMES},
            optimizers={n:optimizers[n].state_dict() for n in NAMES},schedulers={n:schedulers[n].state_dict() for n in NAMES},
            rng=rng_state(),history=history,updates=updates,selectors=selectors,config=config,
            frozen_hash=preflight['frozen_hash'],active_seconds=prev_seconds+time.perf_counter()-started))
        dump(out/'history.json',history);dump(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],gpu_peak_gib=row['gpu_peak_gib'],
            macro_dice={n:vals[n]['macro']['dice'] for n in NAMES},
            train_loss={n:train_total[n]/1886 for n in NAMES})),flush=True)
        if args.stop_after_epoch==epoch or stopping[0]:
            print('STOPPED AFTER CHECKPOINT',epoch,flush=True);return
    assert all(updates[n]==budget*30 for n in NAMES)
    assert all(m.encoder.frozen_state_hash()==preflight['frozen_hash'] for m in models.values())
    for n in NAMES:
        checkpoint(out/n/'last.pth',dict(model=trainable_state(models[n]),epoch=budget,
            metrics=history[-1]['models'][n]['val'],criterion='last',seed=42,frozen_hash=preflight['frozen_hash']))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,resumed=resumed,
        encoder_hash=preflight['frozen_hash'],encoder_unchanged=True,selectors=selectors,
        active_training_seconds=prev_seconds+time.perf_counter()-started))
    print('TRAINING DONE',flush=True)


if __name__=='__main__':main()
