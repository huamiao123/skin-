"""Two independent probes, paired data and resumable effective-batch updates."""
from common import *
import argparse
import fcntl
import signal
from models import BaselineProbe, RegionProbe, region_targets, region_loss_numerators, segmentation_loss, tensor_hash
from stage4_fusion.models import EncoderPair
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode
from engine import evaluate

NAMES = ('paired_baseline','region_only','hiram')
MICRO = 64
EFFECTIVE_BATCH = 64
ENCODER_BATCH = 64


def load_batch_features(encoder, images):
    collected = None
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        for start in range(0,len(images),ENCODER_BATCH):
            out = encoder(images[start:start+ENCODER_BATCH])
            if collected is None:
                collected = [[] for _ in out]
            for dst,x in zip(collected,out):
                dst.append(x.detach())
    return [torch.cat(parts,0) for parts in collected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    assert args.smoke or args.stop_after_epoch is None
    setup()
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    out = ROOT/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=True)
    lock=(out/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():
        print('ALREADY DONE',flush=True)
        return
    stopping=[False]
    def stop(*_): stopping[0]=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    started=time.perf_counter()
    budget=2 if args.smoke else 300
    config=dict(smoke_only=args.smoke,seed=42,epochs=budget,models=list(NAMES),effective_batch=64,
        micro_batch=MICRO,encoder_batch=ENCODER_BATCH,updates_per_epoch=30,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR(T_max=300, eta_min=1e-5), epoch end',
        segmentation_loss='stage1 BCEWithLogits + per-image Soft Dice smooth=1',
        auxiliary_loss='mask >=0.5, official width2 morphology, bilinear 16x16, normalized KL for interior and boundary per branch, weight=1',
        route_training='gumbel hard stochastic',route_evaluation='argmax deterministic',
        augment_seed='42+epoch*100003',precision='BF16 autocast / FP32 loss',clip=1.,
        selection='max val macro Dice; earliest exact tie',secondary='min val segmentation loss; epoch 300',
        data='ISIC2018 1886/808 legacy development split',gpu=torch.cuda.get_device_name(),
        official_hiram_commit='d8aac2de3c33f8f2e79e281fcea8e5239db29645')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config
    else:
        dump(out/'config.json',config)
    data=ImageData(42)
    assert data.audit == json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=EncoderPair().cuda().eval()
    enc_hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    expected=json.loads((ROOT/'preflight.json').read_text())['encoder_hashes']
    assert enc_hashes == expected
    models={'paired_baseline':BaselineProbe().cuda(),
            **{n:RegionProbe(n).cuda() for n in ('region_only','hiram')}}
    optimizers={n:torch.optim.AdamW((p for p in models[n].parameters() if p.requires_grad),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n in NAMES}
    schedulers={n:torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[n],T_max=300,eta_min=1e-5) for n in NAMES}
    print('ENCODING VALIDATION FEATURES',flush=True)
    val_features,val_targets=encode(encoder,data,'val',ENCODER_BATCH)
    start=1
    updates={n:0 for n in NAMES}
    selectors={n:{'best_dice':{'epoch':0,'score':-1.},'best_loss':{'epoch':0,'score':1e30}} for n in NAMES}
    history=[]
    resumed=False
    prev_seconds=0.
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu')
        assert state['config']==config and state['encoder_hashes']==enc_hashes
        for n in NAMES:
            models[n].load_state_dict(state['models'][n],strict=True)
            optimizers[n].load_state_dict(state['optimizers'][n])
            schedulers[n].load_state_dict(state['schedulers'][n])
            assert tree_hash(models[n].state_dict())==tree_hash(state['models'][n]),n
            assert tree_hash(optimizers[n].state_dict())==tree_hash(state['optimizers'][n]),n
            assert tree_hash(schedulers[n].state_dict())==tree_hash(state['schedulers'][n]),n
        restore_rng(state['rng'])
        assert tree_hash(rng_state())==tree_hash(state['rng'])
        start=state['epoch']+1
        updates=state['updates']
        selectors=state['selectors']
        history=state['history']
        prev_seconds=state['active_seconds']
        resumed=True
        dump(out/'resume_audit.json',dict(resumed=True,checkpoint_epoch=state['epoch'],
             start_epoch=start,updates=updates,optimizer_scheduler_and_rng_restored=True))
        del state
    assert all(updates[n]==(start-1)*30 for n in NAMES)
    print('START',json.dumps(dict(smoke=args.smoke,start_epoch=start,budget=budget,resumed=resumed)),flush=True)
    for epoch in range(start,budget+1):
        torch.cuda.synchronize()
        epoch_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        order,matrices=data.geometry(epoch)
        for model in models.values(): model.train()
        train_total={n:0. for n in NAMES}
        train_seg={n:0. for n in NAMES}
        train_aux={n:0. for n in NAMES}
        lrs={n:optimizers[n].param_groups[0]['lr'] for n in NAMES}
        for offset in range(0,len(order),EFFECTIVE_BATCH):
            ids=order[offset:offset+EFFECTIVE_BATCH]
            images,target=data.batch('train',ids,matrices)
            features=load_batch_features(encoder,images)
            soft_regions=region_targets(target)
            denom=[x.sum()+1e-8 for x in soft_regions]
            size=len(target)
            for name in NAMES:
                model=models[name]
                opt=optimizers[name]
                opt.zero_grad(set_to_none=True)
                step_loss=step_seg=step_aux=0.
                for j in range(0,size,MICRO):
                    end=min(j+MICRO,size)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,regions=model([x[j:end] for x in features],return_regions=True)
                    seg=segmentation_loss(logits,target[j:end])
                    if name == 'paired_baseline':
                        aux=seg.new_zeros(())
                    else:
                        numerator=region_loss_numerators(regions,[x[j:end] for x in soft_regions])
                        aux=numerator[0]/denom[0]+numerator[1]/denom[1]
                    loss=seg*((end-j)/size)+aux
                    assert torch.isfinite(loss),(epoch,name,offset,j)
                    loss.backward()
                    step_loss+=float(loss.item())
                    step_seg+=float(seg.item())*(end-j)/size
                    step_aux+=float(aux.item())
                norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                assert torch.isfinite(norm),(epoch,name,offset)
                opt.step()
                updates[name]+=1
                train_total[name]+=step_loss*size
                train_seg[name]+=step_seg*size
                train_aux[name]+=step_aux*size
            del images,target,features,soft_regions,logits,regions,loss
        assert all(updates[n]==epoch*30 for n in NAMES)
        for scheduler in schedulers.values(): scheduler.step()
        vals={n:evaluate(models[n],val_features,val_targets,MICRO)[0] for n in NAMES}
        for name in NAMES:
            dest=out/name
            dest.mkdir(exist_ok=True)
            for criterion,score in [('best_dice',vals[name]['macro']['dice']),('best_loss',vals[name]['loss'])]:
                old=selectors[name][criterion]['score']
                if (score>old if criterion=='best_dice' else score<old):
                    selectors[name][criterion]={'epoch':epoch,'score':score}
                    checkpoint(dest/(criterion+'.pth'),dict(model=models[name].state_dict(),epoch=epoch,
                        metrics=vals[name],criterion=criterion,seed=42,encoder_hashes=enc_hashes))
        if epoch==1 or epoch%25==0 or epoch==budget:
            assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            assert not any(m.training for m in encoder.modules())
            assert all(tensor_hash(getattr(encoder,n))==h for n,h in enc_hashes.items())
        torch.cuda.synchronize()
        row=dict(epoch=epoch,updates=updates.copy(),seconds=time.perf_counter()-epoch_start,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=train_total[n]/1886,train_seg=train_seg[n]/1886,
                train_aux=train_aux[n]/1886,val=vals[n],lr=lrs[n],selectors=selectors[n].copy()) for n in NAMES})
        history.append(row)
        checkpoint(out/'latest.pth',dict(epoch=epoch,models={n:models[n].state_dict() for n in NAMES},
            optimizers={n:optimizers[n].state_dict() for n in NAMES},schedulers={n:schedulers[n].state_dict() for n in NAMES},
            rng=rng_state(),history=history,updates=updates,selectors=selectors,config=config,
            encoder_hashes=enc_hashes,active_seconds=prev_seconds+time.perf_counter()-started))
        dump(out/'history.json',history)
        dump(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],gpu_peak_gib=row['gpu_peak_gib'],
            macro_dice={n:vals[n]['macro']['dice'] for n in NAMES},
            train_loss={n:train_total[n]/1886 for n in NAMES})),flush=True)
        if args.stop_after_epoch==epoch or stopping[0]:
            print('STOPPED AFTER CHECKPOINT',epoch,flush=True)
            return
    assert all(updates[n]==budget*30 for n in NAMES)
    assert all(tensor_hash(getattr(encoder,n))==h for n,h in enc_hashes.items())
    for n in NAMES:
        checkpoint(out/n/'last.pth',dict(model=models[n].state_dict(),epoch=budget,metrics=history[-1]['models'][n]['val'],
            criterion='last',seed=42,encoder_hashes=enc_hashes))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,resumed=resumed,
        encoder_hashes=enc_hashes,encoder_unchanged=True,selectors=selectors,
        active_training_seconds=prev_seconds+time.perf_counter()-started))
    print('TRAINING DONE',flush=True)


if __name__=='__main__': main()
