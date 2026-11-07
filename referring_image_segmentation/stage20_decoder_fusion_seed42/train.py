"""Paired, resumable MambaVision/Swin and HEDS AXIS experiments in BF16."""
from common import *
import argparse
import fcntl
import signal
from models import DecoderProbe, FrozenEncoder, UPSTREAM_SHA256, segmentation_loss, tensor_hash
from stage3_confirmation.data import ImageData
from evaluation import aggregate_counts

NAMES = ('paired_baseline', 'gate_only', 'axis_bridge')
EFFECTIVE_BATCH = 64


@torch.no_grad()
def extract(encoder, images, batch):
    parts = None
    for i in range(0,len(images),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out = encoder(images[i:i+batch])
        if parts is None:
            parts = [[] for _ in out]
        for dest, x in zip(parts,out):
            dest.append(x.detach())
    return [torch.cat(x,0) for x in parts]


@torch.no_grad()
def encode_val(encoder,data,batch):
    n=len(data.ids['val'])
    cache=None
    targets=torch.empty((n,1,256,256),dtype=torch.float32,device='cuda')
    for i in range(0,n,batch):
        end=min(i+batch,n)
        images,gt=data.batch('val',torch.arange(i,end,device='cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feats=encoder(images)
        if cache is None:
            cache=[torch.empty((n,*x.shape[1:]),dtype=torch.bfloat16,device='cuda') for x in feats]
        for dst,x in zip(cache,feats):
            dst[i:end].copy_(x)
        targets[i:end].copy_(gt)
    return cache,targets


@torch.no_grad()
def evaluate(model,features,targets,batch=16,export=False):
    model.eval()
    n=len(targets)
    counts=[]
    loss_sum=0.
    probabilities=np.empty((n,256,256),dtype=np.float32) if export else None
    for i in range(0,n,batch):
        end=min(i+batch,n)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model([x[i:end] for x in features])
        gt=targets[i:end]
        loss_sum += float(segmentation_loss(logits,gt))*(end-i)
        prob=logits.float().sigmoid()
        pred,truth=prob>=.5,gt>=.5
        vals=[(pred & truth).sum((1,2,3)),(pred & ~truth).sum((1,2,3)),
              (~pred & truth).sum((1,2,3)),(~pred & ~truth).sum((1,2,3))]
        counts.append(torch.stack(vals,1).cpu().numpy())
        if export: probabilities[i:end]=prob[:,0].cpu().numpy()
    counts=np.concatenate(counts).astype(np.int64)
    result=aggregate_counts(*counts.T)
    result['loss']=loss_sum/n
    return result,probabilities,counts


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
    micro=capacity['train_micro_batch']
    encoder_batch=capacity['encoder_batch']
    eval_batch=capacity['evaluation_batch']
    out=ROOT/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=True)
    lock=(out/'train.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():
        print('ALREADY COMPLETE',flush=True)
        return
    stop=[False]
    def handle(*_): stop[0]=True
    signal.signal(signal.SIGTERM,handle)
    signal.signal(signal.SIGINT,handle)
    started=time.perf_counter()
    budget=2 if args.smoke else 300
    config=dict(smoke_only=args.smoke,seed=42,epochs=budget,models=list(NAMES),
        data='ISIC2018 original 1886 train / 808 development validation',
        encoder='NVIDIA pretrained MambaVision-S-1K, frozen including buffers',
        adapter_decoder='historic 64-channel adapter and Swin Pixel Decoder',
        comparison='32x32 skip: add vs HEDS context gate vs gate plus coordinate attention',
        heds_commit='a9287da787957b6376a9c403a56ad6693cd04191',axis_source_sha256=UPSTREAM_SHA256,
        effective_batch=64,micro_batch=micro,encoder_batch=encoder_batch,evaluation_batch=eval_batch,
        updates_per_epoch=30,optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR(T_max=300, eta_min=1e-5), epoch end',
        loss='historical BCEWithLogits plus per-image soft Dice, smooth=1',gradient_clip=1.,
        augment_seed='42 + epoch * 100003',precision='BF16 autocast with FP32 segmentation loss',
        precision_reason='Matches the historical stage17 mixed-precision protocol on BF16-capable GPU',
        selection='max validation macro Dice, earliest exact tie',
        secondary='minimum validation loss; last epoch',gpu=torch.cuda.get_device_name())
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config
    else:
        dump(out/'config.json',config)
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    encoder_hash=tensor_hash(encoder)
    assert encoder_hash==preflight['encoder_sha256']
    models={n:DecoderProbe(n).cuda() for n in NAMES}
    optimizers={n:torch.optim.AdamW(models[n].parameters(),lr=.001,weight_decay=.01,
                                   betas=(.9,.999),eps=1e-8) for n in NAMES}
    schedulers={n:torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[n],T_max=300,eta_min=1e-5) for n in NAMES}
    print('ENCODING VALIDATION FEATURES',flush=True)
    val_features,val_targets=encode_val(encoder,data,encoder_batch)
    start=1
    updates={n:0 for n in NAMES}
    selectors={n:{'best_dice':{'epoch':0,'score':-1.},'best_loss':{'epoch':0,'score':1e30}} for n in NAMES}
    history=[]
    prev_seconds=0.
    resumed=False
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu')
        assert state['config']==config and state['encoder_hash']==encoder_hash
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
        train_loss={n:0. for n in NAMES}
        lrs={n:optimizers[n].param_groups[0]['lr'] for n in NAMES}
        for offset in range(0,len(order),EFFECTIVE_BATCH):
            ids=order[offset:offset+EFFECTIVE_BATCH]
            images,target=data.batch('train',ids,matrices)
            features=extract(encoder,images,encoder_batch)
            size=len(target)
            for name in NAMES:
                model=models[name]
                opt=optimizers[name]
                opt.zero_grad(set_to_none=True)
                step_loss=0.
                for j in range(0,size,micro):
                    end=min(j+micro,size)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model([x[j:end] for x in features])
                    loss=segmentation_loss(logits,target[j:end])*(end-j)/size
                    assert torch.isfinite(loss),(epoch,name,offset,j)
                    loss.backward()
                    step_loss+=float(loss.item())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm),(epoch,name,offset)
                opt.step()
                updates[name]+=1
                train_loss[name]+=step_loss*size
            del images,target,features,logits,loss
        assert all(updates[n]==epoch*30 for n in NAMES)
        for scheduler in schedulers.values(): scheduler.step()
        vals={n:evaluate(models[n],val_features,val_targets,eval_batch)[0] for n in NAMES}
        for name in NAMES:
            dest=out/name
            dest.mkdir(exist_ok=True)
            for criterion,score in [('best_dice',vals[name]['macro']['dice']),('best_loss',vals[name]['loss'])]:
                old=selectors[name][criterion]['score']
                if (score>old if criterion=='best_dice' else score<old):
                    selectors[name][criterion]={'epoch':epoch,'score':score}
                    checkpoint(dest/(criterion+'.pth'),dict(model=models[name].state_dict(),epoch=epoch,
                        metrics=vals[name],criterion=criterion,seed=42,encoder_hash=encoder_hash))
        if epoch==1 or epoch%25==0 or epoch==budget:
            assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            assert not any(m.training for m in encoder.modules())
            assert tensor_hash(encoder)==encoder_hash
        torch.cuda.synchronize()
        row=dict(epoch=epoch,updates=updates.copy(),seconds=time.perf_counter()-epoch_start,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=train_loss[n]/1886,val=vals[n],lr=lrs[n],
                           selectors=selectors[n].copy()) for n in NAMES})
        history.append(row)
        checkpoint(out/'latest.pth',dict(epoch=epoch,models={n:models[n].state_dict() for n in NAMES},
            optimizers={n:optimizers[n].state_dict() for n in NAMES},schedulers={n:schedulers[n].state_dict() for n in NAMES},
            rng=rng_state(),history=history,updates=updates,selectors=selectors,config=config,
            encoder_hash=encoder_hash,active_seconds=prev_seconds+time.perf_counter()-started))
        dump(out/'history.json',history)
        dump(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],gpu_peak_gib=row['gpu_peak_gib'],
            macro_dice={n:vals[n]['macro']['dice'] for n in NAMES},
            train_loss={n:train_loss[n]/1886 for n in NAMES})),flush=True)
        if args.stop_after_epoch==epoch or stop[0]:
            print('STOPPED AFTER CHECKPOINT',epoch,flush=True)
            return
    assert all(updates[n]==budget*30 for n in NAMES)
    assert tensor_hash(encoder)==encoder_hash
    for n in NAMES:
        checkpoint(out/n/'last.pth',dict(model=models[n].state_dict(),epoch=budget,
            metrics=history[-1]['models'][n]['val'],criterion='last',seed=42,encoder_hash=encoder_hash))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,resumed=resumed,
        encoder_hash=encoder_hash,encoder_unchanged=True,selectors=selectors,
        active_training_seconds=prev_seconds+time.perf_counter()-started))
    print('TRAINING DONE',flush=True)


if __name__=='__main__':
    main()
