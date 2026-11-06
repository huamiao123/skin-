"""One official frozen S encoder, one seed42 Swin probe; complete resumable epochs."""
from common import *
from models import FrozenEncoder,build_probe,tensor_hash,segmentation_loss
from data import ImageData
from engine import encode,evaluate
import argparse,fcntl,signal

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    assert args.smoke or args.stop_after_epoch is None,'Only smoke can stop early'
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    verify_frozen_sources();setup();session_start=time.perf_counter()
    out=ROOT/('smoke' if args.smoke else 'results');out.mkdir(exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():print('ALREADY_DONE',flush=True);return
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    config=json.loads((ROOT/'config.json').read_text());config=dict(config,epochs=2 if args.smoke else 300,smoke_only=args.smoke)
    budget=config['epochs'];encoder=FrozenEncoder().cuda();probe=build_probe().cuda()
    encoder_hash=tensor_hash(encoder);initial_probe=tensor_hash(probe)
    assert encoder_hash==config['encoder_initial_hash'] and initial_probe==config['probe_initial_hash']
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300,eta_min=1e-5)
    history=[];selectors=dict(best_dice=dict(epoch=0,score=-1.),best_loss=dict(epoch=0,score=1e30))
    start=1;updates=0;resumed=False;previous_seconds=0.;resume_audit=None
    data=ImageData()
    if (out/'latest.pth').exists():
        checkpoint=torch.load(out/'latest.pth',map_location='cpu');assert checkpoint['config']==config
        probe.load_state_dict(checkpoint['probe'],strict=True);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'])
        assert tensor_hash(probe)==checkpoint['probe_hash']!=initial_probe
        assert tree_hash(optimizer.state_dict())==checkpoint['optimizer_hash']
        assert tree_hash(scheduler.state_dict())==checkpoint['scheduler_hash']
        assert tree_hash(rng_state())==checkpoint['rng_hash']
        history=checkpoint['history'];selectors=checkpoint['selectors'];updates=checkpoint['updates']
        start=checkpoint['epoch']+1;resumed=True;previous_seconds=checkpoint['active_seconds']
        resume_audit=dict(checkpoint_epoch=checkpoint['epoch'],start_epoch=start,updates_restored=updates,
            model_hash_restored=True,optimizer_exact=True,scheduler_exact=True,python_numpy_cpu_cuda_rng_exact=True,
            reinitialized=False,probe_hash=checkpoint['probe_hash'])
        dump(out/'resume_audit.json',resume_audit)
    assert updates==(start-1)*30
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    print('START',json.dumps(dict(smoke=args.smoke,start_epoch=start,resumed=resumed,seed=SEED,budget=budget,encoder_hash=encoder_hash)),flush=True)
    val_features,val_targets=encode(encoder,data,'val',config['encoder_batch'])
    for epoch in range(start,budget+1):
        torch.cuda.synchronize();epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        before_encode=time.perf_counter();features,targets=encode(encoder,data,'train',config['encoder_batch'],epoch)
        torch.cuda.synchronize();encode_seconds=time.perf_counter()-before_encode
        probe.train();train_sum=0.;lr=optimizer.param_groups[0]['lr'];epoch_updates=0
        for offset in range(0,len(targets),64):
            batch=[f[offset:offset+64] for f in features];target=targets[offset:offset+64]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=probe(batch)
            loss=segmentation_loss(logits,target);assert torch.isfinite(loss)
            loss.backward()
            if offset==0:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
            norm=torch.nn.utils.clip_grad_norm_(probe.parameters(),1.);assert torch.isfinite(norm)
            optimizer.step();updates+=1;epoch_updates+=1;train_sum+=loss.item()*len(target)
        assert epoch_updates==30 and updates==epoch*30
        scheduler.step()
        val,_,_=evaluate(probe,val_features,val_targets)
        del features,targets,batch,target,logits,loss
        if epoch==1 or epoch%25==0 or epoch==budget:
            assert all(m.routes==4 for m in encoder.modules() if hasattr(m,'routes'))
            assert not any(m.training for m in encoder.modules())
            assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
            assert tensor_hash(encoder)==encoder_hash,'Encoder parameter/buffer changed'
        for criterion,score in [('best_dice',val['macro']['dice']),('best_loss',val['loss'])]:
            previous=selectors[criterion]['score']
            if (score>previous if criterion=='best_dice' else score<previous):
                selectors[criterion]=dict(epoch=epoch,score=score)
                save_checkpoint(out/f'{criterion}.pth',dict(probe=probe.state_dict(),epoch=epoch,metrics=val,
                    criterion=criterion,seed=42,encoder_hash=encoder_hash))
        torch.cuda.synchronize()
        row=dict(epoch=epoch,optimizer_updates=updates,epoch_updates=epoch_updates,
            train_loss=train_sum/1886,val=val,lr=lr,next_lr=optimizer.param_groups[0]['lr'],
            seconds=time.perf_counter()-epoch_start,encode_seconds=encode_seconds,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            best_dice_epoch=selectors['best_dice']['epoch'],best_loss_epoch=selectors['best_loss']['epoch'])
        history.append(row)
        rng=rng_state()
        save_checkpoint(out/'latest.pth',dict(probe=probe.state_dict(),probe_hash=tensor_hash(probe),
            optimizer=optimizer.state_dict(),optimizer_hash=tree_hash(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),scheduler_hash=tree_hash(scheduler.state_dict()),
            rng=rng,rng_hash=tree_hash(rng),epoch=epoch,updates=updates,history=history,
            selectors=selectors,config=config,active_seconds=previous_seconds+time.perf_counter()-session_start))
        dump(out/'history.json',history);dump(out/'status.json',row)
        print('EPOCH',json.dumps(row),flush=True)
        if args.stop_after_epoch==epoch:
            print('SMOKE_STOP_FOR_RESUME',flush=True);return
        if stopping[0]:
            dump(out/'INTERRUPTED.json',dict(epoch=epoch,updates=updates,resumable=True))
            print('STOPPED_AFTER_CHECKPOINT',flush=True);return
    assert len(history)==budget and updates==30*budget
    assert tensor_hash(encoder)==encoder_hash
    verify_frozen_sources()
    save_checkpoint(out/'last.pth',dict(probe=probe.state_dict(),epoch=budget,metrics=history[-1]['val'],
        criterion='last',seed=42,encoder_hash=encoder_hash))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,
        resumed=resumed,resume_audit=resume_audit,encoder_unchanged_including_bn_buffers=True,
        initial_encoder_hash=encoder_hash,final_encoder_hash=tensor_hash(encoder),
        active_training_seconds=previous_seconds+time.perf_counter()-session_start,
        epoch_seconds_sum=sum(r['seconds'] for r in history),selectors=selectors,
        checkpoints={c:dict(path=str(out/f'{c}.pth'),sha256=sha(out/f'{c}.pth')) for c in CRITERIA}))
    print('TRAINING_DONE',json.dumps(dict(epochs=budget,updates=updates,smoke=args.smoke)),flush=True)

if __name__=='__main__':main()
