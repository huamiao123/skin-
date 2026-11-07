"""One seed, one adapted MMSCoPE head; original encoder and data protocol."""
from segman_model import *
import argparse
import fcntl
import signal
from train import encode_val, evaluate, extract
from stage17_mambavision_decoder_hiram_seed42.models import FrozenEncoder
from stage3_confirmation.models import segmentation_loss, tensor_hash
from stage3_confirmation.data import ImageData


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    assert args.smoke or args.stop_after_epoch is None
    setup()
    local=Path(__file__).resolve().parent
    pre=json.loads((local/'preflight.json').read_text())
    cap=json.loads((local/'capacity_preflight.json').read_text())
    assert pre['passed'] and cap['passed'] and cap['train_batch']==64
    out=local/('smoke' if args.smoke else 'results')
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
    config=dict(smoke_only=args.smoke,seed=42,epochs=budget,model='mambavision_segman_adapted',
        data='ISIC2018 original 1886 train / 808 development validation',
        encoder='NVIDIA pretrained MambaVision-S-1K, frozen including buffers',
        decoder='SegMAN-small MMSCoPE adapted to [96,192,384,768] MambaVision channels, binary logits',
        adaptation='standalone framework shims; standard selective scan; not official reproduction',
        upstream_commit=pre['segman_revision'],source_sha256=pre['source_sha256'],
        cross_scan_source_sha256=pre['cross_scan_source_sha256'],
        trainable_parameters=pre['trainable_parameters'],train_batch=64,effective_batch=64,
        encoder_batch=64,evaluation_batch=64,updates_per_epoch=30,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR(T_max=300, eta_min=1e-5), epoch end',
        loss='historical BCEWithLogits plus per-image soft Dice, smooth=1',gradient_clip=1.,
        augment_seed='42 + epoch * 100003',precision='BF16 autocast with FP32 segmentation loss',
        selection='max validation macro Dice, earliest exact tie',
        gpu=torch.cuda.get_device_name())
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config
    else:
        dump(out/'config.json',config)
    data=ImageData(42)
    assert data.audit==json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    encoder_hash=tensor_hash(encoder)
    assert encoder_hash==json.loads((ROOT/'preflight.json').read_text())['encoder_sha256']
    model=MambaVisionSegMAN().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=300,eta_min=1e-5)
    print('ENCODING VALIDATION FEATURES',flush=True)
    val_features,val_targets=encode_val(encoder,data,64)
    start=1
    updates=0
    selectors={'best_dice':{'epoch':0,'score':-1.},'best_loss':{'epoch':0,'score':1e30}}
    history=[]
    prev_seconds=0.
    resumed=False
    if (out/'latest.pth').exists():
        state=torch.load(out/'latest.pth',map_location='cpu')
        assert state['config']==config and state['encoder_hash']==encoder_hash
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        assert tree_hash(model.state_dict())==tree_hash(state['model'])
        assert tree_hash(optimizer.state_dict())==tree_hash(state['optimizer'])
        assert tree_hash(scheduler.state_dict())==tree_hash(state['scheduler'])
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
    assert updates==(start-1)*30
    print('START',json.dumps(dict(smoke=args.smoke,start_epoch=start,budget=budget,resumed=resumed)),flush=True)
    for epoch in range(start,budget+1):
        torch.cuda.synchronize()
        epoch_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        order,matrices=data.geometry(epoch)
        model.train()
        train_loss=0.
        lr=optimizer.param_groups[0]['lr']
        for offset in range(0,len(order),64):
            ids=order[offset:offset+64]
            images,target=data.batch('train',ids,matrices)
            features=extract(encoder,images,64)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(features)
            loss=segmentation_loss(logits,target)
            assert torch.isfinite(loss),(epoch,offset)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm),(epoch,offset)
            optimizer.step()
            updates+=1
            train_loss+=float(loss.item())*len(target)
            del images,target,features,logits,loss
        assert updates==epoch*30
        scheduler.step()
        val=evaluate(model,val_features,val_targets,64)[0]
        for criterion,score in [('best_dice',val['macro']['dice']),('best_loss',val['loss'])]:
            old=selectors[criterion]['score']
            if (score>old if criterion=='best_dice' else score<old):
                selectors[criterion]={'epoch':epoch,'score':score}
                checkpoint(out/(criterion+'.pth'),dict(model=model.state_dict(),epoch=epoch,
                    metrics=val,criterion=criterion,seed=42,encoder_hash=encoder_hash))
        if epoch==1 or epoch%25==0 or epoch==budget:
            assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            assert not any(m.training for m in encoder.modules())
            assert tensor_hash(encoder)==encoder_hash
        torch.cuda.synchronize()
        row=dict(epoch=epoch,updates=updates,seconds=time.perf_counter()-epoch_start,
                 gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                 train_loss=train_loss/1886,val=val,lr=lr,selectors=selectors.copy())
        history.append(row)
        checkpoint(out/'latest.pth',dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=rng_state(),history=history,updates=updates,
            selectors=selectors,config=config,encoder_hash=encoder_hash,
            active_seconds=prev_seconds+time.perf_counter()-started))
        dump(out/'history.json',history)
        dump(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],gpu_peak_gib=row['gpu_peak_gib'],
            macro_dice=val['macro']['dice'],train_loss=row['train_loss'])),flush=True)
        if args.stop_after_epoch==epoch or stop[0]:
            print('STOPPED AFTER CHECKPOINT',epoch,flush=True)
            return
    assert updates==budget*30 and tensor_hash(encoder)==encoder_hash
    checkpoint(out/'last.pth',dict(model=model.state_dict(),epoch=budget,metrics=history[-1]['val'],
               criterion='last',seed=42,encoder_hash=encoder_hash))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,resumed=resumed,
        encoder_hash=encoder_hash,encoder_unchanged=True,selectors=selectors,
        active_training_seconds=prev_seconds+time.perf_counter()-started))
    print('TRAINING DONE',flush=True)


if __name__=='__main__':main()
