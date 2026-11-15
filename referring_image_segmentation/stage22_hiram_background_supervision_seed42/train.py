"""Three same-initialization HiRAM probes differing only in region loss."""
from common import *
import argparse
import fcntl
import signal
from models import (NAMES, DecoderProbe, FrozenEncoder, region_targets, auxiliary_numerators,
                    auxiliary_loss, segmentation_loss, tensor_hash)
from torch.nn import functional as F
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode
from engine import evaluate

MICRO = 64
EFFECTIVE_BATCH = 64
ENCODER_BATCH = 64


def forward_and_observe(model, features):
    """Count actual Gumbel draws without changing draws, outputs or gradients."""
    original=F.gumbel_softmax
    observed=[]
    def wrapper(*args,**kwargs):
        out=original(*args,**kwargs)
        if kwargs.get('hard',False) and out.shape[-1]==3:
            observed.append(out.detach().float().reshape(-1,3).sum(0).double())
        return out
    F.gumbel_softmax=wrapper
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,regions=model(features,return_regions=True)
    finally:
        F.gumbel_softmax=original
    assert len(observed)==2 and len(regions)==2
    return logits,regions,torch.stack(observed)


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
    global MICRO, ENCODER_BATCH, EVAL_BATCH
    capacity=json.loads((ROOT/'capacity_preflight.json').read_text())
    assert capacity['passed'] and capacity['effective_batch']==64
    MICRO=capacity['train_micro_batch']
    assert MICRO==64
    ENCODER_BATCH=capacity['encoder_batch']
    EVAL_BATCH=capacity['evaluation_batch']
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
        micro_batch=MICRO,encoder_batch=ENCODER_BATCH,evaluation_batch=EVAL_BATCH,updates_per_epoch=30,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR(T_max=300, eta_min=1e-5), epoch end',
        segmentation_loss='stage1 BCEWithLogits + per-image Soft Dice smooth=1',
        auxiliary_loss='original boundary+interior; scaled_original=1.5*original; background_supervised=original+background; each class normalized by effective-batch target mass',
        route_training='gumbel hard stochastic',route_evaluation='argmax deterministic',
        common_gumbel_draws=True,train_route_measurement='non-mutating observation of actual gumbel samples',
        augment_seed='42+epoch*100003',precision='BF16 autocast / FP32 loss',clip=1.,
        selection='max val macro Dice; earliest exact tie',secondary='min val segmentation loss; epoch 300',
        data='ISIC2018 1886/808 legacy development split',gpu=torch.cuda.get_device_name(),
        encoder='NVIDIA MambaVision-S-1K',hiram_position='32x32 decoder skip',
        official_hiram_commit='d8aac2de3c33f8f2e79e281fcea8e5239db29645')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config
    else:
        dump(out/'config.json',config)
    data=ImageData(42)
    assert data.audit == json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    encoder=FrozenEncoder().cuda().eval()
    enc_hashes={'mambavision':tensor_hash(encoder)}
    expected={'mambavision':json.loads((ROOT/'preflight.json').read_text())['encoder_sha256']}
    assert enc_hashes == expected
    models={n:DecoderProbe('hiram').cuda() for n in NAMES}
    assert all(tensor_hash(m)==json.loads((ROOT/'preflight.json').read_text())['model_initial_sha256'] for m in models.values())
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
        train_components={n:torch.zeros(3,dtype=torch.float64,device='cuda') for n in NAMES}
        train_gumbel={n:torch.zeros((2,3),dtype=torch.float64,device='cuda') for n in NAMES}
        train_soft={n:torch.zeros((2,3),dtype=torch.float64,device='cuda') for n in NAMES}
        lrs={n:optimizers[n].param_groups[0]['lr'] for n in NAMES}
        for offset in range(0,len(order),EFFECTIVE_BATCH):
            ids=order[offset:offset+EFFECTIVE_BATCH]
            images,target=data.batch('train',ids,matrices)
            features=load_batch_features(encoder,images)
            soft_regions=region_targets(target)
            denom=[x.sum()+1e-8 for x in soft_regions]
            size=len(target)
            common_rng=torch.cuda.get_rng_state()
            after_forward_rng=None
            for name in NAMES:
                model=models[name]
                opt=optimizers[name]
                opt.zero_grad(set_to_none=True)
                torch.cuda.set_rng_state(common_rng)
                logits,regions,gumbel=forward_and_observe(model,features)
                if after_forward_rng is None: after_forward_rng=torch.cuda.get_rng_state()
                else: assert torch.equal(after_forward_rng,torch.cuda.get_rng_state())
                numerator=auxiliary_numerators(regions,soft_regions)
                aux=auxiliary_loss(name,numerator,denom)
                seg=segmentation_loss(logits,target)
                loss=seg+aux
                assert torch.isfinite(loss),(epoch,name,offset)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                assert torch.isfinite(norm),(epoch,name,offset)
                opt.step()
                updates[name]+=1
                train_total[name]+=float(loss.item())*size
                train_seg[name]+=float(seg.item())*size
                train_aux[name]+=float(aux.item())*size
                train_components[name]+=torch.stack([(x/d).detach().double() for x,d in zip(numerator,denom)])*size
                train_gumbel[name]+=gumbel
                train_soft[name]+=torch.stack([r.detach().float().exp().sum((0,2,3)).double() for r in regions])
            del images,target,features,soft_regions,logits,regions,loss
        assert all(updates[n]==epoch*30 for n in NAMES)
        for scheduler in schedulers.values(): scheduler.step()
        vals={n:evaluate(models[n],val_features,val_targets,EVAL_BATCH)[0] for n in NAMES}
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
            assert tensor_hash(encoder)==enc_hashes['mambavision']
        torch.cuda.synchronize()
        row=dict(epoch=epoch,updates=updates.copy(),seconds=time.perf_counter()-epoch_start,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
            models={n:dict(train_loss=train_total[n]/1886,train_seg=train_seg[n]/1886,
                train_aux=train_aux[n]/1886,
                train_aux_components_boundary_interior_background=(train_components[n]/1886).cpu().tolist(),
                train_gumbel_hard_fractions=(train_gumbel[n]/(1886*32*32)).cpu().tolist(),
                train_mean_soft_probabilities=(train_soft[n]/(1886*32*32)).cpu().tolist(),
                val=vals[n],lr=lrs[n],selectors=selectors[n].copy()) for n in NAMES})
        history.append(row)
        checkpoint(out/'latest.pth',dict(epoch=epoch,models={n:models[n].state_dict() for n in NAMES},
            optimizers={n:optimizers[n].state_dict() for n in NAMES},schedulers={n:schedulers[n].state_dict() for n in NAMES},
            rng=rng_state(),history=history,updates=updates,selectors=selectors,config=config,
            encoder_hashes=enc_hashes,active_seconds=prev_seconds+time.perf_counter()-started))
        dump(out/'history.json',history)
        dump(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],gpu_peak_gib=row['gpu_peak_gib'],
            macro_dice={n:vals[n]['macro']['dice'] for n in NAMES},
            val_hard_bg={n:vals[n]['routing']['coarse']['hard_fraction'][0] for n in NAMES},
            train_loss={n:train_total[n]/1886 for n in NAMES})),flush=True)
        if args.stop_after_epoch==epoch or stopping[0]:
            print('STOPPED AFTER CHECKPOINT',epoch,flush=True)
            return
    assert all(updates[n]==budget*30 for n in NAMES)
    assert tensor_hash(encoder)==enc_hashes['mambavision']
    for n in NAMES:
        checkpoint(out/n/'last.pth',dict(model=models[n].state_dict(),epoch=budget,metrics=history[-1]['models'][n]['val'],
            criterion='last',seed=42,encoder_hashes=enc_hashes))
    dump(out/'DONE.json',dict(seed=42,epochs=budget,updates=updates,smoke_only=args.smoke,resumed=resumed,
        encoder_hashes=enc_hashes,encoder_unchanged=True,selectors=selectors,
        active_training_seconds=prev_seconds+time.perf_counter()-started))
    print('TRAINING DONE',flush=True)


if __name__=='__main__': main()
