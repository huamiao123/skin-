"""Three controlled fusion probes; unchanged stage-4 optimizer/evaluation loop."""
import argparse
import fcntl
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import EncoderPair, build_probes, select_features, NAMES, tensor_hash, segmentation_loss, SPEC
from stage3_confirmation.data import ImageData
from stage3_confirmation.engine import encode, evaluate

ROOT = Path(__file__).resolve().parent
CRITERIA = ('best_dice','best_loss','last')


def save_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def save_checkpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    assert args.stop_after_epoch is None or args.smoke
    seed=42;budget=2 if args.smoke else 300
    out=ROOT/('smoke' if args.smoke else 'results')/'seed42'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists(): print('Already complete',flush=True);return
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder=EncoderPair().cuda()
    hashes={n:tensor_hash(getattr(encoder,n)) for n in ('dinov3','vmamba')}
    for n,h in hashes.items():
        assert h==json.loads((ROOT.parent/f'stage1_visual/benchmark_{n}.json').read_text())['frozen_hash']
    probes={n:p.cuda() for n,p in build_probes(seed).items()}
    preflight=json.loads((ROOT/'preflight.json').read_text());assert preflight['passed']
    for n in NAMES:assert tensor_hash(probes[n])==preflight['initial_probe_hashes'][n]
    optimizers={n:torch.optim.AdamW(p.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8) for n,p in probes.items()}
    schedulers={n:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=300,eta_min=1e-5) for n,o in optimizers.items()}
    config=dict(seed=seed,epochs=budget,input_size=256,batch=64,encoder_batch=64,evaluation_batch=64,
        optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,T_max=300,eta_min=1e-5,
        precision='BF16 autocast; FP32 loss',clip=1.,loss='BCEWithLogits + per-image soft Dice; smooth=1',
        encoder_initial_hashes=hashes,pretrained={n:getattr(encoder,n).audit for n in hashes},
        augmentation_seed='42+epoch*100003',primary_selection='maximum validation mean Dice; earliest exact tie',
        secondary_selection='minimum sample-weighted validation loss; earliest exact tie',fixed_selection=f'epoch{budget}',
        split='ISIC2018 legacy 1886/808 development validation',frozen_encoders=True,model_spec=SPEC,
        initial_probe_hashes=preflight['initial_probe_hashes'],parameters=preflight['parameters'])
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:save_json(out/'config.json',config)
    data=ImageData(seed)
    assert data.audit==json.loads((ROOT.parent/'stage1_visual/data_manifest.json').read_text())['members']
    selectors={n:dict(best_loss=dict(epoch=0,score=float('inf')),best_dice=dict(epoch=0,score=-float('inf'))) for n in NAMES}
    history=[];start=1;resumed=False
    if (out/'latest.pth').exists():
        ckpt=torch.load(out/'latest.pth',map_location='cpu');assert ckpt['config']==config
        for n in NAMES:
            probes[n].load_state_dict(ckpt['probes'][n]);optimizers[n].load_state_dict(ckpt['optimizers'][n]);schedulers[n].load_state_dict(ckpt['schedulers'][n])
        torch.set_rng_state(ckpt['rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        history=ckpt['history'];selectors=ckpt['selectors'];start=ckpt['epoch']+1;resumed=True
        del ckpt
    print('START',json.dumps(dict(seed=seed,start_epoch=start,resumed=resumed,models=NAMES)),flush=True)
    val_features,val_targets=encode(encoder,data,'val',64)
    checked={n:False for n in NAMES}
    # An interrupted final export may resume after the last training epoch.
    # Verify gradients without an optimizer update before re-exporting in that case.
    if start>budget:
        for n in NAMES:
            probes[n].train();optimizers[n].zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=probes[n](select_features(n,[f[:2] for f in val_features]))
            loss=segmentation_loss(logits,val_targets[:2]);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probes[n].parameters()),n
            checked[n]=True;optimizers[n].zero_grad(set_to_none=True)
        del logits,loss
    for epoch in range(start,budget+1):
        t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        features,targets=encode(encoder,data,'train',64,epoch)
        losses={n:0. for n in NAMES};lrs={n:optimizers[n].param_groups[0]['lr'] for n in NAMES}
        for p in probes.values():p.train()
        for offset in range(0,len(targets),64):
            batch=[f[offset:offset+64] for f in features];target=targets[offset:offset+64]
            for n in NAMES:
                optimizers[n].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=probes[n](select_features(n,batch))
                loss=segmentation_loss(logits,target);assert torch.isfinite(loss)
                loss.backward()
                if not checked[n]:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probes[n].parameters()),n
                    checked[n]=True
                norm=torch.nn.utils.clip_grad_norm_(probes[n].parameters(),1.);assert torch.isfinite(norm)
                optimizers[n].step();losses[n]+=loss.item()*len(logits)
        for scheduler in schedulers.values():scheduler.step()
        vals={n:evaluate(probes[n],select_features(n,val_features),val_targets)[0] for n in NAMES}
        del features,targets,batch,target,logits,loss
        for n in NAMES:
            destination=out/n;destination.mkdir(exist_ok=True)
            for criterion,score in [('best_loss',vals[n]['loss']),('best_dice',vals[n]['macro']['dice'])]:
                improved=score<selectors[n][criterion]['score'] if criterion=='best_loss' else score>selectors[n][criterion]['score']
                if improved:
                    selectors[n][criterion]=dict(epoch=epoch,score=score)
                    save_checkpoint(destination/f'{criterion}.pth',dict(probe=probes[n].state_dict(),epoch=epoch,metrics=vals[n],criterion=criterion))
        if epoch==1 or epoch==budget or epoch%25==0:
            assert not encoder.training and all(p.grad is None for p in encoder.parameters())
            assert all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items())
        torch.cuda.synchronize()
        row=dict(epoch=epoch,models={n:dict(train_loss=losses[n]/len(data.ids['train']),val=vals[n],lr=lrs[n],
            best_loss_epoch=selectors[n]['best_loss']['epoch'],best_dice_epoch=selectors[n]['best_dice']['epoch']) for n in NAMES},
            seconds=time.perf_counter()-t,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        history.append(row)
        save_checkpoint(out/'latest.pth',dict(probes={n:probes[n].state_dict() for n in NAMES},
            optimizers={n:optimizers[n].state_dict() for n in NAMES},schedulers={n:schedulers[n].state_dict() for n in NAMES},
            epoch=epoch,history=history,selectors=selectors,config=config,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        save_json(out/'history.json',history);save_json(out/'status.json',row)
        print('EPOCH',json.dumps(dict(epoch=epoch,seconds=row['seconds'],peak_gpu_gib=row['peak_gpu_gib'],
            macro_dice={n:vals[n]['macro']['dice'] for n in NAMES})),flush=True)
        if args.stop_after_epoch==epoch:print('SMOKE_STOP_FOR_RESUME',epoch,flush=True);return
    exports={}
    for n in NAMES:
        save_checkpoint(out/n/'last.pth',dict(probe=probes[n].state_dict(),epoch=budget,metrics=history[-1]['models'][n]['val'],criterion='last'))
        exports[n]={}
        for criterion in CRITERIA:
            ckpt=torch.load(out/n/f'{criterion}.pth',map_location='cpu');probes[n].load_state_dict(ckpt['probe'])
            final,probabilities,per_image=evaluate(probes[n],select_features(n,val_features),val_targets,export=True)
            assert final==ckpt['metrics'],(n,criterion,'Export differs from selection metrics')
            destination=out/n/criterion;destination.mkdir(exist_ok=True)
            for row,sid in zip(per_image,data.ids['val']):row['id']=sid
            np.save(destination/'probabilities.npy',probabilities)
            np.save(destination/'masks.npy',(probabilities>=.5).astype(np.uint8))
            save_json(destination/'per_image.json',per_image);save_json(destination/'ids.json',data.ids['val'])
            record=dict(seed=seed,model=n,criterion=criterion,selected_epoch=ckpt['epoch'],final=final)
            save_json(destination/'metrics.json',record);exports[n][criterion]=record
            print('EXPORTED',n,criterion,flush=True)
    unchanged=all(tensor_hash(getattr(encoder,n))==h for n,h in hashes.items());assert unchanged
    save_json(out/'DONE.json',dict(seed=seed,epochs=budget,optimizer_updates_per_model=budget*30,
        smoke_only=args.smoke,resumed=resumed,encoders_unchanged=unchanged,gradient_checks_passed=all(checked.values()),exports=exports))
    print('DONE',seed,flush=True)


if __name__=='__main__':main()

