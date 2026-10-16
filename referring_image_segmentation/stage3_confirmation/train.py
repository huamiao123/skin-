"""Train paired Light/Swin probes with three predeclared checkpoint selectors."""
import argparse
import fcntl
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

from models import FrozenEncoder, Probe, tensor_hash, segmentation_loss
from data import ImageData
from engine import encode, evaluate

ROOT=Path(__file__).resolve().parent
DECODERS=('light','swin')
CRITERIA=('best_dice','best_loss','last')


def save_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def save_checkpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=(43,44,45),required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after-epoch',type=int)
    args=p.parse_args()
    if args.stop_after_epoch and not args.smoke: raise ValueError('Only smoke runs can stop early')
    out=ROOT/('smoke' if args.smoke else 'results')/f'seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():print('Already complete',flush=True);return
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    encoder=FrozenEncoder('dinov3').cuda()
    encoder_hash=tensor_hash(encoder)
    old=json.loads((ROOT.parent/'stage1_visual/benchmark_dinov3.json').read_text())
    assert encoder_hash==old['frozen_hash']
    probes={d:Probe(d,args.seed).cuda() for d in DECODERS}
    checked=json.loads((ROOT/'preflight.json').read_text())['initial_probe_hashes'][str(args.seed)]
    for d in DECODERS:assert tensor_hash(probes[d])==checked[d]
    assert tensor_hash(probes['light'].adapter)==tensor_hash(probes['swin'].adapter)
    assert tensor_hash(probes['light'].decoder.readout)==tensor_hash(probes['swin'].decoder.readout)
    assert set(map(id,probes['light'].parameters())).isdisjoint(set(map(id,probes['swin'].parameters())))
    optimizers={d:torch.optim.AdamW(probes[d].parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8) for d in DECODERS}
    schedulers={d:torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[d],T_max=300,eta_min=1e-5) for d in DECODERS}
    budget=2 if args.smoke else 300
    config=dict(seed=args.seed,epochs=budget,encoder='dinov3',input_size=256,batch=64,encoder_batch=64,
        evaluation_batch=64,optimizer='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        scheduler='CosineAnnealingLR',T_max=300,eta_min=1e-5,clip=1.,precision='BF16 autocast; FP32 loss',
        loss='BCEWithLogits + per-image soft Dice; smooth=1',frozen_encoder=True,
        encoder_initial_hash=encoder_hash,augmentation_seed='seed+epoch*100003',
        primary_selection='maximum validation mean Dice, earliest on exact ties',
        secondary_selection='minimum sample-weighted validation loss, earliest on exact ties',
        fixed_selection='last epoch',split='legacy ISIC2018 1886/808 development validation',
        initial={d:dict(adapter_hash=tensor_hash(probes[d].adapter),decoder_hash=tensor_hash(probes[d].decoder),
            trainable_params=sum(p.numel() for p in probes[d].parameters()),
            decoder_params=sum(p.numel() for p in probes[d].decoder.parameters()),
            total_params=encoder.audit['params']+sum(p.numel() for p in probes[d].parameters())) for d in DECODERS},
        pretrained=encoder.audit)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==config
    else: save_json(out/'config.json',config)
    data=ImageData(args.seed)
    previous=json.loads((ROOT.parent/'stage1_visual/data_manifest.json').read_text())
    assert data.audit==previous['members'],'Dataset changed'
    selectors={d:dict(best_loss=dict(epoch=0,score=float('inf')),best_dice=dict(epoch=0,score=-float('inf'))) for d in DECODERS}
    history=[];start=1;resumed=False
    if (out/'latest.pth').exists():
        ckpt=torch.load(out/'latest.pth',map_location='cpu')
        assert ckpt['config']==config
        for d in DECODERS:
            probes[d].load_state_dict(ckpt['probes'][d]);optimizers[d].load_state_dict(ckpt['optimizers'][d]);schedulers[d].load_state_dict(ckpt['schedulers'][d])
        torch.set_rng_state(ckpt['rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        history=ckpt['history'];selectors=ckpt['selectors'];start=ckpt['epoch']+1;resumed=True
    print('START',json.dumps(dict(config=config,start_epoch=start,resumed=resumed)),flush=True)
    val_features,val_targets=encode(encoder,data,'val',64)
    gradients_checked=False
    for epoch in range(start,budget+1):
        t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        features,targets=encode(encoder,data,'train',64,epoch)
        losses={d:0. for d in DECODERS};lrs={d:optimizers[d].param_groups[0]['lr'] for d in DECODERS}
        for probe in probes.values():probe.train()
        for offset in range(0,len(targets),64):
            batch=[f[offset:offset+64] for f in features];target=targets[offset:offset+64]
            for d in DECODERS:
                optimizers[d].zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=probes[d](batch)
                loss=segmentation_loss(logits,target)
                assert torch.isfinite(loss),'Nonfinite loss'
                loss.backward()
                if not gradients_checked:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probes[d].parameters())
                norm=torch.nn.utils.clip_grad_norm_(probes[d].parameters(),1.)
                assert torch.isfinite(norm),'Nonfinite gradient'
                optimizers[d].step();losses[d]+=loss.item()*len(logits)
            gradients_checked=True
        for scheduler in schedulers.values():scheduler.step()
        vals={d:evaluate(probes[d],val_features,val_targets)[0] for d in DECODERS}
        del features,targets,batch,target,logits,loss
        for d in DECODERS:
            model_out=out/d;model_out.mkdir(exist_ok=True)
            for criterion,score in [('best_loss',vals[d]['loss']),('best_dice',vals[d]['macro']['dice'])]:
                improved=score<selectors[d][criterion]['score'] if criterion=='best_loss' else score>selectors[d][criterion]['score']
                if improved:
                    selectors[d][criterion]=dict(epoch=epoch,score=score)
                    save_checkpoint(model_out/f'{criterion}.pth',dict(probe=probes[d].state_dict(),epoch=epoch,metrics=vals[d],criterion=criterion))
        if epoch==1 or epoch==budget or epoch%25==0:
            assert not encoder.training and all(p.grad is None for p in encoder.parameters())
            assert tensor_hash(encoder)==encoder_hash
        torch.cuda.synchronize()
        row=dict(epoch=epoch,models={d:dict(train_loss=losses[d]/len(data.ids['train']),val=vals[d],lr=lrs[d],
             best_loss_epoch=selectors[d]['best_loss']['epoch'],best_dice_epoch=selectors[d]['best_dice']['epoch']) for d in DECODERS},
             seconds=time.perf_counter()-t,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        history.append(row)
        save_checkpoint(out/'latest.pth',dict(probes={d:probes[d].state_dict() for d in DECODERS},
            optimizers={d:optimizers[d].state_dict() for d in DECODERS},schedulers={d:schedulers[d].state_dict() for d in DECODERS},
            epoch=epoch,history=history,selectors=selectors,config=config,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        save_json(out/'history.json',history);save_json(out/'status.json',row)
        print('EPOCH',json.dumps(row),flush=True)
        if args.stop_after_epoch==epoch:
            print('SMOKE_STOP_FOR_RESUME',epoch,flush=True);return
    exports={}
    for d in DECODERS:
        save_checkpoint(out/d/'last.pth',dict(probe=probes[d].state_dict(),epoch=budget,
            metrics=history[-1]['models'][d]['val'],criterion='last'))
        exports[d]={}
        for criterion in CRITERIA:
            ckpt=torch.load(out/d/f'{criterion}.pth',map_location='cpu')
            probes[d].load_state_dict(ckpt['probe'])
            final,probabilities,per_image=evaluate(probes[d],val_features,val_targets,export=True)
            assert final==ckpt['metrics'],(d,criterion,'Checkpoint export differs from selection metrics')
            destination=out/d/criterion;destination.mkdir(exist_ok=True)
            for row,sid in zip(per_image,data.ids['val']):row['id']=sid
            if not args.smoke:
                np.save(destination/'probabilities.npy',probabilities)
                np.save(destination/'masks.npy',(probabilities>=.5).astype(np.uint8))
                save_json(destination/'per_image.json',per_image);save_json(destination/'ids.json',data.ids['val'])
            record=dict(seed=args.seed,decoder=d,criterion=criterion,selected_epoch=ckpt['epoch'],final=final)
            save_json(destination/'metrics.json',record);exports[d][criterion]=record
    unchanged=tensor_hash(encoder)==encoder_hash
    assert unchanged
    save_json(out/'DONE.json',dict(seed=args.seed,epochs=budget,optimizer_updates_per_decoder=budget*30,
        smoke_only=args.smoke,resumed=resumed,encoder_unchanged=unchanged,gradient_checks_passed=gradients_checked,
        exports=exports))
    print('DONE',args.seed,flush=True)


if __name__=='__main__':main()
