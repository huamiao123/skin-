"""Stage33: independently train original EMCAD and CGA adaptations."""
from pathlib import Path
import argparse, csv, gc, hashlib, json, os, signal, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
from stage28_drop_path import train as base
from author_modules import AuthorModuleDecoder
import numpy as np
import torch
from torch import nn
ARMS = ('emcad64','swin_cga')
NAMES = ARMS
dump, save, cpu_tree, digest = base.dump, base.save, base.cpu_tree, base.digest
STOP = False

class System(base.System):
    def __init__(self,arm='emcad64'):
        super().__init__(finetune=True,stochastic_depth=False)
        assert arm in ARMS
        self.arm=arm
        self.probe.decoder=AuthorModuleDecoder(arm,self.probe.decoder,seed=42)
        self.train()


def cache_new(n):
    return dict(features=[torch.empty((n,c,s,s),dtype=torch.bfloat16,device='cuda')
                          for c,s in [(96,64),(192,32)]],
                mid=torch.empty((n,384,16,16),dtype=torch.bfloat16,device='cuda'),
                target=torch.empty((n,1,256,256),dtype=torch.float32,device='cuda'))

@torch.no_grad()
def fill_cache(model, data, split, cache, matrices=None):
    for begin in range(0,len(data.ids[split]),64):
        end = min(begin+64,len(data.ids[split]))
        ids = torch.arange(begin,end,device='cuda')
        images,gt = data.batch(split,ids,matrices)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f1,f2,mid = model.prefix(images)
        cache['features'][0][begin:end].copy_(f1)
        cache['features'][1][begin:end].copy_(f2)
        cache['mid'][begin:end].copy_(mid)
        cache['target'][begin:end].copy_(gt)

def load_data():
    data = base.ImageData(42)
    historical = json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    assert data.audit == historical
    partition = json.loads((ROOT/'identity_partition.json').read_text())
    assert partition['splits']['train'] == data.ids['train']
    selected = partition['splits']['val']
    positions = {sid:i for i,sid in enumerate(data.ids['val'])}
    ix = torch.tensor([positions[sid] for sid in selected],device='cuda')
    for d in (data.images,data.masks):
        d['val'] = d['val'][ix].contiguous()
    data.ids['val'] = selected
    assert len(selected)==762 and len(data.ids['train'])==1886
    return data,partition

def calibrate(models, cache, out):
    # Match the completed reference runs; no batch-size search based on scores.
    initial={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    opts={n:base.optimizer(m) for n,m in models.items()}
    torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    for n,m in models.items():
        for _ in range(2):
            base.update(m,opts[n],cache,torch.arange(64,device='cuda'),48,check=True)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/base.GIB
    total=torch.cuda.get_device_properties(0).total_memory/base.GIB
    assert peak<total-.7,(peak,total)
    for n,m in models.items():m.load_state_dict(initial[n]);m.zero_grad(set_to_none=True)
    del opts;gc.collect();torch.cuda.empty_cache();base.setup()
    dump(out/'capacity.json',dict(micro_batch=48,effective_batch=64,evaluation_batch=64,
         peak_gib=peak,total_gib=total,seconds=time.perf_counter()-tick,
         reference_micro_batch_matched=True,reserve_gib=.7))
    return 48

def preflight(models,cache,data,val_cache,micro,out):
    initial={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    reference=base.System(finetune=True,stochastic_depth=False).cuda()
    ids=torch.arange(8,device='cuda');images,_=data.batch('train',ids)
    reference.eval()
    shared={}
    for name,m in models.items():
        m.eval()
        assert all(torch.equal(v,m.encoder.state_dict()[k]) for k,v in reference.encoder.state_dict().items())
        assert all(torch.equal(v,m.probe.adapter.state_dict()[k]) for k,v in reference.probe.adapter.state_dict().items())
        if name=='swin_cga':
            base.assert_tree_exact(m.probe.decoder.baseline.state_dict(),reference.probe.decoder.state_dict())
        else:
            base.assert_tree_exact(m.probe.decoder.readout.state_dict(),reference.probe.decoder.readout.state_dict())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            _,official=m.encoder(images)
            f1,f2,mid=m.prefix(images);f3,f4=m.suffix(mid)
            assert all(torch.equal(a,b) for a,b in zip(official,[f1,f2,f3,f4]))
            prediction=m(images)
            assert prediction.shape==(8,1,256,256) and torch.isfinite(prediction).all()
        shared[name]=dict(encoder_and_adapters_match=True,shared_swin_or_readout_match=True,official_encoder_features_identical=True)
    del reference,images,prediction,official,f1,f2,mid,f3,f4
    gc.collect();torch.cuda.empty_cache()
    checks={}
    for name,model in models.items():
        model.train();opt=base.optimizer(model);frozen=base.frozen_hash(model)
        assert all(not mod.training for mod in model.encoder.modules())
        for _ in range(3):
            loss,norms=base.update(model,opt,cache,torch.arange(64,device='cuda'),micro,check=True)
        assert base.frozen_hash(model)==frozen
        grads={k:float(p.grad.abs().sum()) for k,p in model.probe.decoder.named_parameters() if p.requires_grad}
        assert len(grads)==sum(p.requires_grad for p in model.probe.decoder.parameters())
        assert all(np.isfinite(v) for v in grads.values())
        if name=='swin_cga':
            for i in (2,1,0):
                for group in ['mamba_g','mamba_x','combine']:
                    assert sum(v for k,v in grads.items() if k.startswith(f'author_gates.{i}.{group}'))>0
        else:assert sum(v for k,v in grads.items() if k.startswith('author_decoder.'))>0
        state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=base.rng())
        expected_loss,_=base.update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        expected=cpu_tree(model.state_dict())
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);base.set_rng(state['rng'])
        base.assert_tree_exact(model.state_dict(),state['model']);base.assert_tree_exact(opt.state_dict(),state['optimizer'])
        replay_loss,_=base.update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        assert expected_loss==replay_loss,(expected_loss,replay_loss)
        replay={}
        for group,prefix in [('encoder','encoder.'),('decoder','probe.')]:
            d2=u2=max_abs=0.
            for k,v in model.state_dict().items():
                if not k.startswith(prefix):continue
                actual=v.detach().cpu();delta=(actual-expected[k]).double()
                d2+=float(delta.square().sum());u2+=float((expected[k]-state['model'][k]).double().square().sum())
                max_abs=max(max_abs,float(delta.abs().max()))
            rel=(d2/max(u2,1e-30))**.5
            assert rel<=.01 and max_abs<=.0001,(name,group,rel,max_abs)
            replay[group]=dict(relative_l2=rel,max_abs=max_abs)
        checks[name]=dict(shared=shared[name],real_data_loss=loss,gradient_norms=norms,author_gradient_l1=grads,
                         frozen_unchanged=True,restored_state_exact=True,replay_loss_exact=True,replay_update=replay,
                         note='Mamba CUDA backward may not be bitwise deterministic; <=1% update-L2 and1e-4 max_abs tolerance.')
        del opt,state,expected;model.zero_grad(set_to_none=True)
    for n,m in models.items():m.load_state_dict(initial[n])
    base.setup();dump(out/'preflight.json',dict(passed=True,checks=checks))


def source_hashes():
    research=PROJECT.parent.parent/'research'
    files=[ROOT/'train.py',ROOT/'author_modules.py',ROOT/'evaluate.py',ROOT/'run_pipeline.py',ROOT/'PROTOCOL.json',
           ROOT/'identity_partition.json',ROOT/'REFERENCE_MODELS.json',ROOT/'verify_reference.py',
           PROJECT/'stage28_drop_path/train.py',PROJECT/'evaluation.py',
           PROJECT/'stage1_visual/data.py',PROJECT/'stage1_visual/models.py',
           PROJECT/'stage2_decoder/models.py',PROJECT/'stage3_confirmation/models.py',
           PROJECT/'stage3_confirmation/data.py',base.HF/'config.json',
           research/'module_design_20261111/official_code/SLDGroup__EMCAD/lib/decoders.py',
           research/'topconf_open_source_20261111/official_code/maklachur__MambaLiteUNet/models/MambaLiteUNet.py']+list(base.HF.glob('*.py'))
    return {os.path.relpath(p,PROJECT):digest(p) for p in files}


def main():
    global NAMES
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--stop-after-epoch',type=int);args=ap.parse_args();NAMES=(args.arm,)
    base.setup();out=ROOT/args.arm/('smoke' if args.smoke else 'results');out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'TRAINING_DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    assert digest(base.HF/'model.safetensors')==base.WEIGHT_SHA
    data,partition=load_data()
    for row in json.loads((base.HF.parent/'source_download.json').read_text())['files']:
        assert digest(base.HF/row['path'])==row['sha256']
    config=dict(seed=42,epochs=2 if args.smoke else 100,names=NAMES,
        effective_batch=64,evaluation_batch=64,encoder_lr=.0001,decoder_lr=.001,weight_decay=.01,
        selection='max_validation_macro_dice_earliest_tie',precision='BF16_forward_FP32_loss',
        mode='encoder_eval_with_late_parameters_trainable_all_arms',sources=source_hashes(),
        partition_counts=partition['counts'],weights_sha256=base.WEIGHT_SHA,
        criterion='BCE_plus_per_image_soft_Dice_unchanged',lr_schedule='cosine_300_floor0.01',
        test_exposure='867 already exposed; development only on validation; no fresh independent confirmation',
        evaluation_helper_sha256=digest(PROJECT.parent.parent/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py'),
        lr_schedule_horizon=300,reference_reuse='Stage29 control/local first100 schedule and batch matched; validation replay required before formal training')
    config=json.loads(json.dumps(config))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    dump(out/'environment.json',base.environment());dump(out/'active_partition.json',partition)
    if not args.smoke:
        assert json.loads((ROOT/args.arm/'smoke/RESUME_AUDIT.json').read_text())['passed']
        assert json.loads((ROOT/args.arm/'smoke/config.json').read_text())['sources']==config['sources']
    models={n:System(n).cuda() for n in NAMES}
    frozen={n:base.frozen_hash(m) for n,m in models.items()}
    dump(out/'parameters.json',{n:dict(total=sum(p.numel() for p in m.parameters()),
          trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
          decoder=sum(p.numel() for p in m.probe.parameters()),
          encoder_trainable=sum(p.numel() for p in m.encoder.parameters() if p.requires_grad))
          for n,m in models.items()})
    train_cache=cache_new(len(data.ids['train']));val_cache=cache_new(len(data.ids['val']))
    print('BUILDING_FROZEN_PREFIX_CACHE',flush=True)
    fill_cache(models[NAMES[0]],data,'train',train_cache)
    fill_cache(models[NAMES[0]],data,'val',val_cache)
    latest=out/'latest.pth'
    if args.smoke and not latest.exists():
        micro=calibrate(models,train_cache,out)
        preflight(models,train_cache,data,val_cache,micro,out)
    else:
        capacity=json.loads(((out if args.smoke else ROOT/args.arm/'smoke')/'capacity.json').read_text())
        micro=capacity['micro_batch'];dump(out/'capacity.json',capacity)
    opts={n:base.optimizer(m) for n,m in models.items()}
    selectors={n:dict(best_dice=-1.,best_loss=1e9) for n in NAMES}
    history=[];start=1;elapsed=0.
    if latest.exists():
        state=torch.load(latest,map_location='cpu');assert state['config']==config
        for n in NAMES:
            models[n].load_state_dict(state['models'][n]);opts[n].load_state_dict(state['optimizers'][n])
            base.assert_tree_exact(models[n].state_dict(),state['models'][n])
            base.assert_tree_exact(opts[n].state_dict(),state['optimizers'][n])
        selectors=state['selectors'];history=state['history'];start=state['epoch']+1
        elapsed=state['elapsed'];base.set_rng(state['rng']);base.assert_tree_exact(base.rng(),state['rng'])
        dump(out/'RESUME_LOAD_AUDIT.json',dict(passed=True,checkpoint_epoch=state['epoch'],
             start_epoch=start,models_optimizers_rng_exact=True,history_rows=len(history)))
        del state;print('RESUMED',start,flush=True)
    def handler(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    started=time.perf_counter()
    for epoch in range(start,config['epochs']+1):
        tick=time.perf_counter();order,matrices=data.geometry(epoch)
        fill_cache(models[NAMES[0]],data,'train',train_cache,matrices)
        for n in NAMES:models[n].train();base.lr_epoch(opts[n],epoch)
        losses={n:0. for n in NAMES};gradients={}
        torch.cuda.reset_peak_memory_stats()
        for begin in range(0,len(order),64):
            ids=order[begin:begin+64]
            for n in NAMES:
                loss,gradients[n]=base.update(models[n],opts[n],train_cache,ids,micro,check=begin==0)
                losses[n]+=loss*len(ids)
        record=dict(epoch=epoch,train_loss={n:losses[n]/len(order) for n in NAMES},
                    validation={},gradient_norms=gradients)
        for n in NAMES:
            metric,_=base.evaluate(models[n],val_cache,64)
            record['validation'][n]=metric
            for criterion,value,improved in [
                ('best_dice',metric['macro']['dice'],metric['macro']['dice']>selectors[n]['best_dice']),
                ('best_loss',metric['loss'],metric['loss']<selectors[n]['best_loss'])]:
                if improved:
                    selectors[n][criterion]=value
                    if criterion=='best_dice':
                        save(out/f'{n}_{criterion}.pth',dict(epoch=epoch,model=cpu_tree(models[n].state_dict()),metric=metric,config=config))
            assert base.frozen_hash(models[n])==frozen[n]
        record.update(seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/base.GIB)
        history.append(record)
        save(latest,dict(epoch=epoch,models={n:cpu_tree(m.state_dict()) for n,m in models.items()},
             optimizers={n:cpu_tree(o.state_dict()) for n,o in opts.items()},
             rng=base.rng(),selectors=selectors,history=history,config=config,elapsed=elapsed+time.perf_counter()-started))
        dump(out/'history.json',history)
        status=dict(state='running',stage='stage33_author_modules',arm=args.arm,epoch=epoch,total_epochs=config['epochs'],
             updates_per_arm=epoch*30,micro_batch=micro,effective_batch=64,
             validation_macro_dice={n:record['validation'][n]['macro']['dice'] for n in NAMES},
             best=selectors,seconds=time.perf_counter()-tick,peak_allocated_gib=record['peak_allocated_gib'],
             reserved_gib=torch.cuda.memory_reserved()/base.GIB,pid=os.getpid())
        dump(out/'run_status.json',status);print(json.dumps(status),flush=True)
        if STOP or (args.stop_after_epoch and epoch>=args.stop_after_epoch):
            status['state']='stopped_at_epoch_boundary';dump(out/'run_status.json',status);return
    dump(out/'TRAINING_DONE.json',dict(complete=True,epochs=config['epochs'],updates_per_arm=config['epochs']*30))
    dump(out/'run_status.json',dict(state='training_complete_awaiting_evaluation',epochs=config['epochs']))
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        arm=sys.argv[sys.argv.index('--arm')+1] if '--arm' in sys.argv else 'unknown'
        out=ROOT/arm/('smoke' if '--smoke' in sys.argv else 'results')
        dump(out/'FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid()))
        raise
