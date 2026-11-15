"""Stage44: allow early CNN weight adaptation; keep decoder and late update recipe."""
from pathlib import Path
import argparse, csv, gc, hashlib, json, os, signal, sys, time, traceback, math
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
ARMS = ('early_adapt',)
NAMES = ARMS
dump, save, cpu_tree, digest = base.dump, base.save, base.cpu_tree, base.digest
STOP = False

def setup():
    base.setup()
    # Keep deterministic alternatives where available; retain the author's CUDA
    # adaptive max pool instead of changing its math solely for determinism.
    torch.use_deterministic_algorithms(True,warn_only=True)


class System(base.System):
    def __init__(self,arm='early_adapt'):
        super().__init__(finetune=True,stochastic_depth=False)
        assert arm in ('early_adapt','swin_control')
        self.arm=arm;self.early_adapt=arm=='early_adapt'
        self.probe.decoder=AuthorModuleDecoder('swin_control',self.probe.decoder,seed=42)
        if self.early_adapt:
            self.encoder.model.patch_embed.requires_grad_(True)
            for level in self.encoder.model.levels[:2]:level.requires_grad_(True)
        self.train()
    def prefix(self,images):
        # Unlike base.prefix, gradients are allowed through learned early stages.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.early_adapt):
            m=self.encoder.model
            x=m.patch_embed(images);x,f1=m.levels[0](x);x,f2=m.levels[1](x)
        return f1.contiguous(),f2.contiguous(),x.contiguous()
    def from_cache(self,cache,indices):
        return self(cache['images'][indices])
    def frozen_digest(self):
        params=dict(self.named_parameters());h=hashlib.sha256()
        for key,value in self.state_dict().items():
            if key in params and params[key].requires_grad:continue
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

def optimizer(model):
    groups=[dict(params=list(model.probe.parameters()),lr=.001,initial_lr=.001,name='decoder'),
        dict(params=[p for level in model.encoder.model.levels[2:] for p in level.parameters() if p.requires_grad],
             lr=.0001,initial_lr=.0001,name='encoder')]
    if model.early_adapt:
        early=list(model.encoder.model.patch_embed.parameters())+[p for level in model.encoder.model.levels[:2] for p in level.parameters()]
        groups.append(dict(params=early,lr=.00001,initial_lr=.00001,name='early_encoder'))
    used=[id(p) for g in groups for p in g['params']]
    assert len(used)==len(set(used)) and set(used)=={id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.AdamW(groups,weight_decay=.01,betas=(.9,.999),eps=1e-8)



def objective(logits,target):
    assert isinstance(logits,torch.Tensor)
    return base.segmentation_loss(logits,target)
def update(model,opt,cache,ids,micro,check=False):
    opt.zero_grad(set_to_none=True)
    loss_value=0.
    for begin in range(0,len(ids),micro):
        chunk=ids[begin:begin+micro]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model.from_cache(cache,chunk)
        loss=objective(logits,cache['target'][chunk])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        (loss*(len(chunk)/len(ids))).backward()
        loss_value+=float(loss.detach())*len(chunk)/len(ids)
    norms={}
    for group in opt.param_groups:
        norm=torch.nn.utils.clip_grad_norm_(group['params'],1.,error_if_nonfinite=True)
        norms[group['name']]=float(norm)
    if check:
        assert norms['decoder']>0
        if model.finetune:
            assert norms['encoder']>0
            for i in (2,3):
                grads=[p.grad for p in model.encoder.model.levels[i].parameters() if p.requires_grad]
                assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        if model.early_adapt:
            assert norms['early_encoder']>0
            for part in [model.encoder.model.patch_embed,*model.encoder.model.levels[:2]]:
                grads=[p.grad for p in part.parameters() if p.requires_grad]
                assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
                assert sum(float(g.abs().sum()) for g in grads)>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    opt.step()
    return loss_value,norms


def cache_new(n):
    return dict(images=torch.empty(n,3,256,256,dtype=torch.float32,device='cuda'),
        target=torch.empty(n,1,256,256,dtype=torch.float32,device='cuda'))


@torch.no_grad()
def fill_cache(model,data,split,cache,matrices=None):
    for begin in range(0,len(data.ids[split]),64):
        end=min(begin+64,len(data.ids[split]));ids=torch.arange(begin,end,device='cuda')
        images,gt=data.batch(split,ids,matrices)
        assert images.dtype==cache['images'].dtype and gt.dtype==cache['target'].dtype
        cache['images'][begin:end].copy_(images);cache['target'][begin:end].copy_(gt)


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
    opts={n:optimizer(m) for n,m in models.items()}
    torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter()
    for n,m in models.items():
        for _ in range(2):
            update(m,opts[n],cache,torch.arange(64,device='cuda'),64,check=True)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/base.GIB
    total=torch.cuda.get_device_properties(0).total_memory/base.GIB
    assert peak<total-.7,(peak,total)
    for n,m in models.items():m.load_state_dict(initial[n]);m.zero_grad(set_to_none=True)
    del opts;gc.collect();torch.cuda.empty_cache();setup()
    dump(out/'capacity.json',dict(micro_batch=64,effective_batch=64,evaluation_batch=64,
         peak_gib=peak,total_gib=total,seconds=time.perf_counter()-tick,
         reference_micro_batch_matched=True,matched_within_both_new_and_old_arms=True,reserve_gib=.7))
    return 64

def preflight(models,cache,data,val_cache,micro,out):
    initial={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    reference=System('swin_control').cuda().eval()
    ids=torch.arange(64,device='cuda');images,target=data.batch('train',ids)
    checks={}
    for name,model in models.items():
        base.assert_tree_exact(model.state_dict(),reference.state_dict())
        assert torch.equal(cache['images'][ids],images) and torch.equal(cache['target'][ids],target)
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            a=model(images);b=reference(images);c=model.from_cache(cache,ids)
            assert torch.equal(a,b) and torch.equal(a,c)
        model.train()
        assert all(not module.training for module in model.encoder.modules())
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f1,f2,mid=model.prefix(images[:2])
        assert f1.requires_grad and f2.requires_grad and mid.requires_grad
        del f1,f2,mid,a,b,c
        opt=optimizer(model);frozen=model.frozen_digest()
        for _ in range(3):loss,norms=update(model,opt,cache,ids,micro,check=True)
        assert model.frozen_digest()==frozen
        changed={}
        for prefix in ['encoder.model.patch_embed.','encoder.model.levels.0.','encoder.model.levels.1.']:
            changed[prefix]=any(not torch.equal(v.detach().cpu(),initial[name][k]) for k,v in model.state_dict().items() if k.startswith(prefix))
            assert changed[prefix]
        state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=base.rng())
        expected_loss,_=update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        expected=cpu_tree(model.state_dict())
        model.load_state_dict(state['model']);opt.load_state_dict(cpu_tree(state['optimizer']));base.set_rng(state['rng'])
        base.assert_tree_exact(model.state_dict(),state['model']);base.assert_tree_exact(opt.state_dict(),state['optimizer'])
        replay_loss,_=update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        assert expected_loss==replay_loss,(expected_loss,replay_loss)
        replay={}
        for group,prefixes in [
            ('early_encoder',('encoder.model.patch_embed.','encoder.model.levels.0.','encoder.model.levels.1.')),
            ('encoder',('encoder.model.levels.2.','encoder.model.levels.3.')),
            ('decoder',('probe.',))]:
            d2=u2=max_abs=0.
            for k,v in model.state_dict().items():
                if not k.startswith(prefixes):continue
                actual=v.detach().cpu();delta=(actual-expected[k]).double()
                d2+=float(delta.square().sum());u2+=float((expected[k]-state['model'][k]).double().square().sum())
                max_abs=max(max_abs,float(delta.abs().max()))
            rel=(d2/max(u2,1e-30))**.5
            assert rel<=.01 and max_abs<=.0001,(group,rel,max_abs)
            replay[group]=dict(relative_l2=rel,max_abs=max_abs)
        assert model.frozen_digest()==frozen
        checks[name]=dict(initial_model_state_exact=True,initial_forward_exact=True,
            normalized_FP32_input_cache_exact=True,early_features_require_grad=True,
            early_parameter_changes=changed,gradient_norms=norms,loss=loss,
            frozen_parameters_and_all_buffers_unchanged=True,replay_loss_exact=True,replay_update=replay)
        del opt,state,expected
        model.load_state_dict(initial[name]);model.zero_grad(set_to_none=True)
    del reference,images,target,initial;gc.collect();torch.cuda.empty_cache();setup()
    dump(out/'preflight.json',dict(passed=True,checks=checks))



def source_hashes():
    files=[ROOT/name for name in ['train.py','author_modules.py','evaluate.py','run_pipeline.py',
        'verify_integration.py','verify_reference.py','identity_partition.json','REFERENCE_MODELS.json','PROTOCOL.json','README_zh.txt']]
    files += [PROJECT/x for x in ['stage28_drop_path/train.py','evaluation.py','stage1_visual/data.py',
        'stage1_visual/models.py','stage2_decoder/models.py','stage3_confirmation/models.py','stage3_confirmation/data.py',
        'stage37_lr_horizon100/train.py','stage37_lr_horizon100/author_modules.py']]
    files += [base.HF/'config.json']+list(base.HF.glob('*.py'))
    return {os.path.relpath(p,PROJECT):digest(p) for p in files}



def lr_epoch(opt, epoch):
    factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
    for group in opt.param_groups: group['lr']=group['initial_lr']*factor


def main():
    global NAMES
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--stop-after-epoch',type=int);args=ap.parse_args();NAMES=(args.arm,)
    setup();out=ROOT/args.arm/('smoke' if args.smoke else 'results');out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'TRAINING_DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    assert digest(base.HF/'model.safetensors')==base.WEIGHT_SHA
    data,partition=load_data()
    for row in json.loads((base.HF.parent/'source_download.json').read_text())['files']:
        assert digest(base.HF/row['path'])==row['sha256']
    config=dict(seed=42,epochs=2 if args.smoke else 100,names=NAMES,
        effective_batch=64,evaluation_batch=64,encoder_lr=.0001,early_encoder_lr=.00001,decoder_lr=.001,weight_decay=.01,
        selection='max_validation_macro_dice_earliest_tie',precision='BF16_forward_FP32_loss',deterministic_policy='deterministic_requested_warn_only_for_unsupported_ops',
        mode='encoder_eval_all_used_stages_trainable_early_lr1e5_late1e4',sources=source_hashes(),
        intervention=json.loads((ROOT/'PROTOCOL.json').read_text())['changes'][args.arm],
        partition_counts=partition['counts'],weights_sha256=base.WEIGHT_SHA,
        criterion='single_output_BCE_plus_per_image_softDice',lr_schedule='cosine_100_floor0.01',
        test_exposure='867 already exposed; development only on validation; no fresh independent confirmation',
        evaluation_helper_sha256=digest(PROJECT.parent.parent/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py'),
        lr_schedule_horizon=100,reference_reuse='Same Stage37 baseline, physical64, identical architecture/init/input/loss/schedule; only early gradient and parameter group added')
    config=json.loads(json.dumps(config))
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    dump(out/'environment.json',base.environment());dump(out/'active_partition.json',partition)
    if not args.smoke:
        assert json.loads((ROOT/args.arm/'smoke/RESUME_AUDIT.json').read_text())['passed']
        assert json.loads((ROOT/args.arm/'smoke/config.json').read_text())['sources']==config['sources']
    models={n:System(n).cuda() for n in NAMES}
    frozen={n:m.frozen_digest() for n,m in models.items()}
    dump(out/'parameters.json',{n:dict(total=sum(p.numel() for p in m.parameters()),
          trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
          decoder=sum(p.numel() for p in m.probe.parameters()),
          encoder_trainable=sum(p.numel() for p in m.encoder.parameters() if p.requires_grad))
          for n,m in models.items()})
    train_cache=cache_new(len(data.ids['train']));val_cache=cache_new(len(data.ids['val']))
    print('BUILDING_FP32_INPUT_CACHE_NO_FEATURE_CACHE',flush=True)
    fill_cache(models[NAMES[0]],data,'train',train_cache)
    fill_cache(models[NAMES[0]],data,'val',val_cache)
    latest=out/'latest.pth'
    if args.smoke and not latest.exists():
        micro=calibrate(models,train_cache,out)
        preflight(models,train_cache,data,val_cache,micro,out)
    else:
        capacity=json.loads(((out if args.smoke else ROOT/args.arm/'smoke')/'capacity.json').read_text())
        micro=capacity['micro_batch'];dump(out/'capacity.json',capacity)
    opts={n:optimizer(m) for n,m in models.items()}
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
        for n in NAMES:models[n].train();lr_epoch(opts[n],epoch)
        losses={n:0. for n in NAMES};gradients={}
        torch.cuda.reset_peak_memory_stats()
        for begin in range(0,len(order),64):
            ids=order[begin:begin+64]
            for n in NAMES:
                loss,gradients[n]=update(models[n],opts[n],train_cache,ids,micro,check=begin==0)
                losses[n]+=loss*len(ids)
        record=dict(epoch=epoch,train_loss={n:losses[n]/len(order) for n in NAMES},
                    validation={},gradient_norms=gradients,learning_rates={n:{g['name']:g['lr'] for g in opts[n].param_groups} for n in NAMES})
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
            assert models[n].frozen_digest()==frozen[n]
        record.update(seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/base.GIB)
        history.append(record)
        save(latest,dict(epoch=epoch,models={n:cpu_tree(m.state_dict()) for n,m in models.items()},
             optimizers={n:cpu_tree(o.state_dict()) for n,o in opts.items()},
             rng=base.rng(),selectors=selectors,history=history,config=config,elapsed=elapsed+time.perf_counter()-started))
        dump(out/'history.json',history)
        status=dict(state='running',stage='stage44_early_cnn_adaptation',arm=args.arm,epoch=epoch,total_epochs=config['epochs'],
             updates_per_arm=epoch*30,micro_batch=micro,effective_batch=64,
             validation_macro_dice={n:record['validation'][n]['macro']['dice'] for n in NAMES},
             best=selectors,learning_rates=record['learning_rates'],seconds=time.perf_counter()-tick,peak_allocated_gib=record['peak_allocated_gib'],
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
