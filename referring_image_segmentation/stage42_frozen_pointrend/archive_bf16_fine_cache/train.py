"""Stage42: frozen selected baseline; native PointRend MLP only."""
from pathlib import Path
import argparse, csv, gc, hashlib, json, os, signal, sys, time, traceback, math
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
from stage28_drop_path import train as base
from author_modules import PointRendDecoder, SwinCoarseAdapter, make_scope
from torch.nn import functional as F
import numpy as np
import torch
from torch import nn
ARMS = ('frozen_point',)
NAMES = ARMS
dump, save, cpu_tree, digest = base.dump, base.save, base.cpu_tree, base.digest
STOP = False

def setup():
    base.setup()
    # Keep deterministic alternatives where available; retain the author's CUDA
    # adaptive max pool instead of changing its math solely for determinism.
    torch.use_deterministic_algorithms(True,warn_only=True)


class FlatPointCrossEntropy:
    """Same per-point mean CE; flatten N,P to avoid the 3-D NLL CUDA reduction path."""
    def __getattr__(self,name):return getattr(F,name)
    def cross_entropy(self,input,target,**kwargs):
        if input.ndim==3:
            n,c,p=input.shape
            return F.cross_entropy(input.permute(0,2,1).reshape(n*p,c),target.reshape(n*p),**kwargs)
        return F.cross_entropy(input,target,**kwargs)

class CachedSwinCoarse(SwinCoarseAdapter):
    def __init__(self,baseline):
        nn.Module.__init__(self);self.baseline=baseline
    def layers(self,features):
        if '_coarse' in features:return features['_coarse']
        return super().layers(features)

class System(base.System):
    def __init__(self,arm='frozen_point'):
        super().__init__(finetune=False,stochastic_depth=False)
        assert arm in ARMS;self.arm=arm
        ref=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]
        path=Path(ref['checkpoint']);assert digest(path)==ref['sha256']
        ck=torch.load(path,map_location='cpu')
        original={k.replace('probe.decoder.baseline.','probe.decoder.'):v for k,v in ck['model'].items()}
        self.load_state_dict(original,strict=True);del ck,original
        self.probe.decoder=PointRendDecoder('point_uncertain',self.probe.decoder,seed=42)
        h=self.probe.decoder.author_head
        h.forward.__globals__['F']=FlatPointCrossEntropy()
        h.coarse_sem_seg_head=CachedSwinCoarse(h.coarse_sem_seg_head.baseline)
        self.requires_grad_(False);self.point_head.requires_grad_(True)
        self.inference_policy='uncertain'
        self.train()
    @property
    def point_head(self):return self.probe.decoder.author_head.point_head
    def train(self,mode=True):
        nn.Module.train(self,mode)
        self.encoder.eval()
        if hasattr(self,'probe'):
            self.probe.adapter.eval()
            decoder=self.probe.decoder
            (decoder.baseline if hasattr(decoder,'baseline') else decoder).eval()
        return self
    def frozen_digest(self):
        h=hashlib.sha256()
        for key,value in self.state_dict().items():
            if key.startswith('probe.decoder.author_head.point_head.'):continue
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def point_state(self):return cpu_tree(self.point_head.state_dict())
    def load_point_state(self,state):self.point_head.load_state_dict(state,strict=True)
    @torch.no_grad()
    def frozen_features(self,images):
        f1,f2,mid=self.prefix(images);f3,f4=self.suffix(mid)
        features={f'p{i}':a(f) for i,(a,f) in enumerate(zip(self.probe.adapter,[f1,f2,f3,f4]))}
        coarse=self.probe.decoder.author_head.coarse_sem_seg_head.layers(features)
        return features['p0'].contiguous(),coarse.contiguous()
    def _head(self,fine,coarse,target=None):
        h=self.probe.decoder.author_head
        features={'p0':fine,'_coarse':coarse}
        if not self.training and self.inference_policy in ('coarse_direct','coarse_stepwise'):
            if self.inference_policy=='coarse_direct':
                return F.interpolate(coarse[:,1:2]-coarse[:,0:1],size=(256,256),mode='bilinear',align_corners=False)
            x=F.interpolate(F.interpolate(coarse,scale_factor=2,mode='bilinear',align_corners=False),scale_factor=2,mode='bilinear',align_corners=False)
            return x[:,1:2]-x[:,0:1]
        if not self.training and self.inference_policy=='random':
            scope=h.forward.__globals__;old_selector=scope['get_uncertain_point_coords_on_grid']
            rd,_=make_scope(True)
            try:
                scope['get_uncertain_point_coords_on_grid']=rd['get_uncertain_point_coords_on_grid']
                logits,losses=h(features,None)
            finally:scope['get_uncertain_point_coords_on_grid']=old_selector
        else:
            logits,losses=h(features,target[:,0].long() if target is not None else None)
        if self.training:return losses
        return logits[:,1:2]-logits[:,0:1]
    def from_cache(self,cache,indices):
        return self._head(cache['fine'][indices],cache['coarse'][indices],cache['target'][indices] if self.training else None)
    def forward(self,images):
        assert not self.training,'Training uses augmented frozen feature cache and target'
        fine,coarse=self.frozen_features(images)
        return self._head(fine,coarse)

def optimizer(model):
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in trainable}=={id(p) for p in model.point_head.parameters()}
    return torch.optim.AdamW([dict(params=trainable,lr=.001,initial_lr=.001,name='decoder')],
                            weight_decay=.01,betas=(.9,.999),eps=1e-8)




def objective(logits,target):
    if isinstance(logits,dict):
        assert set(logits)=={'loss_sem_seg_coarse','loss_sem_seg_point'}
        assert not logits['loss_sem_seg_coarse'].requires_grad
        return logits['loss_sem_seg_point']
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
        assert all(p.grad is None for p in model.encoder.parameters() if not p.requires_grad)
    opt.step()
    return loss_value,norms


def cache_new(n):
    return dict(fine=torch.empty(n,64,64,64,dtype=torch.bfloat16,device='cuda'),
        coarse=torch.empty(n,2,64,64,dtype=torch.bfloat16,device='cuda'),
        target=torch.empty(n,1,256,256,dtype=torch.float32,device='cuda'))


@torch.no_grad()
def fill_cache(model,data,split,cache,matrices=None):
    for begin in range(0,len(data.ids[split]),64):
        end=min(begin+64,len(data.ids[split]));ids=torch.arange(begin,end,device='cuda')
        images,gt=data.batch(split,ids,matrices)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fine,coarse=model.frozen_features(images)
        cache['fine'][begin:end].copy_(fine);cache['coarse'][begin:end].copy_(coarse)
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
    initial={n:m.point_state() for n,m in models.items()}
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
    for n,m in models.items():m.load_point_state(initial[n]);m.zero_grad(set_to_none=True)
    del opts;gc.collect();torch.cuda.empty_cache();setup()
    dump(out/'capacity.json',dict(micro_batch=64,effective_batch=64,evaluation_batch=64,
         peak_gib=peak,total_gib=total,seconds=time.perf_counter()-tick,
         reference_micro_batch_matched=True,matched_within_both_new_and_old_arms=True,reserve_gib=.7))
    return 64

def preflight(models,cache,data,val_cache,micro,out):
    checks={}
    for name,model in models.items():
        initial=model.point_state();frozen=model.frozen_digest()
        trainable=[k for k,p in model.named_parameters() if p.requires_grad]
        assert trainable and all(k.startswith('probe.decoder.author_head.point_head.') for k in trainable)
        assert all(not m.training for m in model.encoder.modules())
        assert all(not m.training for m in model.probe.adapter.modules())
        assert all(not m.training for m in model.probe.decoder.baseline.modules())
        opt=optimizer(model);model.train()
        for _ in range(3):
            loss,norms=update(model,opt,cache,torch.arange(64,device='cuda'),micro,check=True)
        assert model.frozen_digest()==frozen
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.point_head.parameters())
        assert not all(torch.equal(v,model.point_state()[k]) for k,v in initial.items())
        state=dict(model=model.point_state(),optimizer=cpu_tree(opt.state_dict()),rng=base.rng())
        expected_loss,_=update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        expected=model.point_state()
        model.load_point_state(state['model']);opt.load_state_dict(state['optimizer']);base.set_rng(state['rng'])
        base.assert_tree_exact(model.point_state(),state['model']);base.assert_tree_exact(opt.state_dict(),state['optimizer'])
        actual_loss,_=update(model,opt,cache,torch.arange(64,128,device='cuda'),micro)
        assert expected_loss==actual_loss,(expected_loss,actual_loss,expected_loss-actual_loss)
        base.assert_tree_exact(model.point_state(),expected)
        assert model.frozen_digest()==frozen
        # Same cached outputs and direct image forward under inference.
        model.eval()
        ids=torch.arange(64,device='cuda');images,_=data.batch('val',ids)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached=model.from_cache(val_cache,ids);direct=model(images)
        assert torch.equal(cached,direct)
        # Entire frozen validation prediction must remain the archived baseline.
        model.inference_policy='coarse_direct'
        metric,probs=base.evaluate(model,val_cache,64,export=True)
        ref=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]
        assert metric['counts']==ref['validation']['counts']
        assert abs(metric['macro']['dice']-ref['validation']['macro']['dice'])<1e-12
        prior=np.load(Path(ref['checkpoint']).parent/'swin_control_val_predictions.npz')
        assert np.array_equal(np.packbits((probs>=.5).reshape(len(probs),-1),axis=1),prior['predictions_packed'])
        model.inference_policy='uncertain'
        model.load_point_state(initial);model.zero_grad(set_to_none=True)
        checks[name]=dict(point_head_only_trainable=True,trainable_names=trainable,
            frozen_hash_unchanged=frozen,point_head_changed=True,replayed_optimizer_update_bitwise_exact=True,
            replay_loss_exact=True,cache_vs_direct_logits_exact=True,all762_coarse_masks_exact=True,real_data_loss=loss,gradient_norms=norms)
        del opt,state,expected,probs,images,cached,direct
    setup();dump(out/'preflight.json',dict(passed=True,checks=checks))



def source_hashes():
    author=PROJECT.parent.parent/'research/pointrend_author_20261112/official_code/projects/PointRend/point_rend'
    files=[p for p in ROOT.iterdir() if p.name in {
        'train.py','author_modules.py','evaluate.py','run_pipeline.py','verify_integration.py','verify_reference.py',
        'identity_partition.json','REFERENCE_MODELS.json','PROTOCOL.json','README_zh.txt','AUTHOR_PROVENANCE.json'}]
    files += [PROJECT/x for x in ['stage28_drop_path/train.py','evaluation.py','stage1_visual/data.py',
        'stage1_visual/models.py','stage2_decoder/models.py','stage3_confirmation/models.py','stage3_confirmation/data.py']]
    files += [base.HF/'config.json']+list(base.HF.glob('*.py'))
    files += [author/x for x in ['point_head.py','point_features.py','semantic_seg.py']]
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
        effective_batch=64,evaluation_batch=64,encoder_lr=0.,decoder_lr=.001,weight_decay=.01,
        selection='max_validation_macro_dice_earliest_tie',precision='BF16_forward_FP32_loss',deterministic_policy='deterministic_requested_warn_only_for_unsupported_ops',
        mode='all_baseline_parameters_buffers_frozen_eval_only_point_head_trainable',sources=source_hashes(),
        intervention=json.loads((ROOT/'PROTOCOL.json').read_text())['changes'][args.arm],
        partition_counts=partition['counts'],weights_sha256=base.WEIGHT_SHA,
        criterion='native_point_CE_only_frozen_coarse_loss_no_gradient',lr_schedule='cosine_100_floor0.01',
        test_exposure='867 already exposed; development only on validation; no fresh independent confirmation',
        evaluation_helper_sha256=digest(PROJECT.parent.parent/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py'),
        lr_schedule_horizon=100,reference_reuse='Frozen selected Stage37 Swin baseline plus100head-onlyepochs; small checkpoint depends on recorded baseline hash',checkpoint_format='point_head_only',reference_checkpoint=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]['checkpoint'],reference_sha256=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]['sha256'])
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
    print('BUILDING_FROZEN_FINE_AND_COARSE_CACHE',flush=True)
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
            models[n].load_point_state(state['models'][n]);opts[n].load_state_dict(state['optimizers'][n])
            base.assert_tree_exact(models[n].point_state(),state['models'][n])
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
                        save(out/f'{n}_{criterion}.pth',dict(epoch=epoch,model=models[n].point_state(),metric=metric,config=config))
            assert models[n].frozen_digest()==frozen[n]
        record.update(seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/base.GIB)
        history.append(record)
        save(latest,dict(epoch=epoch,models={n:m.point_state() for n,m in models.items()},
             optimizers={n:cpu_tree(o.state_dict()) for n,o in opts.items()},
             rng=base.rng(),selectors=selectors,history=history,config=config,elapsed=elapsed+time.perf_counter()-started))
        dump(out/'history.json',history)
        status=dict(state='running',stage='stage42_frozen_pointrend',arm=args.arm,epoch=epoch,total_epochs=config['epochs'],
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
