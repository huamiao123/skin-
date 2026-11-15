"""Paired late-stage fine-tuning: only late DropPath training behavior differs."""
from pathlib import Path
import argparse, contextlib, copy, csv, gc, hashlib, importlib.util, json, math
import os, random, signal, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from safetensors.torch import load_file
from stage3_confirmation.models import Probe
from stage3_confirmation.data import ImageData
from stage1_visual.models import segmentation_loss, tensor_hash
from evaluation import aggregate_counts, summarize_rows
from scipy.ndimage import binary_erosion, distance_transform_edt

HF = PROJECT/'stage14_mambavision_s_isic2018_seed42/official/hf'
WEIGHT_SHA = '006af39903f2d88c5b455431d087d0fcbf5d5932114aef52542c0d2f0b1ec180'
NAMES = ('control', 'drop_path')
GIB = 1024**3
STOP = False

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def save(path, obj):
    tmp=Path(path).with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def cpu_tree(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj,dict): return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list): return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple): return tuple(cpu_tree(v) for v in obj)
    return obj

def assert_tree_exact(a,b):
    if torch.is_tensor(a): assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_tree_exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b): assert_tree_exact(left,right)
    else: assert a==b

def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

def setup():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

class System(nn.Module):
    def __init__(self, finetune=False, stochastic_depth=False):
        super().__init__()
        self.stochastic_depth=bool(stochastic_depth)
        assert not self.stochastic_depth or finetune
        self.encoder,info=AutoModel.from_pretrained(str(HF),trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
        state=load_file(str(HF/'model.safetensors'))
        self.encoder.load_state_dict(state,strict=True)
        assert all(torch.equal(v,self.encoder.state_dict()[k]) for k,v in state.items())
        self.encoder.requires_grad_(False)
        if finetune:
            for level in self.encoder.model.levels[2:]: level.requires_grad_(True)
        self.probe=Probe(decoder='swin',seed=42,channels=(96,192,384,768))
        self.finetune=finetune
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.stochastic_depth:
            for level in self.encoder.model.levels[2:]:
                for module in level.modules():
                    if type(module).__name__=='DropPath': module.train(mode)
        self.probe.train(mode)
        return self

    @torch.no_grad()
    def prefix(self, images):
        base=self.encoder.model
        x=base.patch_embed(images)
        x,f1=base.levels[0](x)
        x,f2=base.levels[1](x)
        return f1.contiguous(),f2.contiguous(),x.contiguous()

    def suffix(self, mid):
        context=contextlib.nullcontext if self.finetune else torch.no_grad
        with context():
            x,f3=self.encoder.model.levels[2](mid)
            _,f4=self.encoder.model.levels[3](x)
        return f3.contiguous(),f4.contiguous()

    def from_cache(self, cache, indices):
        f1,f2=cache['features'][0][indices],cache['features'][1][indices]
        if self.finetune: f3,f4=self.suffix(cache['mid'][indices])
        else: f3,f4=cache['features'][2][indices],cache['features'][3][indices]
        return self.probe([f1,f2,f3,f4])

    def forward(self, images):
        f1,f2,mid=self.prefix(images)
        f3,f4=self.suffix(mid)
        return self.probe([f1,f2,f3,f4])

def optimizer(model):
    groups=[dict(params=list(model.probe.parameters()),lr=.001,initial_lr=.001,name='decoder')]
    late=[p for p in model.encoder.parameters() if p.requires_grad]
    if late: groups.append(dict(params=late,lr=.0001,initial_lr=.0001,name='encoder'))
    return torch.optim.AdamW(groups,weight_decay=.01,betas=(.9,.999),eps=1e-8)

def lr_epoch(opt, epoch):
    factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/300))/2
    for group in opt.param_groups: group['lr']=group['initial_lr']*factor

def frozen_hash(model):
    h=hashlib.sha256()
    for key,value in model.encoder.state_dict().items():
        if model.finetune and key.startswith(('model.levels.2.','model.levels.3.')): continue
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def cache_new(n):
    return dict(features=[torch.empty((n,c,s,s),dtype=torch.bfloat16,device='cuda') for c,s in [(96,64),(192,32),(384,16),(768,8)]],
                mid=torch.empty((n,384,16,16),dtype=torch.bfloat16,device='cuda'),
                target=torch.empty((n,1,256,256),dtype=torch.float32,device='cuda'))

@torch.no_grad()
def fill_cache(model,data,split,cache,matrices=None,batch=64):
    for begin in range(0,len(data.ids[split]),batch):
        end=min(begin+batch,len(data.ids[split])); ids=torch.arange(begin,end,device='cuda')
        images,gt=data.batch(split,ids,matrices)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f1,f2,mid=model.prefix(images);f3,f4=model.suffix(mid)
        for dst,src in zip(cache['features'],[f1,f2,f3,f4]):dst[begin:end].copy_(src)
        cache['mid'][begin:end].copy_(mid);cache['target'][begin:end].copy_(gt)

def update(model,opt,cache,ids,micro,check=False):
    opt.zero_grad(set_to_none=True)
    loss_value=0.
    for begin in range(0,len(ids),micro):
        chunk=ids[begin:begin+micro]
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model.from_cache(cache,chunk)
        loss=segmentation_loss(logits,cache['target'][chunk])
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
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

@torch.no_grad()
def evaluate(model,cache,micro=64,export=False):
    model.eval();counts=[];loss_total=0.;predictions=[]
    bce_total=0.;soft_dice_total=0.;wrong_n=0;wrong_confidence=0.;wrong99=0
    n=len(cache['target'])
    for begin in range(0,n,micro):
        ids=torch.arange(begin,min(begin+micro,n),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model.from_cache(cache,ids)
        target=cache['target'][ids]
        loss_total+=float(segmentation_loss(logits,target))*len(ids)
        prob=logits.float().sigmoid();pred=prob>=.5;truth=target>=.5
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),target.float())
        soft_dice=1-((2*(prob*target).sum((1,2,3))+1)/(prob.sum((1,2,3))+target.sum((1,2,3))+1)).mean()
        bce_total+=float(bce)*len(ids);soft_dice_total+=float(soft_dice)*len(ids)
        wrong=pred!=truth;confidence=torch.where(pred,prob,1-prob)
        wrong_n+=int(wrong.sum());wrong_confidence+=float(confidence[wrong].double().sum())
        wrong99+=int((wrong&(confidence>.99)).sum())
        counts.append(torch.stack([(pred&truth).sum((1,2,3)),(pred&~truth).sum((1,2,3)),(~pred&truth).sum((1,2,3)),(~pred&~truth).sum((1,2,3))],1).cpu().numpy())
        if export:predictions.append(prob[:,0].cpu().numpy())
    c=np.concatenate(counts);result=aggregate_counts(*c.T);result['loss']=loss_total/n
    assert abs(result['loss']-(bce_total+soft_dice_total)/n)<2e-6
    result['diagnostics']=dict(bce_soft=bce_total/n,soft_dice_loss=soft_dice_total/n,
        wrong_pixels=wrong_n,wrong_mean_confidence=wrong_confidence/max(wrong_n,1),
        wrong_confidence_gt99_fraction=wrong99/max(wrong_n,1))
    model.train()
    return result,np.concatenate(predictions) if export else None

def calibrate(models,opts,cache,out):
    initial={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    records=[];chosen=None
    memory_limit=torch.cuda.get_device_properties(0).total_memory-1.2*GIB
    for micro in (64,48,32,24,16,8,4):
        try:
            model=models['drop_path'];model.load_state_dict(initial['drop_path']);opt=optimizer(model)
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for _ in range(2): update(model,opt,cache,torch.arange(64,device='cuda'),micro,check=True)
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
            records.append(dict(micro=micro,peak_gib=peak/GIB,seconds=(time.perf_counter()-start)/2,passed=peak<memory_limit))
            if peak<memory_limit:chosen=micro
            del opt
            model.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
            if chosen:break
        except torch.cuda.OutOfMemoryError:
            records.append(dict(micro=micro,passed=False,error='CUDA OOM'))
            try: del opt
            except UnboundLocalError: pass
            models['drop_path'].zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
    if chosen is None:raise RuntimeError('No safe microbatch')
    for n,m in models.items():m.load_state_dict(initial[n]);m.zero_grad(set_to_none=True);opts[n]=optimizer(m)
    dump(out/'capacity.json',dict(micro_batch=chosen,effective_batch=64,records=records,
         cache_allocated_gib=torch.cuda.memory_allocated()/GIB,total_gib=torch.cuda.get_device_properties(0).total_memory/GIB,
         reserve_gib=1.2,cache_policy='GPU input data and both-arm shared frozen-prefix features; no stale trainable features'))
    return chosen

def preflight(models,opts,data,cache,micro,out):
    before={n:cpu_tree(m.state_dict()) for n,m in models.items()}
    ids=torch.arange(8,device='cuda');images,_=data.batch('train',ids)
    checks={}
    for model in models.values(): model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        _,official=models['control'].encoder(images)
        f1,f2,mid=models['control'].prefix(images);f3,f4=models['control'].suffix(mid)
        checks['official_feature_max_abs']=[float((a-b).abs().max()) for a,b in zip(official,[f1,f2,f3,f4])]
        assert all(v==0 for v in checks['official_feature_max_abs'])
        a=models['control'](images);b=models['drop_path'](images)
        assert torch.equal(a,b);checks['initial_logits_identical']=True
    checks['initial_probe_hash']=tensor_hash(models['control'].probe)
    assert checks['initial_probe_hash']==tensor_hash(models['drop_path'].probe)
    for name,model in models.items():
        model.train()
        entries=[dict(name=k,training=v.training,prob=v.drop_prob)
                 for k,v in model.encoder.named_modules() if type(v).__name__=='DropPath']
        late=[x for x in entries if x['name'].startswith(('model.levels.2.','model.levels.3.'))]
        assert len(late)==12
        assert sum(x['training'] for x in late)==(12 if name=='drop_path' else 0)
        assert all(not x['training'] for x in entries if x not in late)
        assert all(not b.training for b in model.encoder.modules() if isinstance(b,nn.modules.batchnorm._BatchNorm))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            state_rng=rng();a=model(images);b=model(images)
            if name=='drop_path': assert not torch.equal(a,b),'DropPath inactive in training'
            else: assert torch.equal(a,b),'Unexpected stochastic control behavior'
            set_rng(state_rng);replayed=model(images)
            assert torch.equal(a,replayed),'Stochastic forward RNG replay failed'
            model.eval();a=model(images);b=model(images);assert torch.equal(a,b)
        assert all(not x.training for x in model.encoder.modules() if type(x).__name__=='DropPath')
        model.train()
        checks[name+'_mode_policy']=dict(drop_paths=entries,evaluation_deterministic=True,train_rng_replay_exact=True)
    for name,model in models.items():
        h=frozen_hash(model)
        loss,norms=update(model,opts[name],cache,torch.arange(64,device='cuda'),micro,check=True)
        assert h==frozen_hash(model)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),before[name][k])]
        assert any(k.startswith('probe.') for k in changed)
        if model.finetune:
            assert any(k.startswith('encoder.model.levels.2.') for k in changed)
            assert any(k.startswith('encoder.model.levels.3.') for k in changed)
        else:assert not any(k.startswith('encoder.') for k in changed)
        # Replay one optimizer update from a serialized model/optimizer/RNG checkpoint.
        state=dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opts[name].state_dict()),rng=rng())
        save(out/(name+'_recovery_test.pth'),state)
        expected_loss,_=update(model,opts[name],cache,torch.arange(64,128,device='cuda'),micro)
        expected=cpu_tree(model.state_dict())
        restored=torch.load(out/(name+'_recovery_test.pth'),map_location='cpu')
        model.load_state_dict(restored['model']);opts[name].load_state_dict(restored['optimizer']);set_rng(restored['rng'])
        assert_tree_exact(model.state_dict(),restored['model'])
        assert_tree_exact(opts[name].state_dict(),restored['optimizer'])
        assert_tree_exact(rng(),restored['rng'])
        actual_loss,_=update(model,opts[name],cache,torch.arange(64,128,device='cuda'),micro)
        assert actual_loss==expected_loss
        diff_squared=0.;update_squared=0.;max_abs=0.;different=0
        for key,value in model.state_dict().items():
            actual=value.detach().cpu();reference=expected[key]
            if not (model.finetune and key.startswith(('encoder.model.levels.2.','encoder.model.levels.3.'))):
                assert torch.equal(actual,reference),key
                continue
            difference=(actual-reference).double()
            change=(reference-restored['model'][key]).double()
            diff_squared+=float(difference.square().sum());update_squared+=float(change.square().sum())
            max_abs=max(max_abs,float(difference.abs().max()))
            different+=int(not torch.equal(actual,reference))
        relative_l2=math.sqrt(diff_squared/max(update_squared,1e-30))
        # The installed selective-scan backward is empirically non-bitwise-repeatable,
        # even with deterministic PyTorch settings. State restoration itself is exact.
        assert relative_l2<=.01 and max_abs<=.0001,(relative_l2,max_abs)
        checks[name]=dict(frozen_unchanged=True,changed_tensors=len(changed),gradient_norms=norms,
            restored_model_optimizer_rng_exact=True,resume_loss_exact=True,resume_update_exact=different==0,
            decoder_update_exact=True,encoder_replay_difference_relative_to_update_l2=relative_l2,
            encoder_replay_max_abs=max_abs,encoder_different_tensors=different,
            replay_limits=dict(relative_to_update_l2=.01,max_abs=.0001))
    for name,model in models.items():model.load_state_dict(before[name]);model.zero_grad(set_to_none=True);opts[name]=optimizer(model)
    setup();dump(out/'preflight.json',dict(passed=True,checks=checks))

def boundary_row(pred,gt):
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum());tn=int((~pred&~gt).sum())
    pb=pred^binary_erosion(pred,border_value=0);gb=gt^binary_erosion(gt,border_value=0)
    diagonal=np.hypot(*gt.shape)
    if not pb.any() or not gb.any():
        bf1=float(not pb.any() and not gb.any());hd95=0. if bf1 else diagonal
    else:
        dp=distance_transform_edt(~pb)[gb];dg=distance_transform_edt(~gb)[pb]
        bp,br=(dg<=.005*diagonal).mean(),(dp<=.005*diagonal).mean()
        bf1=2*bp*br/(bp+br) if bp+br else 0.
        hd95=float(np.percentile(np.concatenate([dp,dg]),95))
    result=dict(tp=tp,fp=fp,fn=fn,tn=tn,boundary_f1=float(bf1),hd95_pixels=float(hd95),hd95_diagonal=float(hd95/diagonal))
    result.update(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                  precision=tp/(tp+fp) if tp+fp else float(not gt.any()),recall=tp/(tp+fn) if tp+fn else 1.)
    return result

def final_evaluate(models,cache,data,micro,out):
    gt=cache['target'][:,0].cpu().numpy()>=.5
    summary={};best_probs={};all_rows=[]
    for name,model in models.items():
        for criterion in ('best_dice','best_loss','last'):
            state=torch.load(out/f'{name}_{criterion}.pth',map_location='cpu');model.load_state_dict(state['model'])
            basic,probs=evaluate(model,cache,micro,export=True)
            with ThreadPoolExecutor(8) as pool:rows=list(pool.map(lambda pair:boundary_row(pair[0]>=.5,pair[1]),zip(probs,gt)))
            for sid,row in zip(data.ids['val'],rows):row.update(id=sid,model=name,criterion=criterion)
            full=summarize_rows(rows);full.update(epoch=state['epoch'],loss=basic['loss'],diagnostics=basic['diagnostics']);summary[name+'/'+criterion]=full
            all_rows.extend(rows)
            if criterion=='best_dice':
                best_probs[name]=probs;np.save(out/(name+'_probabilities.npy'),probs)
            dump(out/f'{name}_{criterion}_metrics.json',full)
    with (out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    base=best_probs['control']>=.5;fine=best_probs['drop_path']>=.5
    diffs=np.array([boundary_row(f,g)['dice']-boundary_row(b,g)['dice'] for f,b,g in zip(fine,base,gt)])
    generator=np.random.default_rng(42);boot=np.array([diffs[generator.integers(len(diffs),size=len(diffs))].mean() for _ in range(2000)])
    summary['paired']=dict(macro_dice_delta=float(diffs.mean()),image_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),
          corrected_pixels=int(((base!=gt)&(fine==gt)).sum()),introduced_pixels=int(((base==gt)&(fine!=gt)).sum()),
          note='Exploratory development-validation image bootstrap; not seed variability or independent-test inference')
    dump(out/'summary.json',summary)
    return summary

def environment():
    import importlib.metadata as md
    names=['torch','torchvision','timm','transformers','mamba-ssm','numpy','scipy','Pillow','safetensors']
    import selective_scan_cuda
    return dict(python=sys.version,versions={k:md.version(k) for k in names},gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        selective_scan_cuda_path=selective_scan_cuda.__file__,selective_scan_cuda_sha256=digest(selective_scan_cuda.__file__))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args();setup()
    out=ROOT/('smoke' if args.smoke else 'results');out.mkdir(exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    assert digest(HF/'model.safetensors')==WEIGHT_SHA
    data=ImageData(42)
    historical=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    assert data.audit==historical,'Dataset differs from archived experiment'
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    assert partition['splits']['train']==data.ids['train']
    selected=partition['splits']['val']
    positions={sid:i for i,sid in enumerate(data.ids['val'])}
    indices=torch.tensor([positions[sid] for sid in selected],device='cuda')
    data.images['val']=data.images['val'][indices].contiguous()
    data.masks['val']=data.masks['val'][indices].contiguous()
    data.ids['val']=selected
    assert len(selected)==partition['counts']['val']
    dump(out/'active_partition.json',partition)
    for row in json.loads((HF.parent/'source_download.json').read_text())['files']:
        assert digest(HF/row['path'])==row['sha256'],('Official source mismatch',row['path'])
    source_paths=[ROOT/'train.py',ROOT/'analyze.py',ROOT/'run_pipeline.py',ROOT/'PROTOCOL.md',ROOT/'test_protocol.json',ROOT/'identity_partition.json',PROJECT/'evaluation.py',PROJECT/'stage1_visual/data.py',PROJECT/'stage1_visual/models.py',PROJECT/'stage2_decoder/models.py',PROJECT/'stage3_confirmation/models.py',PROJECT/'stage3_confirmation/data.py']+list(HF.glob('*.py'))+[HF/'config.json']
    provenance={str(p.relative_to(PROJECT)):digest(p) for p in source_paths}
    config=dict(seed=42,epochs=2 if args.smoke else 300,effective_batch=64,encoder_lr=.0001,decoder_lr=.001,weight_decay=.01,
      selection='max_validation_macro_dice_earliest_tie',data_manifest_sha256=digest(PROJECT/'stage1_visual/data_manifest.json'),
      weights_sha256=WEIGHT_SHA,sources=provenance,encoder_mode_policy='eval_except_candidate_late_DropPath_during_training',drop_path_rates='unchanged_pretrained_schedule_0.070588_to_0.2',precision='BF16_autocast_FP32_loss',
      deterministic_algorithms=True,cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
      partition_counts=partition['counts'],partition='image_identity_screened_v1')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:dump(out/'config.json',config)
    dump(out/'environment.json',environment())
    if not args.smoke:
        assert json.loads((ROOT/'smoke/RESUME_AUDIT.json').read_text())['passed']
    models={name:System(finetune=True,stochastic_depth=name=='drop_path').cuda() for name in NAMES}
    assert tensor_hash(models['control'].probe)==tensor_hash(models['drop_path'].probe)
    opts={name:optimizer(model) for name,model in models.items()}
    frozen_before={n:frozen_hash(m) for n,m in models.items()}
    dump(out/'parameters.json',{n:dict(total=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),encoder_trainable=sum(p.numel() for p in m.encoder.parameters() if p.requires_grad),probe_hash=tensor_hash(m.probe)) for n,m in models.items()})
    train_cache=cache_new(len(data.ids['train']));val_cache=cache_new(len(data.ids['val']))
    print('BUILDING_FROZEN_FEATURE_CACHES',flush=True)
    fill_cache(models['control'],data,'val',val_cache)
    fill_cache(models['control'],data,'train',train_cache)
    if args.smoke:
        micro=calibrate(models,opts,train_cache,out)
        preflight(models,opts,data,train_cache,micro,out)
    else:
        assert json.loads((ROOT/'smoke/preflight.json').read_text())['passed']
        assert (ROOT/'smoke/DONE.json').exists()
        assert json.loads((ROOT/'smoke/config.json').read_text())['sources']==provenance
        micro=json.loads((ROOT/'smoke/capacity.json').read_text())['micro_batch']
        dump(out/'capacity.json',json.loads((ROOT/'smoke/capacity.json').read_text()))
    selectors={n:dict(best_dice=-1.,best_loss=1e9) for n in NAMES};history=[];start=1;elapsed=0.
    latest=out/'latest.pth'
    if latest.exists():
        state=torch.load(latest,map_location='cpu');assert state['config']==config
        for n in NAMES:models[n].load_state_dict(state['models'][n]);opts[n].load_state_dict(state['optimizers'][n])
        selectors=state['selectors'];history=state['history'];start=state['epoch']+1;elapsed=state['elapsed'];set_rng(state['rng'])
        print('RESUMED',start,flush=True)
    started=time.perf_counter()
    def stop_handler(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    for epoch in range(start,config['epochs']+1):
        tick=time.perf_counter();order,matrices=data.geometry(epoch)
        fill_cache(models['control'],data,'train',train_cache,matrices)
        for n in NAMES:models[n].train();lr_epoch(opts[n],epoch)
        losses={n:0. for n in NAMES};gradients={}
        torch.cuda.reset_peak_memory_stats()
        for begin in range(0,len(order),64):
            ids=order[begin:begin+64]
            for n in NAMES:
                loss,gradients[n]=update(models[n],opts[n],train_cache,ids,micro,check=begin==0)
                losses[n]+=loss*len(ids)
        record=dict(epoch=epoch,train_loss={n:losses[n]/len(order) for n in NAMES},validation={},gradient_norms=gradients,
             learning_rates={n:{g['name']:g['lr'] for g in opts[n].param_groups} for n in NAMES})
        for n in NAMES:
            metric,_=evaluate(models[n],val_cache,micro)
            record['validation'][n]=metric
            for criterion,value,improved in [('best_dice',metric['macro']['dice'],metric['macro']['dice']>selectors[n]['best_dice']),('best_loss',metric['loss'],metric['loss']<selectors[n]['best_loss'])]:
                if improved:
                    selectors[n][criterion]=value
                    save(out/f'{n}_{criterion}.pth',dict(epoch=epoch,model=cpu_tree(models[n].state_dict()),metric=metric,config=config))
            assert frozen_hash(models[n])==frozen_before[n]
        record.update(seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,reserved_gib=torch.cuda.memory_reserved()/GIB)
        history.append(record)
        save(latest,dict(epoch=epoch,models={n:cpu_tree(m.state_dict()) for n,m in models.items()},optimizers={n:cpu_tree(o.state_dict()) for n,o in opts.items()},rng=rng(),selectors=selectors,history=history,config=config,elapsed=elapsed+time.perf_counter()-started))
        dump(out/'history.json',history)
        status=dict(state='running',epoch=epoch,total_epochs=config['epochs'],updates_per_arm=epoch*30,micro_batch=micro,
            validation_macro_dice={n:record['validation'][n]['macro']['dice'] for n in NAMES},best=selectors,
            seconds=record['seconds'],peak_allocated_gib=record['peak_allocated_gib'],reserved_gib=record['reserved_gib'],pid=os.getpid())
        dump(out/'run_status.json',status);print(json.dumps(status),flush=True)
        if STOP or (args.stop_after_epoch and epoch>=args.stop_after_epoch):
            status['state']='stopped_at_epoch_boundary';dump(out/'run_status.json',status);return
    for n in NAMES:save(out/f'{n}_last.pth',dict(epoch=config['epochs'],model=cpu_tree(models[n].state_dict()),config=config))
    dump(out/'run_status.json',dict(state='evaluating',epochs=config['epochs'],pid=os.getpid()))
    summary=final_evaluate(models,val_cache,data,micro,out)
    dump(out/'DONE.json',dict(complete=True,epochs=config['epochs'],updates_per_arm=config['epochs']*30,summary='summary.json'))
    dump(out/'run_status.json',dict(state='complete',epochs=config['epochs'],paired=summary['paired']))
    print('COMPLETE',json.dumps(summary['paired']),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        failure=dict(state='failed',error=traceback.format_exc(),pid=os.getpid())
        dump(ROOT/('smoke' if '--smoke' in sys.argv else 'results')/'failure.json',failure)
        raise
