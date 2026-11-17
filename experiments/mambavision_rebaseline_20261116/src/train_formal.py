"""Stage A formal 100-epoch training and validation; test performed separately once available."""
import argparse,contextlib,csv,hashlib,json,math,os,random,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from model import PROJECT,ARMS,SegmentationSystem,load_pretrained_encoder,objective
sys.path.insert(0,str(PROJECT))
from stage45_upernet_decoder import train as reference
from stage28_drop_path import train as base
from stage1_visual.analyze import metrics as sample_metrics
from evaluation import summarize_rows
ROOT=Path(__file__).resolve().parents[1]

def accumulate(model,opt,cache,indices,micro,device_type='cuda',use_amp=True,clip=1.0):
    """Accumulate a sample-weighted mean, including a short final microbatch."""
    opt.zero_grad(set_to_none=True);total=0.
    for start in range(0,len(indices),micro):
        ix=indices[start:start+micro]
        ctx=torch.autocast(device_type,dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
        with ctx:
            out=model.from_cache(cache['features'][0][ix],cache['features'][1][ix],cache['mid'][ix],cache['target'].shape[-2:])
        loss,_=objective(out,cache['target'][ix])
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        weight=len(ix)/len(indices); (loss*weight).backward();total+=float(loss.detach())*weight
    norms={}
    for group in opt.param_groups:
        parameters=group['params']
        if clip is not None:
            norms[group.get('name','group')]=float(torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=True))
        elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('nonfinite gradients')
    opt.step();return total,norms

def evaluate(model,cache,ids,micro):
    model.eval();rows=[];loss_sum=0.
    with torch.no_grad():
        for start in range(0,len(ids),micro):
            ix=torch.arange(start,min(start+micro,len(ids)),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.from_cache(cache['features'][0][ix],cache['features'][1][ix],cache['mid'][ix],(256,256))
            loss,_=objective(output,cache['target'][ix]);loss_sum+=float(loss)*len(ix)
            pred=(output['main'].float().sigmoid()>=.5).cpu().numpy()[:,0]
            gt=(cache['target'][ix]>=.5).cpu().numpy()[:,0]
            def score(item):
                j,p,g=item; row=sample_metrics(p,g);row['id']=ids[start+j];return row
            with ThreadPoolExecutor(max_workers=8) as pool:rows.extend(pool.map(score,[(j,p,g) for j,(p,g) in enumerate(zip(pred,gt))]))
    result=summarize_rows(rows);result['loss']=loss_sum/len(ids);model.train();return result,rows

def real_data_replay_preflight(model,opt,data,cache,order,geometry,out):
    """Test real cache equivalence and replay, restoring all model/optimizer/RNG state."""
    ids=order[:2]; images,_=data.batch('train',ids,geometry); model.eval()
    # Match the original prefix cache's batch64 geometry and numerical execution.
    # Independently compare microbatch direct inference for transparency.
    # Augmentation must execute outside autocast, exactly as in fill_cache.
    rebuilt=[]
    with torch.no_grad():
        for sid in ids.tolist():
            start=(sid//64)*64
            group=torch.arange(start,min(start+64,len(data.ids['train'])),device='cuda')
            group_images,group_targets=data.batch('train',group,geometry)
            assert torch.equal(group_targets[sid-start],cache['target'][sid])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                features=model.prefix(group_images)
            rebuilt.append(tuple(v[sid-start:sid-start+1] for v in features))
        f1,f2,mid=(torch.cat([row[j] for row in rebuilt]) for j in range(3))
        assert torch.equal(f1,cache['features'][0][ids])
        assert torch.equal(f2,cache['features'][1][ids])
        assert torch.equal(mid,cache['mid'][ids])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            independent=model(images)['main']
            direct=model.from_cache(f1,f2,mid,(256,256))['main']
            cached=model.from_cache(cache['features'][0][ids],cache['features'][1][ids],cache['mid'][ids],(256,256))['main']
    independent_max_diff=float((independent.float()-cached.float()).abs().max())
    max_diff=float((direct.float()-cached.float()).abs().max())
    if not torch.allclose(direct.float(),cached.float(),atol=1e-3,rtol=1e-3):raise RuntimeError(f'real data cache mismatch: {max_diff}')
    model.train();snapshot={'model':base.cpu_tree(model.state_dict()),'optimizer':base.cpu_tree(opt.state_dict()),'rng':rng_state()}
    first_loss,_=accumulate(model,opt,cache,ids,2);expected=base.cpu_tree(model.state_dict())
    model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);restore_rng(snapshot['rng'])
    second_loss,_=accumulate(model,opt,cache,ids,2)
    named=set(dict(model.named_parameters()));d2=0.;u2=0.;max_abs=0.;buffer_errors=[]
    for name,current in model.state_dict().items():
        actual=current.detach().cpu();reference=expected[name]
        if name not in named:
            if not torch.equal(actual,reference):buffer_errors.append(name)
            continue
        delta=(actual-reference).double();update=(reference-snapshot['model'][name]).double()
        d2+=float(delta.square().sum());u2+=float(update.square().sum());max_abs=max(max_abs,float(delta.abs().max()))
    rel=(d2/max(u2,1e-30))**.5
    model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);restore_rng(snapshot['rng'])
    if buffer_errors or abs(first_loss-second_loss)>1e-6 or rel>.01 or max_abs>2.1e-4:
        raise RuntimeError(f'GPU replay failed loss={first_loss,second_loss} rel={rel} max={max_abs} buffers={buffer_errors}')
    atomic_json(out/'real_data_preflight.json',{'passed':True,'n':2,'cache_direct_max_abs':max_diff,'independent_microbatch_direct_max_abs':independent_max_diff,'prefix_batch':64,'replay_loss_1':first_loss,'replay_loss_2':second_loss,'replay_relative_update_l2':rel,'replay_max_parameter_abs':max_abs,'replay_relative_limit':0.01,'replay_absolute_limit':0.00021,'limit_rationale':'Empirical native repeat max 0.0001984; initial encoder AdamW lr 0.0001; up to 2*lr sign divergence. Decoder is independently audited in later checks.','buffer_errors':buffer_errors,'training_state_restored':True,'preflight_optimizer_steps_excluded_from_training':2})
    del snapshot,expected

def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])
def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def atomic_save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=tuple(ARMS),required=True);ap.add_argument('--micro-batch',type=int,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--stop-after-epoch',type=int,choices=tuple(range(1,101)));args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.micro_batch<2 or args.micro_batch>64:raise ValueError('microbatch must be 2..64')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 required by protocol')
    # Every compared arm must share the exact microbatch selected by the preflight.
    expected_micro=2
    if args.micro_batch!=expected_micro:raise ValueError('Only the jointly preflighted microbatch=2 is currently permitted')
    for arm in ARMS:
        smoke=ROOT/'results'/'smoke'/arm
        if json.loads((smoke/'status.json').read_text())['state']!='complete':raise RuntimeError('All smoke arms must complete first')
        if not json.loads((smoke/'resume_load_audit.json').read_text())['passed']:raise RuntimeError('Smoke restart audit required')
    if not (ROOT/'reports/validation_cache_comparison.json').exists():raise RuntimeError('Validation cache comparison required')
    preflight=ROOT/'reports'/f'gpu_preflight_{args.arm}_mb{expected_micro}.json'
    if not json.loads(preflight.read_text())['passed']:raise RuntimeError('preflight not passed')
    out=ROOT/'results'/'formal'/args.arm;latest=out/'latest.pth'
    if out.exists() and not args.resume:raise FileExistsError('formal output already exists; use --resume only for this exact run')
    if args.resume and not latest.exists():raise FileNotFoundError('resume checkpoint missing')
    out.mkdir(parents=True,exist_ok=True)
    reference.setup(); torch.use_deterministic_algorithms(True,warn_only=True)
    data,partition=reference.load_data()
    model=SegmentationSystem(load_pretrained_encoder(),args.arm,seed=42).cuda()
    opt=torch.optim.AdamW([{'params':list(model.probe.parameters()),'lr':1e-3,'initial_lr':1e-3,'name':'decoder'},
                           {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':1e-4,'initial_lr':1e-4,'name':'encoder'}],weight_decay=.01,betas=(.9,.999),eps=1e-8)
    config={'arm':args.arm,'seed':42,'epochs':100,'input_size':256,'micro_batch':args.micro_batch,'effective_batch':64,'optimizer':'AdamW','encoder_lr':1e-4,'decoder_lr':1e-3,'weight_decay':.01,'scheduler':'cosine_100_floor_0.01','threshold':.5,'selection':'validation_macro_dice_earliest_tie','test_evaluated':False,'synthetic_only':False,'smoke_only':False,'partition_sha256':hashlib.sha256((PROJECT/'stage45_upernet_decoder/identity_partition.json').read_bytes()).hexdigest(), 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((ROOT/'src/model.py').read_bytes()).hexdigest(),'preflight_sha256':hashlib.sha256(preflight.read_bytes()).hexdigest(),'fcn_sha256':hashlib.sha256((ROOT/'vendor/fcn_head.py').read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(0)}
    if args.resume:
        if not (out/'config.json').is_file() or json.loads((out/'config.json').read_text())!=config:raise RuntimeError('existing config does not match')
    else:atomic_json(out/'config.json',config)
    val_cache=reference.cache_new(len(data.ids['val']));reference.fill_cache(model,data,'val',val_cache)
    train_cache=reference.cache_new(len(data.ids['train']))
    history=[];start_epoch=1;best=-1.
    if args.resume:
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['config']!=config:raise RuntimeError('resume config/source mismatch')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);base.assert_tree_exact(model.state_dict(),state['model']);base.assert_tree_exact(opt.state_dict(),state['optimizer']);base.assert_tree_exact(rng_state(),state['rng']);history=state['history'];best=state['best'];start_epoch=state['epoch']+1
        if start_epoch>100:raise RuntimeError('formal training already completed')
        atomic_json(out/'resume_load_audit.json',{'passed':True,'checkpoint_epoch':state['epoch'],'model_optimizer_rng_exact':True})
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,101):
        tick=time.perf_counter();order,geometry=data.geometry(epoch);reference.fill_cache(model,data,'train',train_cache,geometry)
        model.train()
        if epoch==1 and not args.resume:real_data_replay_preflight(model,opt,data,train_cache,order,geometry,out)
        factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        loss_sum=0.;n=0
        for start in range(0,len(order),64):
            ids=order[start:start+64];value,_=accumulate(model,opt,train_cache,ids,args.micro_batch)
            loss_sum+=value*len(ids);n+=len(ids)
        validation,rows=evaluate(model,val_cache,data.ids['val'],args.micro_batch)
        record={'epoch':epoch,'train_loss':loss_sum/n,'validation':validation,'seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'updates':math.ceil(n/64)};history.append(record)
        if validation['macro']['dice']>best:
            best=validation['macro']['dice'];atomic_save(out/'best_dice.pth',{'epoch':epoch,'model':base.cpu_tree(model.state_dict()),'validation':validation,'config':config})
        atomic_save(latest,{'epoch':epoch,'model':base.cpu_tree(model.state_dict()),'optimizer':base.cpu_tree(opt.state_dict()),'rng':rng_state(),'best':best,'history':history,'config':config})
        atomic_json(out/'history.json',history)
        with (out/f'val_epoch{epoch}_per_sample.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps({'arm':args.arm,'epoch':epoch,'train_loss':record['train_loss'],'val_macro_dice':validation['macro']['dice'],'seconds':record['seconds'],'peak_bytes':record['peak_allocated_bytes']}),flush=True)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:break
    atomic_json(out/'status.json',{'state':'complete' if history and history[-1]['epoch']==100 else 'stopped_at_epoch_boundary','epochs_recorded':len(history),'total_seconds':time.perf_counter()-started,'test_evaluated':False})
if __name__=='__main__':main()
