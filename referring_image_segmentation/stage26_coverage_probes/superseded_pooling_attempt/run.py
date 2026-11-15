"""Frozen-model feature readouts, fitted only on the training set."""
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,csv,datetime,fcntl,gc,hashlib,json,math,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stage22_partial_finetune.train import System,setup,dump,digest,boundary_row,tensor_hash
from stage3_confirmation.data import ImageData
from evaluation import summarize_rows
POINTS=('encoder_f3','encoder_f4','decoder_coarse','decoder_final')
STAGES=((42,'stage22_partial_finetune'),(43,'stage24_seed43_confirmation'))
REG=1e-4
LIMIT=50

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_rows(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare_data():
    data=ImageData(42)
    historical=json.loads((PROJECT/'stage1_visual/data_manifest.json').read_text())['members']
    assert data.audit==historical
    partition=json.loads((PROJECT/'stage22_partial_finetune/identity_partition.json').read_text())
    assert data.ids['train']==partition['splits']['train']
    positions={sid:i for i,sid in enumerate(data.ids['val'])}
    selected=partition['splits']['val']
    idx=torch.tensor([positions[sid] for sid in selected],device='cuda')
    data.images['val']=data.images['val'][idx].contiguous()
    data.masks['val']=data.masks['val'][idx].contiguous();data.ids['val']=selected
    assert len(data.ids['train'])==1886 and len(selected)==762
    return data,partition

@torch.no_grad()
def extract(model,data,split,arm,limit=None):
    n=min(len(data.ids[split]),limit or len(data.ids[split]))
    store={k:[] for k in POINTS};outputs=[];targets=[];captured={}
    def hook(name):
        def capture(module,inputs,output):
            captured[name]=F.adaptive_avg_pool2d(output.detach().float(),8)
        return capture
    handles=[model.probe.decoder.blocks[3].register_forward_hook(hook('decoder_coarse')),
             model.probe.decoder.blocks[0].register_forward_hook(hook('decoder_final'))]
    try:
        for begin in range(0,n,64):
            idx=torch.arange(begin,min(begin+64,n),device='cuda')
            images,gt=data.batch(split,idx)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                f1,f2,mid=model.prefix(images)
                if arm=='frozen':
                    f3,f4=model.suffix(mid)
                    feats=[f.to(torch.bfloat16) for f in (f1,f2,f3,f4)]
                else:
                    f3,f4=model.suffix(mid.to(torch.bfloat16))
                    feats=[f1.to(torch.bfloat16),f2.to(torch.bfloat16),f3,f4]
                logits=model.probe(feats)
            captured['encoder_f3']=F.adaptive_avg_pool2d(feats[2].float(),8)
            captured['encoder_f4']=F.adaptive_avg_pool2d(feats[3].float(),8)
            for k in POINTS:
                assert captured[k].shape[-2:]==(8,8)
                store[k].append(captured[k].detach())
            if split=='val':outputs.append(logits.float().sigmoid()[:,0].cpu().numpy())
            targets.append(F.adaptive_avg_pool2d(gt,8))
        return {k:torch.cat(v) for k,v in store.items()},torch.cat(targets),np.concatenate(outputs) if outputs else None
    finally:
        for h in handles:h.remove()

def fit(x,y,log_path,regularization=REG,max_outer=LIMIT):
    # Flatten positions but keep channel identity; all statistics use training only.
    x=x.permute(0,2,3,1).reshape(-1,x.shape[1]).double()
    target=y.reshape(-1).double()
    mu=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mu)/sd
    w=nn.Parameter(torch.zeros(x.shape[1]+1,device='cuda',dtype=torch.float64))
    with torch.no_grad():w[-1].copy_(torch.logit(target.mean().clamp(1e-8,1-1e-8)))
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=20,max_eval=30,history_size=50,
                         tolerance_grad=1e-8,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    history=[];calls=0;start=time.perf_counter()
    def closure():
        nonlocal calls
        opt.zero_grad(set_to_none=True)
        logits=x@w[:-1]+w[-1]
        loss=F.binary_cross_entropy_with_logits(logits,target)+.5*regularization*w[:-1].square().sum()
        assert torch.isfinite(loss)
        loss.backward();calls+=1
        return loss
    converged=False
    for step in range(1,max_outer+1):
        opt.step(closure)
        loss=closure()
        grad=float(w.grad.abs().max())
        history.append(dict(block=step,objective=float(loss.detach()),gradient_max=grad,
                            closure_calls=calls,seconds=time.perf_counter()-start))
        if log_path:dump(log_path,history)
        if grad<=1e-5:
            converged=True;break
    state=dict(weight=w[:-1].detach().cpu(),bias=w[-1].detach().cpu(),
               mean=mu.cpu(),std=sd.cpu(),regularization=regularization,
               converged=converged,history=history,parameters=w.numel())
    return state

@torch.no_grad()
def predict(state,features):
    x=features.permute(0,2,3,1).double()
    logits=((x-state['mean'].cuda())/state['std'].cuda())@state['weight'].cuda()+state['bias'].cuda()
    return logits[:,None].float()

def summarize_coarse(prob,gt8,ids,gt,cutpoints,full_probs=None):
    truth8=gt8[:,0].cpu().numpy()
    pc=prob[:,0].cpu().numpy()
    if full_probs is None:
        logits=torch.logit(prob.clamp(1e-7,1-1e-7))
        full_probs=F.interpolate(logits,size=(256,256),mode='bilinear',align_corners=False).sigmoid()[:,0].cpu().numpy()
    with ThreadPoolExecutor(8) as pool:
        rows=list(pool.map(lambda pair:boundary_row(pair[0]>=.5,pair[1]),zip(full_probs,gt)))
    for i,(sid,row) in enumerate(zip(ids,rows)):
        area=int(gt[i].sum())
        row.update(id=sid,area_quartile=int(np.searchsorted(cutpoints,area,side='right'))+1,
                   coarse_mse=float(np.mean((pc[i]-truth8[i])**2)),
                   coarse_mae=float(np.mean(abs(pc[i]-truth8[i]))),
                   signed_area_error=float(pc[i].mean()-truth8[i].mean()),
                   absolute_area_error=float(abs(pc[i].mean()-truth8[i].mean())),
                   all_foreground_dice=2*area/(area+256*256))
    result=summarize_rows(rows)
    for key in ('coarse_mse','coarse_mae','signed_area_error','absolute_area_error'):
        result[key]=float(np.mean([r[key] for r in rows]))
    result['area_quartiles']={}
    for q in range(1,5):
        sub=[r for r in rows if r['area_quartile']==q]
        result['area_quartiles'][str(q)]=dict(n=len(sub),macro_dice=float(np.mean([r['dice'] for r in sub])),
            coarse_mse=float(np.mean([r['coarse_mse'] for r in sub])),
            absolute_area_error=float(np.mean([r['absolute_area_error'] for r in sub])))
    result['recurring_cases']=[r for r in rows if r['id'] in ('305','581','791')]
    return result,rows

def synthetic_check():
    g=torch.Generator(device='cuda').manual_seed(901)
    x=torch.randn((20,4,8,8),generator=g,device='cuda')
    w=torch.tensor([.2,-.4,.6,.1],device='cuda')
    y=(x.permute(0,2,3,1)@w+.3).sigmoid()[:,None]
    state=fit(x,y,None,regularization=0.,max_outer=10)
    predicted=predict(state,x).sigmoid()
    error=float((predicted-y).abs().max())
    assert state['converged'] and error<1e-4,(state['history'][-1],error)
    return dict(passed=True,synthetic_max_probability_error=error)

def main(smoke):
    setup()
    out=ROOT/('smoke' if smoke else 'results');out.mkdir(exist_ok=True)
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    if not smoke:
        assert (ROOT/'smoke/DONE.json').exists()
        passed=json.loads((ROOT/'smoke/preflight.json').read_text())
        assert passed['passed'] and passed['source_sha256']==digest(ROOT/'run.py')
    data,partition=prepare_data()
    dump(out/'partition.json',partition)
    gt=(data.masks['val'][:,0]>=128).cpu().numpy()
    cutpoints=np.quantile(gt.sum((1,2)),[.25,.5,.75])
    config=dict(kind='diagnostic_linear_coverage_probe',created_utc=stamp(),
        training_views='original_training_images_only',regularization=REG,
        max_lbfgs_iterations=LIMIT*20,selection='training_gradient_convergence_only',
        checkpoint_selection='previously_selected_best_dice',points=POINTS,
        input_size=256,coverage_size=8,encoding_batch=64,n_train=1886,n_val=762,
        test_evaluated=False,area_cutpoints=cutpoints.tolist(),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in [
            ROOT/'run.py',ROOT/'PROTOCOL.txt',PROJECT/'stage22_partial_finetune/train.py',
            PROJECT/'stage1_visual/data.py',PROJECT/'stage3_confirmation/data.py',
            PROJECT/'stage2_decoder/models.py',PROJECT/'stage3_confirmation/models.py',
            PROJECT/'evaluation.py']})
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        assert old['source_sha256']==config['source_sha256']
    else:dump(out/'config.json',config)
    check=synthetic_check();summary={}
    if not smoke:
        target8=F.adaptive_avg_pool2d(data.masks['train'].float()/255.,8)
        val8=F.adaptive_avg_pool2d(data.masks['val'].float()/255.,8)
        prior=target8.mean(0,keepdim=True).expand(len(gt),-1,-1,-1)
        metric,rows=summarize_coarse(prior,val8,data.ids['val'],gt,cutpoints)
        summary['training_spatial_prior']=metric
        dump(out/'training_spatial_prior.json',metric);write_rows(out/'training_spatial_prior.csv',rows)
    for seed,stage in STAGES if not smoke else STAGES[:1]:
        for arm in ('frozen','finetune34'):
            name=f'seed{seed}_{arm}';folder=out/name;folder.mkdir(exist_ok=True)
            if (folder/'DONE.json').exists():
                summary[name]=json.loads((folder/'metrics.json').read_text());continue
            status=dict(state='extracting_features',model=name,pid=os.getpid(),time=stamp())
            dump(out/'run_status.json',status);print(json.dumps(status),flush=True)
            checkpoint=PROJECT/stage/'results'/(arm+'_best_dice.pth')
            state=torch.load(checkpoint,map_location='cpu')
            model=System(False).cuda();model.load_state_dict(state['model'],strict=True)
            model.requires_grad_(False);model.eval()
            before=tensor_hash(model)
            cache_train,train8,_=extract(model,data,'train',arm,limit=64 if smoke else None)
            cache_val,val8,reproduced=extract(model,data,'val',arm,limit=64 if smoke else None)
            reference=np.load(PROJECT/stage/'results'/(arm+'_probabilities.npy'),mmap_mode='r')
            ref=np.asarray(reference[:len(reproduced)])
            max_error=float(np.max(abs(reproduced-ref)));changed=int(((reproduced>=.5)!=(ref>=.5)).sum())
            assert max_error<=1e-6 and changed==0,(name,max_error,changed)
            parity=dict(max_probability_error=max_error,changed_mask_pixels=changed,
                checkpoint_sha256=digest(checkpoint),checkpoint_epoch=state['epoch'],
                model_hash_before=before,trainable_source_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                feature_shapes={k:list(v.shape) for k,v in cache_train.items()})
            dump(folder/'reproduction.json',parity)
            metrics={}
            if not smoke:
                original8=F.adaptive_avg_pool2d(torch.from_numpy(reproduced[:,None]).cuda(),8)
                original,rows=summarize_coarse(original8,val8,data.ids['val'],gt,cutpoints,full_probs=reproduced)
                metrics['original_decoder']=original;write_rows(folder/'original_decoder.csv',rows)
            for point in POINTS if not smoke else POINTS[:1]:
                status.update(state='fitting_probe',point=point,time=stamp())
                dump(out/'run_status.json',status);print(json.dumps(status),flush=True)
                fitted=fit(cache_train[point],train8,folder/(point+'_training.json'),
                           max_outer=10 if smoke else LIMIT)
                torch.save(fitted,folder/(point+'_probe.pth'))
                logits=predict(fitted,cache_val[point])
                assert torch.isfinite(logits).all()
                if not smoke:
                    full_prob=F.interpolate(logits,size=(256,256),mode='bilinear',align_corners=False).sigmoid()[:,0].cpu().numpy()
                    metric,rows=summarize_coarse(logits.sigmoid(),val8,data.ids['val'],gt,cutpoints,full_probs=full_prob)
                    metric.update(converged=fitted['converged'],training_final=fitted['history'][-1],probe_parameters=fitted['parameters'])
                    metrics[point]=metric;write_rows(folder/(point+'_per_image.csv'),rows)
                    np.save(folder/(point+'_coarse_probabilities.npy'),logits.sigmoid().cpu().numpy())
                print(name,point,'CONVERGED',fitted['converged'],'GRAD',fitted['history'][-1]['gradient_max'],flush=True)
            after=tensor_hash(model);assert after==before
            parity['model_hash_after']=after;parity['source_unchanged']=True
            dump(folder/'reproduction.json',parity);dump(folder/'metrics.json',metrics)
            dump(folder/'DONE.json',dict(complete=True,time=stamp(),all_probe_gradients_converged=all(v.get('converged',True) for v in metrics.values())))
            summary[name]=metrics
            del model,state,cache_train,cache_val,train8,val8,reproduced,fitted,logits
            gc.collect();torch.cuda.empty_cache()
    if smoke:
        dump(out/'preflight.json',dict(check,source_sha256=digest(ROOT/'run.py'),
             prediction_reproduction=True,source_frozen=True,passed=True))
    dump(out/'summary.json',summary)
    dump(out/'DONE.json',dict(complete=True,time=stamp(),test_evaluated=False))
    dump(out/'run_status.json',dict(state='complete',time=stamp(),test_evaluated=False))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:main(args.smoke)
    except Exception:
        dump(ROOT/('smoke' if args.smoke else 'results')/'failure.json',dict(traceback=traceback.format_exc(),time=stamp()))
        raise
