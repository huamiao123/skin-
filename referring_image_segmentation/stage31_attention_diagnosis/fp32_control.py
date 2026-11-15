"""Validation-only post-training mechanism audit; never opens test data."""
from pathlib import Path
import sys,json,csv,gc,types,time,traceback,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT))
from stage30_channel_fusion.train import System,NAMES,base
import importlib.util
hp=PROJECT.parent.parent/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py'
spec=importlib.util.spec_from_file_location('audit_helpers',hp)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
SOURCE=PROJECT/'stage30_channel_fusion'
OUT=ROOT/'fp32_results'
NAMES=('cross_channel_scale',)
dump=base.dump
def status(state,**kw):
    v=dict(state=state,time_utc=helpers.now(),**kw);dump(OUT/'status.json',v)
    print(json.dumps(v),flush=True)
def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def distribution(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max()),
                q05=float(np.quantile(x,.05)),q50=float(np.quantile(x,.5)),q95=float(np.quantile(x,.95)))
def safe_corr(x,y):
    return None if np.std(x)<1e-12 or np.std(y)<1e-12 else float(spearmanr(x,y).statistic)
def paired(a,b):
    d=np.array([r['dice'] for r in b])-np.array([r['dice'] for r in a])
    rng=np.random.default_rng(310042)
    boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(2000)])
    return dict(macro_dice_delta=float(d.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        improved=int((d>0).sum()),worsened=int((d<0).sum()),
        gains_over_5pp=int((d>.05).sum()),losses_over_5pp=int((d<-.05).sum()),
        note='Posttraining intervention; image bootstrap is not seed/patient uncertainty.')
def metric_rows(logits,targets,ids):
    pred=logits.sigmoid().numpy()>=.5
    with ThreadPoolExecutor(8) as pool:
        rows=list(pool.map(lambda z:base.boundary_row(*z),zip(pred,targets)))
    for sid,row in zip(ids,rows):row['id']=sid
    metric=base.summarize_rows(rows)
    metric['severe_undercoverage']=sum(r['recall']<.5 for r in rows)
    return metric,rows,pred
def learned_weights_summary(raw,kind,areas,errors):
    t=torch.from_numpy(raw)
    if kind=='scale':
        weights=t.reshape(len(t),2,32).softmax(1)[:,0].numpy()
        identity=.5
    else:
        weights=(2*t.sigmoid()).reshape(len(t),64).numpy()
        identity=1.
    sample_delta=np.sqrt(np.mean((weights-weights.mean(0,keepdims=True))**2,axis=1))
    return dict(values=distribution(weights),mean_absolute_change_from_identity=float(np.abs(weights-identity).mean()),
      between_image_std_per_channel=distribution(weights.std(0)),
      sample_distance_from_average=distribution(sample_delta),
      image_mean_weight_vs_gt_area_spearman=safe_corr(weights.mean(1),areas),
      image_weight_distance_vs_dice_error_spearman=safe_corr(sample_delta,errors),
      fraction_within_001_of_identity=float((np.abs(weights-identity)<.01).mean()),
      mean_weight_by_channel=weights.mean(0).tolist())
@torch.inference_mode()
def audit_model(name,data,ids,perm):
    status('auditing_model',model=name)
    dest=OUT/name;dest.mkdir(exist_ok=True)
    checkpoint=SOURCE/'results'/f'{name}_best_dice.pth'
    state=torch.load(checkpoint,map_location='cpu')
    for rel,sha in state['config']['sources'].items():assert helpers.sha(PROJECT/rel)==sha,rel
    checkpoint_sha=helpers.sha(checkpoint)
    model=System(name).cuda();model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False).eval()
    expected=state['metric'];epoch=state['epoch'];del state
    decoder=model.probe.decoder
    raw={};activations={str(i):[] for i in range(3)}
    head_modules={};handles=[];current={}
    for i,block in enumerate(decoder.fusion):
        for kind,attr in [('channel','eca' if name=='eca_channel' else 'channel_head'),('scale','scale_head')]:
            if not hasattr(block,attr):continue
            key=f'{i}_{kind}';raw[key]=[];head_modules[key]=getattr(block,attr)
            def collect_head(mod,args,output,key=key):
                raw[key].append(output.detach().float().cpu().numpy())
            handles.append(head_modules[key].register_forward_hook(collect_head))
        def branch_hook(mod,args,output,i=i):
            current[i]['dw3']=output
        def branch5_hook(mod,args,output,i=i):
            current[i]['dw5']=output
        def before_fusion(mod,args,i=i):
            current[i]={'context':args[0],'skip':args[1]}
        def after_fusion(mod,args,output,i=i):
            current[i]['residual']=output
        def before_swin(mod,args,i=i):
            q=current[i];r=q['residual'].float();merged=args[0].float();original=merged-r
            a=q['dw3'].float().flatten(1);b=q['dw5'].float().flatten(1)
            rms=lambda z:z.float().flatten(1).square().mean(1).sqrt()
            rr=rms(r);br=rms(original)
            rows=torch.stack([rr,br,rr/br.clamp_min(1e-12),
              rms(q['context']),rms(q['skip']),
              torch.nn.functional.cosine_similarity(a,b,dim=1),
              (a-b).square().mean(1).sqrt() / (((a.square().mean(1)+b.square().mean(1))/2).sqrt().clamp_min(1e-12))],1)
            activations[str(i)].append(rows.cpu().numpy())
            current.pop(i)
        handles.extend([block.register_forward_pre_hook(before_fusion),
            block.dw3.register_forward_hook(branch_hook),block.dw5.register_forward_hook(branch5_hook),
            block.register_forward_hook(after_fusion),decoder.blocks[i].register_forward_pre_hook(before_swin)])
    xs,ys=data;target=ys>=127.5
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    features=[];outputs=[]
    for start in range(0,len(ids),64):
        x=torch.from_numpy(xs[start:start+64]).permute(0,3,1,2).contiguous().cuda().float()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f1,f2,mid=model.prefix((x-mean)/std);f3,f4=model.suffix(mid)
            ff=[f1,f2,f3,f4]
        ff=[f.float() for f in ff]
        logits=model.probe(ff)
        features.append([f.detach() for f in ff]);outputs.append(logits[:,0].float().cpu())
    for h in handles:h.remove()
    baseline_logits=torch.cat(outputs);metric,rows,original=metric_rows(baseline_logits,target,ids)
    metric['difference_from_archived_bf16_dice']=metric['macro']['dice']-expected['macro']['dice']
    old=np.load(SOURCE/'results'/f'{name}_val_predictions.npz')
    assert list(old['ids'])==ids
    metric['changed_pixels_from_archived_bf16']=int(np.count_nonzero(np.unpackbits(old['predictions_packed'],axis=1,count=256*256).reshape(-1,256,256)!=original))
    raw={k:np.concatenate(v) for k,v in raw.items()}
    activation_names=['residual_rms','base_merge_rms','residual_to_base_rms','context_rms','skip_rms','dw3_dw5_cosine','dw3_dw5_relative_distance']
    acts={k:np.concatenate(v) for k,v in activations.items()}
    np.savez_compressed(dest/'attention_and_activations.npz',ids=np.array(ids),
         **raw,**{'activation_'+k:v for k,v in acts.items()})
    areas=target.sum((1,2));errors=1-np.array([r['dice'] for r in rows])
    stats={k:learned_weights_summary(v,k.split('_')[1],areas,errors) for k,v in raw.items()}
    actstats={k:{n:distribution(v[:,j]) for j,n in enumerate(activation_names)} for k,v in acts.items()}
    all_rows=[]
    for i,sid in enumerate(ids):
        for layer,v in acts.items():
            all_rows.append(dict(id=sid,layer=int(layer),**{n:float(v[i,j]) for j,n in enumerate(activation_names)}))
    write_rows(dest/'activation_per_image.csv',all_rows)
    dump(dest/'activation_schema.json',dict(columns=activation_names,layer_sizes={'0':64,'1':32,'2':16},
        denominator='Swin input minus residual, approximates original merge after finite-precision addition'))
    dump(dest/'statistics.json',dict(attention=stats,activations=actstats,epoch=epoch,checkpoint_sha256=checkpoint_sha))
    modes=['channel_identity','scale_equal','joint_permuted','residual_off']
    result=dict(epoch=epoch,checkpoint_sha256=checkpoint_sha,baseline=metric,statistics=stats,activations=actstats,interventions={})
    for mode in modes:
        status('intervention',model=name,mode=mode)
        handles=[];cursor=[0,0]
        for key,module in head_modules.items():
            kind=key.split('_')[1]
            if mode.startswith(kind+'_') or mode=='joint_permuted':
                operation='permuted' if mode=='joint_permuted' else mode.split('_',1)[1]
                saved=torch.from_numpy(raw[key]).cuda()
                def replace_head(mod,args,output,saved=saved,operation=operation):
                    a,b=cursor
                    if operation in ('identity','equal'):return torch.zeros_like(output)
                    if operation=='mean':return saved.mean(0,keepdim=True).expand_as(output).to(output.dtype)
                    if operation=='permuted':return saved[torch.as_tensor(perm[a:b],device='cuda')].to(output.dtype)
                    raise ValueError(operation)
                handles.append(module.register_forward_hook(replace_head))
        decoder.disable_fusion=mode=='residual_off'
        if mode.startswith('residual_off_level'):
            layer=int(mode[-1])
            handles.append(decoder.fusion[layer].register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
        logits=[]
        for batch,ff in enumerate(features):
            cursor[:]=[batch*64,min((batch+1)*64,len(ids))]
            out=model.probe(ff)
            logits.append(out[:,0].float().cpu())
        for h in handles:h.remove()
        decoder.disable_fusion=False
        changed_logits=torch.cat(logits);m,rs,pred=metric_rows(changed_logits,target,ids)
        comparison=paired(rows,rs)
        comparison.update(changed_pixel_fraction=float((pred!=original).mean()),
          images_with_changed_prediction=int(np.any(pred!=original,axis=(1,2)).sum()),
          probability_mean_absolute_change=float((changed_logits.sigmoid()-baseline_logits.sigmoid()).abs().mean()),
          logit_mean_absolute_change=float((changed_logits-baseline_logits).abs().mean()))
        result['interventions'][mode]=dict(metrics=m,comparison=comparison)
        write_rows(dest/f'{mode}_per_image.csv',rs)
        dump(dest/'summary.json',result)
        print('RESULT',name,mode,round(m['macro']['dice']*100,6),'delta_pp',round(comparison['macro_dice_delta']*100,6),flush=True)
        del changed_logits,logits,pred
    # Stored parameters are inspected but no last-epoch inference or selection is done.
    last=torch.load(SOURCE/'results'/f'{name}_last.pth',map_location='cpu')['model']
    norms={}
    for key,value in model.state_dict().items():
        if '.fusion.' in key and any(token in key for token in ['eca.','channel_head.','scale_head.','out.']):
            norms[key]=dict(best_norm=float(value.float().norm()),last_norm=float(last[key].float().norm()),
                           best_to_last_l2=float((last[key].float()-value.float().cpu()).norm()))
    result['parameter_norms']=norms
    dump(dest/'summary.json',result)
    assert helpers.sha(checkpoint)==checkpoint_sha
    del model,features,last,baseline_logits,raw;gc.collect();torch.cuda.empty_cache()
    return result
def main():
    base.setup();OUT.mkdir(exist_ok=True)
    assert not (OUT/'DONE.json').exists()
    partition=json.loads((SOURCE/'identity_partition.json').read_text())
    ids=partition['splits']['val'];assert len(ids)==762
    protocol=json.loads((ROOT/'FP32_CONTROL_PROTOCOL.json').read_text())
    dump(OUT/'provenance.json',dict(protocol=protocol,script_sha256=helpers.sha(Path(__file__)),
        source_module_sha256=helpers.sha(SOURCE/'channel_fusion.py'),partition_sha256=helpers.sha(SOURCE/'identity_partition.json')))
    rng=np.random.default_rng(310042)
    while True:
        perm=rng.permutation(len(ids))
        if np.all(perm!=np.arange(len(ids))):break
    dump(OUT/'permutation.json',dict(seed=310042,indices=perm.tolist(),fixed_points=0))
    data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
    all_results={}
    for name in NAMES:
        all_results[name]=audit_model(name,data,ids,perm)
        dump(OUT/'summary.json',all_results)
    dump(OUT/'DONE.json',dict(complete=True,validation_n=762,models=1,test_read=False,training_run=False,
                             no_weights_modified=True))
    status('complete',models=1,validation_n=762,test_read=False)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);dump(OUT/'FAILURE.json',dict(error=traceback.format_exc()));raise
