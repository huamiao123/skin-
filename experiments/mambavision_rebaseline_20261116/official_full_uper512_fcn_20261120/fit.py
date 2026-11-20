"""Full-network binary fine-tuning. All sets start from the exact epoch-zero artifact."""
import argparse
import gc
import math
import time
from official import *
evaluation=runpy.run_path(str(ROOT/'evaluate.py'),run_name='full_segmentation_evaluation')
measure=evaluation['measure']
result=evaluation['result']

def optimizer(net):
    groups=[dict(name='encoder',params=list(net.backbone.parameters()),lr=1e-4),
            dict(name='decoder',params=list(net.probe.parameters()),lr=1e-3)]
    for group in groups:group['initial_lr']=group['lr']
    return torch.optim.AdamW(groups,weight_decay=.01)

def gradient_norm(module):
    grads=[p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    if not grads:raise RuntimeError('No gradients')
    norm=torch.stack(grads).norm().item()
    if not (math.isfinite(norm) and norm>0):raise RuntimeError(f'Invalid gradient norm: {norm}')
    return norm

def checked_groups(net):
    return {'stem':net.backbone.patch_embed,
            **{f'stage{i+1}':m for i,m in enumerate(net.backbone.levels)},
            **{f'outnorm{i}':getattr(net.backbone,f'outnorm{i}') for i in range(4)},
            'uper':net.probe.decode_head,'fcn':net.probe.auxiliary_head}

def update(net,opt,data,batch,geometry,micro,check=False):
    opt.zero_grad(set_to_none=True)
    total=torch.zeros((),device='cuda')
    pieces=list(batch.split(micro))
    if len(pieces)>1 and len(pieces[-1])==1:
        pair=torch.cat(pieces[-2:])
        pieces[-2:]=[pair[:-2],pair[-2:]]
    for ix in pieces:
        x,y=data.batch('train',ix,geometry)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=net(x)
        assert 'aux' in out
        loss,detail=objective(out,y,aux_weight=.4)
        assert detail['aux'] is not None
        if not torch.isfinite(loss):raise FloatingPointError('training loss')
        (loss*(len(ix)/len(batch))).backward()
        total+=loss.detach()*len(ix)
    audit=None
    if check:
        modules=checked_groups(net)
        assert all(m.training for m in modules.values())
        norms={k:gradient_norm(m) for k,m in modules.items()}
        probes={}
        for k,m in modules.items():
            weight=next(p for p in m.parameters() if p.grad is not None and p.grad.detach().abs().amax().item()>0)
            pos=weight.grad.detach().abs().reshape(-1).argmax()
            probes[k]=(weight,pos,weight.detach().reshape(-1)[pos].item())
    for group in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'],1.,error_if_nonfinite=True)
    opt.step()
    if check:
        deltas={k:abs(p.detach().reshape(-1)[idx].item()-v) for k,(p,idx,v) in probes.items()}
        assert all(v>0 for v in deltas.values()),deltas
        audit=dict(gradient_norms=norms,absolute_parameter_updates=deltas,aux_loss_weight=.4,
                   effective_batch=len(batch),microbatch=micro,loss_finite=True)
    return total.item(),audit

def run(ds,micro):
    start_sha=json.loads((ROOT/'initialization.json').read_text())['initial_sha256']
    for key in DATASETS:
        zero=json.loads((ROOT/key/'epoch0_test.json').read_text())
        assert zero['model_sha256']==start_sha and zero['epoch']==0
        assert zero['per_image_sha256']==sha(ROOT/key/'epoch0_test_per_image.csv')
    folder=ROOT/ds
    cfg=dict(dataset=ds,init_sha256=start_sha,split_sha256=SPLITS[ds]['split_sha256'],seed=42,
             max_epochs=100,patience=25,batch=64,microbatch=micro,aux_weight=.4,
             learning_rate_encoder=1e-4,learning_rate_decoder=1e-3,weight_decay=.01,schedule='cosine-to-1-percent',
             windows=[16,8],size=256,full_encoder_train_mode=True,metric='mean per-image foreground Dice at 0.5')
    if (folder/'config.json').exists():assert cfg==json.loads((folder/'config.json').read_text())
    write(folder/'config.json',cfg)
    if (folder/'final_test.json').exists():return
    net=load_model()
    opt=optimizer(net)
    train=dataset(ds,['train','val'])
    latest=folder/'resume.pth';hist=[];best=None
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        assert ck['config']==cfg
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng']);hist=ck['history'];best=ck['best'];del ck
    for epoch in range(len(hist)+1,101):
        if best and epoch-1-best['epoch']>=25:break
        start=time.monotonic();net.train();torch.cuda.reset_peak_memory_stats()
        order,geometry=train.geometry(epoch)
        factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        loss_sum=0.
        for step,ix in enumerate(recipe.batches(order)):
            loss,diag=update(net,opt,train,ix,geometry,micro,check=(epoch==1 and step==0))
            loss_sum+=loss
            if diag:
                write(folder/'first_update.json',diag)
                print('VERIFIED REAL UPDATE',ds,epoch,flush=True)
        score,_=measure(net,train,'val')
        row=dict(epoch=epoch,train_loss=loss_sum/len(order),validation_macro_dice=score,
                 compute_s=round(time.monotonic()-start,2),peak_gib=round(torch.cuda.max_memory_allocated()/1024**3,3))
        hist.append(row)
        improved=not best or score>best['score']
        if improved:
            name=f'best_epoch_{epoch}.pth'
            save(folder/name,net.state_dict())
            best=dict(epoch=epoch,score=score,file=name)
        finished=epoch==100 or epoch-best['epoch']>=25
        if improved or epoch%5==0 or finished:
            save(latest,dict(config=cfg,model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),history=hist,best=best))
            for old in folder.glob('best_epoch_*.pth'):
                if old.name!=best['file']:old.unlink()
        write(folder/'history.json',hist)
        print('EPOCH',ds,epoch,'best_epoch',best['epoch'],'seconds',round(time.monotonic()-start,1),flush=True)
        if finished:break
    del opt;net.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    ckpt=folder/best['file']
    net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
    replay,_=measure(net,train,'val');assert abs(replay-best['score'])<1e-10,(replay,best)
    digest=sha(ckpt);write(folder/'fit_complete.json',dict(epochs=len(hist),best_epoch=best['epoch'],best_checkpoint=best['file'],sha256=digest))
    result(net,ds,'final_test',best['epoch'],digest)
    latest.unlink(missing_ok=True)

def smoke(micro):
    net=load_model();net.train()
    data=dataset('ISIC2018',['train']);order,geom=data.geometry(1)
    torch.cuda.reset_peak_memory_stats()
    _,audit=update(net,optimizer(net),data,order[:64],geom,micro,check=True)
    audit['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    write(ROOT/'batch_probe.json',audit)
    print('BATCH PROBE OK',micro,audit['peak_allocated_gib'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe',*DATASETS])
    parser.add_argument('--micro',type=int)
    args=parser.parse_args()
    if args.action=='probe':
        try:smoke(args.micro)
        except torch.cuda.OutOfMemoryError:sys.exit(42)
    else:
        run(args.action,json.loads((ROOT/'batch_probe.json').read_text())['microbatch'])
