"""Matched SDI and arithmetic-mean branches, with pre-existing UPer64 as control."""
from pathlib import Path
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch

ROOT=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/decoder_literature_20261119/sdi_formal_20261119')
TMP=Path('/tmp/mv_s_uper64_sdi_formal_20261119')
REF=Path('/home/featurize/work/mambavision_seg_rebaseline_20261116/uper64_feature_audit_20261119')/'runs/add'
sys.path.insert(0,str(ROOT.parent))
from smoke_sdi import FusionUPer
import train_feature as train

def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fixed(path,record):
    if path.exists():
        assert read(path)==record, f'configuration changed: {path}'
    else:train.atomic_json(path,record)
def setup(seed):
    train.reference.setup()
    if seed!=42:
        random.seed(seed);np.random.seed(seed)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def build(arm,seed):
    model=train.build('add',seed=seed)
    if arm!='add':
        assert arm in ('mean','product')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261119+seed-42)
            model.probe.decode_head=FusionUPer(model.probe.decode_head,arm).cuda()
    return model
def run_dir(arm,seed):
    return REF if (arm,seed)==('add',42) else TMP/f'{arm}_{seed}'
def verify_baseline():
    cfg=read(REF/'config.json');comp=read(REF/'complete.json')
    assert (cfg['arm'],cfg['seed'],cfg['batch'],cfg['epochs'])==('add',42,64,100)
    assert cfg['encoder_trainable_stages']==[3,4] and cfg['head_parameters']==1089089
    for file,digest in cfg['source_hashes'].items():assert sha(Path(file))==digest,file
    assert sha(ROOT.parent/'verified_sources/sdi.py')==read(ROOT.parent/'sdi_smoke.json')['source_sha256']
    setup(42);model=build('add',42)
    ck=torch.load(REF/'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']);del ck
    data,_=train.reference.load_data();cache=train.reference.cache_new(762)
    train.reference.fill_cache(model,data,'val',cache)
    sc,rows=train.validate(model,cache,data.ids['val'],None)
    historic=list(csv.DictReader((REF/'best_validation.csv').open()))
    assert [r['id'] for r in historic]==[r['id'] for r in rows]
    assert all(abs(float(a['dice'])-float(b['dice']))<1e-10 for a,b in zip(historic,rows))
    assert abs(sc['val_macro_dice']-comp['best']['val_macro_dice'])<1e-10
    fixed(TMP/'reference_verified.json',dict(epoch=comp['best']['epoch'],validation=sc['val_macro_dice'],checkpoint_sha256=sha(REF/'best.pth'),all_source_hashes_match=True,per_image_replay_n=len(rows)))
def fit(arm,seed):
    dst=run_dir(arm,seed);dst.mkdir(exist_ok=True)
    if (dst/'complete.json').exists():return
    setup(seed);model=build(arm,seed);opt=train.optimizer(model)
    data,_=train.reference.load_data();data.seed=seed
    cfg=dict(arm=arm,seed=seed,ref_sha=sha(REF/'config.json'),adapter_sha=sha(ROOT.parent/'smoke_sdi.py'),
             training_sha=sha(ROOT/'fit_sdi.py'),source_sha=sha(ROOT.parent/'verified_sources/sdi.py'),
             head_parameters=sum(p.numel() for p in model.probe.parameters()),epochs=100,batch=64)
    fixed(dst/'config.json',cfg)
    cache=train.reference.cache_new(1886);val=train.reference.cache_new(762)
    train.reference.fill_cache(model,data,'val',val)
    history=[];last=dst/'latest.pth'
    if last.exists():
        obj=torch.load(last,map_location='cpu',weights_only=False);assert obj['config']==cfg
        model.load_state_dict(obj['model']);opt.load_state_dict(obj['optimizer']);train.restore_rng(obj['rng'])
        history=obj['history'];del obj
    best=max((r['val_macro_dice'] for r in history),default=-1.)
    for epoch in range(len(history)+1,101):
        start=time.monotonic();order,geo=data.geometry(epoch)
        train.reference.fill_cache(model,data,'train',cache,geo)
        factor=.01+.99*(1+math.cos(math.pi*(epoch-1)/100))/2
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        model.train();loss_sum=0.
        for ids in order.split(64):
            loss,norms=train.accumulate(model,opt,cache,ids,64)
            if epoch==1 and loss_sum==0.:
                assert all(math.isfinite(v) and v>0 for v in norms.values())
                if seed==42:assert abs(loss-read(REF/'first_update.json')['loss'])<1e-5,(arm,loss)
                train.atomic_json(dst/'first_update.json',dict(loss=loss,norms=norms))
            loss_sum+=loss*len(ids)
        scores,rows=train.validate(model,val,data.ids['val'],None)
        rec=dict(epoch=epoch,train_loss=loss_sum/len(order),**scores,seconds=round(time.monotonic()-start,3))
        history.append(rec)
        if rec['val_macro_dice']>best:
            best=rec['val_macro_dice']
            train.atomic_save(dst/'best.pth',dict(epoch=epoch,model=train.base.cpu_tree(model.state_dict()),config=cfg,scores=scores))
            train.write_rows(dst/'best_validation.csv',rows)
        if epoch%5==0 or epoch==1:
            train.atomic_save(last,dict(model=train.base.cpu_tree(model.state_dict()),optimizer=train.base.cpu_tree(opt.state_dict()),
                                        rng=train.rng_state(),history=history,config=cfg))
        train.atomic_json(dst/'history.json',history)
        print(arm,seed,epoch,round(rec['val_macro_dice'],6),'best',round(best,6),'sec',rec['seconds'],flush=True)
    ck=torch.load(dst/'best.pth',map_location='cpu',weights_only=False)
    bestrow=max(history,key=lambda x:x['val_macro_dice'])
    assert ck['epoch']==bestrow['epoch']
    model.load_state_dict(ck['model']);del ck
    replay,rows=train.validate(model,val,data.ids['val'],None)
    assert abs(replay['val_macro_dice']-best)<1e-10
    fixed(dst/'complete.json',dict(best=bestrow,epochs=100,test_used=False,replay=replay['val_macro_dice'],sha256=sha(dst/'best.pth')))
def main():
    with (TMP/'train.lock').open('w') as fp:
        fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=dict(candidate_arms=['product','mean'],initial_seed=42,epochs=100,threshold=.5,
                      baseline='MambaVision-S ImageNet-1K + native UPer64; late two stages trainable; BCE + soft Dice',
                      reference_config=read(REF/'config.json'),samples=dict(train=1886,val=762,test=867),
                      primary='mean per-image foreground Dice',selection='earliest highest validation epoch',
                      confirmation='If either candidate exceeds native validation at seed 42, train the highest-validation candidate and native from scratch at seed 43; ties prefer mean. Lock every configuration and best epoch before test. If neither improves, test seed42 only and stop.',
                      test_history='exposed in earlier work; also score pre-existing 814-image group-isolated subset')
        fixed(TMP/'protocol.json',protocol)
        if not (TMP/'reference_verified.json').exists():verify_baseline()
        for arm in ('product','mean'):
            fit(arm,42);gc.collect();torch.cuda.empty_cache()
        ranks={a:read(run_dir(a,42)/'complete.json')['best']['val_macro_dice'] for a in ('add','mean','product')}
        winner=max(('mean','product'),key=ranks.get)
        decision=dict(scores=ranks,leader=winner,confirm=ranks[winner]>ranks['add'])
        fixed(TMP/'validation_decision.json',decision)
        if decision['confirm']:
            for arm in ('add',winner):
                fit(arm,43);gc.collect();torch.cuda.empty_cache()
        fixed(TMP/'fit_done.json',dict(complete=True,decision=decision))
if __name__=='__main__':main()
