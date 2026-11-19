"""Locked paired evaluation of two existing MambaVision-S checkpoints and their fixed probability mean."""
import csv
import fcntl
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from analyze import SEED,setup,train,model_for,read,sha,fixed
from fit_sdi import build,REF,ROOT as SDIROOT
from fit_bypass import START
from test_bypass import test_inputs,GROUPS,REFERENCE
from score_sdi import paired

def outcomes(prob,gt):
    is_fg=prob>=.5
    tp=(is_fg & gt).sum((1,2));fp=(is_fg & ~gt).sum((1,2));fn=(~is_fg & gt).sum((1,2));den=2*tp+fp+fn
    dice=torch.where(den>0,2*tp.double()/den,1.)
    return {'dice':dice,'tp':tp,'fp':fp,'fn':fn}

def compare(rows,groups):
    plain=np.array([r['plain_dice'] for r in rows]);sdi=np.array([r['sdi_dice'] for r in rows]);mix=np.array([r['ensemble_dice'] for r in rows])
    out={}
    for name,sel in [('all',np.ones(len(rows),bool)),('filtered',np.array([g['retain'] for g in groups],dtype=bool))]:
        gr=np.array([g['group'] for g in groups])[sel];p=plain[sel];s=sdi[sel];e=mix[sel]
        names,ix=np.unique(gr,return_inverse=True);N=len(names);count=np.bincount(ix)
        # Resample known groups together for both differences, including when the better endpoint changes across replicates.
        rng=np.random.default_rng(42);pair_draw=[]
        gains=np.stack([np.bincount(ix,weights=e-p),np.bincount(ix,weights=e-s)])
        for i in range(5000):
            b=rng.integers(N,size=N)
            pair_draw.append(gains[:,b].sum(1).min()/count[b].sum())
        out[name]={'n':int(sel.sum()),'groups':N,'dice':{'plain':float(p.mean()),'sdi':float(s.mean()),'ensemble':float(e.mean())},
                   'vs_plain':paired(p,e,gr),'vs_sdi':paired(s,e,gr),
                   'vs_best':dict(delta=float(e.mean()-max(s.mean(),p.mean())),paired_group_bootstrap95=np.quantile(pair_draw,[.025,.975]).tolist())}
    return out
@torch.inference_mode()
def main():
    prior=read(SDIROOT/'runs/locked_before_test.json')['models']
    path_plain=REF/'best.pth';path_sdi=START
    assert sha(path_plain)==prior['add_42']['sha256'] and sha(path_sdi)==prior['product_42']['sha256']
    partition=train.PROJECT/'stage45_upernet_decoder/identity_partition.json'
    protocol=dict(weights={'plain':dict(sha=sha(path_plain),epoch=prior['add_42']['epoch'],path=str(path_plain)),'sdi':dict(sha=sha(path_sdi),epoch=prior['product_42']['epoch'],path=str(path_sdi))},seed=SEED,
       design='plain native UPer64 with its own encoder; product SDI weak2 with its own encoder; 0.5*(p_plain+p_sdi) after sigmoid in FP32, threshold >=0.5; three outputs only. No fitting or choice of ensemble weight.',
       endpoint='test per-image foreground Dice, averaged equally over all cases; ensemble minus each endpoint and minus better endpoint; known-group paired bootstrap (5000, seed 42)',
       secondary='814 prespecified retained cases using existing group map. Shared subset, not an independent replication.',
       source=sha(HERE/'score.py'),dev_design=sha(HERE.parent/'decoder_protocol.json'),dev_source=sha(HERE.parent/'decoder_check.py'),
       reference_files=[sha(REFERENCE),sha(SDIROOT/'runs/test_per_image.csv')],group_sha=sha(GROUPS),partition_sha=sha(partition),
       inference_impl=sha(Path(sys.modules['fit_bypass'].__file__)),split_history='test previously used. No alternative outputs or threshold sweeps.')
    fixed(HERE/'locked_before_test.json',protocol)
    q=read(partition);ids=q['splits']['test'];groups=read(GROUPS)['cases']['test']
    with REFERENCE.open() as f:old_sdi=list(csv.DictReader(f))
    with (SDIROOT/'runs/test_per_image.csv').open() as f:old_plain=list(csv.DictReader(f))
    assert ids==[r['id'] for r in groups]==[r['id'] for r in old_plain]==[r['id'] for r in old_sdi] and len(set(ids))==867
    x,y=test_inputs(q);setup(SEED)
    sdi=model_for('control').eval()
    plain=build('add',SEED).eval()
    st=torch.load(path_plain,map_location='cpu',weights_only=False)
    assert st['epoch']==prior['add_42']['epoch'];plain.load_state_dict(st['model']);del st
    mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    rows=[]
    for image,g in zip(x.split(64),y.split(64)):
        image=(image.cuda().float()/255.-mu)/sd;g=g.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a=plain(image)['main'][:,0].float().sigmoid();b=sdi(image)['main'][:,0].float().sigmoid()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        predictions={'plain':a,'sdi':b,'ensemble':.5*(a+b)}
        data={name+'_'+k:v.cpu().numpy() for name,p in predictions.items() for k,v in outcomes(p,g).items()}
        start=len(rows)
        for j in range(len(g)):
            i=start+j;row={k:float(v[j]) if k.endswith('_dice') else int(v[j]) for k,v in data.items()}
            assert abs(row['plain_dice']-float(old_plain[i]['add_42']))<1e-12
            assert abs(row['sdi_dice']-float(old_sdi[i]['weak2']))<1e-12
            rows.append(dict(id=ids[i],group=groups[i]['group'],retain=groups[i]['retain'],**row))
    result=compare(rows,groups);assert result['all']['n']==867 and result['filtered']['n']==814
    train.write_rows(HERE/'test_per_image.csv',rows)
    train.atomic_json(HERE/'RESULTS.json',dict(test=result,verified_reference_cases={'plain':len(rows),'sdi':len(rows)},protocol_sha=sha(HERE/'locked_before_test.json')))
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    with (HERE/'evaluation.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
