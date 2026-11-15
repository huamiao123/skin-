"""Evaluate a fixed validation-selected Stage43 checkpoint with common metrics."""
from pathlib import Path
import argparse,csv,gc,importlib.util,json
import numpy as np
import torch
from train import ROOT,PROJECT,ARMS,System,base,dump,digest
RESEARCH_ROOT=PROJECT.parent.parent
spec=importlib.util.spec_from_file_location('stage43_fixed_eval_helpers',
    RESEARCH_ROOT/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def refrows(path,ids):
    rows=[]
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            rows.append({k:(v if k=='id' else int(v) if k in ['tp','fp','fn','tn'] else float(v)) for k,v in r.items()})
    assert [r['id'] for r in rows]==ids
    return rows
def compare(a,b):
    delta=np.array([r['dice']-s['dice'] for r,s in zip(b,a)])
    gen=np.random.default_rng(42)
    boot=np.array([delta[gen.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
    return dict(macro_dice_delta=float(delta.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
        improved=int((delta>0).sum()),worsened=int((delta<0).sum()),
        gains_over5pp=int((delta>.05).sum()),losses_over5pp=int((delta<-.05).sum()),
        net_reduced_fn=sum(r['fn'] for r in a)-sum(r['fn'] for r in b),
        net_added_fp=sum(r['fp'] for r in b)-sum(r['fp'] for r in a),
        severe_undercoverage=[sum(r['recall']<.5 for r in a),sum(r['recall']<.5 for r in b)],
        note='Single seed; paired image bootstrap is exploratory, not patient/seed uncertainty.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=ARMS)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    base.setup();out=ROOT/args.arm/('smoke' if args.smoke else 'results')
    trained=json.loads((out/'TRAINING_DONE.json').read_text());assert trained['complete']
    config=json.loads((out/'config.json').read_text())
    assert trained['epochs']==config['epochs']==(2 if args.smoke else 100)
    for rel,sha in config['sources'].items():assert digest(PROJECT/rel)==sha,rel
    assert digest(RESEARCH_ROOT/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py')==config['evaluation_helper_sha256']
    if not args.smoke:
        for arm in ARMS:assert json.loads((ROOT/arm/'results/TRAINING_DONE.json').read_text())['complete']
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    references=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    p=out/f'{args.arm}_best_dice.pth';sha=digest(p)
    state=torch.load(p,map_location='cpu');assert state['config']==config
    selection=dict(arm=args.arm,checkpoint=str(p),sha256=sha,epoch=state['epoch'],validation=state['metric'])
    protocol=dict(selection=selection,selection_rule='max_validation_macro_dice_earliest_tie',threshold=.5,
        batch64=True,input_resolution=256,test_already_exposed=True,test_used_for_selection=False,
        patient_disjointness_verified=False,split_sha256=digest(ROOT/'identity_partition.json'))
    dest=out/'FIXED_EVALUATION_PROTOCOL.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol
    else:dump(dest,protocol)
    model=System(args.arm).cuda();model.load_point_state(state['model']);del state
    model.requires_grad_(False).eval()
    summary=dict(arm=args.arm,seed=42,epochs=config['epochs'],selection=selection,validation={},test={},
        comparisons={},test_evaluated=False,complete=False,train_n=1886,validation_n=762,test_n=0,
        normalization='all existing baseline frozen in eval; only point MLP trained',lr_horizon=100)
    for split in (['val'] if args.smoke else ['val','test']):
        ids=partition['splits'][split]
        if split=='val':
            data=helpers.load_data(ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
        else:
            mp=RESEARCH_ROOT/'desktop_backup_20261110/MambaVision_stage22_20261109/test_assets/test_manifest.json'
            assert digest(mp)==partition['audit_sha256']['test_manifest.json']
            manifest={r['id']:r for r in json.loads(mp.read_text())};assert len(ids)==867
            data=helpers.load_data(ids,RESEARCH_ROOT/'heldout_data/test/images',RESEARCH_ROOT/'heldout_data/test/masks',manifest)
            dump(out/'TEST_INPUT_VERIFIED.json',dict(passed=True,n=867,all_hashes_match=True))
        metric,rows=helpers.evaluate(model,data,ids,base,out/f'{args.arm}_{split}_predictions.npz')
        secondary={}
        for policy in ['coarse_direct','coarse_stepwise','random']:
            model.inference_policy=policy
            sm,srows=helpers.evaluate(model,data,ids,base,out/f'{args.arm}_{split}_{policy}_predictions.npz')
            if policy=='coarse_direct':
                expected=references[0]['validation' if split=='val' else 'test']
                assert sm['counts']==expected['counts'] and abs(sm['macro']['dice']-expected['macro']['dice'])<1e-12
            write_rows(out/f'{args.arm}_{split}_{policy}_per_image.csv',srows)
            secondary[policy]=dict(metrics=sm,primary_minus_policy=compare(srows,rows))
        model.inference_policy='uncertain'
        dump(out/f'{split}_SECONDARY_POLICIES.json',secondary)
        key='validation' if split=='val' else 'test'
        if split=='val':
            assert metric['counts']==selection['validation']['counts'],(args.arm,'validation replay counts')
            assert abs(metric['macro']['dice']-selection['validation']['macro']['dice'])<1e-12
            dump(out/'VALIDATION_REPLAY_VERIFIED.json',dict(passed=True,n=762,confusion_counts_identical=True))
        metric.update(epoch=selection['epoch'],checkpoint_sha256=sha)
        summary[key][args.arm]=metric;summary['comparisons'][key]={}
        dump(out/f'{args.arm}_{key}_metrics.json',metric);write_rows(out/f'{args.arm}_{key}_per_image.csv',rows)
        for r in references:
            assert digest(Path(r['checkpoint']))==r['sha256']
            summary[key][r['arm']]=r[key]
            prior=refrows(r['val_rows' if split=='val' else 'test_rows'],ids)
            summary['comparisons'][key][r['arm']+'_vs_'+args.arm]=compare(prior,rows)
        if split=='test':summary.update(test_evaluated=True,test_n=867)
        print('EVALUATED',args.arm,key,'n',len(ids),'macro_dice',metric['macro']['dice'],'pooled_dice',metric['pooled']['dice'],flush=True)
        del data;gc.collect()
    assert digest(p)==sha
    summary['complete']=True
    m=summary['validation'][args.arm];best_ref=max((r['validation'] for r in references),key=lambda m:m['macro']['dice'])
    summary['primary_comparison']='Frozen baseline plus point-only training'
    summary['archived_reference_caveat']='Stage37 selected baseline frozen; additional100head-onlyepochs; not equal-total-training-budget comparison'
    summary['validation_only_screen']=dict(status='defer_to_final_paired_validation',test_used_for_selection=False)
    dump(out/'summary.json',summary)
    dump(out/'DONE.json',dict(complete=True,test_evaluated=not args.smoke,test_n=0 if args.smoke else 867,arm=args.arm))
    dump(out/'run_status.json',dict(state='complete',epochs=config['epochs'],arm=args.arm,test_evaluated=not args.smoke))
if __name__=='__main__':main()

