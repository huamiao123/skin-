"""Stage30: replay validation, then test each fixed new checkpoint; reuse archived references."""
from pathlib import Path
import argparse,csv,gc,importlib.util,json
import numpy as np
import torch
from train import ROOT,PROJECT,NAMES,System,base,dump,digest
RESEARCH_ROOT=PROJECT.parent.parent
spec=importlib.util.spec_from_file_location('stage30_fixed_eval_helpers',
    RESEARCH_ROOT/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def read_reference_rows(path,ids):
    result=[]
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            result.append({k:(v if k=='id' else int(v) if k in ['tp','fp','fn','tn'] else float(v))
                           for k,v in row.items()})
    assert [r['id'] for r in result]==ids
    return result

def comparison(rows,other,gt):
    a=np.array([r['dice'] for r in rows]);b=np.array([r['dice'] for r in other]);delta=b-a
    gen=np.random.default_rng(42)
    boot=np.array([delta[gen.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
    areas=(gt>=127.5).sum((1,2));cuts=np.quantile(areas,[.25,.5,.75])
    group=np.searchsorted(cuts,areas,side='left')
    return dict(macro_dice_delta=float(delta.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
      improved=int((delta>0).sum()),worsened=int((delta<0).sum()),
      gains_over_5pp=int((delta>.05).sum()),losses_over_5pp=int((delta<-.05).sum()),
      severe_undercoverage=[sum(r['recall']<.5 for r in rows),sum(r['recall']<.5 for r in other)],
      net_reduced_fn=int(sum(r['fn'] for r in rows)-sum(r['fn'] for r in other)),
      net_added_fp=int(sum(r['fp'] for r in other)-sum(r['fp'] for r in rows)),
      area_quartile_cuts=cuts.tolist(),
      area_quartiles=[dict(n=int((group==i).sum()),macro_dice_delta=float(delta[group==i].mean())) for i in range(4)],
      note='Exploratory image bootstrap, not seed/patient uncertainty; historical reference not retrained.')

def pairwise(all_rows,gt):
    pairs=[(r,n) for r in ['control','local_fusion','anchor_fusion'] for n in NAMES]
    pairs+=[('eca_channel','cross_channel'),('cross_channel','cross_channel_scale')]
    return {a+'_vs_'+b:comparison(all_rows[a],all_rows[b],gt) for a,b in pairs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    base.setup();out=ROOT/('smoke' if args.smoke else 'results')
    trained=json.loads((out/'TRAINING_DONE.json').read_text());assert trained['complete']
    config=json.loads((out/'config.json').read_text())
    assert trained['epochs']==config['epochs']==(2 if args.smoke else 100)
    assert digest(RESEARCH_ROOT/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py')==config['evaluation_helper_sha256']
    for rel,sha in config['sources'].items():assert digest(PROJECT/rel)==sha,rel
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    references=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models']
    for record in references:assert digest(Path(record['checkpoint']))==record['sha256']
    records=[]
    for name in NAMES:
        p=out/f'{name}_best_dice.pth';s=torch.load(p,map_location='cpu')
        assert s['config']==config
        records.append(dict(arm=name,checkpoint=str(p),checkpoint_sha256=digest(p),
                            epoch=s['epoch'],validation_record=s['metric']))
        del s
    protocol=dict(models=records,references=[dict(arm=x['arm'],checkpoint=x['checkpoint'],
           checkpoint_sha256=x['sha256']) for x in references],epochs=config['epochs'],
       partition_sha256=digest(ROOT/'identity_partition.json'),threshold=.5,input_resolution=256,
       selection='validation_macro_dice_earliest_tie',evaluation_batch=64,
       test_already_exposed=True,test_used_for_selection=False,test_subset=True,
       patient_disjointness_verified=False,lr_schedule_horizon=300)
    pp=out/'FIXED_EVALUATION_PROTOCOL.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:dump(pp,protocol)
    val_ids=partition['splits']['val']
    val=helpers.load_data(val_ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
    summary=dict(validation={},test={},comparisons={},ablation_validation={})
    all_val={}
    for record in records:
        name=record['arm'];model=System(name).cuda()
        state=torch.load(record['checkpoint'],map_location='cpu');model.load_state_dict(state['model'],strict=True)
        del state;model.requires_grad_(False).eval()
        metric,rows=helpers.evaluate(model,val,val_ids,base,out/f'{name}_val_predictions.npz')
        assert metric['counts']==record['validation_record']['counts'],(name,'validation replay mismatch')
        assert abs(metric['macro']['dice']-record['validation_record']['macro']['dice'])<1e-12
        metric.update(epoch=record['epoch'],checkpoint_sha256=record['checkpoint_sha256'])
        summary['validation'][name]=metric;all_val[name]=rows
        dump(out/f'{name}_validation_metrics.json',metric);write_rows(out/f'{name}_validation_per_image.csv',rows)
        if not args.smoke and name=='cross_channel_scale':
            for block in model.probe.decoder.fusion:block.disable_scale=True
            ablated,_=helpers.evaluate(model,val,val_ids,base)
            summary['ablation_validation']['fixed_equal_scale_weights']=ablated
            for block in model.probe.decoder.fusion:
                block.disable_scale=False;block.disable_channel=True
            ablated,_=helpers.evaluate(model,val,val_ids,base)
            summary['ablation_validation']['channel_weights_fixed_to_one']=ablated
            summary['ablation_validation']['interpretation']='Post-training interventions on validation only; not independently retrained controls and not selection.'
        del model;gc.collect();torch.cuda.empty_cache()
        print('VALIDATION_REPLAY_OK',name,metric['macro']['dice'],flush=True)
    for r in references:
        summary['validation'][r['arm']]=r['validation']
        all_val[r['arm']]=read_reference_rows(r['val_rows'],val_ids)
    summary['comparisons']['validation']=pairwise(all_val,val[1])
    dump(out/'VALIDATION_REPLAY_VERIFIED.json',dict(passed=True,new_models=3,reused_references=3,
         new_confusion_counts_identical=True,reference_preflight_replay=True,n=762))
    del val
    if not args.smoke:
        manifest_path=RESEARCH_ROOT/'desktop_backup_20261110/MambaVision_stage22_20261109/test_assets/test_manifest.json'
        assert digest(manifest_path)==partition['audit_sha256']['test_manifest.json']
        manifest={r['id']:r for r in json.loads(manifest_path.read_text())}
        test_ids=partition['splits']['test'];assert len(test_ids)==867
        test=helpers.load_data(test_ids,RESEARCH_ROOT/'heldout_data/test/images',RESEARCH_ROOT/'heldout_data/test/masks',manifest)
        dump(out/'TEST_INPUT_VERIFIED.json',dict(passed=True,n=867,all_hashes_match=True))
        all_test={}
        for record in records:
            name=record['arm'];model=System(name).cuda()
            state=torch.load(record['checkpoint'],map_location='cpu');model.load_state_dict(state['model'],strict=True)
            del state;model.requires_grad_(False).eval()
            metric,rows=helpers.evaluate(model,test,test_ids,base,out/f'{name}_test_predictions.npz')
            metric.update(epoch=record['epoch'],checkpoint_sha256=record['checkpoint_sha256'])
            summary['test'][name]=metric;all_test[name]=rows
            dump(out/f'{name}_test_metrics.json',metric);write_rows(out/f'{name}_test_per_image.csv',rows)
            print('TEST_RESULT',name,json.dumps(metric),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        for r in references:
            summary['test'][r['arm']]=r['test']
            all_test[r['arm']]=read_reference_rows(r['test_rows'],test_ids)
        summary['comparisons']['test']=pairwise(all_test,test[1])
    for r in records:assert digest(Path(r['checkpoint']))==r['checkpoint_sha256']
    for r in references:assert digest(Path(r['checkpoint']))==r['sha256']
    summary.update(complete=True,seed=42,epochs=config['epochs'],lr_schedule_horizon=300,
                   test_evaluated=not args.smoke,train_n=1886,validation_n=762,
                   test_n=0 if args.smoke else 867,newly_trained_models=3,reused_reference_models=3)
    if not args.smoke:
        anchor=summary['validation']['anchor_fusion'];local=summary['validation']['local_fusion']
        summary['development_screen']={}
        for name in NAMES:
            m=summary['validation'][name]
            summary['development_screen'][name]=dict(
               macro_dice_delta_vs_local=m['macro']['dice']-local['macro']['dice'],
               fp_delta_vs_anchor=m['counts']['fp']-anchor['counts']['fp'],
               recall_delta_vs_anchor=m['macro']['recall']-anchor['macro']['recall'],
               severe_delta_vs_local=m['severe_undercoverage']-local['severe_undercoverage'],
               qualifies_for_seed_confirmation=bool(
                   m['macro']['dice']>=local['macro']['dice']+.002 and
                   m['counts']['fp']<=anchor['counts']['fp']*.95 and
                   m['macro']['recall']>=anchor['macro']['recall']-.005 and
                   m['severe_undercoverage']<=local['severe_undercoverage']))
    dump(out/'summary.json',summary)
    table=[]
    for n in ['control','local_fusion','anchor_fusion',*NAMES]:
        row=dict(arm=n,origin='new' if n in NAMES else 'reused_stage29',
                 epoch=summary['validation'][n]['epoch'])
        for split in ('validation','test'):
            if n in summary[split]:
                m=summary[split][n]
                row.update({split+'_macro_dice':m['macro']['dice'],split+'_pooled_dice':m['pooled']['dice'],
                     split+'_macro_iou':m['macro']['iou'],split+'_macro_precision':m['macro']['precision'],
                     split+'_macro_recall':m['macro']['recall'],split+'_severe_undercoverage':m['severe_undercoverage']})
        table.append(row)
    write_rows(out/'validation_test_comparison.csv',table)
    dump(out/'DONE.json',{k:v for k,v in summary.items() if k not in
         ['validation','test','comparisons','ablation_validation','development_screen']})
    dump(out/'run_status.json',dict(state='complete',epochs=config['epochs'],test_evaluated=not args.smoke))
    print('ALL_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
