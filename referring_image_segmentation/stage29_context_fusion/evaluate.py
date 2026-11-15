"""Freeze all validation-selected checkpoints, replay validation, then evaluate test."""
from pathlib import Path
import argparse, csv, gc, importlib.util, json, sys
import numpy as np
import torch
from train import ROOT, PROJECT, NAMES, System, base, dump, digest

RESEARCH_ROOT = PROJECT.parent.parent
old = RESEARCH_ROOT/'test_best_checkpoints_20261111/evaluate_fixed_checkpoints.py'
spec=importlib.util.spec_from_file_location('fixed_test_helpers',old)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def write_rows(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def comparison(rows,other,gt):
    a=np.array([r['dice'] for r in rows]);b=np.array([r['dice'] for r in other]);delta=b-a
    gen=np.random.default_rng(42)
    boot=np.array([delta[gen.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
    areas=(gt>=127.5).sum((1,2))
    cuts=np.quantile(areas,[.25,.5,.75]);group=np.searchsorted(cuts,areas,side='left')
    return dict(macro_dice_delta=float(delta.mean()),image_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
       improved=int((delta>0).sum()),worsened=int((delta<0).sum()),
       gains_over_5pp=int((delta>.05).sum()),losses_over_5pp=int((delta<-.05).sum()),
       severe_undercoverage=[sum(r['recall']<.5 for r in rows),sum(r['recall']<.5 for r in other)],
       corrected_fn=int(sum(r['fn'] for r in rows)-sum(r['fn'] for r in other)),
       added_fp=int(sum(r['fp'] for r in other)-sum(r['fp'] for r in rows)),
       area_quartile_cuts=cuts.tolist(),
       area_quartiles=[dict(n=int((group==i).sum()),macro_dice_delta=float(delta[group==i].mean())) for i in range(4)],
       note='Exploratory image-level comparison; not patient-level or multi-seed uncertainty.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    base.setup();out=ROOT/('smoke' if args.smoke else 'results')
    assert json.loads((out/'TRAINING_DONE.json').read_text())['complete']
    config=json.loads((out/'config.json').read_text())
    for rel,sha in config['sources'].items():assert digest(PROJECT/rel)==sha,rel
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    records=[]
    for name in NAMES:
        p=out/f'{name}_best_dice.pth';s=torch.load(p,map_location='cpu')
        assert s['config']==config
        records.append(dict(arm=name,checkpoint=str(p),checkpoint_sha256=digest(p),epoch=s['epoch'],validation_record=s['metric']))
        del s
    protocol=dict(models=records,partition_sha256=digest(ROOT/'identity_partition.json'),
       threshold=.5,input_resolution=256,selection='validation_macro_dice_earliest_tie',
       evaluation_batch=64,test_already_exposed=True,test_used_for_selection=False,
       test_subset=True,patient_disjointness_verified=False)
    pp=out/'FIXED_EVALUATION_PROTOCOL.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:dump(pp,protocol)
    val_ids=partition['splits']['val']
    val=helpers.load_data(val_ids,PROJECT/'data/isic2018/val/images',PROJECT/'data/isic2018/val/masks')
    summary={'validation':{},'test':{},'comparisons':{},'ablation_validation':{}}
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
        if not args.smoke and name=='anchor_fusion':
            model.probe.decoder.route='local_fusion'
            ablated,_=helpers.evaluate(model,val,val_ids,base)
            summary['ablation_validation']['anchor_weights_using_current_decoder_context']=ablated
            model.probe.decoder.route='anchor_fusion';model.probe.decoder.disable_fusion=True
            ablated,_=helpers.evaluate(model,val,val_ids,base)
            summary['ablation_validation']['anchor_weights_without_residual_branch']=ablated
            summary['ablation_validation']['interpretation']='Post-training interventions only; not retrained architecture comparisons and not checkpoint selection.'
        del model;gc.collect();torch.cuda.empty_cache()
        print('VALIDATION_REPLAY_OK',name,metric['macro']['dice'],flush=True)
    for candidate in NAMES[1:]:
        summary['comparisons']['validation_control_vs_'+candidate]=comparison(all_val['control'],all_val[candidate],val[1])
    summary['comparisons']['validation_local_vs_anchor']=comparison(all_val['local_fusion'],all_val['anchor_fusion'],val[1])
    dump(out/'VALIDATION_REPLAY_VERIFIED.json',dict(passed=True,models=3,counts_identical=True,n=762))
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
        for candidate in NAMES[1:]:
            summary['comparisons']['test_control_vs_'+candidate]=comparison(all_test['control'],all_test[candidate],test[1])
        summary['comparisons']['test_local_vs_anchor']=comparison(all_test['local_fusion'],all_test['anchor_fusion'],test[1])
    for r in records:assert digest(r['checkpoint'])==r['checkpoint_sha256']
    summary.update(complete=True,seed=42,epochs=config['epochs'],test_evaluated=not args.smoke,
                   train_n=1886,validation_n=762,test_n=0 if args.smoke else 867)
    dump(out/'summary.json',summary)
    comparison_rows=[]
    for n in NAMES:
        row=dict(arm=n,epoch=summary['validation'][n]['epoch'])
        for split in ('validation','test'):
            if n in summary[split]:
                m=summary[split][n]
                row.update({split+'_macro_dice':m['macro']['dice'],split+'_pooled_dice':m['pooled']['dice'],
                            split+'_macro_iou':m['macro']['iou'],split+'_severe_undercoverage':m['severe_undercoverage']})
        comparison_rows.append(row)
    write_rows(out/'validation_test_comparison.csv',comparison_rows)
    dump(out/'DONE.json',{k:v for k,v in summary.items() if k not in ['validation','test','comparisons','ablation_validation']})
    dump(out/'run_status.json',dict(state='complete',epochs=config['epochs'],test_evaluated=not args.smoke))
    print('ALL_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
