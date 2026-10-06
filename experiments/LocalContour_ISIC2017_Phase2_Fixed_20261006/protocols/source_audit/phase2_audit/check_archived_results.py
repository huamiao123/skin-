"""Recompute archived result summaries without images or checkpoints."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();p=a.repo/'experiments/LocalContour_ISIC2017_Phase2_20261005'
    errors=[];out={'scope':'Archive consistency checks, NOT replay of training or image inference'}
    splitmeta=json.loads((p/'protocols/split_provenance.json').read_text())
    ids={s:set((p/'splits'/f'{s}.txt').read_text().splitlines()) for s in ('train','val','test')}
    out['split_counts']={s:len(v) for s,v in ids.items()}
    out['split_hash_matches']={s:hashlib.sha256((p/'splits'/f'{s}.txt').read_bytes()).hexdigest()==splitmeta['sha256'][f'{s}.txt'] for s in ids}
    out['split_overlap_counts']={f'{x}_{y}':len(ids[x]&ids[y]) for x,y in [('train','val'),('train','test'),('val','test')]}
    fm=json.loads((p/'results/cache_provenance/feature_cache_COMPLETE.json').read_text())
    out['archived_feature_records']={'count':len(fm['records']),'splits':sorted({r['split'] for r in fm['records']}),'gt_shape_verified_for_all':fm.get('gt_shape_verified_for_all')}
    final=json.loads((p/'results/phase2_summary/summary.json').read_text())
    allrows=[];maxerr=0.
    for seed in (17,23,42):
        rows=read(p/f'results/transformer/chosen_keep_seed{seed}/per_image.csv')
        sm=json.loads((p/f'results/transformer/chosen_keep_seed{seed}/summary.json').read_text())
        for mod,metrics in sm['models'].items():
            group=[r for r in rows if r['model']==mod and r['role']=='locked_verification100']
            if len(group)!=100:errors.append(f'count {seed} {mod}')
            for met in ('dice','iou','bf1','hd95','assd'):
                maxerr=max(maxerr,abs(float(np.mean([float(r[met]) for r in group]))-metrics[met]))
        allrows.extend(r for r in rows if r['role']=='locked_verification100')
    out['maximum_per_seed_summary_absolute_difference']=maxerr
    out['recomputed_main_table']={}
    for mod in final['model_means']:
        group=[r for r in allrows if r['model']==mod]
        result={m:float(np.mean([float(r[m]) for r in group])) for m in ('dice','iou','bf1','hd95','assd')}
        result['delta_dice_percentage_points_vs_CNN']=(result['dice']-final['model_means']['CNN']['dice_mean'])*100
        out['recomputed_main_table'][mod]=result
        for met in ('dice','iou','bf1','hd95','assd'):
            if abs(result[met]-final['model_means'][mod][met+'_mean'])>1e-12:errors.append(f'main mean {mod} {met}')
    manifest=read(p/'results/training_records/training_manifest.csv')
    logerrors=[]
    for row in manifest:
        f=p/'results/training_records'/f"{row['family']}_seed{row['seed']}_training_log.csv"
        if hashlib.sha256(f.read_bytes()).hexdigest()!=row['training_log_sha256']:logerrors.append('hash '+f.name)
        best=min(read(f),key=lambda r:float(r['val_loss']))
        if best['epoch']!=row['best_epoch']:logerrors.append('epoch '+f.name)
    out['training_logs']={'count':len(manifest),'hash_or_best_epoch_errors':logerrors,
                          'checkpoints_not_present':'Actual checkpoint tensors cannot be hash-verified from this ZIP'}
    zero_rows=read(p/'results/transformer/gated_seed17/per_image.csv')
    cnn={r['image_id']:r for r in zero_rows if r['model']=='CNN' and r['role']=='locked_verification100'}
    gate={r['image_id']:r for r in zero_rows if r['model']=='global_gate' and r['role']=='locked_verification100'}
    diffs=np.array([float(gate[i]['dice'])-float(cnn[i]['dice']) for i in cnn])
    out['zero_motion_vs_original_CNN_on_val100']={
        'moved_nodes_fraction':max(float(r['moved_fraction']) for r in gate.values() if r['moved_fraction']),
        'number_of_images_with_nonzero_Dice_difference':int((diffs!=0).sum()),
        'mean_Dice_difference':float(diffs.mean()),'maximum_absolute_Dice_difference':float(abs(diffs).max()),
        'CNN_mean_Dice':float(np.mean([float(r['dice']) for r in cnn.values()])),
        'zero_motion_mean_Dice':float(np.mean([float(r['dice']) for r in gate.values()]))}
    candidates=read(p/'results/candidates/per_image.csv');table=read(p/'results/candidates/summary.csv')
    cerrors=[]
    for row in table:
        group=[r for r in candidates if r['config']==row['config']]
        coverage=sum(int(r['covered_nodes']) for r in group)/sum(int(r['effective_nodes']) for r in group)
        oracle=float(np.mean([float(r['oracle_dice']) for r in group]))
        if abs(coverage-float(row['candidate_coverage_pooled']))>1e-12:cerrors.append('coverage '+row['config'])
        if abs(oracle-float(row['oracle_dice_mean']))>1e-12:cerrors.append('oracle mean '+row['config'])
    out['candidate_table_consistency_errors']=cerrors
    out['errors']=errors+logerrors+cerrors
    out['status']='PASS_ARCHIVE_ARITHMETIC_ONLY' if not out['errors'] else 'FAIL'
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
