"""Three-seed paired development comparison and explicit stage-C stop decision."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=(17,23,42)
MODELS=('CNN','G1_zero_reconstruction','local','local_large','local8','medium32','global')
METRICS=('dice','iou','bf1','hd95','assd','moved_fraction','repair_precision','repair_recall','keep_rate_on_initially_correct','move_rate_on_initially_correct','distance_increase_rate','correct_to_incorrect_rate')
COMPARISONS=(('global','CNN'),('global','G1_zero_reconstruction'),('global','local_large'),('global','local8'),('global','medium32'),('local_large','CNN'))

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    all_rows=[]
    for seed in SEEDS:
        file=ROOT/'results/transformer'/f'chosen_keep_seed{seed}'/'per_image.csv'
        rows=read(file)
        seen={(r['role'],r['model'],r['image_id']) for r in rows}
        if len(rows)!=len(seen) or any(sum(r['role']=='locked_verification100' and r['model']==m for r in rows)!=100 for m in MODELS):
            raise RuntimeError(f'Incomplete or duplicated val100 seed {seed}')
        all_rows.extend(r for r in rows if r['role']=='locked_verification100')
    ids=sorted({r['image_id'] for r in all_rows})
    if len(ids)!=100:raise RuntimeError('Expected 100 reused development images')
    lookup={(int(r['seed']),r['model'],r['image_id']):r for r in all_rows}
    summary={'scope':'ISIC2017 official val100, previously exposed to the public CNN and phase1; all val150 also selected checkpoints in this run; exploratory development only',
             'seeds':SEEDS,'n_images':100,'official_test_images_or_GT_opened':0,
             'model_means':{},'paired_comparisons':{},'oracle_context':{},'decision':None}
    for model in MODELS:
        seed_stats=[]
        for seed in SEEDS:
            group=[lookup[(seed,model,i)] for i in ids]
            values={'seed':seed}
            for metric in METRICS:
                present=[float(r[metric]) for r in group if r.get(metric) not in (None,'')]
                values[metric]=float(np.mean(present)) if present else None
            values['alpha']=None if model in ('CNN','G1_zero_reconstruction') else float(group[0]['selected_alpha'])
            seed_stats.append(values)
        payload={'per_seed':seed_stats}
        for metric in METRICS:
            numbers=np.array([r[metric] for r in seed_stats if r[metric] is not None],dtype=float)
            payload[metric+'_mean']=float(numbers.mean()) if len(numbers) else None
            payload[metric+'_seed_std']=float(numbers.std(ddof=1)) if len(numbers)>1 else None
        summary['model_means'][model]=payload
    rng=np.random.default_rng(17)
    draws=rng.integers(0,len(ids),size=(2000,len(ids)))
    for a,b in COMPARISONS:
        for metric in ('dice','bf1','hd95'):
            # Average each image's paired difference across seeds before image bootstrap.
            per_image=np.array([np.mean([float(lookup[(s,a,i)][metric])-float(lookup[(s,b,i)][metric]) for s in SEEDS]) for i in ids])
            samples=per_image[draws].mean(axis=1)
            summary['paired_comparisons'][f'{a}_minus_{b}_{metric}']={
                'mean':float(per_image.mean()),'median':float(np.median(per_image)),
                'ci95':[float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))],
                'improved_images':int((per_image>0).sum()) if metric!='hd95' else int((per_image<0).sum()),
                'harmed_images':int((per_image<0).sum()) if metric!='hd95' else int((per_image>0).sum()),
                'equal_images':int((per_image==0).sum()),
                'resampling_unit':'paired image, averaging three seeds within image',
                'bootstrap_seed':17,'bootstrap_repetitions':2000}
    oracle=read(ROOT/'results/candidates/radius32_oracle_per_image.csv')
    valids={r['image_id'] for r in all_rows}
    part=[r for r in oracle if r['image_id'] in valids]
    if len(part)!=100:raise RuntimeError('R32 oracle image IDs differ')
    summary['oracle_context']={'n_images':100,'r32_dense_oracle_dice_mean':float(np.mean([float(r['r32_dense_oracle_dice']) for r in part])),
                              'r32_dense_oracle_bf1_mean':float(np.mean([float(r['r32_dense_oracle_bf1']) for r in part])),
                              'GT_used_to_choose_candidates':True,'deployable':False}
    global_vs_large=summary['paired_comparisons']['global_minus_local_large_dice']
    global_vs_local8=summary['paired_comparisons']['global_minus_local8_dice']
    global_vs_cnn=summary['paired_comparisons']['global_minus_CNN_dice']
    global_bf1=summary['paired_comparisons']['global_minus_CNN_bf1']
    global_hd95=summary['paired_comparisons']['global_minus_CNN_hd95']
    individual=[summary['model_means']['global']['per_seed'][j]['dice']>summary['model_means']['local_large']['per_seed'][j]['dice']
                and summary['model_means']['global']['per_seed'][j]['dice']>summary['model_means']['local8']['per_seed'][j]['dice']
                and summary['model_means']['global']['per_seed'][j]['dice']>summary['model_means']['CNN']['per_seed'][j]['dice']
                for j in range(3)]
    passes=all(individual) and global_vs_large['ci95'][0]>0 and global_vs_local8['ci95'][0]>0 and global_vs_cnn['ci95'][0]>0 and global_bf1['mean']>0 and global_hd95['mean']<0
    summary['decision']={'status':'PROCEED_TO_INDEPENDENT_TEST' if passes else 'HOLD_GLOBAL_PRIMARY_NO_INDEPENDENT_TEST',
                         'strict_success_rule_met':passes,
                         'rule_source':'taskbook section 15 success and section 16 failure conditions',
                         'per_seed_global_beats_CNN_local8_local_large':individual,
                         'paired_CI_global_vs_strong_local_positive':global_vs_large['ci95'][0]>0,
                         'paired_CI_global_vs_local_attention_positive':global_vs_local8['ci95'][0]>0,
                         'CNN_BF1_and_HD95_direction_both_improved':global_bf1['mean']>0 and global_hd95['mean']<0,
                         'test_access_after_decision':False,
                         'caution':'Phase1 reused val150, and this run selected all model checkpoints on val150 before cal50 penalty tuning. All CIs exploratory; failure of the strict rule does not establish zero global-context benefit.'}
    out=ROOT/'results/phase2_summary';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'model_seed_table.csv').open('w',newline='') as f:
        fieldnames=['model','seed','alpha',*METRICS]
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for model in MODELS:
            for row in summary['model_means'][model]['per_seed']:
                w.writerow({'model':model,**row})
    print(json.dumps({'decision':summary['decision'],'model_dice':{m:summary['model_means'][m]['dice_mean'] for m in MODELS},
                      'global_minus_large':global_vs_large,'global_minus_local8':global_vs_local8},indent=2),flush=True)

if __name__=='__main__':main()
