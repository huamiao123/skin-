"""Aggregate three-seed exact-DP controls and explicit-gate ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
SEEDS=(17,23,42)

def main():
    dp=[];gate=[]
    for seed in SEEDS:
        for family,label,path in [
            ('local','StrongLocal_DP',ROOT/'results/local'/f'seed{seed}/summary.json'),
            ('local_large','LocalLarge_DP',ROOT/'results/transformer/local_large'/f'seed{seed}/summary.json'),
            ('global','GlobalTransformer_DP',ROOT/'results/transformer/global'/f'seed{seed}/summary.json')]:
            summary=json.loads(path.read_text())
            method=summary['methods'][label]
            dp.append({'seed':seed,'method':family+'_DP','parameters':summary.get('model_parameters',summary.get('local_parameters')),
                       'smooth_lambda':summary['best_dp']['smooth_lambda'],
                       'displacement_penalty':summary['best_dp']['displacement_penalty'],
                       'dice':method['dice'],'bf1':method['bf1'],'hd95':method['hd95']})
        gated=json.loads((ROOT/'results/transformer'/f'gated_seed{seed}'/'summary.json').read_text())
        for family in ('local8_gate','global_gate'):
            model=gated['models'][family]
            gate.append({'seed':seed,'method':family,'threshold':model['threshold'],
                         'dice':model['dice'],'bf1':model['bf1'],'hd95':model['hd95'],
                         'moved_fraction':model['moved_fraction']})
    output=ROOT/'results/phase2_summary';output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('dp_comparison.csv',dp),('gated_comparison.csv',gate)]:
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={'scope':'previously exposed official ISIC2017 val150 for DP hyperparameter selection/readout; gate threshold cal50 and val100 reused development verification',
             'official_test_images_or_GT_opened':0,
             'dp_mean':{family:{metric:float(np.mean([r[metric] for r in dp if r['method']==family+'_DP'])) for metric in ('dice','bf1','hd95')}
                        for family in ('local','local_large','global')},
             'gated_mean':{family:{metric:float(np.mean([r[metric] for r in gate if r['method']==family])) for metric in ('dice','bf1','hd95','moved_fraction')}
                           for family in ('local8_gate','global_gate')},
             'gate_thresholds_all_0_95':all(r['threshold']==0.95 for r in gate),
             'gated_models_move_no_nodes_after_calibration':all(r['moved_fraction']==0 for r in gate),
             'DP_selection_on_val150_not_independent':True}
    (output/'control_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
