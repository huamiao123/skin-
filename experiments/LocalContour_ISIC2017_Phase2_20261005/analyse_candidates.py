"""Audit phase-A output and report the original search-band denominator."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from project_paths import PHASE1

import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
OLD=PHASE1/'results/per_image_results.csv'

def read(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def main():
    rows=read(ROOT/'results/candidates/per_image.csv')
    original_summary={r['config']:r for r in read(ROOT/'results/candidates/summary.csv')}
    older={r['image_id']:r for r in read(OLD) if r['method']=='G0'}
    assert len(rows)==900 and len(older)==150
    old3=[r for r in rows if r['config']=='old_3']
    assert all(float(r['oracle_dice'])==float(r['historical_g4_dice']) for r in old3)
    bands=[float(older[r['image_id']]['search_band_coverage']) for r in old3 if older[r['image_id']]['search_band_coverage']]
    summary=[]
    for name in ('old_3','fixed_3','fixed_4','fixed_8','fixed_16','dense'):
        a=[r for r in rows if r['config']==name]
        n=sum(int(r['effective_nodes']) for r in a)
        pooled=sum(int(r['covered_nodes']) for r in a)/n
        d=np.array([float(r['coverage']) for r in a if r['coverage']])
        summary.append({'config':name,'n_images':150,'effective_nodes':n,
            'candidate_coverage_pooled':pooled,
            'candidate_coverage_image_mean':float(d.mean()),
            'candidate_coverage_image_median':float(np.median(d)),
            'candidate_coverage_image_p10':float(np.percentile(d,10)),
            'candidate_coverage_image_p90':float(np.percentile(d,90)),
            'oracle_dice_mean':float(np.mean([float(r['oracle_dice']) for r in a])),
            'oracle_bf1_mean':float(np.mean([float(r['oracle_bf1']) for r in a])),
            'oracle_minus_g0_dice_mean':float(np.mean([float(r['oracle_dice'])-float(r['g0_dice']) for r in a])),
            'mean_nearest_gt_distance':float(original_summary[name]['mean_nearest_gt_distance']),
            'median_nearest_gt_distance':float(original_summary[name]['median_nearest_gt_distance']),
            'p90_nearest_gt_distance':float(original_summary[name]['p90_nearest_gt_distance']),
            'p95_nearest_gt_distance':float(original_summary[name]['p95_nearest_gt_distance']),
            'coverage_vs_g0_dice_spearman':float(spearmanr([float(r['coverage']) for r in a if r['coverage']],
                [float(r['g0_dice']) for r in a if r['coverage']]).statistic) if name!='dense' else None,
            'coverage_vs_oracle_dice_spearman':float(spearmanr([float(r['coverage']) for r in a if r['coverage']],
                [float(r['oracle_dice']) for r in a if r['coverage']]).statistic) if name!='dense' else None,
            **{f'candidate_recall_at_{k}_pooled':sum(float(r[f'recall_at_{k}'])*int(r['effective_nodes']) for r in a if r[f'recall_at_{k}'])/n for k in (1,2,4,8,16)}
        })
    result={'scope':'official ISIC2017 val150; previously exposed development set, not independent test',
        'sample_count':150,'search_band_coverage_macro':float(np.mean(bands)),
        'search_band_coverage_median':float(np.median(bands)),
        'normal_validity':'single GT intersection within +/-16 pixels and non-collinear; dense 100% coverage is conditional on this definition and does not cover missing/ambiguous nodes',
        'candidate_coverage_denominator':summary[0]['effective_nodes'],
        'all_planned_nodes':150*256,
        'old_G4_reproduction_max_absolute_dice_difference':0.0,
        'groups':summary,
        'decision':'PEAK_SELECTION_BOTTLENECK; dense candidates pass conditional node-coverage gate; use dense for next local baseline',
        'limitations':['CNN public checkpoint trained on official train+val mixture',
                       'oracle uses GT at selection time and is undeployable',
                       'GT beyond current search band, multi-intersections, and other components remain structural limits',
                       'dense selection imposes 33 candidates per node and must be tested against strong local model']}
    (ROOT/'results/candidates/analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (ROOT/'results/candidates/corrected_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    with (ROOT/'results/candidates/summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    print(json.dumps({'band_macro':result['search_band_coverage_macro'],
          'old_cov':summary[0]['candidate_coverage_pooled'],
          'fixed8_cov':summary[3]['candidate_coverage_pooled'],
          'dense_cov':summary[5]['candidate_coverage_pooled'],
          'old_oracle':summary[0]['oracle_dice_mean'],
          'dense_oracle':summary[5]['oracle_dice_mean']},indent=2))

if __name__=='__main__':main()
