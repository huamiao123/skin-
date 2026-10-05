"""Post-A radius-only diagnostic: identical 256 source nodes/normals, R16 vs R32."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from project_paths import PHASE1

import numpy as np

SOURCE=PHASE1
sys.path.insert(0,str(SOURCE))
from contour.geometry import coverage_diagnostics, extract_geometry, generate_candidates

ROOT=Path(__file__).resolve().parent

def dense(score, geo):
    old=generate_candidates(score,geo)
    off=np.arange(-int(geo.search_radius),int(geo.search_radius)+1,dtype=np.float64)
    points=geo.points[:,None,:]+off[None,:,None]*geo.normals[:,None,:]
    from contour.geometry import CandidateField
    return CandidateField(points,np.tile(off,(len(points),1)),old.sample_scores,old.sample_valid,
                          old.sample_offsets,old.sample_scores,old.sample_valid)

def main():
    paths=sorted((SOURCE/'results/cases').glob('*.npz'))
    rows=[]
    for n,p in enumerate(paths,1):
        with np.load(p) as z:
            pred=z['masks_G0'];gt=z['gt_mask'];q=z['boundary_probability'];role=str(z['role'])
        base=extract_geometry(pred,256,16)
        if base is None:
            for rad in (16,32):rows.append({'image_id':p.stem,'role':role,'radius':rad,'status':'empty_prediction','search_band_coverage':0.0,'valid_nodes':0,'covered_nodes':0})
            continue
        for rad in (16,32):
            geo=base if rad==16 else replace(base,search_radius=32.0)
            coverage=coverage_diagnostics(geo,dense(q,geo),gt)
            rows.append({'image_id':p.stem,'role':role,'radius':rad,'status':'ok',
                         'search_band_coverage':coverage['search_band_coverage'],
                         'valid_nodes':coverage['effective_normal_nodes'],
                         'covered_nodes':coverage['candidate_covered_effective_nodes'],
                         'all_nodes':coverage['total_nodes'],
                         'no_intersection_nodes':coverage['no_intersection_nodes'],
                         'multiple_intersection_nodes':coverage['multiple_intersection_nodes']})
        if n%10==0:print(f'{n}/{len(paths)}',flush=True)
    out=ROOT/'results/candidates'
    with (out/'radius_sensitivity_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    result={}
    for rad in (16,32):
        a=[r for r in rows if r['radius']==rad]
        result[str(rad)]={'search_band_coverage_macro':float(np.mean([r['search_band_coverage'] for r in a])),
                          'valid_nodes':sum(r['valid_nodes'] for r in a),
                          'covered_nodes':sum(r['covered_nodes'] for r in a),
                          'total_planned_nodes':len(a)*256,
                          'no_intersection_nodes':sum(r.get('no_intersection_nodes',0) for r in a),
                          'multiple_intersection_nodes':sum(r.get('multiple_intersection_nodes',0) for r in a)}
    result['scope']='post-A development diagnostic, same CNN and same 256 node positions/normals; only radius changes; no model tuning or GT-free result claimed'
    (out/'radius_sensitivity_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
