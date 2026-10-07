"""Count how often the archived wrong keep target applies on actual train/val candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candidate_layout import ZERO_INDEX

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'


def main():
    records=json.loads((CACHE/'COMPLETE.json').read_text())['records']
    base=CACHE/'candidates_R32'
    valid=np.load(base/'valid.npy',mmap_mode='r')
    distance=np.load(base/'gt_distance.npy',mmap_mode='r')
    has=np.load(base/'has_contour.npy',mmap_mode='r')
    result={}
    for split in ('train','val'):
        totals={key:0 for key in ('images','contour_images','nodes','ordinary_wrong_keep_nodes',
          'ordinary_wrong_keep_invalid_index0','gate_wrong_keep_nodes','gate_wrong_keep_invalid_index0',
          'gate_repair_nodes','initially_within_2px')}
        for record in records:
            if record['split']!=split:continue
            totals['images']+=1
            i=record['index']
            if not has[i]:continue
            totals['contour_images']+=1;totals['nodes']+=256
            v=np.asarray(valid[i],bool);d=np.asarray(distance[i],np.float32)
            if not v[:,ZERO_INDEX].all():raise ValueError('Invalid zero candidate')
            unreachable=np.min(np.where(v,d,1e4),axis=-1)>4
            other=v.copy();other[:,ZERO_INDEX]=False
            closest=np.min(np.where(other,d,1e4),axis=-1)
            zero=d[:,ZERO_INDEX]
            repair=(zero>2)&(closest+0.5<zero)&(closest<=4)
            keep=~repair
            totals['ordinary_wrong_keep_nodes']+=int(unreachable.sum())
            totals['ordinary_wrong_keep_invalid_index0']+=int((unreachable&~v[:,0]).sum())
            totals['gate_wrong_keep_nodes']+=int(keep.sum())
            totals['gate_wrong_keep_invalid_index0']+=int((keep&~v[:,0]).sum())
            totals['gate_repair_nodes']+=int(repair.sum())
            totals['initially_within_2px']+=int((zero<=2).sum())
        n=max(totals['nodes'],1)
        totals.update({key+'_fraction':totals[key]/n for key in ('ordinary_wrong_keep_nodes',
          'ordinary_wrong_keep_invalid_index0','gate_wrong_keep_nodes',
          'gate_wrong_keep_invalid_index0','gate_repair_nodes','initially_within_2px')})
        result[split]=totals
    output=ROOT/'results/repair_validation';output.mkdir(parents=True,exist_ok=True)
    (output/'label_bug_incidence.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
