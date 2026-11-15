"""Validation-only post-training error decomposition; preserves diagnostic baseline files."""
from pathlib import Path
import ast,csv,json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
def main():
    tree=ast.parse((ROOT/'diagnose_validation.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['profile','read_preds']]
    ids=json.loads((ROOT/'identity_partition.json').read_text())['splits']['val']
    scope=dict(np=np,ndi=ndi,ids=ids)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'diagnostic_definitions','exec'),scope)
    targets=np.stack([np.asarray(Image.open(P/'data/isic2018/val/masks'/f'{sid}.png').convert('L'))>=127.5 for sid in ids])
    prior=json.loads((ROOT/'diagnosis/SUMMARY.json').read_text())['models']['uper64']
    report={'validation_only':True,'baseline':prior,'arms':{},'rule':'Fixed same3pixel and component definitions from pretraining diagnosis; all762 retained.'}
    for arm in ['cfam','conv_control']:
        out=ROOT/arm/'results'
        assert json.loads((out/'DONE.json').read_text())['test_n']==867
        pred=scope['read_preds'](out/f'{arm}_val_predictions.npz')
        rows=[scope['profile'](a,b,sid) for a,b,sid in zip(pred,targets,ids)]
        with (out/'validation_error_profiles.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        totals={k:sum(x[k] for x in rows) for k in ['tp','fp','fn','fp_near3','fp_far3','disconnected_fp','connected_fp','fn_boundary3','fn_deep3']}
        delta={k:totals[k]-prior[k] for k in totals}
        totals.update(macro_dice=float(np.mean([x['dice'] for x in rows])),severe_undercoverage=sum(x['recall']<.5 for x in rows))
        report['arms'][arm]=dict(totals=totals,delta_vs_baseline=delta)
    (ROOT/'VALIDATION_ERROR_COMPARISON.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
