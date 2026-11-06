"""Independently gate the completed experiment and fixed checkpoint selectors."""
from common import *


def main():
    out=ROOT/'results'
    preflight=json.loads((ROOT/'preflight.json').read_text())
    capacity=json.loads((ROOT/'capacity_preflight.json').read_text())
    smoke=json.loads((ROOT/'smoke/DONE.json').read_text())
    resume=json.loads((ROOT/'smoke/resume_audit.json').read_text())
    done=json.loads((out/'DONE.json').read_text())
    evaluation=json.loads((out/'evaluation.json').read_text())
    history=json.loads((out/'history.json').read_text())
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    assert preflight['passed'] and capacity['passed'] and capacity['effective_batch']==64 and smoke['epochs']==2 and resume['resumed']
    assert all(preflight['stage17_initial_hashes_match'].values())
    assert preflight['region_only_forward_max_abs']==0 and preflight['zero_residual_forward_max_abs']==0
    assert done['epochs']==300 and done['seed']==42 and done['encoder_unchanged']
    assert all(n==9000 for n in done['updates'].values())
    assert len(history)==300 and [r['epoch'] for r in history]==list(range(1,301))
    assert evaluation['passed'] and evaluation['n_val']==808
    for name in ('region_only','hiram'):
        selectors=done['selectors'][name]
        best=max(history,key=lambda r:r['models'][name]['val']['macro']['dice'])
        low=min(history,key=lambda r:r['models'][name]['val']['loss'])
        assert best['epoch']==selectors['best_dice']['epoch']==evaluation['results'][name]['best_dice']['selected_epoch']
        assert low['epoch']==selectors['best_loss']['epoch']==evaluation['results'][name]['best_loss']['selected_epoch']
        assert evaluation['results'][name]['last']['selected_epoch']==300
        summary_name={'region_only':'post_region','hiram':'post_hiram'}[name]
        assert summary['models'][summary_name]['n']==808
        assert summary['models'][summary_name]['counts']==evaluation['results'][name]['best_dice']['metrics']['counts']
        path=out/name/'best_dice/probabilities.npy'
        assert np.load(path,mmap_mode='r').shape==(808,256,256)
    if (ROOT/'launch_provenance.json').exists():
        prov=json.loads((ROOT/'launch_provenance.json').read_text())
        for group in ('source_sha256','historical_sha256'):
            for p,h in prov[group].items():assert sha(p)==h,('changed source',p)
    assert (ROOT/'analysis/REPORT.md').exists() and (ROOT/'analysis/learning_curves.png').exists()
    result=dict(passed=True,seed=42,epochs=300,updates=done['updates'],n_val=808,
        fixed_selectors_verified=True,frozen_encoder_verified=True,complete_region_and_boundary_metrics=True,
        official_single_image_equivalence=preflight['official_single_forward_max_abs'],
        resume_smoke_verified=True,historical_initialization_hash_matches=preflight['historical_initial_hash_matches'])
    dump(ROOT/'validation.json',result)
    print('VALIDATION PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
