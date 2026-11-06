"""Gate completion on the full fixed protocol and source identity."""
from common import *


def main():
    out=ROOT/'results'
    preflight=json.loads((ROOT/'preflight.json').read_text())
    capacity=json.loads((ROOT/'capacity_preflight.json').read_text())
    smoke=json.loads((ROOT/'smoke/DONE.json').read_text())
    resume=json.loads((ROOT/'smoke/resume_audit.json').read_text())
    smoke_eval=json.loads((ROOT/'smoke/evaluation.json').read_text())
    done=json.loads((out/'DONE.json').read_text())
    evaluation=json.loads((out/'evaluation.json').read_text())
    history=json.loads((out/'history.json').read_text())
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    assert preflight['passed'] and capacity['passed'] and capacity['effective_batch']==64
    assert preflight['two_variants_identical_initial_state']
    assert preflight['official_native_feature_max_abs']==0 and preflight['one_region_original_order_max_abs']<1e-6
    assert smoke['epochs']==2 and resume['resumed'] and smoke_eval['passed']
    assert done['epochs']==300 and done['seed']==42 and done['encoder_unchanged']
    assert all(n==9000 for n in done['updates'].values())
    assert len(history)==300 and [r['epoch'] for r in history]==list(range(1,301))
    assert evaluation['passed'] and evaluation['n_val']==808
    for name in ('ordered_prior','region_scan'):
        selectors=done['selectors'][name]
        best=max(history,key=lambda r:r['models'][name]['val']['macro']['dice'])
        low=min(history,key=lambda r:r['models'][name]['val']['loss'])
        assert best['epoch']==selectors['best_dice']['epoch']==evaluation['results'][name]['best_dice']['selected_epoch']
        assert low['epoch']==selectors['best_loss']['epoch']==evaluation['results'][name]['best_loss']['selected_epoch']
        assert evaluation['results'][name]['last']['selected_epoch']==300
        assert summary['models'][name]['n']==808
        assert summary['models'][name]['counts']==evaluation['results'][name]['best_dice']['metrics']['counts']
        path=out/name/'best_dice/probabilities.npy'
        assert np.load(path,mmap_mode='r').shape==(808,256,256)
    prov=json.loads((ROOT/'launch_provenance.json').read_text())
    for group in ('source_sha256','historical_sha256'):
        for p,h in prov[group].items():assert sha(p)==h,('changed source',p)
    assert (ROOT/'analysis/REPORT.md').exists() and (ROOT/'analysis/learning_curves.png').exists()
    result=dict(passed=True,seed=42,epochs=300,updates=done['updates'],n_val=808,
        fixed_selectors_verified=True,original_encoder_frozen_verified=True,
        matching_initial_parameters=True,complete_region_and_boundary_metrics=True,
        official_native_feature_max_abs=preflight['official_native_feature_max_abs'],
        resume_smoke_verified=True,train_micro_batch=capacity['train_micro_batch'])
    dump(ROOT/'validation.json',result)
    print('VALIDATION PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
