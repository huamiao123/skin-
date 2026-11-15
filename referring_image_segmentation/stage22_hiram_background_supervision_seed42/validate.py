"""Mark complete only after all three models and per-image audits finish."""
from common import *
from models import NAMES


def main():
    done=json.loads((ROOT/'results/DONE.json').read_text())
    histories=json.loads((ROOT/'results/history.json').read_text())
    evaluation=json.loads((ROOT/'results/evaluation.json').read_text())
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    routing=json.loads((ROOT/'analysis/route_summary.json').read_text())
    assert done['seed']==42 and done['epochs']==300 and done['encoder_unchanged']
    assert not done['smoke_only'] and all(done['updates'][n]==9000 for n in NAMES)
    assert len(histories)==300 and evaluation['passed'] and evaluation['n_val']==808 and routing['n']==808
    for n in NAMES:
        scores=[r['models'][n]['val']['macro']['dice'] for r in histories]
        chosen=done['selectors'][n]['best_dice']
        assert scores.index(max(scores))+1==chosen['epoch'] and max(scores)==chosen['score']
        assert abs(summary['models'][n]['macro']['dice']-chosen['score'])<1e-12
        assert evaluation['results'][n]['best_dice']['selected_epoch']==chosen['epoch']
        for branch in ('coarse','skip'):
            route=routing['models'][n][branch]
            assert abs(sum(route['hard_fraction'])-1)<1e-6
            assert abs(sum(route['mean_soft_probability'])-1)<1e-5
            assert abs(sum(sum(row) for row in route['soft_target_by_hard_route'])-1)<1e-5
    result=dict(passed=True,seed=42,epochs=300,updates_per_model=9000,models=list(NAMES),
        n_val=808,fixed_dice_selection_verified=True,per_image_counts_and_full_metrics_verified=True,
        routing_audit_verified=True,encoder_frozen_including_buffers=True,
        interpretation='single seed; development validation is not independent test')
    dump(ROOT/'validation.json',result)
    print('VALIDATION',json.dumps(result),flush=True)


if __name__=='__main__':main()
