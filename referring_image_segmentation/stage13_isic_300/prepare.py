"""Verify and reuse read-only inputs; all five trainable models are new runs."""
from common import *
from datetime import datetime,timezone

def main():
    assert json.loads((EARLY/'validation.json').read_text())['passed']
    source=json.loads((EARLY/'launch_provenance.json').read_text())
    for part in ('source_sha256','record_sha256'):
        for p,h in source[part].items():assert sha(p)==h,p
    for name,folder in [('data_manifest.json',PREVIOUS),('text_provenance.json',PREVIOUS),('token_provenance.json',EARLY)]:
        original=folder/name;target=ROOT/name
        if target.exists():assert target.read_bytes()==original.read_bytes()
        else:target.write_bytes(original.read_bytes())
    controls=json.loads((CONTROLS/'preflight.json').read_text());early=json.loads((EARLY/'preflight.json').read_text())
    expected=dict(controls['initial_hashes']);expected.update(early['initial_hashes'])
    old_report=json.loads((EARLY/'analysis/summary.json').read_text())
    lineage=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,epochs=EPOCHS,
        counts=dict(train=806,val=90,test=379),fresh_models=list(NAMES),previous_trained_weights_loaded=False,
        expected_initial_hashes=expected,expected_frozen_visual_hash=controls['frozen_visual_hash'],
        assets_sha256={str(p):sha(p) for p in ASSETS.iterdir() if p.is_file()},
        token_assets_sha256={str(p):sha(p) for p in TOKEN_ASSETS.iterdir() if p.is_file()},
        previous_100_epoch_metrics=old_report['models'],previous_100_epoch_report_sha256=sha(EARLY/'analysis/summary.json'),
        patient_disjointness='not established',test_set_status='Already used in earlier development; budget comparison is exploratory, not an independent new test',
        budget_change='100 to300; cosine reset for full15300 updates; no continuation from trained checkpoints; all other ISIC2016 training settings retained')
    target=ROOT/'lineage.json'
    if target.exists():
        old=json.loads(target.read_text())
        for k in ('expected_initial_hashes','assets_sha256','token_assets_sha256','previous_100_epoch_report_sha256'):assert old[k]==lineage[k],k
    else:dump(target,lineage)
    print('PREPARED','five fresh seed42 runs','300 epochs','806/90/379',flush=True)

if __name__=='__main__':main()
