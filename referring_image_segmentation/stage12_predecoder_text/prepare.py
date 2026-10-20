"""Keep the completed ISIC split and all earlier results intact."""
from common import *
from datetime import datetime,timezone

def main():
    TOKEN_ASSETS.mkdir(exist_ok=True)
    assert json.loads((CONTROLS/'validation.json').read_text())['passed']
    frozen=json.loads((CONTROLS/'launch_provenance.json').read_text())
    for section in ('source_sha256','record_sha256'):
        for path,digest in frozen[section].items():assert sha(path)==digest,path
    for name in ('data_manifest.json','text_provenance.json'):
        original=PREVIOUS/name;dest=ROOT/name
        if dest.exists():assert dest.read_bytes()==original.read_bytes()
        else:dest.write_bytes(original.read_bytes())
    historical=json.loads((CONTROLS/'analysis/summary.json').read_text())
    checkpoints=historical['checkpoints']
    for value in checkpoints.values():assert sha(value['path'])==value['sha256']
    lineage=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        historical_stage=str(CONTROLS),data_counts=dict(train=806,val=90,test=379),
        asset_sha256={str(p):sha(p) for p in ASSETS.iterdir() if p.is_file()},
        previous_checkpoints=checkpoints,previous_metrics=historical['models'],
        previous_report_sha256=sha(CONTROLS/'analysis/summary.json'),
        historical_preflight_sha256=sha(CONTROLS/'preflight.json'),
        patient_disjointness='not established',test_set_status='Previously used during development; this is another controlled exploratory experiment, not a new independent test',
        new_models=list(NAMES),new_seed=SEED,previous_segmentation_weights_loaded=False)
    path=ROOT/'lineage.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('asset_sha256','previous_checkpoints','previous_report_sha256','historical_preflight_sha256'):assert old[key]==lineage[key],key
    else:dump(path,lineage)
    print('PREPARED',lineage['data_counts'],'two paired predecoder runs, seed42',flush=True)

if __name__=='__main__':main()
