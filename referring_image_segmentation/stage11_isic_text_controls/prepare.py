"""Verify the completed historical experiment and snapshot its immutable manifests."""
from common import *
from datetime import datetime,timezone

def main():
    historical=json.loads((PREVIOUS/'launch_provenance.json').read_text())
    assert json.loads((PREVIOUS/'validation.json').read_text())['passed']
    for section in ('source_sha256','records_sha256','data_files_sha256'):
        for path,digest in historical[section].items():assert sha(path)==digest,path
    records={}
    for name in ('data_manifest.json','text_provenance.json'):
        source=PREVIOUS/name;dest=ROOT/name;blob=source.read_bytes()
        if dest.exists():assert dest.read_bytes()==blob
        else:dest.write_bytes(blob)
        records[str(source)]=sha(source)
    config=json.loads((PREVIOUS/'results/seed42/config.json').read_text())
    assert config['seed']==SEED and config['epochs']==EPOCHS and config['batch']==BATCH
    assert config['train_samples']==806 and config['validation_samples']==90
    assert config['lr']==1e-4 and config['warmup_steps']==255 and config['total_schedule_steps']==5100
    checkpoints={n:dict(source_name=old,path=str(PREVIOUS/'results/seed42'/old/'best_dice.pth'),
                        sha256=sha(PREVIOUS/'results/seed42'/old/'best_dice.pth')) for n,old in REUSED.items()}
    source_metrics={n:json.loads((PREVIOUS/'analysis/summary.json').read_text())['models'][old] for n,old in REUSED.items()}
    lineage=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_stage=str(PREVIOUS),
        reuse_reason='E1 and E2 already trained independently under the requested architecture. Add only the missing, exactly matched visual control.',
        historical_protocol_sha256=sha(PREVIOUS/'PROTOCOL.md'),historical_training_config=config,
        source_record_sha256=records,historical_frozen_assets=historical['data_files_sha256'],
        reused_checkpoints=checkpoints,reused_metrics=source_metrics,
        historical_artifacts_preserved=True,new_training_models=[NAMES[0]],new_training_seed=SEED,
        patient_disjointness='not established',test_set_status='Previously evaluated during stage9; this is a developmental control completion, not a new untouched external test')
    dest=ROOT/'lineage.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        for k in ('historical_protocol_sha256','historical_training_config','source_record_sha256','historical_frozen_assets','reused_checkpoints'):assert old[k]==lineage[k],k
    else:dump(dest,lineage)
    print('PREPARED','806/90/379','only E0 is newly trained',flush=True)

if __name__=='__main__':main()
