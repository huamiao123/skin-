"""Stage45: native UPerHead width64/128, actual restart, full val/test evaluation."""
from pathlib import Path
import datetime,fcntl,json,os,shutil,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
RESEARCH=PROJECT.parent.parent
ARMS=['uper64','uper128']
ACTIVE=None
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def current(state,**kw):
    obj=dict(stage=ROOT.name,state=state,arms=ARMS,epochs_per_arm=100,lr_schedule_horizon=100,
             pipeline_pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             status_file=str(ROOT/'pipeline_status.json'),resume_command=[sys.executable,str(ROOT/'run_pipeline.py')])
    obj.update(kw);dump(ROOT/'pipeline_status.json',obj);dump(RESEARCH/'current_experiment.json',obj)
def run(args,log,**status):
    global ACTIVE
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        process=subprocess.Popen([sys.executable]+args,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        ACTIVE=process.pid
        current(status.pop('state','running'),training_pid=process.pid,**status)
        code=process.wait();ACTIVE=None
        if code:raise RuntimeError(f'Child exited {code}; inspect {log}')
def retain_best(arm):
    folder=(ROOT/arm).resolve();assert folder.is_relative_to(ROOT.resolve())
    removed=[]
    targets=[folder/'results/latest.pth']+list((folder/'smoke').glob('*.pth'))
    for path in targets:
        assert path.resolve().is_relative_to(folder)
        if path.is_file():removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    dump(folder/'CHECKPOINT_RETENTION.json',dict(best_only_after_full_test=True,removed=removed))
def main():
    f=(ROOT/'pipeline.lock').open('w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'PIPELINE_DONE.json').exists():print('ALREADY_COMPLETE',flush=True);return
    old=RESEARCH/'current_experiment.json'
    if old.exists() and not (ROOT/'previous_experiment_status.json').exists():shutil.copy2(old,ROOT/'previous_experiment_status.json')
    if not (ROOT/'INTEGRATION_VERIFIED.json').exists():
        run([str(ROOT/'verify_integration.py')],ROOT/'integration_verification.log',state='integration_verification')
    if not (ROOT/'REFERENCE_REPLAY_VERIFIED.json').exists():
        run([str(ROOT/'verify_reference.py')],ROOT/'reference_replay.log',state='reference_validation_replay')
    for arm in ARMS:
        s=ROOT/arm/'smoke';s.mkdir(parents=True,exist_ok=True)
        if (s/'RESUME_AUDIT.json').exists():continue
        if not (s/'latest.pth').exists():
            run([str(ROOT/'train.py'),'--arm',arm,'--smoke','--stop-after-epoch','1'],s/'first.log',state='smoke_training',arm=arm)
        before=json.loads((s/'history.json').read_text())
        if len(before)==1:
            run([str(ROOT/'train.py'),'--arm',arm,'--smoke'],s/'resume.log',state='smoke_resume',arm=arm)
        after=json.loads((s/'history.json').read_text());assert after[:len(before)]==before and len(after)==2
        load=json.loads((s/'RESUME_LOAD_AUDIT.json').read_text());assert load['passed'] and load['start_epoch']==2
        import torch
        ck=torch.load(s/'latest.pth',map_location='cpu');assert ck['epoch']==2
        for opt in ck['optimizers'].values():assert {int(x['step']) for x in opt['state'].values() if 'step' in x}=={60}
        del ck
        run([str(ROOT/'evaluate.py'),'--arm',arm,'--smoke'],s/'evaluation.log',state='smoke_validation_replay',arm=arm)
        dump(s/'RESUME_AUDIT.json',dict(passed=True,actual_process_restart=True,second_epoch=2,optimizer_steps=60,
             validation_replay_verified=True,test_read=False,first_epoch_preserved=True))
    for arm in ARMS:
        if (ROOT/arm/'results/TRAINING_DONE.json').exists():continue
        run([str(ROOT/'train.py'),'--arm',arm],ROOT/arm/'formal.log',state='formal_training',arm=arm,
            status_file=str(ROOT/arm/'results/run_status.json'),completed_trainings=[a for a in ARMS if (ROOT/a/'results/TRAINING_DONE.json').exists()])
    for arm in ARMS:
        if not (ROOT/arm/'results/DONE.json').exists():
            run([str(ROOT/'evaluate.py'),'--arm',arm],ROOT/arm/'evaluation.log',state='fixed_validation_and_test',arm=arm)
        done=json.loads((ROOT/arm/'results/DONE.json').read_text());assert done['complete'] and done['test_evaluated'] and done['test_n']==867
        retain_best(arm)
    summaries={a:json.loads((ROOT/a/'results/summary.json').read_text()) for a in ARMS}
    ref=json.loads((ROOT/'REFERENCE_MODELS.json').read_text())['models'][0]
    rows=[dict(arm=ref['arm'],best_epoch=ref['epoch'],val_macro_dice=ref['validation']['macro']['dice'],
        test_macro_dice=ref['test']['macro']['dice'],test_pooled_dice=ref['test']['pooled']['dice'])]
    screens={}
    for arm in ARMS:
        s=summaries[arm]
        params=json.loads((ROOT/arm/'results/parameters.json').read_text())[arm]
        hist=json.loads((ROOT/arm/'results/history.json').read_text())
        rows.append(dict(arm=arm,best_epoch=s['selection']['epoch'],val_macro_dice=s['validation'][arm]['macro']['dice'],
            test_macro_dice=s['test'][arm]['macro']['dice'],test_pooled_dice=s['test'][arm]['pooled']['dice'],
            parameters=params,max_peak_allocated_gib=max(v['peak_allocated_gib'] for v in hist),
            mean_epoch_seconds=sum(v['seconds'] for v in hist)/len(hist)))
        val=s['validation'][arm];baseline=ref['validation']
        screen=dict(validation_dice_gain_pp=100*(val['macro']['dice']-baseline['macro']['dice']),
            validation_boundary_f1_delta=val['macro']['boundary_f1']-baseline['macro']['boundary_f1'],
            rule='macro Dice gain >=0.1pp and boundary F1 no decrease; single seed validation screen',
            test_used_for_screen=False)
        screen['passed']=screen['validation_dice_gain_pp']>=.1 and screen['validation_boundary_f1_delta']>=0
        screen['decision']='candidate_for_paired_seed_replication' if screen['passed'] else 'no_validated_screen_pass_review_paired_validation_errors'
        screens[arm]=screen
    from evaluate import refrows,compare
    partition=json.loads((ROOT/'identity_partition.json').read_text())
    width_pair={}
    for split,key in [('val','validation'),('test','test')]:
        a=refrows(ROOT/ARMS[0]/'results'/f'{ARMS[0]}_{key}_per_image.csv',partition['splits'][split])
        b=refrows(ROOT/ARMS[1]/'results'/f'{ARMS[1]}_{key}_per_image.csv',partition['splits'][split])
        width_pair[key]=compare(a,b)
    dump(ROOT/'RESULTS_TABLE.json',rows);dump(ROOT/'DECISION.json',screens)
    dump(ROOT/'summary.json',dict(complete=True,arms=summaries,screen=screens,width64_vs128=width_pair,test_n_per_arm=867))
    (ROOT/'RESULTS_zh.txt').write_text('Stage45: native UPerHead64/128, unchanged late-only encoder adaptation and common soft BCE+Dice;100 epochs per arm, selected by762 validation images and evaluated on867 test images. Decoder-package comparison; not an isolated attribution to feature compression. Single seed; exposed test.\\n'+json.dumps(rows,indent=2,ensure_ascii=False)+'\\n'+json.dumps(screens,indent=2,ensure_ascii=False))
    dump(ROOT/'PIPELINE_DONE.json',dict(complete=True,trained=ARMS,test_evaluated=ARMS))
    current('complete',training_pid=None,pipeline_pid=None,results_summary=str(ROOT/'summary.json'))
    latest=RESEARCH/'LATEST_RESEARCH_20261112.json'
    if latest.exists():
        record=json.loads(latest.read_text())
        if record.get('active_experiment')==str(ROOT):
            record.update(active_experiment=None,active_pipeline_pid=None,active_arms=[],latest_completed_experiment=str(ROOT))
            dump(latest,record)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'PIPELINE_FAILURE.json',dict(traceback=traceback.format_exc(),active_child=ACTIVE))
        current('failed',training_pid=ACTIVE,error_record=str(ROOT/'PIPELINE_FAILURE.json'))
        raise

