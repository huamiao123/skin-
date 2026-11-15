"""User-requested cap at epoch 100; original training sources/config remain immutable."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
RESEARCH=PROJECT.parent.parent
OUT=ROOT/'results'
AUDIT=ROOT/'budget100_amendment'
CAP=100
NAMES=('control','local_fusion','anchor_fusion')
def dump(p,x):
    tmp=p.with_name(p.name+'.budget100.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
    tmp.replace(p)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'
def status(phase,**extra):
    s=json.loads((OUT/'run_status.json').read_text()) if (OUT/'run_status.json').exists() else {}
    s.update(state=phase,total_epochs=CAP,original_planned_epochs=300,
             lr_schedule_horizon=300,budget_amendment=str(AUDIT/'AMENDMENT.json'),
             supervisor_pid=os.getpid())
    s.update(extra);dump(OUT/'budget100_status.json',s)
    p=RESEARCH/'current_experiment.json'
    current=json.loads(p.read_text())
    current.update(state=phase,total_epochs=CAP,original_planned_epochs=300,
        lr_schedule_horizon=300,pipeline_pid=os.getpid(),last_observed_epoch=s.get('epoch',0),
        status_file=str(OUT/'budget100_status.json'),budget_amendment=str(AUDIT/'AMENDMENT.json'),
        resume_command=[sys.executable,str(ROOT/'run_budget100.py')],
        last_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    current.update({k:v for k,v in extra.items() if k in ['training_complete','analysis_complete',
        'test_evaluated','test_evaluation_complete','training_pid','pipeline_pid']})
    dump(p,current)
def validate_sources(config):
    for rel,expected in config['sources'].items():
        assert sha(PROJECT/rel)==expected,rel
def main():
    lock=(ROOT/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=json.loads((AUDIT/'AMENDMENT.json').read_text())
    oldpid=request['training_pid_at_handoff']
    status('waiting_for_epoch_boundary',training_pid=oldpid)
    deadline=time.monotonic()+300
    while alive(oldpid):
        assert time.monotonic()<deadline,'Trainer did not stop at boundary'
        time.sleep(1)
    import torch
    config=json.loads((OUT/'config.json').read_text());validate_sources(config)
    state=torch.load(OUT/'latest.pth',map_location='cpu')
    assert state['config']==config and state['epoch']<=CAP
    first_epoch=state['epoch']
    before=state['history']
    assert len(before)==first_epoch
    assert json.loads((OUT/'history.json').read_text())==before
    for opt in state['optimizers'].values():
        assert {int(s['step']) for s in opt['state'].values() if 'step' in s}=={first_epoch*30}
    dump(AUDIT/'HANDOFF_CHECKPOINT_VERIFIED.json',dict(passed=True,epoch=first_epoch,
         optimizer_steps=first_epoch*30,sha256=sha(OUT/'latest.pth'),
         unchanged_original_config=True,unchanged_original_sources=True))
    dump(AUDIT/'history_at_handoff.json',before)
    del state
    if first_epoch<CAP:
        status('running',training_pid=None)
        with (ROOT/'formal_budget100.log').open('a') as f:
            proc=subprocess.Popen([sys.executable,str(ROOT/'train.py'),'--stop-after-epoch',str(CAP)],
                cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            observed=None
            while proc.poll() is None:
                s=json.loads((OUT/'run_status.json').read_text())
                if s.get('epoch')!=observed:
                    observed=s.get('epoch');status('running',training_pid=proc.pid)
                time.sleep(2)
            assert proc.returncode==0,('training failed',proc.returncode)
    status('finalizing_training',epoch=CAP,training_pid=None)
    state=torch.load(OUT/'latest.pth',map_location='cpu')
    assert state['epoch']==CAP and state['config']==config
    assert state['history'][:first_epoch]==before
    assert len(state['history'])==CAP
    for n in NAMES:
        steps={int(s['step']) for s in state['optimizers'][n]['state'].values() if 'step' in s}
        assert steps=={CAP*30},(n,steps)
        selected=max(state['history'],key=lambda row:row['validation'][n]['macro']['dice'])
        best=torch.load(OUT/(n+'_best_dice.pth'),map_location='cpu')
        assert best['epoch']==selected['epoch']
        assert best['metric']==selected['validation'][n]
        del best
    history=state['history'];del state
    validate_sources(config)
    dump(AUDIT/'CAP100_COMPLETION_AUDIT.json',dict(passed=True,actual_epochs=CAP,
        original_planned_epochs=300,lr_schedule_horizon=300,optimizer_steps_per_arm=CAP*30,
        history_before_handoff_unchanged=True,selection_verified_all_arms=True,
        sources_unchanged=True))
    dump(OUT/'TRAINING_DONE.json',dict(complete=True,epochs=CAP,original_planned_epochs=300,
        updates_per_arm=CAP*30,lr_schedule_horizon=300,
        completion_reason='user_requested_compute_budget_amendment',budget_amendment=str(AUDIT/'AMENDMENT.json')))
    status('evaluating',epoch=CAP,training_complete=True,training_pid=None)
    with (ROOT/'evaluation_budget100.log').open('a') as f:
        subprocess.run([sys.executable,str(ROOT/'evaluate_budget100.py')],cwd=PROJECT,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
    done=json.loads((OUT/'DONE.json').read_text())
    assert done['complete'] and done['test_evaluated'] and done['test_n']==867 and done['epochs']==CAP
    dump(ROOT/'PIPELINE_DONE.json',dict(complete=True,epochs=CAP,training=True,validation=True,test=True,
        budget_amendment=str(AUDIT/'AMENDMENT.json')))
    status('complete',epoch=CAP,training_complete=True,analysis_complete=True,test_evaluated=True,
           test_evaluation_complete=True,training_pid=None,pipeline_pid=None)
    print('CAP100_TRAIN_VALIDATION_TEST_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        dump(AUDIT/'FAILURE.json',dict(traceback=traceback.format_exc()))
        status('failed',failure=str(AUDIT/'FAILURE.json'))
        raise
