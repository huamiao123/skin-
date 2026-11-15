"""Run verified smoke/resume, formal training, validation replay and fixed test."""
from pathlib import Path
import datetime, json, os, shutil, subprocess, sys, traceback
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
RESEARCH=PROJECT.parent.parent
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def run(args,name):
    with (ROOT/name).open('a') as f:
        subprocess.run([sys.executable]+args,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
def current(state,**kwargs):
    p=RESEARCH/'current_experiment.json'
    obj=json.loads(p.read_text()) if p.exists() else {}
    obj.update(active_stage=ROOT.name,state=state,source_root=str(ROOT),
               arms=['main_context_channel'],pipeline_pid=os.getpid(),
               training_complete=False,analysis_complete=False,test_evaluated=False,
               test_evaluation_complete=False,gpu='NVIDIA RTX A4000',gpu_memory_bytes=16376*1024**2,
               partition_counts=dict(train=1886,val=762,test=867),total_epochs=100,
               status_file=str(ROOT/'results/run_status.json'),test_result_directory=str(ROOT/'results'),
               last_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    obj.update(lr_schedule_horizon=300,original_planned_epochs=100,budget_amendment=None,
               resume_command=[sys.executable,str(ROOT/'run_pipeline.py')],reused_reference_models=4)
    for key in ['analysis_result','analysis_log','epoch_seconds','resume_audit_directory','checkpoint_retention']:
        obj.pop(key,None)
    obj.update(kwargs);dump(p,obj)
def main():
    import fcntl
    f=(ROOT/'pipeline.lock').open('w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=RESEARCH/'current_experiment.json'
    if not (ROOT/'previous_stage30_status.json').exists():shutil.copy2(old,ROOT/'previous_stage30_status.json')
    current('smoke_testing',training_pid=None,last_observed_epoch=0)
    smoke=ROOT/'smoke'
    if not (smoke/'RESUME_AUDIT.json').exists():
        if not (smoke/'latest.pth').exists():
            run([str(ROOT/'train.py'),'--smoke','--stop-after-epoch','1'],'smoke_first.log')
        history=json.loads((smoke/'history.json').read_text())
        assert len(history)==1 and history[0]['epoch']==1
        run([str(ROOT/'train.py'),'--smoke'],'smoke_resume.log')
        after=json.loads((smoke/'history.json').read_text())
        assert after[:1]==history and len(after)==2
        load=json.loads((smoke/'RESUME_LOAD_AUDIT.json').read_text())
        assert load['passed'] and load['start_epoch']==2
        import torch
        s=torch.load(smoke/'latest.pth',map_location='cpu')
        assert s['epoch']==2
        for arm,opt in s['optimizers'].items():
            steps={int(x['step']) for x in opt['state'].values() if 'step' in x}
            assert steps=={60},(arm,steps)
        del s
        run([str(ROOT/'evaluate.py'),'--smoke'],'smoke_evaluation.log')
        dump(smoke/'RESUME_AUDIT.json',dict(passed=True,actual_process_restart=True,
             first_epoch_unchanged=True,second_epoch=2,optimizer_steps=60,
             validation_replay_verified=True,test_read=False))
    current('running',last_observed_epoch=0)
    run([str(ROOT/'train.py')],'formal.log')
    current('evaluating',training_complete=True,last_observed_epoch=100)
    run([str(ROOT/'evaluate.py')],'evaluation.log')
    done=json.loads((ROOT/'results/DONE.json').read_text())
    assert done['complete'] and done['test_evaluated'] and done['test_n']==867
    current('complete',training_complete=True,analysis_complete=True,test_evaluated=True,
            test_evaluation_complete=True,test_evaluation_models=1,last_observed_epoch=100,
            pipeline_pid=None,training_pid=None,results_summary=str(ROOT/'results/summary.json'))
    dump(ROOT/'PIPELINE_DONE.json',dict(complete=True,training=True,validation=True,test=True))
if __name__=='__main__':
    try:main()
    except Exception:
        dump(ROOT/'PIPELINE_FAILURE.json',dict(traceback=traceback.format_exc()))
        current('failed',error_record=str(ROOT/'PIPELINE_FAILURE.json'))
        raise
