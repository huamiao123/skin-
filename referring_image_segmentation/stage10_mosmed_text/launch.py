"""Detached supervisor: verified smoke -> common-budget pilot -> nine runs -> locked test."""
from common import *
import os, subprocess, fcntl, signal
from datetime import datetime, timezone

def stamp():return datetime.now(timezone.utc).isoformat()

def source_hashes():
    paths=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',PROJECT/'evaluation.py']
    for stage in ('stage1_visual','stage2_decoder','stage3_confirmation'):
        paths+=list((PROJECT/stage).glob('*.py'))
    return {str(p):sha(p) for p in paths}

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:return
    status=dict(state='preparing',stage='mosmed_text_gain',seeds=list(SEEDS),models=list(NAMES),supervisor_pid=os.getpid(),started_utc=stamp(),events=[])
    interrupted=[False];child=[None]
    def stop(*_):
        interrupted[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(script,args,logname,phase):
        if interrupted[0]:raise InterruptedError('stopped by user')
        status.update(state=phase,command=[script]+args);dump(ROOT/'run_status.json',status)
        env=dict(os.environ,PYTHONHASHSEED='42',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with (ROOT/logname).open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            child[0]=p;status['child_pid']=p.pid;dump(ROOT/'run_status.json',status);code=p.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if interrupted[0]:raise InterruptedError('stopped by user')
        if code:raise RuntimeError(f'{phase} failed ({code}); inspect {logname}')
    def assert_hashes(obj):
        for key in ('source_sha256','record_sha256'):
            for p,h in obj[key].items():assert sha(p)==h,p
    try:
        if not (ROOT/'data_manifest.json').exists():
            run('fetch_data.py',[],'fetch.log','fetching_data');run('prepare_data.py',[],'data_prepare.log','preparing_data')
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed']
        if not (ROOT/'text_provenance.json').exists():run('prepare_text.py',[],'text_prepare.log','preparing_text')
        assert json.loads((ROOT/'text_provenance.json').read_text())['passed']
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        done=ROOT/'smoke/seed42/DONE.json'
        if not done.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():run('train.py',['--phase','smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--phase','smoke'],'smoke.log','smoke_resume')
        smoke=json.loads(done.read_text());assert smoke['passed'] and smoke['resumed'] and smoke['updates_per_model']==2
        if not (ROOT/'smoke_analysis/validation.json').exists():run('evaluate.py',['--smoke'],'smoke_evaluate.log','smoke_native_evaluation')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        records={str(ROOT/f'{n}.json'):sha(ROOT/f'{n}.json') for n in ('data_manifest','text_provenance','preflight')}
        records.update({str(p):sha(p) for p in ASSETS.glob('*') if p.is_file()})
        pilot_provenance=dict(created_utc=stamp(),source_sha256=source_hashes(),record_sha256=records,smoke_resume_and_evaluation_passed=True)
        pp=ROOT/'pilot_provenance.json'
        if pp.exists():pilot_provenance=json.loads(pp.read_text());assert_hashes(pilot_provenance)
        else:dump(pp,pilot_provenance)
        run('train.py',['--phase','pilot'],'pilot.log','pilot_training')
        pilot=json.loads((ROOT/'pilot/seed42/DONE.json').read_text());assert pilot['passed'] and pilot['epochs']==PILOT_EPOCHS
        assert_hashes(pilot_provenance)
        if not (ROOT/'budget_decision.json').exists():
            history=json.loads((ROOT/'pilot/seed42/history.json').read_text());assert len(history)==PILOT_EPOCHS
            checks={}
            for n in NAMES:
                curve=[r['models'][n]['val']['macro']['dice'] for r in history]
                earlier=max(curve[:40]);late=max(curve[40:]);checks[n]=dict(best_epoch=pilot['best'][n]['epoch'],
                    first40_best=earlier,last10_best=late,late_gain=late-earlier,
                    needs_longer=pilot['best'][n]['epoch']>40 or late-earlier>=.002)
            budget=150 if any(v['needs_longer'] for v in checks.values()) else 100
            dump(ROOT/'budget_decision.json',dict(passed=True,formal_epochs=budget,pilot_epochs=PILOT_EPOCHS,checks=checks,
                rule='150 if any best epoch >40 or last10 best minus first40 best >=.002; otherwise100',
                based_on_validation_only=True,formal_seed42_reinitialized=True,pilot_history_sha256=sha(ROOT/'pilot/seed42/history.json')))
        status['formal_epochs']=json.loads((ROOT/'budget_decision.json').read_text())['formal_epochs']
        provenance=dict(created_utc=stamp(),source_sha256=source_hashes(),record_sha256=dict(records,**{str(ROOT/'budget_decision.json'):sha(ROOT/'budget_decision.json')}))
        fp=ROOT/'formal_provenance.json'
        if fp.exists():provenance=json.loads(fp.read_text());assert_hashes(provenance)
        else:dump(fp,provenance)
        for seed in SEEDS:
            run('train.py',['--phase','formal','--seed',str(seed)],f'train_seed{seed}.log',f'formal_training_seed{seed}')
            done=json.loads((ROOT/f'results/seed{seed}/DONE.json').read_text());assert done['passed'] and done['epochs']==status['formal_epochs']
            assert_hashes(provenance)
        run('evaluate.py',[],'evaluate.log','evaluating_all_locked_checkpoints')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed'];assert_hashes(provenance)
        best={str(p):sha(p) for p in (ROOT/'results').glob('seed*/*/best_dice.pth')};assert len(best)==9
        removed=[]
        for p in list((ROOT/'smoke').rglob('*.pth'))+list((ROOT/'results').glob('seed*/latest.pth'))+[ROOT/'pilot/seed42/latest.pth']:
            if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
        assert all(sha(p)==h for p,h in best.items())
        dump(ROOT/'retention.json',dict(retained_formal_best=best,removed=removed,pilot_best_retained=True))
        dump(ROOT/'validation.json',dict(passed=True,seeds=list(SEEDS),models=list(NAMES),all_nine_formal_runs_complete=True,
            original_resolution_test_evaluation_passed=True,source_and_data_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except InterruptedError as e:status.update(state='stopped_by_user',reason=str(e),ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)

if __name__=='__main__':main()
