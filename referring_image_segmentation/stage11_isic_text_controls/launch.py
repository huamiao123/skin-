"""Independent supervisor for one new E0 run and the completed three-group comparison."""
from common import *
import os,subprocess,fcntl,signal
from datetime import datetime,timezone

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:return
    status=dict(state='preparing',stage='isic2016_text_controls',seed=SEED,newly_trained=[NAMES[0]],reused=list(REUSED),
        epochs=EPOCHS,supervisor_pid=os.getpid(),started_utc=stamp(),events=[])
    interrupted=[False];child=[None]
    def stop(*_):
        interrupted[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(script,args,logname,phase):
        if interrupted[0]:raise InterruptedError('stopped by user')
        status.update(state=phase,command=[script]+args);dump(ROOT/'run_status.json',status)
        env=dict(os.environ,PYTHONHASHSEED=str(SEED),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with (ROOT/logname).open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            child[0]=p;status['child_pid']=p.pid;dump(ROOT/'run_status.json',status);code=p.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if interrupted[0]:raise InterruptedError('stopped by user')
        if code:raise RuntimeError(f'{phase} failed ({code}); inspect {logname}')
    try:
        if not (ROOT/'lineage.json').exists():run('prepare.py',[],'prepare.log','preparing_historical_lineage')
        assert json.loads((ROOT/'lineage.json').read_text())['passed']
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight_historical_compatibility')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        done=ROOT/'smoke/seed42/DONE.json'
        if not done.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():run('train.py',['--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--smoke'],'smoke.log','smoke_resume')
        smoke=json.loads(done.read_text());assert smoke['passed'] and smoke['resumed'] and smoke['updates']==4
        if not (ROOT/'smoke_analysis/validation.json').exists():run('evaluate.py',['--smoke'],'smoke_evaluate.log','smoke_native_evaluation')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        paths=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',PROJECT/'evaluation.py',PROJECT/'stage10_mosmed_text/models.py']
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage9_isic_text'):paths+=list((PROJECT/stage).glob('*.py'))
        lineage=json.loads((ROOT/'lineage.json').read_text())
        records={str(ROOT/f'{n}.json'):sha(ROOT/f'{n}.json') for n in ('data_manifest','text_provenance','lineage','preflight')}
        records.update(lineage['historical_frozen_assets'])
        records.update({v['path']:v['sha256'] for v in lineage['reused_checkpoints'].values()})
        provenance=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in paths},record_sha256=records)
        target=ROOT/'launch_provenance.json'
        if target.exists():
            prior=json.loads(target.read_text())
            for k in ('source_sha256','record_sha256'):assert prior[k]==provenance[k],k
        else:dump(target,provenance)
        run('train.py',[],'train.log','training_E0_seed42')
        done=json.loads((ROOT/'results/seed42/DONE.json').read_text());assert done['passed'] and done['epochs']==EPOCHS and done['updates']==5100
        run('evaluate.py',[],'evaluate.log','evaluating_three_locked_groups')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        for k in ('source_sha256','record_sha256'):
            for p,h in provenance[k].items():assert sha(p)==h,p
        best=ROOT/'results/seed42'/NAMES[0]/'best_dice.pth';best_hash=sha(best);removed=[]
        for p in [ROOT/'results/seed42/latest.pth']+list((ROOT/'smoke').rglob('*.pth')):
            if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
        assert sha(best)==best_hash
        dump(ROOT/'retention.json',dict(retained_new_best={str(best):best_hash},removed=removed,historical_checkpoints_not_changed=True))
        dump(ROOT/'validation.json',dict(passed=True,seed=SEED,newly_trained=[NAMES[0]],historical_reused=list(REUSED),
            all_three_groups_available=True,historical_replay_matches=True,source_and_data_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except InterruptedError as e:status.update(state='stopped_by_user',reason=str(e),ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)

if __name__=='__main__':main()
