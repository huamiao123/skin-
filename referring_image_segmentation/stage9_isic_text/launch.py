"""Detached supervisor: preflight, restart smoke, frozen protocol, train, evaluate."""
from common import *
import os,subprocess,fcntl,signal
from datetime import datetime,timezone

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:return
    status=dict(state='preparing',stage='isic2016_text_baselines',seed=SEED,epochs=EPOCHS,models=list(NAMES),
        supervisor_pid=os.getpid(),started_utc=stamp(),events=[])
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
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            child[0]=proc;status['child_pid']=proc.pid;dump(ROOT/'run_status.json',status);code=proc.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if interrupted[0]:raise InterruptedError('stopped by user')
        if code:raise RuntimeError(f'{phase} failed ({code}); inspect {logname}')
    try:
        if not (ROOT/'data_manifest.json').exists():run('prepare_data.py',[],'data_prepare.log','preparing_data')
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed']
        if not (ROOT/'text_provenance.json').exists():run('prepare_text.py',[],'text_prepare.log','preparing_text')
        assert json.loads((ROOT/'text_provenance.json').read_text())['passed']
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        done_path=ROOT/'smoke/seed42/DONE.json'
        if not done_path.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():run('train.py',['--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--smoke'],'smoke.log','smoke_resume')
        smoke=json.loads(done_path.read_text());assert smoke['passed'] and smoke['resumed'] and smoke['epochs']==2 and smoke['updates_per_model']==4
        if not (ROOT/'smoke_analysis/validation.json').exists():run('evaluate.py',['--smoke'],'smoke_evaluate.log','smoke_evaluation')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        sources=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',PROJECT/'evaluation.py']
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage4_fusion'):
            sources+=list((PROJECT/stage).glob('*.py'))
        provenance=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in sources},
            records_sha256={str(ROOT/f'{n}.json'):sha(ROOT/f'{n}.json') for n in ('data_manifest','text_provenance','preflight')},
            data_files_sha256={str(p):sha(p) for p in ASSETS.glob('*') if p.is_file()},smoke_restart_and_native_eval_passed=True)
        if (ROOT/'launch_provenance.json').exists():
            old=json.loads((ROOT/'launch_provenance.json').read_text())
            for k in ('source_sha256','records_sha256','data_files_sha256'):assert old[k]==provenance[k],k
        else:dump(ROOT/'launch_provenance.json',provenance)
        run('train.py',[],'train.log','training')
        done=json.loads((ROOT/'results/seed42/DONE.json').read_text());assert done['passed'] and done['epochs']==EPOCHS and not done['smoke_only']
        run('evaluate.py',[],'evaluate.log','evaluating_locked_checkpoints')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        for section in ('source_sha256','records_sha256','data_files_sha256'):
            for p,digest in provenance[section].items():assert sha(p)==digest,p
        dump(ROOT/'validation.json',dict(passed=True,seed=SEED,epochs=EPOCHS,main_models=3,fixed_text_controls=1,
            native_evaluation_passed=True,sources_and_data_unchanged=True))
        best={str(p):sha(p) for p in (ROOT/'results/seed42').glob('*/best_dice.pth')};assert len(best)==4
        removed=[]
        paths=[ROOT/'results/seed42/latest.pth']+list((ROOT/'smoke').rglob('*.pth'))
        for p in paths:
            if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
        assert all(sha(p)==h for p,h in best.items())
        dump(ROOT/'retention.json',dict(retained_best=best,removed=removed,after_successful_evaluation=True))
        status.update(state='complete',ended_utc=stamp())
    except InterruptedError as e:status.update(state='stopped_by_user',reason=str(e),ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)

if __name__=='__main__':main()
