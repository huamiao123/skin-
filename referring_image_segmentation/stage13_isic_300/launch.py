"""Detached supervisor for five fresh 300-epoch runs and a complete locked report."""
from common import *
import os,subprocess,fcntl,signal
from datetime import datetime,timezone

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:return
    status=dict(state='preparing',stage='isic2016_five_300',seed=SEED,epochs=EPOCHS,models=list(NAMES),
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
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            child[0]=p;status['child_pid']=p.pid;dump(ROOT/'run_status.json',status);code=p.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if interrupted[0]:raise InterruptedError('stopped by user')
        if code:raise RuntimeError(f'{phase} failed ({code}); inspect {logname}')
    try:
        if not (ROOT/'lineage.json').exists():run('prepare.py',[],'prepare.log','preparing_lineage')
        assert json.loads((ROOT/'lineage.json').read_text())['passed']
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        done=ROOT/'smoke/seed42/DONE.json'
        if not done.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():run('train.py',['--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--smoke'],'smoke.log','smoke_resume')
        smoke=json.loads(done.read_text());assert smoke['passed'] and smoke['resumed'] and smoke['updates_per_model']==4
        if not (ROOT/'smoke_analysis/validation.json').exists():run('evaluate.py',['--smoke'],'smoke_evaluate.log','smoke_native_evaluation')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        paths=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT/'analysis/finalize_analysis.py',PROJECT/'evaluation.py',PROJECT/'stage10_mosmed_text/models.py']
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage9_isic_text','stage12_predecoder_text'):paths+=list((PROJECT/stage).glob('*.py'))
        lineage=json.loads((ROOT/'lineage.json').read_text())
        records={str(ROOT/f'{n}.json'):sha(ROOT/f'{n}.json') for n in ('data_manifest','text_provenance','token_provenance','lineage','preflight')}
        records.update(lineage['assets_sha256']);records.update(lineage['token_assets_sha256'])
        records[str(EARLY/'analysis/summary.json')]=lineage['previous_100_epoch_report_sha256']
        for n in NAMES:
            folder=EARLY if n in PRE_NAMES else CONTROLS
            p=folder/'analysis'/n/'test_per_image.json';records[str(p)]=sha(p)
        provenance=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in paths},record_sha256=records)
        target=ROOT/'launch_provenance.json'
        if target.exists():
            old=json.loads(target.read_text())
            for k in ('source_sha256','record_sha256'):assert old[k]==provenance[k],k
        else:dump(target,provenance)
        run('train.py',[],'train.log','training_five_models_300')
        done=json.loads((ROOT/'results/seed42/DONE.json').read_text());assert done['passed'] and done['epochs']==300 and done['updates_per_model']==15300
        run('evaluate.py',[],'evaluate.log','evaluating_locked_300_models')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        run('analysis/finalize_analysis.py',[],'analysis.log','independent_report_and_figures')
        assert json.loads((ROOT/'analysis/independent_report_check.json').read_text())['passed']
        for k in ('source_sha256','record_sha256'):
            for p,h in provenance[k].items():assert sha(p)==h,p
        best={str(p):sha(p) for p in (ROOT/'results/seed42').glob('*/best_dice.pth')};assert len(best)==5
        removed=[]
        for p in [ROOT/'results/seed42/latest.pth']+list((ROOT/'smoke').rglob('*.pth')):
            if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
        assert all(sha(p)==h for p,h in best.items())
        dump(ROOT/'retention.json',dict(retained_new_best=best,removed=removed,historical_files_unchanged=True))
        dump(ROOT/'validation.json',dict(passed=True,seed=SEED,epochs=EPOCHS,models=list(NAMES),all_five_fresh_runs_complete=True,
            original_resolution_evaluation_passed=True,independent_report_check_passed=True,sources_and_data_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except InterruptedError as e:status.update(state='stopped_by_user',reason=str(e),ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)

if __name__=='__main__':main()
