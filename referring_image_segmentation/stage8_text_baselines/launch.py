"""Detached resumable supervisor for dataset, text, engineering checks and RIS training."""
from common import *
import fcntl,os,subprocess,time
from datetime import datetime,timezone

def stamp():return datetime.now(timezone.utc).isoformat()

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:return
    status=dict(state='preparing',stage='refcoco_text_baselines',seed=42,epochs=30,models=list(NAMES),supervisor_pid=os.getpid(),started_utc=stamp(),events=[])
    def run(script,args,logname,phase):
        status.update(state=phase,command=[script]+args);dump(ROOT/'run_status.json',status)
        env=dict(os.environ,PYTHONHASHSEED='42',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with (ROOT/logname).open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=proc.pid;dump(ROOT/'run_status.json',status);code=proc.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if code:raise RuntimeError(f'{phase} failed ({code}); inspect {logname}')
    try:
        if not (ROOT/'data_manifest.json').exists():run('prepare_data.py',[],'data_prepare.log','preparing_data')
        assert json.loads((ROOT/'data_manifest.json').read_text())['passed']
        if not (ROOT/'text_provenance.json').exists():run('prepare_text.py',[],'text_prepare.log','preparing_text')
        assert json.loads((ROOT/'text_provenance.json').read_text())['passed']
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        smoke=ROOT/'smoke/seed42/DONE.json'
        if not smoke.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():run('train.py',['--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--smoke'],'smoke.log','smoke_resume')
        done=json.loads(smoke.read_text());assert done['passed'] and done['smoke_only'] and done['epochs']==2 and done['resumed']
        run('evaluate.py',['--smoke'],'smoke_evaluate.log','smoke_evaluation')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        sources=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',PROJECT/'evaluation.py',PROJECT/'stage1_visual/analyze.py']
        history=[]
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage4_fusion','stage5_selective_fusion','stage6_fusion_confirmation','stage7_cross_attention'):
            folder=PROJECT/stage;sources.extend(folder.glob('*.py'));history.extend(folder.glob('analysis/summary.json'))
            history.extend(folder.glob('results/**/config.json'));history.extend(folder.glob('results/**/DONE.json'))
        provenance=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in sources},historical_sha256={str(p):sha(p) for p in history},
            data_manifest_sha256=sha(ROOT/'data_manifest.json'),text_provenance_sha256=sha(ROOT/'text_provenance.json'),
            preflight_sha256=sha(ROOT/'preflight.json'),resume_and_native_evaluation_smoke_passed=True)
        if (ROOT/'launch_provenance.json').exists():
            old=json.loads((ROOT/'launch_provenance.json').read_text())
            for k in ('source_sha256','historical_sha256','data_manifest_sha256','text_provenance_sha256','preflight_sha256'):assert old[k]==provenance[k],k
        else:dump(ROOT/'launch_provenance.json',provenance)
        run('train.py',[],'train.log','training')
        done=json.loads((ROOT/'results/seed42/DONE.json').read_text())
        assert done['passed'] and done['epochs']==30 and done['updates_per_model']==226170 and not done['smoke_only']
        run('evaluate.py',[],'evaluate.log','evaluating_locked_checkpoints')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        for section in ('source_sha256','historical_sha256'):
            for path,digest in provenance[section].items():assert sha(path)==digest,path
        for name in ('data_manifest','text_provenance','preflight'):assert sha(ROOT/f'{name}.json')==provenance[f'{name}_sha256']
        dump(ROOT/'validation.json',dict(passed=True,seed=42,epochs=30,main_models=3,fixed_text_controls=1,
            native_evaluation_passed=True,historical_sources_and_records_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)

if __name__=='__main__':main()
