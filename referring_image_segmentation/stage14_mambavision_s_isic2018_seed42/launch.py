"""Single-run supervisor; complete only after all independent final audits pass."""
from common import *
import subprocess,fcntl,signal
from datetime import datetime,timezone

def utc():return datetime.now(timezone.utc).isoformat()

def freeze_sources():
    path=ROOT/'launch_provenance.json'
    if path.exists():verify_frozen_sources();return
    names=['stage1_visual/models.py','stage1_visual/data.py','stage1_visual/analyze.py','stage1_visual/data_manifest.json',
        'stage2_decoder/models.py','stage3_confirmation/models.py','stage3_confirmation/data.py','stage3_confirmation/engine.py',
        'stage4_fusion/models.py','stage4_fusion/PROTOCOL.md','stage4_fusion/preflight.json','stage4_fusion/analysis/REPORT.md',
        'stage4_fusion/analysis/summary.json','stage4_fusion/analysis/per_image_best_dice.json','EVALUATION.md','evaluation.py']
    new=list(ROOT.glob('*.py'))+[ROOT/'config.json',ROOT/'PROTOCOL.md',ROOT/'requirements.txt',ROOT/'preflight.json',ROOT/'repository_inspection.json']
    official=[p for p in (ROOT/'official').rglob('*') if p.is_file()]
    digest=dict(created_utc=utc(),historical_sha256={str(PROJECT/n):sha(PROJECT/n) for n in names},
        new_source_sha256={str(p):sha(p) for p in new},official_sha256={str(p):sha(p) for p in official},
        dataset_manifest_sha256=sha(PROJECT/'stage1_visual/data_manifest.json'),
        formal_models=1,seeds=[42],epochs=300,updates=9000,no_remote_push=True)
    dump(path,digest)

def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='preparing',started_utc=utc(),supervisor_pid=__import__('os').getpid(),seed=42,epochs=300,events=[])
    stopping=[False];child=[None]
    def stop(*_):
        stopping[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(phase,args,log):
        if stopping[0]:raise InterruptedError('Supervisor stopped')
        state.update(state=phase,command=args);dump(ROOT/'run_status.json',state)
        with (ROOT/log).open('a') as output:
            process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            child[0]=process;state['child_pid']=process.pid;dump(ROOT/'run_status.json',state)
            code=process.wait();child[0]=None
        state['events'].append(dict(phase=phase,returncode=code,ended_utc=utc()))
        dump(ROOT/'run_status.json',state)
        if stopping[0]:raise InterruptedError('Stopped; checkpoint preserved')
        if code:raise RuntimeError(f'{phase} exited {code}; see {log}')
    try:
        if not (ROOT/'official/remote_weight_identity.json').exists():run('prepare_official',['prepare_official.py'],'prepare_official.log')
        if not (ROOT/'preflight.json').exists():run('preflight',['preflight.py'],'preflight.log')
        if not (ROOT/'smoke/DONE.json').exists():
            if not (ROOT/'smoke/latest.pth').exists():run('smoke_first_epoch',['train.py','--smoke','--stop-after-epoch','1'],'smoke_first.log')
            run('smoke_resume',['train.py','--smoke'],'smoke_resume.log')
        if not (ROOT/'smoke_analysis/evaluation_validation.json').exists():run('smoke_export',['evaluate.py','--smoke'],'smoke_evaluate.log')
        assert json.loads((ROOT/'smoke_analysis/evaluation_validation.json').read_text())['passed']
        freeze_sources()
        run('training_300',['train.py'],'train.log')
        assert json.loads((ROOT/'results/DONE.json').read_text())['updates']==9000
        run('evaluating_three_selectors',['evaluate.py'],'evaluate.log')
        run('benchmarking_full_model',['benchmark.py'],'benchmark.log')
        run('analyzing',['analyze.py'],'analysis.log')
        run('independent_completion_audit',['validate.py'],'validate.log')
        assert json.loads((ROOT/'validation.json').read_text())['passed'];verify_frozen_sources()
        state.update(state='complete',ended_utc=utc(),validation_passed=True)
    except InterruptedError as exc:state.update(state='stopped',ended_utc=utc(),reason=str(exc))
    except Exception as exc:
        state.update(state='failed',ended_utc=utc(),error=repr(exc));dump(ROOT/'run_status.json',state);raise
    dump(ROOT/'run_status.json',state)

if __name__=='__main__':main()
