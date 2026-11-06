"""Persistent smoke, training, evaluation and analysis supervisor."""
from common import *
import fcntl
import os
import signal
import subprocess
from datetime import datetime,timezone


def utc():return datetime.now(timezone.utc).isoformat()


def freeze_sources():
    fresh=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT/'preflight.json',ROOT/'capacity_preflight.json']
    names=['AGENTS.md','EVALUATION.md','evaluation.py','stage1_visual/models.py','stage1_visual/data.py',
        'stage1_visual/analyze.py','stage1_visual/data_manifest.json','stage2_decoder/models.py',
        'stage3_confirmation/models.py','stage3_confirmation/data.py','stage3_confirmation/engine.py',
        'stage4_fusion/models.py','stage14_mambavision_s_isic2018_seed42/models.py',
        'stage14_mambavision_s_isic2018_seed42/config.json','stage14_mambavision_s_isic2018_seed42/official/source_download.json',
        'stage16_hiram_dual_seed42/models.py','stage16_hiram_dual_seed42/vendor_hiram/HiRAM.py',
        'stage16_hiram_dual_seed42/vendor_hiram/common.py','stage16_hiram_dual_seed42/vendor_hiram/__init__.py',
        'stage16_hiram_dual_seed42/vendor_hiram/LICENSE']
    digest=dict(created_utc=utc(),source_sha256={str(p):sha(p) for p in fresh},
        historical_sha256={str(PROJECT/n):sha(PROJECT/n) for n in names},seed=42,epochs=300,
        models=['paired_baseline','region_only','hiram'],no_github_push=True)
    if (ROOT/'launch_provenance.json').exists():
        existing=json.loads((ROOT/'launch_provenance.json').read_text())
        assert existing['source_sha256']==digest['source_sha256'] and existing['historical_sha256']==digest['historical_sha256'], 'source changed since launch'
    else:
        dump(ROOT/'launch_provenance.json',digest)


def main():
    lock=(ROOT/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='preparing',started_utc=utc(),supervisor_pid=os.getpid(),seed=42,epochs=300,events=[])
    child=[None];stopping=[False]
    def stop(*_):
        stopping[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].terminate()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def run(phase,args,log):
        if stopping[0]:raise InterruptedError('supervisor stopped')
        state.update(state=phase,command=args)
        dump(ROOT/'run_status.json',state)
        with (ROOT/log).open('a') as output:
            p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            child[0]=p;state['child_pid']=p.pid;dump(ROOT/'run_status.json',state)
            code=p.wait();child[0]=None
        state['events'].append(dict(phase=phase,returncode=code,ended_utc=utc()))
        dump(ROOT/'run_status.json',state)
        if stopping[0]:raise InterruptedError('stopped after checkpoint')
        if code:raise RuntimeError(f'{phase} exited {code}; inspect {log}')
    try:
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        assert json.loads((ROOT/'capacity_preflight.json').read_text())['passed']
        if not (ROOT/'smoke/DONE.json').exists():
            if not (ROOT/'smoke/latest.pth').exists():
                run('smoke_first_epoch',['train.py','--smoke','--stop-after-epoch','1'],'smoke.log')
            run('smoke_resume',['train.py','--smoke'],'smoke.log')
        assert json.loads((ROOT/'smoke/resume_audit.json').read_text())['resumed']
        if not (ROOT/'smoke/evaluation.json').exists():run('smoke_evaluation',['evaluate.py','--smoke'],'smoke_evaluate.log')
        freeze_sources()
        run('training',['train.py'],'train.log')
        assert json.loads((ROOT/'results/DONE.json').read_text())['epochs']==300
        run('evaluation',['evaluate.py'],'evaluate.log')
        run('benchmark',['benchmark.py'],'benchmark.log')
        run('analysis',['analyze.py'],'analysis.log')
        run('audit',['validate.py'],'validate.log')
        state.update(state='complete',ended_utc=utc(),validation_passed=True)
    except InterruptedError as e:
        state.update(state='stopped',ended_utc=utc(),reason=str(e))
    except Exception as e:
        state.update(state='failed',ended_utc=utc(),error=repr(e))
        dump(ROOT/'run_status.json',state)
        raise
    dump(ROOT/'run_status.json',state)


if __name__=='__main__':main()
