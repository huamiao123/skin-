"""Detached supervisor for cross-attention, matched convolution and retrieval audits."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def stamp():return datetime.now(timezone.utc).isoformat()


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:
        print('Already complete',flush=True);return
    status=dict(state='preparing',stage='window_cross_attention',seed=42,epochs=300,
        supervisor_pid=os.getpid(),started_utc=stamp(),events=[])
    env=dict(os.environ,PYTHONHASHSEED='42',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def run(script,args,logname,phase):
        status.update(state=phase,command=[script]+args);dump(ROOT/'run_status.json',status)
        with (ROOT/logname).open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=p.pid;dump(ROOT/'run_status.json',status)
            code=p.wait()
        status['events'].append(dict(phase=phase,returncode=code,ended_utc=stamp()))
        if code:raise RuntimeError(f'{phase} exited with code {code}; see {logname}')
    try:
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        smoke_path=ROOT/'smoke/seed42/DONE.json'
        if not smoke_path.exists():
            if not (ROOT/'smoke/seed42/latest.pth').exists():
                run('train.py',['--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--smoke'],'smoke.log','smoke_resume_and_export')
        smoke=json.loads(smoke_path.read_text())
        assert smoke['smoke_only'] and smoke['epochs']==2 and smoke['resumed']
        assert smoke['encoders_unchanged'] and smoke['gradient_checks_passed']
        run('analyze.py',['--smoke'],'smoke_analysis.log','smoke_analysis')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        run('attention_audit.py',['--smoke'],'smoke_attention_audit.log','smoke_attention_audit')
        assert json.loads((ROOT/'smoke_analysis/attention_audit/summary.json').read_text())['passed']
        history=json.loads((ROOT/'smoke/seed42/history.json').read_text())
        peak=max(h['peak_gpu_gib'] for h in history)
        total=json.loads((ROOT/'preflight.json').read_text())['total_gpu_gib']
        assert peak<total*.85,'Insufficient GPU memory headroom'
        sources=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT.parent/'evaluation.py',ROOT.parent/'evaluate_runs.py',ROOT.parent/'EVALUATION.md']
        historical=[]
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage4_fusion','stage5_selective_fusion','stage6_fusion_confirmation'):
            folder=ROOT.parent/stage
            sources.extend(folder.glob('*.py'))
            historical.extend(folder.glob('analysis/*.json'))
            historical.extend(folder.glob('results/**/*.pth'))
            historical.extend(folder.glob('results/**/config.json'))
            historical.extend(folder.glob('results/**/DONE.json'))
        for stage in ('stage4_fusion','stage5_selective_fusion'):
            historical.extend((ROOT.parent/stage/'results/seed42').glob('**/probabilities.npy'))
        provenance=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in sources},
            historical_sha256={str(p):sha(p) for p in historical},preflight_sha256=sha(ROOT/'preflight.json'),
            smoke_resume_and_analysis_passed=True,smoke_peak_gpu_gib=peak,available_gpu_gib=total,
            estimated_training_seconds=300*sum(h['seconds'] for h in history)/len(history))
        if (ROOT/'launch_provenance.json').exists():
            prior=json.loads((ROOT/'launch_provenance.json').read_text())
            assert prior['source_sha256']==provenance['source_sha256']
            assert prior['historical_sha256']==provenance['historical_sha256']
            assert prior['preflight_sha256']==provenance['preflight_sha256']
        else:dump(ROOT/'launch_provenance.json',provenance)
        run('train.py',[],'train.log','training')
        done=json.loads((ROOT/'results/seed42/DONE.json').read_text())
        assert done['epochs']==300 and not done['smoke_only'] and done['encoders_unchanged']
        run('analyze.py',[],'analysis.log','analyzing')
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        run('attention_audit.py',[],'attention_audit.log','attention_audit')
        assert json.loads((ROOT/'analysis/attention_audit/summary.json').read_text())['passed']
        for section in ('source_sha256','historical_sha256'):
            for path,digest in provenance[section].items():assert sha(Path(path))==digest,path
        assert sha(ROOT/'preflight.json')==provenance['preflight_sha256']
        dump(ROOT/'validation.json',dict(passed=True,epochs=300,all_two_models_complete=True,
            full_metrics_passed=True,attention_audit_passed=True,original_artifacts_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp())
        dump(ROOT/'run_status.json',status)
        raise
    dump(ROOT/'run_status.json',status)


if __name__=='__main__':main()

