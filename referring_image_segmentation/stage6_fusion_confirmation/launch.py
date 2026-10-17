"""Detached, resumable supervisor for six new runs and three audited baselines."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SEEDS=(43,44,45)


def stamp():return datetime.now(timezone.utc).isoformat()


def dump(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text())['passed']:
        print('Already complete',flush=True);return
    status=dict(state='preparing',stage='fusion_seed_confirmation',seeds=list(SEEDS),epochs=300,
        new_training_runs=6,reused_baseline_runs=3,supervisor_pid=os.getpid(),started_utc=stamp(),events=[],completed_seeds=[])
    def run(script,args,logname,phase,seed=43):
        status.update(state=phase,seed=seed,command=[script]+args);dump(ROOT/'run_status.json',status)
        env=dict(os.environ,PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (ROOT/logname).open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)]+args,cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=p.pid;dump(ROOT/'run_status.json',status);code=p.wait()
        status['events'].append(dict(phase=phase,seed=seed,returncode=code,ended_utc=stamp()))
        if code:raise RuntimeError(f'{phase} seed{seed} exited {code}; see {logname}')
    try:
        if not (ROOT/'preflight.json').exists():run('preflight.py',[],'preflight.log','preflight')
        pre=json.loads((ROOT/'preflight.json').read_text());assert pre['passed']
        smoke=ROOT/'smoke/seed43/DONE.json'
        if not smoke.exists():
            if not (ROOT/'smoke/seed43/latest.pth').exists():
                run('train.py',['--seed','43','--smoke','--stop-after-epoch','1'],'smoke.log','smoke_first_epoch')
            run('train.py',['--seed','43','--smoke'],'smoke.log','smoke_resume_and_export')
        done=json.loads(smoke.read_text())
        assert done['smoke_only'] and done['epochs']==2 and done['resumed'] and done['encoders_unchanged'] and done['gradient_checks_passed']
        run('analyze.py',['--seed','43','--smoke','--aggregate'],'smoke_analysis.log','smoke_analysis')
        assert json.loads((ROOT/'smoke_analysis/validation.json').read_text())['passed']
        hist=json.loads((ROOT/'smoke/seed43/history.json').read_text())
        peak=max(r['peak_gpu_gib'] for r in hist);assert peak<pre['total_gpu_gib']*.85
        sources=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT.parent/'evaluation.py',ROOT.parent/'evaluate_runs.py',ROOT.parent/'EVALUATION.md']
        historical=[ROOT.parent/'stage1_visual/data_manifest.json']
        for stage in ('stage1_visual','stage2_decoder','stage3_confirmation','stage4_fusion','stage5_selective_fusion'):
            folder=ROOT.parent/stage;sources.extend(folder.glob('*.py'))
            historical.extend(folder.glob('results/**/config.json'));historical.extend(folder.glob('results/**/DONE.json'))
            historical.extend(folder.glob('analysis/summary.json'));historical.extend(folder.glob('preflight.json'))
        for seed in SEEDS:
            folder=ROOT.parent/f'stage3_confirmation/results/seed{seed}'
            historical.extend(folder.glob('swin/*.pth'));historical.extend(folder.glob('swin/*/*.npy'))
            historical.extend(folder.glob('swin/*/*.json'));historical.append(folder/'history.json')
        prov=dict(created_utc=stamp(),source_sha256={str(p):sha(p) for p in sources},
            historical_sha256={str(p):sha(p) for p in historical},preflight_sha256=sha(ROOT/'preflight.json'),
            reused_baselines=pre['reused_baselines'],smoke_resume_and_full_analysis_passed=True,
            concurrent_seed_processes=1,smoke_peak_gpu_gib=peak,smoke_peak_reserved_gib=max(r['peak_reserved_gib'] for r in hist),
            total_gpu_gib=pre['total_gpu_gib'],estimated_training_seconds=900*sum(r['seconds'] for r in hist)/len(hist))
        if (ROOT/'launch_provenance.json').exists():
            old=json.loads((ROOT/'launch_provenance.json').read_text())
            for key in ('source_sha256','historical_sha256','preflight_sha256'):assert old[key]==prov[key],key
        else:dump(ROOT/'launch_provenance.json',prov)
        for seed in SEEDS:
            run('train.py',['--seed',str(seed)],f'train_seed{seed}.log','training',seed)
            done=json.loads((ROOT/f'results/seed{seed}/DONE.json').read_text())
            assert done['epochs']==300 and not done['smoke_only'] and done['encoders_unchanged'] and done['gradient_checks_passed']
            run('analyze.py',['--seed',str(seed)],f'analysis_seed{seed}.log','analyzing_seed',seed)
            assert json.loads((ROOT/f'analysis/seed{seed}/validation.json').read_text())['passed']
            status['completed_seeds'].append(seed)
        run('analyze.py',['--aggregate'],'analysis.log','aggregating',45)
        assert json.loads((ROOT/'analysis/validation.json').read_text())['passed']
        for section in ('source_sha256','historical_sha256'):
            for path,digest in prov[section].items():assert sha(path)==digest,path
        assert sha(ROOT/'preflight.json')==prov['preflight_sha256']
        dump(ROOT/'validation.json',dict(passed=True,seeds=list(SEEDS),epochs_per_run=300,
            new_training_runs=6,reused_baseline_runs=3,full_metrics_checkpoints=27,
            all_training_and_analysis_complete=True,original_artifacts_unchanged=True))
        status.update(state='complete',ended_utc=stamp())
    except Exception as e:
        status.update(state='failed',error=repr(e),ended_utc=stamp());dump(ROOT/'run_status.json',status);raise
    dump(ROOT/'run_status.json',status)


if __name__=='__main__':main()
