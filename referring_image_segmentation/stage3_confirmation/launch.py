"""Supervise three independent paired seed runs, then complete all evaluation."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def stamp():return datetime.now(timezone.utc).isoformat()


def save_status(value):
    p=ROOT/'run_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    lock=(ROOT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'preflight.json').read_text())['passed']
    smoke=json.loads((ROOT/'smoke/seed43/DONE.json').read_text())
    assert smoke['smoke_only'] and smoke['epochs']==2 and smoke['resumed'] and smoke['encoder_unchanged'] and smoke['gradient_checks_passed']
    history=json.loads((ROOT/'smoke/seed43/history.json').read_text())
    peak=max(r['peak_gpu_gib'] for r in history)
    assert peak*3<40,'Insufficient reserved GPU headroom for three pairs'
    sources=list(ROOT.glob('*.py'))+[ROOT/'PROTOCOL.md',ROOT.parent/'evaluation.py',ROOT.parent/'EVALUATION.md',
        ROOT.parent/'stage1_visual/models.py',ROOT.parent/'stage1_visual/data.py',ROOT.parent/'stage1_visual/analyze.py',ROOT.parent/'stage2_decoder/models.py']
    historical=[ROOT.parent/'stage1_visual/analysis/summary.json',ROOT.parent/'stage2_decoder/analysis/summary.json']
    for stage in ('stage1_visual','stage2_decoder'):
        for model in ('dinov3','vmamba'):
            historical.extend([ROOT.parent/stage/'results'/model/'best.pth',ROOT.parent/stage/'results'/model/'DONE.json'])
    provenance=dict(created_utc=stamp(),source_sha256={str(p):sha256(p) for p in sources},
        historical_sha256={str(p):sha256(p) for p in historical},preflight_passed=True,resume_smoke_passed=True,
        paired_smoke_peak_gib=peak,concurrent_seed_processes=3)
    provenance_path=ROOT/'launch_provenance.json'
    if provenance_path.exists():
        prior=json.loads(provenance_path.read_text());assert prior['source_sha256']==provenance['source_sha256']
    else:provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    status=dict(state='running',stage='decoder_multiseed_confirmation',seeds=[43,44,45],epochs=300,models=['light','swin'],started_utc=stamp(),processes={})
    processes={};logs={}
    for seed in (43,44,45):
        if (ROOT/'results'/f'seed{seed}'/'DONE.json').exists():
            status['processes'][str(seed)]=dict(state='complete_preexisting');continue
        env=dict(os.environ,PYTHONHASHSEED=str(seed),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        logs[seed]=(ROOT/f'seed{seed}.log').open('a')
        processes[seed]=subprocess.Popen([sys.executable,'-u',str(ROOT/'train.py'),'--seed',str(seed)],cwd=ROOT,env=env,stdout=logs[seed],stderr=subprocess.STDOUT)
        status['processes'][str(seed)]=dict(state='running',pid=processes[seed].pid)
    save_status(status)
    while processes:
        for seed,p in list(processes.items()):
            code=p.poll()
            if code is not None:
                success=code==0 and (ROOT/'results'/f'seed{seed}'/'DONE.json').exists()
                status['processes'][str(seed)].update(state='complete' if success else 'failed',returncode=code,ended_utc=stamp())
                logs[seed].close();del processes[seed];save_status(status)
        if processes:time.sleep(5)
    if all(p['state'].startswith('complete') for p in status['processes'].values()):
        status['state']='analyzing';save_status(status)
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (ROOT/'analysis.log').open('a') as log:
            code=subprocess.call([sys.executable,'-u',str(ROOT/'analyze.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        valid=(ROOT/'validation.json').exists() and json.loads((ROOT/'validation.json').read_text()).get('passed',False)
        status.update(state='complete' if code==0 and valid else 'analysis_failed',analysis_returncode=code)
    else:status['state']='training_failed'
    status['ended_utc']=stamp();save_status(status)


if __name__=='__main__':main()
