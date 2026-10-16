"""Run the two independent probes concurrently, then export paired diagnostics."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def write_status(value):
    p=ROOT/'run_status.json'
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def main():
    env=dict(os.environ,PYTHONHASHSEED='42',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    status=dict(state='running',stage='frozen_visual_probes',seed=42,epochs=300,
                started_utc=datetime.now(timezone.utc).isoformat(),processes={})
    processes={};logs={}
    for name in ['dinov3','vmamba']:
        if (ROOT/'results'/name/'DONE.json').exists():
            status['processes'][name]=dict(state='complete_preexisting');continue
        logs[name]=(ROOT/f'{name}.log').open('a')
        processes[name]=subprocess.Popen([sys.executable,'-u','train.py',name],cwd=ROOT,env=env,
                                        stdout=logs[name],stderr=subprocess.STDOUT)
        status['processes'][name]=dict(pid=processes[name].pid,state='running')
    write_status(status)
    while processes:
        for name,p in list(processes.items()):
            code=p.poll()
            if code is not None:
                ok=code==0 and (ROOT/'results'/name/'DONE.json').exists()
                status['processes'][name].update(state='complete' if ok else 'failed',returncode=code,
                              ended_utc=datetime.now(timezone.utc).isoformat())
                logs[name].close();del processes[name];write_status(status)
        if processes:time.sleep(5)
    if all(x['state'].startswith('complete') for x in status['processes'].values()):
        status['state']='analyzing';write_status(status)
        with (ROOT/'analysis.log').open('a') as log:
            code=subprocess.call([sys.executable,'-u','analyze.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(state='complete' if code==0 else 'analysis_failed',analysis_returncode=code)
    else:status['state']='training_failed'
    status['ended_utc']=datetime.now(timezone.utc).isoformat();write_status(status)


if __name__=='__main__':main()
