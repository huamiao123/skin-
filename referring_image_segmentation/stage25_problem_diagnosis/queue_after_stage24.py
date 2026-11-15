"""Wait for Stage24 to finish and exit, then run its CPU-only diagnostics."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'stage24_seed43_confirmation'
OUTPUT=ROOT/'queue_status.json'


def save(state, **kwargs):
    tmp=OUTPUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        next_action='CPU-only saved-prediction diagnosis; no new training',
        test_evaluated=False,**kwargs),indent=2)+'\n')
    tmp.replace(OUTPUT)


def active_training():
    status=PRIOR/'results/run_status.json'
    if not status.exists():
        return False
    pid=json.loads(status.read_text()).get('pid')
    command=Path('/proc')/str(pid)/'cmdline'
    return command.exists() and 'stage24_seed43_confirmation/train.py' in command.read_bytes().decode(errors='replace')


def main():
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'results/stage24_seed43_confirmation/DONE.json').exists():
            save('already_complete')
            return
        frozen_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('diagnose.py','PROTOCOL.txt','queue_after_stage24.py')}
        deadline=time.time()+6*3600
        while True:
            failures=[PRIOR/'results/failure.json',PRIOR/'pipeline_failure.json']
            for failure in failures:
                if failure.exists():
                    raise RuntimeError('Prior stage failed: '+str(failure))
            done=(PRIOR/'results/DONE.json').exists()
            active=active_training()
            if done and not active:
                break
            if time.time()>deadline:
                raise TimeoutError('Six-hour wait for Stage24 exceeded')
            if not active and not done:
                raise RuntimeError('Stage24 process stopped without DONE.json')
            save('waiting_for_stage24',prior_done=done,prior_training_active=active,
                 source_sha256=frozen_hashes)
            time.sleep(15)
        assert frozen_hashes=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in frozen_hashes}, 'Queued source changed'
        save('running_diagnosis',prior_done=True,prior_training_active=False,
             source_sha256=frozen_hashes)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (ROOT/'stage24_diagnosis.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'diagnose.py'),'--stages',
                            'stage24_seed43_confirmation'],cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        save('complete',source_sha256=frozen_hashes)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        save('failed',error=repr(error),traceback=traceback.format_exc())
        raise
