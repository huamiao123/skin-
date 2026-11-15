"""Resume-tested smoke, formal paired training, routing audit and analysis."""
from common import *
import os
import subprocess
import traceback


def status(phase,**extra):
    dump(ROOT/'run_status.json',dict(phase=phase,pid=os.getpid(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))


def call(*args):
    print('PIPELINE',' '.join(args),flush=True)
    subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)


def main():
    try:
        assert json.loads((ROOT/'preflight.json').read_text())['passed']
        assert json.loads((ROOT/'capacity_preflight.json').read_text())['passed']
        smoke=ROOT/'smoke'
        if not (smoke/'DONE.json').exists():
            if not (smoke/'latest.pth').exists():
                status('smoke_epoch_1');call('train.py','--smoke','--stop-after-epoch','1')
            status('smoke_resume_epoch_2');call('train.py','--smoke')
        d=json.loads((smoke/'DONE.json').read_text())
        resume=json.loads((smoke/'resume_audit.json').read_text())
        assert d['smoke_only'] and d['epochs']==2 and d['resumed'] and d['encoder_unchanged']
        assert all(v==60 for v in d['updates'].values())
        assert resume['checkpoint_epoch']==1 and resume['optimizer_scheduler_and_rng_restored']
        for phase,script in [('training','train.py'),('evaluation','evaluate.py'),
                             ('routing_audit','route_audit.py'),('benchmark','benchmark.py'),
                             ('analysis','analyze.py'),('validation','validate.py')]:
            status(phase,script=script);call(script)
        status('complete',seed=42,epochs=300,report='analysis/REPORT.md')
        print('PIPELINE COMPLETE',flush=True)
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise


if __name__=='__main__':main()
