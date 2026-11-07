"""Smoke, verify resume, then train and evaluate the one HGM adaptation."""
from common import *
import os
import subprocess
import traceback


def status(phase, **extra):
    dump(ROOT/'run_status.json', dict(phase=phase, pid=os.getpid(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra))


def call(*args):
    print('PIPELINE', ' '.join(args), flush=True)
    subprocess.run([sys.executable, '-u', *args], cwd=ROOT, check=True)


def main():
    try:
        pre = json.loads((ROOT/'preflight.json').read_text())
        cap = json.loads((ROOT/'capacity_preflight.json').read_text())
        assert pre['passed'] and cap['passed'] and cap['train_batch'] == 64
        smoke = ROOT/'smoke'
        if not (smoke/'DONE.json').exists():
            if not (smoke/'latest.pth').exists():
                status('smoke_epoch_1')
                call('train.py', '--smoke', '--stop-after-epoch', '1')
            status('smoke_resume_epoch_2')
            call('train.py', '--smoke')
        done = json.loads((smoke/'DONE.json').read_text())
        resume = json.loads((smoke/'resume_audit.json').read_text())
        assert done['smoke_only'] and done['epochs']==2 and done['updates']==60
        assert done['resumed'] and done['encoder_unchanged']
        assert resume['checkpoint_epoch']==1 and resume['optimizer_scheduler_and_rng_restored']
        for phase, script in [('training','train.py'), ('evaluation','evaluate.py'),
                              ('benchmark','benchmark.py'), ('analysis','analyze.py')]:
            status(phase, script=script)
            call(script)
        status('complete', seed=42, epochs=300, report='analysis/REPORT.md')
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
