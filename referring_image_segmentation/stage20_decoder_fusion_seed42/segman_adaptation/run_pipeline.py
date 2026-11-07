"""Queue the independently smoke-tested adapted SegMAN experiment."""
from segman_model import *
import os
import subprocess
import traceback

LOCAL=Path(__file__).resolve().parent


def status(phase,**extra):
    dump(LOCAL/'run_status.json',dict(phase=phase,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))


def main():
    try:
        pre=json.loads((LOCAL/'preflight.json').read_text())
        cap=json.loads((LOCAL/'capacity_preflight.json').read_text())
        smoke=json.loads((LOCAL/'smoke/DONE.json').read_text())
        resume=json.loads((LOCAL/'smoke/resume_audit.json').read_text())
        assert pre['passed'] and cap['passed'] and cap['train_batch']==64
        assert smoke['smoke_only'] and smoke['epochs']==2 and smoke['resumed']
        assert smoke['encoder_unchanged'] and smoke['updates']==60
        assert resume['optimizer_scheduler_and_rng_restored'] and resume['checkpoint_epoch']==1
        status('waiting_for_axis_pipeline',reason='avoid GPU contention during its training, evaluation and timing')
        while True:
            parent=json.loads((ROOT/'run_status.json').read_text())
            if parent['phase']=='complete':
                break
            if parent['phase']=='failed':
                raise RuntimeError('AXIS pipeline failed; fix it before starting the SegMAN comparison')
            try:
                os.kill(parent['pid'],0)
            except ProcessLookupError as exc:
                raise RuntimeError('AXIS pipeline stopped without completion') from exc
            time.sleep(20)
        for phase,script in [('training','train_segman.py'),('evaluation','evaluate_segman.py'),
                             ('benchmark','benchmark_segman.py'),('analysis','analyze_segman.py')]:
            status(phase,script=script)
            print('PIPELINE',phase,flush=True)
            subprocess.run([sys.executable,'-u',str(LOCAL/script)],cwd=LOCAL,check=True)
        status('complete',epochs=300,seed=42,analysis='analysis/REPORT.md')
        print('PIPELINE COMPLETE',flush=True)
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise


if __name__=='__main__':main()
