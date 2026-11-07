"""Start the full experiment only after an independently resumed BF16 smoke."""
from common import *
import os
import subprocess
import traceback


def status(phase, **extra):
    dump(ROOT/'run_status.json',dict(phase=phase,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))


def main():
    try:
        pre=json.loads((ROOT/'preflight.json').read_text())
        cap=json.loads((ROOT/'capacity_preflight.json').read_text())
        smoke=json.loads((ROOT/'smoke/DONE.json').read_text())
        resume=json.loads((ROOT/'smoke/resume_audit.json').read_text())
        config=json.loads((ROOT/'smoke/config.json').read_text())
        assert pre['passed'] and pre['bf16_supported'] and pre['gpu']==torch.cuda.get_device_name()
        assert cap['passed'] and cap['effective_batch']==64 and cap['train_micro_batch']==64
        assert smoke['smoke_only'] and smoke['epochs']==2 and smoke['resumed']
        assert smoke['encoder_unchanged'] and all(x==60 for x in smoke['updates'].values())
        assert resume['optimizer_scheduler_and_rng_restored'] and resume['checkpoint_epoch']==1
        assert config['precision']=='BF16 autocast with FP32 segmentation loss'
        for phase,script in [('training','train.py'),('evaluation','evaluate.py'),
                             ('benchmark','benchmark.py'),('analysis','analyze.py')]:
            status(phase,script=script)
            print('PIPELINE',phase,flush=True)
            subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
        status('complete',epochs=300,seed=42,analysis='analysis/REPORT.md')
        print('PIPELINE COMPLETE',flush=True)
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise


if __name__=='__main__':main()
