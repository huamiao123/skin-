from pathlib import Path
import subprocess, sys, json, os, time
ROOT=Path(__file__).resolve().parent

def run(args,name):
    with (ROOT/(name+'.log')).open('a',buffering=1) as stream:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'train.py'),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(name+' failed; see '+str(ROOT/(name+'.log')))

if __name__=='__main__':
    try:
        if not (ROOT/'smoke/DONE.json').exists():
            run(['--smoke','--stop-after-epoch','1'],'smoke_first')
            run(['--smoke'],'smoke_resume')
        run([],'formal')
    except Exception as exc:
        (ROOT/'pipeline_failure.json').write_text(json.dumps(dict(error=str(exc),time=time.time(),pid=os.getpid()),indent=2))
        raise
