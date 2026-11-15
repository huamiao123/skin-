from pathlib import Path
import subprocess,sys,json,datetime
ROOT=Path(__file__).resolve().parent
def main():
    for label,args in [('smoke',['--smoke']),('formal',[])]:
        with (ROOT/(label+'.log')).open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'run.py')]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'pipeline_failure.json').write_text(json.dumps(dict(error=repr(e),time=datetime.datetime.now(datetime.timezone.utc).isoformat())))
        raise
