from pathlib import Path
import subprocess,sys,json,os,time,hashlib
ROOT=Path(__file__).resolve().parent
def run(args,name):
 with (ROOT/(name+'.log')).open('a',buffering=1) as stream:
  subprocess.run([sys.executable,'-B','-u',str(ROOT/'train.py'),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
def main():
 smoke=ROOT/'smoke'
 if not (smoke/'DONE.json').exists():
  if not (smoke/'latest.pth').exists():run(['--smoke','--stop-after-epoch','1'],'smoke_first')
  before=json.loads((smoke/'history.json').read_text())
  assert len(before)==1 and before[0]['epoch']==1
  checkpoint_hash=hashlib.file_digest((smoke/'latest.pth').open('rb'),'sha256').hexdigest()
  run(['--smoke'],'smoke_resume')
  after=json.loads((smoke/'history.json').read_text())
  preflight=json.loads((smoke/'preflight.json').read_text())
  assert after[0]==before[0] and [x['epoch'] for x in after]==[1,2]
  assert 'RESUMED 2' in (ROOT/'smoke_resume.log').read_text()
  assert preflight['passed'] and (smoke/'DONE.json').exists()
  (smoke/'RESUME_AUDIT.json').write_text(json.dumps(dict(passed=True,epoch1_checkpoint_sha256=checkpoint_hash,first_epoch_history_preserved=True,resumed_at_epoch=2,final_epoch=2,preflight_passed=True,time=time.time()),indent=2))
 assert json.loads((smoke/'RESUME_AUDIT.json').read_text())['passed']
 run([],'formal')
 with (ROOT/'analysis.log').open('a') as stream:
  subprocess.run([sys.executable,'-B','-u',str(ROOT/'analyze.py')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  (ROOT/'pipeline_failure.json').write_text(json.dumps(dict(error=repr(exc),time=time.time(),pid=os.getpid()),indent=2));raise
